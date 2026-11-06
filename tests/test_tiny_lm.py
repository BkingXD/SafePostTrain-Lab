"""Small CPU invariants for the optional, real-Transformer teaching lab.

Run with ``.venv-teaching/bin/python -m unittest discover -s tests -p
test_tiny_lm.py -v``. The standard-library-only test environment deliberately
skips this module's tests when the optional torch dependency is absent.
"""
import copy
from pathlib import Path
import sys
import tempfile
import unittest


try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    torch = None

if torch is not None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "labs"))
    from tiny_lm.data import assess, brief, datasets, final_probe, record, verbose
    from tiny_lm.model import (
        ModelConfig, PAD, TinyLM, apply_lora, checkpoint, collate, encode,
        load_checkpoint, parameter_counts, sft_loss, token_statistics,
    )
    from tiny_lm.training import frozen, gae_tensor, preference_loss, train_preference


@unittest.skipIf(torch is None, "optional PyTorch lab: use .venv-teaching/bin/python")
class TinyLMInvariants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(47)
        self.model = TinyLM(ModelConfig(width=16, heads=4, layers=1, max_length=24))
        self.rows = [record(["copy"], "A"), record(["please", "repeat"], "B", "verbose")]

    def assert_tensors_equal(self, actual, expected):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def assert_nested_equal(self, actual, expected):
        if isinstance(expected, torch.Tensor):
            self.assert_tensors_equal(actual, expected)
        elif isinstance(expected, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for key in expected:
                self.assert_nested_equal(actual[key], expected[key])
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(actual), len(expected))
            for a, e in zip(actual, expected):
                self.assert_nested_equal(a, e)
        else:
            self.assertEqual(actual, expected)

    def test_attention_cannot_read_future_tokens_or_their_embeddings(self):
        prefix = self.rows[0]["prompt"]
        original = torch.tensor([encode(prefix + ["B", "<eos>"])])
        changed = torch.tensor([encode(prefix + ["H", "G"])])
        original_logits = self.model(original)
        changed_logits = self.model(changed)
        self.assert_tensors_equal(original_logits[:, :len(prefix)], changed_logits[:, :len(prefix)])
        # Also check the dependency graph, not only two matching forward values.
        changed_logits[0, len(prefix) - 1].square().sum().backward()
        for token in ["H", "G"]:
            grad = self.model.embedding.weight.grad[encode([token])[0]]
            self.assert_tensors_equal(grad, torch.zeros_like(grad))
        self.assertGreater(self.model.embedding.weight.grad.abs().sum().item(), 0)

    def test_sft_shifts_once_and_supervises_only_answer_and_eos(self):
        ids, labels = collate(self.rows)
        stats = token_statistics(self.model, ids, labels)
        logits = self.model(ids[:, :-1])
        expected_logps = []
        for index, row in enumerate(self.rows):
            prompt_length = len(row["prompt"])
            response = encode(row["response"])
            sequence = encode(row["prompt"] + row["response"])
            self.assertEqual(ids[index, :len(sequence)].tolist(), sequence)
            self.assertEqual(labels[index, :prompt_length].tolist(), [-100] * prompt_length)
            self.assertEqual(labels[index, prompt_length:len(sequence)].tolist(), response)
            expected_positions = list(range(prompt_length - 1, len(sequence) - 1))
            self.assertEqual(torch.where(stats["mask"][index])[0].tolist(), expected_positions)
            self.assertEqual(stats["lengths"][index].item(), len(response))
            for position, target in zip(expected_positions, response):
                logp = logits[index, position].log_softmax(-1)[target]
                torch.testing.assert_close(stats["logps"][index, position], logp)
                expected_logps.append(logp)
        # Independent target-position calculation catches either a missing or
        # a double causal shift, and proves EOS is a supervised response token.
        expected_loss = -torch.stack(expected_logps).mean()
        torch.testing.assert_close(sft_loss(self.model, ids, labels), expected_loss)
        self.assert_tensors_equal(stats["logps"][~stats["mask"]], torch.zeros_like(stats["logps"][~stats["mask"]]))

    def test_right_padding_leaves_valid_loss_and_gradients_unchanged(self):
        ids, labels = collate(self.rows)
        longer_ids = torch.nn.functional.pad(ids, (0, 4), value=PAD)
        longer_labels = torch.nn.functional.pad(labels, (0, 4), value=-100)
        padded_model = copy.deepcopy(self.model)
        original_loss = sft_loss(self.model, ids, labels)
        padded_loss = sft_loss(padded_model, longer_ids, longer_labels)
        torch.testing.assert_close(padded_loss, original_loss, atol=1e-6, rtol=1e-6)
        original_loss.backward()
        padded_loss.backward()
        for original, padded in zip(self.model.parameters(), padded_model.parameters()):
            self.assertTrue(torch.isfinite(padded.grad).all().item())
            torch.testing.assert_close(padded.grad, original.grad, atol=1e-6, rtol=1e-5)

    def test_empty_batches_and_empty_supervision_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "empty batch"):
            collate([])
        with self.assertRaisesRegex(ValueError, "supervised token"):
            collate(self.rows[:1], [[]])
        ids, labels = collate(self.rows)
        with self.assertRaisesRegex(ValueError, "empty assistant supervision"):
            sft_loss(self.model, ids, torch.full_like(labels, -100))

    def test_lora_zero_initialization_and_adapter_only_update(self):
        ids, labels = collate(self.rows)
        baseline = self.model(ids).detach()
        apply_lora(self.model, rank=2)
        self.assert_tensors_equal(self.model(ids), baseline)
        trainable = [name for name, param in self.model.named_parameters() if param.requires_grad]
        self.assertTrue(trainable)
        self.assertTrue(all(name.endswith((".a", ".b")) for name in trainable))
        counts = parameter_counts(self.model)
        self.assertLess(counts["trainable"], counts["total"])
        before = {name: param.detach().clone() for name, param in self.model.named_parameters()}
        optimizer = torch.optim.SGD([p for p in self.model.parameters() if p.requires_grad], lr=.1)
        sft_loss(self.model, ids, labels).backward()
        # B starts at zero: dL/dA must be zero on the first step, while B learns.
        for name, param in self.model.named_parameters():
            if name.endswith(".a"):
                self.assert_tensors_equal(param.grad, torch.zeros_like(param.grad))
            elif not param.requires_grad:
                self.assertIsNone(param.grad)
        optimizer.step()
        changed = []
        for name, param in self.model.named_parameters():
            if not torch.equal(param, before[name]):
                changed.append(name)
            if not param.requires_grad:
                self.assert_tensors_equal(param, before[name])
        self.assertTrue(changed)
        self.assertTrue(all(name.endswith(".b") for name in changed))

    def test_dpo_gradient_increases_relative_chosen_log_probability(self):
        chosen = torch.tensor([-2., -3.], dtype=torch.float64, requires_grad=True)
        rejected = torch.tensor([-4., -2.], dtype=torch.float64, requires_grad=True)
        ref_chosen = torch.tensor([-3., -3.], dtype=torch.float64)
        ref_rejected = torch.tensor([-4., -2.], dtype=torch.float64)
        loss = preference_loss({"sums": chosen}, {"sums": rejected},
                               {"sums": ref_chosen}, {"sums": ref_rejected})
        loss.backward()
        margin = (chosen.detach() - rejected.detach()) - (ref_chosen - ref_rejected)
        expected = -.2 * torch.sigmoid(-.2 * margin) / len(chosen)
        torch.testing.assert_close(chosen.grad, expected)
        torch.testing.assert_close(rejected.grad, -expected)
        self.assertTrue((chosen.grad < 0).all().item())
        self.assertTrue((rejected.grad > 0).all().item())

    def test_preference_training_keeps_reference_frozen_and_independent(self):
        reference = frozen(self.model)
        reference_before = copy.deepcopy(reference.state_dict())
        policy_before = copy.deepcopy(self.model.state_dict())
        self.assertFalse(reference.training)
        for policy_param, reference_param in zip(self.model.parameters(), reference.parameters()):
            self.assertNotEqual(policy_param.data_ptr(), reference_param.data_ptr())
            self.assertFalse(reference_param.requires_grad)
        train_preference(self.model, reference, self.rows, steps=1, method="dpo")
        self.assertTrue(any(not torch.equal(value, policy_before[name])
                            for name, value in self.model.state_dict().items()))
        self.assert_nested_equal(reference.state_dict(), reference_before)
        self.assertTrue(all(param.grad is None for param in reference.parameters()))

    def test_gae_terminal_boundary_and_padding_do_not_leak_into_answers(self):
        # The recipe defines EOS and maximum output horizon as terminal. This
        # API does not claim to bootstrap a nonterminal sampling truncation.
        mask = torch.tensor([[False, True, True, True, False],
                             [False, False, True, False, False]])
        values = torch.tensor([[100., .2, .3, .4, -100.],
                               [100., -100., .25, 99., 100.]], dtype=torch.float64)
        rewards = torch.tensor([[50., 0., 0., 1., -50.],
                                [50., -50., .75, 50., -50.]], dtype=torch.float64)
        advantages, returns = gae_tensor(rewards, values, mask, gamma=.9, lam=.95)
        expected = torch.tensor([[0., .559915, .573, .6, 0.],
                                 [0., 0., .5, 0., 0.]], dtype=torch.float64)
        torch.testing.assert_close(advantages, expected, atol=1e-12, rtol=0)
        torch.testing.assert_close(returns[mask], (expected + values)[mask], atol=1e-12, rtol=0)
        # Arbitrary critic estimates/rewards in prompt and PAD positions must
        # affect neither valid advantages nor valid value targets.
        changed_values, changed_rewards = values.clone(), rewards.clone()
        changed_values[~mask] = 1e6
        changed_rewards[~mask] = -1e6
        other_advantages, other_returns = gae_tensor(changed_rewards, changed_values, mask, gamma=.9, lam=.95)
        self.assert_tensors_equal(other_advantages, advantages)
        self.assert_tensors_equal(other_returns[mask], returns[mask])

    def test_gae_lambda_zero_is_one_step_td_and_gamma_zero_is_immediate(self):
        mask = torch.tensor([[False, True, True, True, False]])
        values = torch.tensor([[99., .2, .3, .4, 99.]], dtype=torch.float64)
        rewards = torch.tensor([[99., 0., 0., 1., 99.]], dtype=torch.float64)
        advantages, _ = gae_tensor(rewards, values, mask, gamma=.9, lam=0.)
        torch.testing.assert_close(advantages[mask], torch.tensor([.07, .06, .6], dtype=torch.float64))
        advantages, returns = gae_tensor(rewards, values, mask, gamma=0., lam=.95)
        torch.testing.assert_close(advantages[mask], (rewards - values)[mask])
        torch.testing.assert_close(returns[mask], rewards[mask])

    def test_checkpoint_restores_optimizer_rng_and_identical_next_update(self):
        for lora_rank in [None, 2]:
            with self.subTest(lora_rank=lora_rank), tempfile.TemporaryDirectory() as temporary:
                model = copy.deepcopy(self.model)
                if lora_rank is not None:
                    apply_lora(model, lora_rank)
                optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.003)

                def update(target_model, target_optimizer, row):
                    ids, labels = collate([row])
                    target_optimizer.zero_grad(set_to_none=True)
                    loss = sft_loss(target_model, ids, labels)
                    loss.backward()
                    target_optimizer.step()
                    return loss.detach()

                update(model, optimizer, self.rows[0])  # Populate Adam moments.
                before = copy.deepcopy(model.state_dict())
                optimizer_before = copy.deepcopy(optimizer.state_dict())
                path = Path(temporary) / "resume.pt"
                checkpoint(path, model, optimizer, step=1, extra={"lora_rank": lora_rank})
                expected_random = torch.rand(4)
                expected_row = torch.randint(len(self.rows), (1,)).item()
                expected_loss = update(model, optimizer, self.rows[expected_row])

                restored, state = load_checkpoint(path)
                self.assertEqual(state["step"], 1)
                self.assertEqual(restored.config, model.config)
                self.assert_nested_equal(restored.state_dict(), before)
                restored_optimizer = torch.optim.AdamW([p for p in restored.parameters() if p.requires_grad], lr=.5)
                restored_optimizer.load_state_dict(state["optimizer"])
                self.assert_nested_equal(restored_optimizer.state_dict(), optimizer_before)
                # load_checkpoint returns state; callers explicitly restore
                # optimizer/RNG after model construction consumes random draws.
                torch.set_rng_state(state["torch_rng"])
                self.assert_tensors_equal(torch.rand(4), expected_random)
                resumed_row = torch.randint(len(self.rows), (1,)).item()
                self.assertEqual(resumed_row, expected_row)
                actual_loss = update(restored, restored_optimizer, self.rows[resumed_row])
                self.assert_tensors_equal(actual_loss, expected_loss)
                self.assert_nested_equal(restored.state_dict(), model.state_dict())
                self.assert_nested_equal(restored_optimizer.state_dict(), optimizer.state_dict())

    def test_data_splits_hold_out_templates_and_verifier_checks_full_response(self):
        train, development = datasets()
        probe = final_probe()
        families = [{r["template_family"] for r in split} for split in [train, development, probe]]
        self.assertFalse(families[0] & families[1])
        self.assertFalse(families[0] & families[2])
        self.assertFalse(families[1] & families[2])
        # These are held-out templates, not held-out symbols/intents.
        self.assertEqual({r["symbol"] for r in train}, {r["symbol"] for r in probe})
        row = train[0]
        self.assertEqual(assess(row, brief(row)), {"correct": True, "brief": True, "reward": 1.})
        self.assertEqual(assess(row, verbose(row)), {"correct": True, "brief": False, "reward": .5})
        for invalid in [[], [row["symbol"]], brief(row) + ["A"], ["H", "<eos>"]]:
            self.assertEqual(assess(row, invalid), {"correct": False, "brief": False, "reward": 0.})


if __name__ == "__main__":
    unittest.main()
