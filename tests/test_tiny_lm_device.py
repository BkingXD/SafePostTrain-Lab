"""GPU-first preflight tests and conditional, real-CUDA teaching-path tests.

CPU/mocked preflight checks are not CUDA execution evidence. The CUDA class
actually runs the algorithms and explicitly skips if its interpreter has no
usable CUDA device. No test runs the full training recipe or judges accuracy.
"""
import copy
import inspect
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    torch = None

CUDA_AVAILABLE = torch is not None and torch.version.cuda is not None and torch.cuda.is_available()
CUDA_SKIP = ("real CUDA execution requires a CUDA-enabled PyTorch build and an accessible NVIDIA GPU; "
             "CPU/mocked checks do not validate GPU execution")

if torch is not None:
    sys.path.insert(0, str(ROOT / "labs"))
    from tiny_lm import run as runner
    from tiny_lm import runtime
    from tiny_lm.data import assess, record
    from tiny_lm.model import (
        ModelConfig, TinyLM, apply_lora, checkpoint, collate, generate,
        load_checkpoint, model_device, sft_loss, token_statistics,
    )
    from tiny_lm.training import (
        frozen, sample_and_filter, train_online, train_preference, train_reward, train_sft,
    )


@unittest.skipIf(torch is None, "optional PyTorch lab: use a teaching environment with PyTorch")
class DevicePreflightTests(unittest.TestCase):
    def test_gpu_is_default_and_cpu_requires_explicit_selection(self):
        parameters = inspect.signature(runtime.resolve_device).parameters
        self.assertEqual(next(iter(parameters.values())).default, "cuda")
        self.assertEqual(inspect.signature(runner.run).parameters["device"].default, "cuda")
        with mock.patch.object(torch.version, "cuda", None), mock.patch.object(torch.cuda, "is_available", return_value=False):
            self.assertEqual(runtime.resolve_device("cpu"), torch.device("cpu"))
            with self.assertRaisesRegex(RuntimeError, "CUDA|cuda"):
                runtime.resolve_device()

    def test_cuda_build_without_accessible_gpu_fails_without_cpu_fallback(self):
        with mock.patch.object(torch.version, "cuda", "12.8"), mock.patch.object(torch.cuda, "is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CUDA|cuda"):
                runtime.resolve_device("cuda")

    def test_cpu_only_build_cannot_claim_cuda_even_if_availability_is_mocked(self):
        # An availability flag alone must not overrule the installed build.
        with mock.patch.object(torch.version, "cuda", None), mock.patch.object(torch.cuda, "is_available", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "CUDA|cuda"):
                runtime.resolve_device("cuda")

    def test_run_checks_device_before_creating_model_or_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "must-not-exist"
            with mock.patch.object(runner, "resolve_device", side_effect=RuntimeError("CUDA unavailable")) as resolve, \
                    mock.patch.object(runner, "TinyLM") as constructor:
                with self.assertRaisesRegex(RuntimeError, "CUDA unavailable"):
                    runner.run(output)
                resolve.assert_called_once_with("cuda")
                constructor.assert_not_called()
                self.assertFalse(output.exists())

    def test_cli_cpu_device_check_does_not_start_training_or_create_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "preflight-only"
            result = subprocess.run(
                [sys.executable, "-m", "labs.tiny_lm.run", "--device", "cpu", "--check-device", "--output", str(output)],
                cwd=ROOT, capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("cpu", result.stdout.lower())
            self.assertFalse(output.exists())

    def test_explicit_cpu_training_does_not_report_gpu_measurements(self):
        # One update tests the CPU metadata contract, not the CUDA backend.
        model = TinyLM(ModelConfig(width=8, heads=2, layers=1, max_length=16))
        stats, _ = train_sft(model, [record(["copy"], "A")], steps=1)
        self.assertEqual(stats["device"], "cpu")
        self.assertIsNone(stats["gpu_peak_allocated_bytes"])
        self.assertIsNone(stats["gpu_peak_reserved_bytes"])
        self.assertGreater(stats["elapsed_seconds"], 0)


@unittest.skipUnless(CUDA_AVAILABLE, CUDA_SKIP)
class RealCUDATeachingPaths(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Exercise the actual preflight before allocating training tensors;
        # this also establishes the deterministic cuBLAS workspace setting.
        cls.device = runtime.resolve_device("cuda")

    def setUp(self):
        torch.manual_seed(73)
        torch.cuda.manual_seed_all(73)
        self.model = TinyLM(ModelConfig(width=16, heads=4, layers=1, max_length=24)).to(self.device)
        self.rows = [record(["copy"], "A"), record(["please", "repeat"], "B")]

    def assert_model_device(self, model):
        self.assertEqual(model_device(model), self.device)
        self.assertTrue(all(param.device == self.device for param in model.parameters()))
        self.assertTrue(all(buffer.device == self.device for buffer in model.buffers()))

    def assert_training_measurement(self, stats):
        self.assertEqual(torch.device(stats["device"]), self.device)
        self.assertEqual(stats["steps"], 1)
        self.assertGreater(stats["effective_training_tokens"], 0)
        self.assertTrue(math.isfinite(stats["elapsed_seconds"]))
        self.assertGreater(stats["elapsed_seconds"], 0)
        self.assertGreater(stats["tokens_per_second"], 0)
        self.assertGreater(stats["gpu_peak_allocated_bytes"], 0)
        self.assertGreaterEqual(stats["gpu_peak_reserved_bytes"], stats["gpu_peak_allocated_bytes"])
        self.assertTrue(stats["gpu_memory_scope"])
        self.assertTrue(stats["history"])
        for entry in stats["history"]:
            self.assertTrue(math.isfinite(entry["loss_before_update"]))

    def assert_optimizer_device(self, optimizer):
        self.assertTrue(optimizer.state)
        for state in optimizer.state.values():
            # Adam's scalar step counter may remain on CPU in noncapturable
            # mode; parameter-sized moments must follow their parameters.
            for key in ["exp_avg", "exp_avg_sq"]:
                if key in state:
                    self.assertEqual(state[key].device, self.device)

    def test_model_collation_generation_and_sampling_stay_on_cuda(self):
        self.assertEqual(runtime.resolve_device("cuda").type, "cuda")
        self.assert_model_device(self.model)
        ids, labels = collate(self.rows, device=self.device)
        self.assertEqual(ids.device, self.device)
        self.assertEqual(labels.device, self.device)
        for value in token_statistics(self.model, ids, labels).values():
            self.assertEqual(value.device, self.device)
        observed_devices = []
        hook = self.model.register_forward_pre_hook(lambda module, inputs: observed_devices.append(inputs[0].device))
        try:
            first = generate(self.model, self.rows[0]["prompt"], sample=True,
                             generator=torch.Generator(device=self.device).manual_seed(7))
            second = generate(self.model, self.rows[0]["prompt"], sample=True,
                              generator=torch.Generator(device=self.device).manual_seed(7))
            self.assertEqual(first, second)
            self.assertTrue(observed_devices)
            self.assertTrue(all(device == self.device for device in observed_devices))
            with self.assertRaisesRegex(ValueError, "same device"):
                generate(self.model, self.rows[0]["prompt"], sample=True, generator=torch.Generator(device="cpu"))
        finally:
            hook.remove()

    def test_sft_lora_reward_and_preference_algorithms_run_on_cuda(self):
        # These spies still execute the real CUDA operations. Positive timing
        # numbers alone would not establish that queued kernels were timed.
        with mock.patch.object(torch.cuda, "synchronize", wraps=torch.cuda.synchronize) as synchronize, \
                mock.patch.object(torch.cuda, "reset_peak_memory_stats", wraps=torch.cuda.reset_peak_memory_stats) as reset_peak:
            stats, optimizer = train_sft(self.model, self.rows, steps=1)
            self.assertGreaterEqual(sum(call.args == (self.device,) for call in synchronize.call_args_list), 2)
            reset_peak.assert_called_once_with(self.device)
        self.assert_training_measurement(stats)
        self.assert_optimizer_device(optimizer)
        reference = frozen(self.model)
        for method in ["dpo", "dpo_anchor", "simpo"]:
            with self.subTest(method=method):
                policy = copy.deepcopy(self.model)
                stats, optimizer = train_preference(policy, reference, self.rows, steps=1, method=method)
                self.assert_model_device(policy)
                self.assert_training_measurement(stats)
                self.assert_optimizer_device(optimizer)
        self.assertTrue(all(not parameter.requires_grad and parameter.grad is None for parameter in reference.parameters()))

        adapted = apply_lora(copy.deepcopy(self.model), rank=2)
        self.assert_model_device(adapted)
        before = {name: parameter.detach().clone() for name, parameter in adapted.named_parameters()}
        stats, optimizer = train_sft(adapted, self.rows, steps=1)
        self.assert_training_measurement(stats)
        self.assert_optimizer_device(optimizer)
        changed = []
        for name, parameter in adapted.named_parameters():
            if not torch.equal(parameter, before[name]):
                changed.append(name)
            if not parameter.requires_grad:
                torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
        self.assertTrue(changed)
        self.assertTrue(all(name.endswith((".a", ".b")) for name in changed))

        reward, stats = train_reward(self.model, self.rows, self.rows, steps=1)
        self.assert_model_device(reward)
        self.assert_training_measurement(stats)
        ids, _ = collate(self.rows, device=self.device)
        self.assertEqual(reward(ids).device, self.device)
        self.assertTrue(torch.isfinite(reward(ids)).all().item())

    def test_online_ppo_grpo_and_generation_filter_run_on_cuda(self):
        reference = frozen(self.model)
        for method in ["ppo", "grpo"]:
            with self.subTest(method=method):
                policy = copy.deepcopy(self.model)
                stats, optimizer = train_online(policy, reference, self.rows, steps=1, method=method, seed=9)
                self.assert_model_device(policy)
                self.assert_training_measurement(stats)
                self.assert_optimizer_device(optimizer)
                self.assertGreater(stats["generated_tokens"], 0)
                if method == "ppo":
                    self.assertGreater(stats["critic_parameters"]["trainable"], 0)
                    self.assertTrue(math.isfinite(stats["history"][0]["critic_loss"]))
        selected, summary = sample_and_filter(self.model, self.rows, seed=17)
        self.assertEqual(summary["generated"], 4 * len(self.rows))
        self.assertEqual(summary["selected"], len(selected))
        for row in selected:
            self.assertTrue(assess(row, row["response"])["correct"])
            self.assertIn("generated-by-", row["source"])
        # No correctness threshold and no inserted gold-answer fallback: an
        # untrained teacher is allowed to yield no distillation examples.

    def test_cuda_checkpoint_restores_model_optimizer_and_both_rng_streams(self):
        for rank in [None, 2]:
            with self.subTest(lora_rank=rank), tempfile.TemporaryDirectory() as temporary:
                model = copy.deepcopy(self.model)
                if rank is not None:
                    apply_lora(model, rank)
                _, optimizer = train_sft(model, self.rows, steps=1)
                path = Path(temporary) / "cuda.pt"
                checkpoint(path, model, optimizer, step=1, extra={"lora_rank": rank})
                expected_cpu_random = torch.rand(5)
                expected_cuda_random = torch.rand(5, device=self.device)
                restored, state = load_checkpoint(path, device=self.device, restore_rng=True)
                self.assert_model_device(restored)
                self.assertEqual(state["torch_cuda_rng"]["device"], str(self.device))
                self.assertEqual(state["torch_cuda_rng"]["state"].device.type, "cpu")
                torch.testing.assert_close(torch.rand(5), expected_cpu_random, rtol=0, atol=0)
                torch.testing.assert_close(torch.rand(5, device=self.device), expected_cuda_random, rtol=0, atol=0)
                restored_optimizer = torch.optim.AdamW([p for p in restored.parameters() if p.requires_grad], lr=.5)
                restored_optimizer.load_state_dict(state["optimizer"])
                self.assert_optimizer_device(restored_optimizer)
                ids, labels = collate(self.rows, device=self.device)
                torch.testing.assert_close(model(ids), restored(ids), rtol=0, atol=0)
                for target, target_optimizer in [(model, optimizer), (restored, restored_optimizer)]:
                    target_optimizer.zero_grad(set_to_none=True)
                    sft_loss(target, ids, labels).backward()
                    target_optimizer.step()
                for original, resumed in zip(model.parameters(), restored.parameters()):
                    torch.testing.assert_close(original, resumed, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
