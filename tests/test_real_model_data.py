"""Dataset and provenance checks for the real-model teaching workflow."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from labs.real_model.data_validation import load_data
from labs.real_model.run import MODEL_REVISION, examples, snapshot_provenance


class RealModelDataTests(unittest.TestCase):
    def test_dataset_has_distinct_pairs_and_separate_probe(self):
        data, digest = load_data()
        self.assertEqual(len(digest), 64)
        self.assertGreaterEqual(len(data['train']), 4)
        self.assertGreaterEqual(len(data['probe']), 2)
        self.assertEqual(len({r['id'] for r in data['train'] + data['probe']}), len(data['train']) + len(data['probe']))
        self.assertEqual(set(r['prompt'] for r in data['train']) & set(r['prompt'] for r in data['probe']), set())
        for row in data['train']:
            self.assertNotEqual(row['chosen'], row['rejected'])

    def test_rejects_identical_preference_and_leaked_prompt(self):
        data, _ = load_data()
        for case in ('equal_pair', 'leaked_prompt'):
            changed = copy.deepcopy(data)
            if case == 'equal_pair':
                changed['train'][0]['rejected'] = changed['train'][0]['chosen']
            else:
                changed['probe'][0]['prompt'] = changed['train'][0]['prompt']
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'data.json'
                path.write_text(json.dumps(changed, ensure_ascii=False))
                with self.assertRaises(ValueError):
                    load_data(path)

    def test_sft_and_dpo_use_the_same_prompt_and_non_thinking_mode(self):
        data, _ = load_data()
        sft, dpo = examples(data['train'], 'sft'), examples(data['train'], 'dpo')
        for row, a, b in zip(data['train'], sft, dpo):
            self.assertEqual(a['prompt'], b['prompt'])
            self.assertEqual(a['completion'], b['chosen'])
            self.assertEqual(a['completion'][0]['content'], row['chosen'])
            self.assertFalse(a['chat_template_kwargs']['enable_thinking'])
            self.assertFalse(b['chat_template_kwargs']['enable_thinking'])

    def test_local_snapshot_metadata_is_checked_not_invented(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ('config.json', 'tokenizer.json', 'model.safetensors'):
                (root / name).write_bytes(name.encode())
            self.assertFalse(snapshot_provenance(root)['revision_from_all_hub_metadata'])
            metadata = root / '.cache/huggingface/download'
            metadata.mkdir(parents=True)
            for name in ('config.json', 'tokenizer.json', 'model.safetensors'):
                (metadata / (name + '.metadata')).write_text(MODEL_REVISION + '\n')
            self.assertTrue(snapshot_provenance(root)['revision_from_all_hub_metadata'])
            (metadata / 'config.json.metadata').write_text('different-revision\n')
            with self.assertRaisesRegex(ValueError, 'different revision'):
                snapshot_provenance(root)


if __name__ == '__main__':
    unittest.main()
