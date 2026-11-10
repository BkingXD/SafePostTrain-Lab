"""Audit the small real-model demonstrations without loading the weights."""
import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports/real_model'


class RealModelReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = {name: json.loads((REPORTS / (name + '.json')).read_text())
                 for name in ('base', 'sft', 'dpo', 'sft_probe', 'dpo_probe')}

    def test_runs_reference_the_current_sources_and_same_model_data(self):
        sources = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'labs/real_model').glob('*.py')}
        dataset = hashlib.sha256((ROOT/'labs/real_model/data/demo_pairs.json').read_bytes()).hexdigest()
        model = self.r['base']['model']
        for name, report in self.r.items():
            with self.subTest(name=name):
                self.assertEqual(report['source_sha256'], sources)
                self.assertEqual(report['dataset_sha256'], dataset)
                self.assertEqual(report['model'], model)
                self.assertTrue(report['model']['revision_from_all_hub_metadata'])
                self.assertTrue(report['environment']['device'].startswith('cuda'))
                self.assertEqual(report['dataset_counts'], {'train':8,'probe':3})

    def test_training_uses_masked_answers_and_frozen_sft_reference(self):
        sft, dpo = self.r['sft']['metrics'], self.r['dpo']['metrics']
        self.assertEqual(sft['mask_check']['examples_checked'], 8)
        self.assertTrue(sft['mask_check']['first_sample']['eos_supervised'])
        self.assertGreater(sft['mask_check']['first_sample']['prompt_masked_tokens'], 0)
        self.assertEqual(dpo['preference_pairs_checked'], 8)
        self.assertTrue(dpo['reference_unchanged'])
        self.assertEqual(self.r['dpo']['input_adapter_sha256'], sft['output_adapter_sha256'])
        for metrics in (sft,dpo):
            self.assertEqual(metrics['global_step'], 2)
            self.assertGreater(metrics['trainable_parameters'], 0)
            self.assertGreater(metrics['updated_parameter_tensors'], 0)

    def test_probes_share_inputs_decoding_and_correct_adapter_versions(self):
        base, sft, dpo = (self.r[name] for name in ('base','sft_probe','dpo_probe'))
        probes = [[(x['id'],x['prompt']) for x in r['metrics']['outputs']] for r in (base,sft,dpo)]
        self.assertEqual(probes[0],probes[1])
        self.assertEqual(probes[0],probes[2])
        self.assertEqual(base['metrics']['decode'], sft['metrics']['decode'])
        self.assertEqual(base['metrics']['decode'], dpo['metrics']['decode'])
        self.assertEqual(sft['input_adapter_sha256'], self.r['sft']['metrics']['output_adapter_sha256'])
        self.assertEqual(dpo['input_adapter_sha256'], self.r['dpo']['metrics']['output_adapter_sha256'])
        for report in (base,sft,dpo):
            for item in report['metrics']['outputs']:
                self.assertTrue(item['generated'])
                self.assertIn('hit_max_new_tokens',item)
                if item['hit_max_new_tokens']:
                    self.assertFalse(item['ended_on_eos'])


if __name__ == '__main__':
    unittest.main()
