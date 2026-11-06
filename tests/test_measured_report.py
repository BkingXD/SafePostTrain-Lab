"""Recompute GPU evidence from saved raw records; never start training."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('evaluate_measured', ROOT/'labs/evaluate_measured.py')
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)
RESOURCE_FIELDS = {'elapsed_seconds', 'tokens_per_second', 'process_peak_rss_mb',
                   'gpu_peak_allocated_bytes', 'gpu_peak_reserved_bytes'}


def without_resource_measurements(value):
    if isinstance(value, dict):
        return {key: without_resource_measurements(item) for key, item in value.items() if key not in RESOURCE_FIELDS}
    if isinstance(value, list):
        return [without_resource_measurements(item) for item in value]
    return value


class MeasuredReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads((ROOT/'reports/tiny_lm/gpu_seed7.json').read_text())
        cls.replay = json.loads((ROOT/'reports/tiny_lm/gpu_replay_seed7.json').read_text())

    def test_gpu_metrics_recalculate_and_current_sources_match(self):
        for report in [self.report, self.replay]:
            with self.subTest(device=report['environment']['device']):
                AUDIT.validate_source(report, ROOT/'labs/tiny_lm')
                summary = AUDIT.validate_results(report)
                self.assertEqual(len(summary), 22)  # Eleven stages × development/final.
                self.assertEqual({item['stage'] for item in summary},
                                 {'random', 'warmup', 'sft', 'lora', 'chosen_sft', 'dpo', 'dpo_anchor', 'simpo', 'ppo', 'grpo', 'distill'})

    def test_changed_raw_response_is_detected(self):
        changed = copy.deepcopy(self.report)
        stage = next(s for s in changed['stages'] if s['name']=='sft')
        stage['evaluation']['outputs'][0]['generated'] = ['<eos>']
        with self.assertRaisesRegex(ValueError, 'raw-output mismatch'):
            AUDIT.validate_results(changed)

    def test_changed_row_flag_is_detected_even_when_aggregate_is_unchanged(self):
        changed = copy.deepcopy(self.report)
        row = changed['stages'][0]['evaluation']['outputs'][0]
        row['correct'] = not row['correct']
        with self.assertRaisesRegex(ValueError, 'raw-output mismatch'):
            AUDIT.validate_results(changed)

    def test_changed_aggregate_is_detected(self):
        changed = copy.deepcopy(self.report)
        changed['stages'][0]['evaluation']['accuracy'] += .125
        with self.assertRaisesRegex(ValueError, 'raw-output mismatch'):
            AUDIT.validate_results(changed)

    def test_expected_response_must_agree_with_original_copy_prompt(self):
        changed = copy.deepcopy(self.report)
        changed['stages'][0]['evaluation']['outputs'][0]['expected'] = ['<eos>']
        with self.assertRaisesRegex(ValueError, 'invalid expected response'):
            AUDIT.validate_results(changed)

    def test_non_cuda_report_environments_are_rejected(self):
        for device in ['cpu', 'mps', 'cuda-fallback', '']:
            with self.subTest(device=device):
                changed = copy.deepcopy(self.report)
                changed['environment']['device'] = device
                with self.assertRaisesRegex(ValueError, 'CUDA report environment'):
                    AUDIT.validate_results(changed)

    def test_cuda_name_without_runtime_or_hardware_evidence_is_rejected(self):
        for key in ['cuda_runtime', 'gpu_name']:
            with self.subTest(missing=key):
                changed = copy.deepcopy(self.report)
                del changed['environment'][key]
                with self.assertRaisesRegex(ValueError, 'runtime and GPU'):
                    AUDIT.validate_results(changed)

    def test_training_devices_and_gpu_measurements_must_match_report(self):
        for location in ['reward_model', 'sft']:
            with self.subTest(location=location):
                changed = copy.deepcopy(self.report)
                training = (changed['reward_model'] if location == 'reward_model' else
                            next(s['training'] for s in changed['stages'] if s['name'] == location))
                training['device'] = 'cpu'
                with self.assertRaisesRegex(ValueError, 'training device mismatch'):
                    AUDIT.validate_results(changed)
        changed = copy.deepcopy(self.report)
        changed['reward_model']['gpu_peak_allocated_bytes'] = None
        with self.assertRaisesRegex(ValueError, 'CUDA memory measurements'):
            AUDIT.validate_results(changed)

    def test_modified_source_file_is_detected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for source in (ROOT/'labs/tiny_lm').glob('*.py'):
                shutil.copyfile(source, directory/source.name)
            source = directory/'model.py'
            source.write_bytes(source.read_bytes() + b'\n# changed source revision\n')
            with self.assertRaisesRegex(ValueError, 'different source revision'):
                AUDIT.validate_source(self.report, directory)

    def test_reproducibility_claim_recomputes_from_both_complete_reports(self):
        comparison = json.loads((ROOT/'reports/tiny_lm/gpu_reproducibility.json').read_text())
        self.assertEqual(set(comparison['excluded_fields']), RESOURCE_FIELDS)
        self.assertEqual(len(comparison['excluded_fields']), len(RESOURCE_FIELDS))
        self.assertEqual(comparison['same_seed'], self.report['seed'])
        self.assertEqual(comparison['same_seed'], self.replay['seed'])
        self.assertEqual(comparison['same_environment'], self.report['environment'])
        self.assertEqual(comparison['same_environment'], self.replay['environment'])
        self.assertEqual(comparison['source_sha256'], self.report['source_sha256'])
        self.assertEqual(comparison['source_sha256'], self.replay['source_sha256'])
        self.assertEqual(without_resource_measurements(self.report), without_resource_measurements(self.replay))
        self.assertIs(comparison['training_and_all_evaluation_records_equal'], True)
        for key, expected_name in [('first_report', 'gpu_seed7.json'), ('second_report', 'gpu_replay_seed7.json')]:
            self.assertEqual(comparison[key], expected_name)
            digest = hashlib.sha256((ROOT/'reports/tiny_lm'/expected_name).read_bytes()).hexdigest()
            self.assertEqual(comparison[key + '_sha256'], digest)

    def test_cli_has_no_source_override(self):
        result = subprocess.run([sys.executable, str(ROOT/'labs/evaluate_measured.py'), '--help'],
                                capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--report', result.stdout)
        self.assertNotIn('--source-dir', result.stdout)


if __name__ == '__main__':
    unittest.main()
