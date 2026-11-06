import copy
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('evaluate_measured', ROOT/'labs/evaluate_measured.py')
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class MeasuredReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads((ROOT/'reports/tiny_lm/verified_seed7.json').read_text())

    def test_metrics_recalculate_and_sources_match(self):
        AUDIT.validate_source(self.report, ROOT/'reports/tiny_lm/cpu_reference_source')
        summary = AUDIT.validate_results(self.report)
        self.assertEqual(len(summary), 22)  # Eleven stages × development/final.

    def test_changed_raw_response_is_detected(self):
        changed = copy.deepcopy(self.report)
        stage = next(s for s in changed['stages'] if s['name']=='sft')
        stage['evaluation']['outputs'][0]['generated'] = ['<eos>']
        with self.assertRaisesRegex(ValueError, 'raw-output mismatch'):
            AUDIT.validate_results(changed)

    def test_pilot_is_not_passed_off_as_current_evidence(self):
        pilot = json.loads((ROOT/'reports/tiny_lm/pilot_seed7.json').read_text())
        with self.assertRaises(ValueError):
            AUDIT.validate_results(pilot)

    def test_cpu_reference_is_not_claimed_as_current_gpu_source(self):
        with self.assertRaisesRegex(ValueError, 'different source revision'):
            AUDIT.validate_source(self.report, ROOT/'labs/tiny_lm')

    def test_gpu_report_has_actual_cuda_provenance_and_current_sources(self):
        report = json.loads((ROOT/'reports/tiny_lm/gpu_seed7.json').read_text())
        self.assertTrue(report['environment']['device'].startswith('cuda'))
        self.assertTrue(report['environment']['gpu_name'])
        AUDIT.validate_source(report, ROOT/'labs/tiny_lm')
        self.assertEqual(len(AUDIT.validate_results(report)), 22)
