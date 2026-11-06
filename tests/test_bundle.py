from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]


class DownloadBundleTests(unittest.TestCase):
    def test_download_is_self_contained_and_matches_current_sources(self):
        expected = {"labs/core_math.py", "labs/toy_training.py", "labs/reward_model.py", "labs/data_audit.py", "labs/README.md", "docs/LEARNING_CHECKLIST.md"}
        archive_path = ROOT / "web/dist/downloads/safeposttrain-teaching-labs.zip"
        with ZipFile(archive_path) as archive, tempfile.TemporaryDirectory(prefix="safeposttrain-labs-") as target:
            self.assertEqual(set(archive.namelist()), expected)
            for name in expected:
                self.assertEqual(archive.read(name), (ROOT / name).read_bytes())
            archive.extractall(target)  # Exact allowlisted paths above.
            for script in ["core_math", "toy_training", "reward_model", "data_audit"]:
                result = subprocess.run([sys.executable, f"labs/{script}.py"], cwd=target, capture_output=True, text=True, timeout=15)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertTrue(result.stdout.strip())

    def test_tiny_lm_bundle_sources_and_metric_audit_work_without_torch(self):
        archive_path = ROOT / "web/dist/downloads/safeposttrain-tiny-lm.zip"
        with ZipFile(archive_path) as archive, tempfile.TemporaryDirectory(prefix="safeposttrain-tiny-") as target:
            names=archive.namelist()
            required={"labs/__init__.py","labs/evaluate_measured.py","labs/tiny_lm/model.py","labs/tiny_lm/data.py","labs/tiny_lm/training.py","labs/tiny_lm/run.py","labs/tiny_lm/requirements-lock.txt","labs/tiny_lm/README.md","reports/tiny_lm/verified_seed7.json","tests/test_tiny_lm.py"}
            self.assertTrue(required.issubset(set(names)))
            for name in names:
                self.assertFalse(name.startswith('/'))
                self.assertNotIn('..',Path(name).parts)
                self.assertEqual(archive.read(name),(ROOT/name).read_bytes())
            archive.extractall(target)
            result=subprocess.run([sys.executable,"labs/evaluate_measured.py"],cwd=target,capture_output=True,text=True,timeout=15)
            self.assertEqual(result.returncode,0,result.stderr)


if __name__ == "__main__":
    unittest.main()
