"""Create a reproducible ZIP with the directory layout used by lab commands."""
from pathlib import Path
from zipfile import ZipFile, ZipInfo, ZIP_STORED

ROOT = Path(__file__).resolve().parents[1]
FILES = ["labs/README.md", "labs/core_math.py", "labs/toy_training.py", "labs/reward_model.py", "labs/data_audit.py", "docs/LEARNING_CHECKLIST.md"]
OUT = ROOT / "web/dist/downloads/safeposttrain-teaching-labs.zip"


def build():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(OUT, "w", compression=ZIP_STORED) as archive:
        for name in FILES:
            info = ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.external_attr = 0o100644 << 16
            archive.writestr(info, (ROOT / name).read_bytes())
    print(f"Teaching bundle: {OUT.name} ({len(FILES)} files)")
    tiny_files = ["labs/__init__.py", "labs/evaluate_measured.py", "tests/test_tiny_lm.py", "tests/test_tiny_lm_device.py", "tests/test_measured_report.py", "reports/tiny_lm/gpu_seed7.json"]
    tiny_files += ["reports/tiny_lm/gpu_replay_seed7.json", "reports/tiny_lm/gpu_reproducibility.json"]
    tiny_files += [str(p.relative_to(ROOT)) for p in sorted((ROOT/'labs/tiny_lm').iterdir())
                   if p.suffix == '.py' or p.name in {'README.md', 'requirements-gpu.txt', 'requirements-gpu-lock.txt'}]
    with ZipFile(OUT.parent/'safeposttrain-tiny-lm.zip', 'w', compression=ZIP_STORED) as archive:
        for name in tiny_files:
            info=ZipInfo(name,date_time=(2026,1,1,0,0,0));info.external_attr=0o100644<<16
            archive.writestr(info,(ROOT/name).read_bytes())
    print(f"TinyLM source bundle: {len(tiny_files)} files")
    real_files = ["labs/__init__.py", "labs/tiny_lm/requirements-gpu.txt", "tests/test_real_model_data.py", "tests/test_real_model_reports.py",
                  "labs/real_model/__init__.py", "labs/real_model/run.py", "labs/real_model/data_validation.py",
                  "labs/real_model/requirements.txt", "labs/real_model/requirements-lock.txt",
                  "labs/real_model/README.md", "labs/real_model/data/demo_pairs.json"]
    real_files += [f"reports/real_model/{name}.json" for name in ("base", "sft", "dpo", "sft_probe", "dpo_probe")]
    with ZipFile(OUT.parent/'safeposttrain-real-model.zip', 'w', compression=ZIP_STORED) as archive:
        for name in real_files:
            info=ZipInfo(name,date_time=(2026,1,1,0,0,0));info.external_attr=0o100644<<16
            archive.writestr(info,(ROOT/name).read_bytes())
    print(f"Real-model source bundle: {len(real_files)} files")


if __name__ == "__main__":
    build()
