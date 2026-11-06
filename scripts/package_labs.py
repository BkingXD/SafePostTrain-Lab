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
    tiny_files = ["labs/__init__.py", "labs/evaluate_measured.py", "tests/test_tiny_lm.py", "tests/test_tiny_lm_device.py", "reports/tiny_lm/verified_seed7.json", "reports/tiny_lm/gpu_seed7.json"]
    tiny_files += ["reports/tiny_lm/gpu_replay_seed7.json", "reports/tiny_lm/gpu_reproducibility.json"]
    tiny_files += [str(p.relative_to(ROOT)) for p in sorted((ROOT/'reports/tiny_lm/cpu_reference_source').iterdir()) if p.suffix in {'.py', '.md'}]
    tiny_files += [str(p.relative_to(ROOT)) for p in sorted((ROOT/'labs/tiny_lm').iterdir()) if p.suffix in {'.py','.txt','.md'}]
    with ZipFile(OUT.parent/'safeposttrain-tiny-lm.zip', 'w', compression=ZIP_STORED) as archive:
        for name in tiny_files:
            info=ZipInfo(name,date_time=(2026,1,1,0,0,0));info.external_attr=0o100644<<16
            archive.writestr(info,(ROOT/name).read_bytes())
    print(f"TinyLM source bundle: {len(tiny_files)} files")


if __name__ == "__main__":
    build()
