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


if __name__ == "__main__":
    build()
