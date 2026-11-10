"""Validate the tiny synthetic learning dataset without model dependencies."""
import hashlib
import json
from pathlib import Path

DATA = Path(__file__).parent / "data" / "demo_pairs.json"


def load_data(path=DATA):
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    train, probe = value.get("train"), value.get("probe")
    if not isinstance(train, list) or not train or not isinstance(probe, list) or not probe:
        raise ValueError("train and probe must be non-empty arrays")
    ids, prompts = set(), set()
    for split, rows in (("train", train), ("probe", probe)):
        for row in rows:
            fields = ["id", "category", "prompt"] + (["chosen", "rejected"] if split == "train" else [])
            if not isinstance(row, dict) or any(not isinstance(row.get(k), str) or not row[k].strip() for k in fields):
                raise ValueError(f"{split} row has an empty or invalid field")
            if row["id"] in ids or row["prompt"].strip() in prompts:
                raise ValueError("duplicate id or prompt across train/probe")
            if split == "train" and row["chosen"].strip() == row["rejected"].strip():
                raise ValueError("preference answers must differ")
            ids.add(row["id"])
            prompts.add(row["prompt"].strip())
    return value, hashlib.sha256(raw).hexdigest()
