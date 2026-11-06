"""A tiny, synthetic data audit. Group splitting happens before augmentation."""
from collections import Counter
import hashlib
import json

SAMPLES = [
    {"id": "a1", "group": "summary", "text": "请把这段天气说明概括成一句话。", "label": "allowed"},
    {"id": "a2", "group": "summary", "text": "用一句话概括这段天气说明。", "label": "allowed"},
    {"id": "b1", "group": "fraud-awareness", "text": "怎样识别可疑的诈骗邮件？", "label": "allowed"},
    {"id": "b2", "group": "fraud-awareness", "text": "怎样识别可疑的诈骗邮件？", "label": "allowed"},
    {"id": "c1", "group": "disallowed-placeholder", "text": "[抽象的不允许请求占位；无危险操作细节]", "label": "disallowed"},
    {"id": "c2", "group": "disallowed-placeholder", "text": "[同一不允许意图的另一种表达占位]", "label": "disallowed"},
]


def audit_and_split(rows):
    seen = set()
    duplicate_ids, clean = [], []
    for row in rows:
        if not all(isinstance(row.get(k), str) and row[k].strip() for k in ["id", "group", "text", "label"]):
            raise ValueError("missing or invalid required field")
        normalized = " ".join(row["text"].split())
        digest = hashlib.sha256(normalized.encode()).hexdigest()
        if digest in seen:
            duplicate_ids.append(row["id"])
        else:
            seen.add(digest)
            clean.append(row)
    # Deterministic group assignment for a teaching fixture, not balanced sampling.
    groups = sorted({r["group"] for r in clean})
    assignment = {g: ["train", "dev", "test"][i % 3] for i, g in enumerate(groups)}
    splits = {name: [r for r in clean if assignment[r["group"]] == name] for name in ["train", "dev", "test"]}
    return {"scope": "synthetic teaching data, not a benchmark", "duplicates": duplicate_ids, "counts": {k: len(v) for k, v in splits.items()}, "splits": splits}


if __name__ == "__main__":
    print(json.dumps(audit_and_split(SAMPLES), ensure_ascii=False, indent=2))
