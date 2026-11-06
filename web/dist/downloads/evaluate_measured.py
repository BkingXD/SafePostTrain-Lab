"""Recalculate measured copy-task metrics from raw outputs, without PyTorch."""
import argparse
import hashlib
import json
from pathlib import Path


def validate_results(report):
    if report.get("schema_version") != 2:
        raise ValueError("expected the verified schema-2 report; pilot reports are historical")
    summary = []
    for stage in report["stages"]:
        for split in ["evaluation", "final_evaluation"]:
            if split not in stage:
                continue
            evaluation = stage[split]
            rows = evaluation["outputs"]
            if not rows:
                raise ValueError("empty evaluation split")
            correct = sum(r["generated"] in (r["expected"], ["answer", *r["expected"]]) for r in rows)
            brief = sum(r["generated"] == r["expected"] for r in rows)
            expected = {"sample_count": len(rows), "correct_count": correct, "accuracy": correct/len(rows), "brief_rate": brief/len(rows)}
            for key, value in expected.items():
                if evaluation[key] != value:
                    raise ValueError(f"raw-output mismatch: {stage['name']}/{split}/{key}")
            summary.append({"stage":stage["name"], "split":split, "correct":correct, "total":len(rows), "brief":brief})
    by_name = {stage["name"]:stage for stage in report["stages"]}
    if by_name["sft"]["training"]["effective_training_tokens"] != by_name["lora"]["training"]["effective_training_tokens"]:
        raise ValueError("SFT/LoRA token budgets are not matched")
    if by_name["chosen_sft"]["training"]["effective_training_tokens"] != by_name["dpo"]["training"]["effective_training_tokens"]:
        raise ValueError("chosen-only SFT/DPO token budgets are not matched")
    selection = by_name["distill"]["selection"]
    if selection["accepted"] != sum(candidate["accepted"] for candidate in selection["candidates"]):
        raise ValueError("selection counts disagree")
    if report["checkpoint_reload_max_logit_difference"] != 0:
        raise ValueError("checkpoint reload changed logits")
    return summary


def validate_source(report, directory):
    sources = {path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in Path(directory).glob('*.py')}
    if sources != report["source_sha256"]:
        raise ValueError("report belongs to a different source revision; rerun or explicitly preserve it as history")


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', default=str(root/'reports/tiny_lm/verified_seed7.json'))
    args = parser.parse_args()
    report = json.loads(Path(args.report).read_text())
    validate_source(report, root/'labs/tiny_lm')
    print(json.dumps(validate_results(report), ensure_ascii=False, indent=2))
