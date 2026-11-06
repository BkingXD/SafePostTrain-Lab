"""Audit CUDA copy-task reports from raw outputs, without starting training."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re


def validate_cuda_training(training, device, name):
    if not isinstance(training, dict) or training.get("device") != device:
        raise ValueError(f"training device mismatch: {name}; expected {device}")
    allocated = training.get("gpu_peak_allocated_bytes")
    reserved = training.get("gpu_peak_reserved_bytes")
    if not isinstance(allocated, int) or allocated <= 0 or not isinstance(reserved, int) or reserved < allocated:
        raise ValueError(f"invalid CUDA memory measurements: {name}")
    elapsed = training.get("elapsed_seconds")
    if not isinstance(elapsed, (float, int)) or not math.isfinite(elapsed) or elapsed <= 0:
        raise ValueError(f"invalid CUDA timing: {name}")


def validate_results(report):
    if report.get("schema_version") != 2:
        raise ValueError("expected the CUDA schema-2 report")
    environment = report.get("environment", {})
    device = environment.get("device", "")
    if not isinstance(device, str) or re.fullmatch(r"cuda(?::\d+)?", device) is None:
        raise ValueError("expected a CUDA report environment")
    if not environment.get("cuda_runtime") or not environment.get("gpu_name"):
        raise ValueError("CUDA report must identify its runtime and GPU")
    if environment.get("timing") != "CUDA synchronized wall clock":
        raise ValueError("CUDA report must declare synchronized timing")
    validate_cuda_training(report.get("reward_model"), device, "reward_model")
    summary = []
    for stage in report["stages"]:
        if stage["name"] != "random":
            validate_cuda_training(stage.get("training"), device, stage["name"])
        for split in ["evaluation", "final_evaluation"]:
            if split not in stage:
                continue
            evaluation = stage[split]
            rows = evaluation["outputs"]
            if not rows:
                raise ValueError("empty evaluation split")
            for row in rows:
                if row["expected"] != [row["prompt"][-2], "<eos>"] or row["prompt"][-1] != "<assistant>":
                    raise ValueError(f"invalid expected response: {stage['name']}/{split}")
                is_brief = row["generated"] == row["expected"]
                is_correct = is_brief or row["generated"] == ["answer", *row["expected"]]
                expected_flags = {"correct": is_correct, "brief": is_brief,
                                  "reward": 1. if is_brief else .5 if is_correct else 0.}
                if any(row.get(key) != value for key, value in expected_flags.items()):
                    raise ValueError(f"raw-output mismatch: {stage['name']}/{split}/row flags")
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
        raise ValueError("report belongs to a different source revision; rerun the CUDA recipe")


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', default=str(root/'reports/tiny_lm/gpu_seed7.json'))
    args = parser.parse_args()
    report = json.loads(Path(args.report).read_text())
    validate_source(report, root/'labs/tiny_lm')
    print(json.dumps(validate_results(report), ensure_ascii=False, indent=2))
