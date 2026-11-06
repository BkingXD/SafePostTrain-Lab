"""Run a reproducible CPU post-training course, never a safety benchmark."""
import argparse
import copy
import hashlib
import json
import platform
import resource
import subprocess
import sys
from pathlib import Path

import torch

from .data import assess, brief, data_card, datasets, preference_rows
from .model import ModelConfig, TinyLM, apply_lora, checkpoint, collate, generate, load_checkpoint, parameter_counts, token_statistics
from .training import entropy, exact_kl, frozen, sample_and_filter, train_online, train_preference, train_reward, train_sft


@torch.no_grad()
def evaluate(model, rows, reference=None):
    model.eval()
    outputs = []
    for row in rows:
        answer = generate(model, row["prompt"])
        outputs.append({"id": row["id"], "prompt": row["prompt"], "expected": brief(row), "generated": answer, **assess(row, answer)})
    ids, labels = collate(rows, [brief(r) for r in rows])
    stats = token_statistics(model, ids, labels)
    result = {"sample_count": len(rows), "correct_count": sum(o["correct"] for o in outputs),
              "accuracy": sum(o["correct"] for o in outputs)/len(rows),
              "brief_rate": sum(o["brief"] for o in outputs)/len(rows),
              "brief_target_nll": (-stats["logps"].sum()/stats["mask"].sum()).item(),
              "entropy_on_brief_target_prefixes": entropy(stats).item(), "outputs": outputs}
    if reference is not None:
        result["kl_to_sft_on_brief_target_prefixes"] = exact_kl(stats, token_statistics(reference, ids, labels)).item()
    return result


def run(output, seed=7, quick=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output/"report.json").exists():
        raise ValueError("report.json already exists; choose a new output directory to preserve results")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    train, heldout = datasets()
    pairs = preference_rows()
    budgets = {"warmup": 10 if quick else 30, "sft": 20 if quick else 120,
               "lora": 20 if quick else 120, "reward": 15 if quick else 80,
               "preference": 10 if quick else 60, "online": 3 if quick else 25, "distill": 20 if quick else 100}
    root = Path(__file__).parent
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob('*.py'))}
    report = {"schema_version": 1, "scope": "Measured CPU teaching runs on a tiny synthetic copy task. NOT a large-model or safety benchmark.",
              "seed": seed, "quick": quick, "environment": {"python": sys.version.split()[0], "torch": torch.__version__,
              "platform": platform.system(), "device": "cpu", "torch_threads": 1, "deterministic_algorithms": True},
              "config": vars(ModelConfig()), "budgets": budgets, "data_card": data_card(), "source_sha256": source_hashes,
              "stages": [], "limitations": ["Only 8 held-out symbol cases; do not infer general LLM capability.",
              "Held-out templates share symbols/intents with training; this is not unseen-intent OOD.",
              "CPU timing depends on host load; optimizer bytes omit activation and allocator memory.",
              "DPO/SimPO/PPO/GRPO use different objectives and compute; their loss values are not rankings.",
              "No QLoRA or distributed GPU execution is claimed."]}
    models = {}
    def record(name, model, training=None, extra=None, optimizer=None):
        models[name] = copy.deepcopy(model)
        entry = {"name": name, "training": training, "evaluation": evaluate(model, heldout)}
        if extra:
            entry.update(extra)
        report["stages"].append(entry)
        checkpoint(output/(name+'.pt'), model, optimizer, (training or {}).get('steps',0), {"lora_rank": 4} if name=='lora' else {})
        print(f'{name}: accuracy={entry["evaluation"]["correct_count"]}/{len(heldout)}, brief={entry["evaluation"]["brief_rate"]:.3f}', flush=True)

    base = TinyLM()
    report["model_parameters"] = parameter_counts(base)["total"]
    record("random", base)
    warmup, warm_optimizer = train_sft(base, train, budgets["warmup"])
    record("warmup", base, warmup, optimizer=warm_optimizer)
    full = copy.deepcopy(base)
    full_stats, optimizer = train_sft(full, train, budgets["sft"])
    record("sft", full, full_stats, optimizer=optimizer)
    adapted = apply_lora(copy.deepcopy(base), rank=4)
    adapted_stats, adapted_optimizer = train_sft(adapted, train, budgets["lora"], lr=.01)
    record("lora", adapted, adapted_stats, {"comparison": "Same warmup checkpoint, same training rows and update/token budget as full SFT; different learning rates."}, adapted_optimizer)
    reference = frozen(full)
    reward, reward_stats = train_reward(full, pairs, heldout, budgets["reward"])
    report["reward_model"] = reward_stats
    torch.save(reward.state_dict(), output/'reward.pt')
    print(f'reward-model held-out pair accuracy={reward_stats["heldout"]["pair_accuracy"]:.3f}', flush=True)
    for method in ["dpo", "simpo"]:
        model = copy.deepcopy(full)
        stats, optimizer = train_preference(model, reference, pairs, budgets["preference"], method)
        record(method, model, stats, {"start": "same sft checkpoint", "preference": "brief correct over verbose correct"}, optimizer)
    for method in ["ppo", "grpo"]:
        model = copy.deepcopy(full)
        stats, optimizer = train_online(model, reference, pairs, budgets["online"], method)
        record(method, model, stats, {"start": "same sft checkpoint"}, optimizer)
    selected, selection = sample_and_filter(models["dpo"], pairs)
    if not selected:
        raise RuntimeError("teacher produced no accepted data; preserve this failure rather than inserting gold answers")
    student = copy.deepcopy(base)
    stats, optimizer = train_sft(student, selected, budgets["distill"])
    record("distill", student, stats, {"selection": selection, "start": "warmup checkpoint; teacher is the DPO branch"}, optimizer)
    for entry in report["stages"]:
        entry["evaluation"] = evaluate(models[entry["name"]], heldout, reference)
    loaded, state = load_checkpoint(output/'sft.pt')
    ids, _ = collate(heldout)
    with torch.no_grad():
        max_difference = (loaded(ids)-full(ids)).abs().max().item()
    report["checkpoint_reload_max_logit_difference"] = max_difference
    report["process_peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 if platform.system() != "Darwin" else 1024**2)
    report["raw_outputs"] = "Each stage includes every held-out prompt, expected response and generated response. Recompute correctness from these, not from train loss."
    (output/'data.json').write_text(json.dumps({"train":train,"heldout":heldout},ensure_ascii=False,indent=2))
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(f'Report: {output / "report.json"}', flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='artifacts/tiny-lm-seed7')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--quick', action='store_true', help='short smoke run; report marks reduced budgets')
    args = parser.parse_args()
    run(args.output, args.seed, args.quick)
