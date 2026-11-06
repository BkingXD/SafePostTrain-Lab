"""Run the CUDA-first post-training course, never a safety benchmark."""
import argparse
import copy
import hashlib
import json
import platform
import resource
from pathlib import Path

import torch

from .data import assess, brief, data_card, datasets, final_probe, preference_rows
from .model import ModelConfig, TOKENS, TinyLM, apply_lora, checkpoint, collate, generate, load_checkpoint, parameter_counts, token_statistics
from .training import entropy, exact_kl, frozen, sample_and_filter, train_online, train_preference, train_reward, train_sft
from .runtime import configure_runtime, environment, resolve_device


@torch.no_grad()
def evaluate(model, rows, reference=None):
    model.eval()
    outputs = []
    for row in rows:
        answer = generate(model, row["prompt"])
        outputs.append({"id": row["id"], "prompt": row["prompt"], "expected": brief(row), "generated": answer, **assess(row, answer)})
    ids, labels = collate(rows, [brief(r) for r in rows], device=next(model.parameters()).device)
    stats = token_statistics(model, ids, labels)
    result = {"sample_count": len(rows), "correct_count": sum(o["correct"] for o in outputs),
              "accuracy": sum(o["correct"] for o in outputs)/len(rows),
              "brief_rate": sum(o["brief"] for o in outputs)/len(rows),
              "brief_target_nll": (-stats["logps"].sum()/stats["mask"].sum()).item(),
              "entropy_on_brief_target_prefixes": entropy(stats).item(), "outputs": outputs}
    if reference is not None:
        result["kl_to_sft_on_brief_target_prefixes"] = exact_kl(stats, token_statistics(reference, ids, labels)).item()
    return result


def run(output, seed=7, final_eval=False, device="cuda"):
    device = resolve_device(device)  # Fail before creating any output artifacts.
    configure_runtime(device, seed)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output/"report.json").exists():
        raise ValueError("report.json already exists; choose a new output directory to preserve results")
    train, heldout = datasets()
    pairs = preference_rows()
    budgets = {"warmup":30, "sft":120, "lora":120, "reward":80,
               "preference":60, "online":25, "distill":100}
    root = Path(__file__).parent
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob('*.py'))}
    report = {"schema_version": 2, "scope": f"Measured {device} teaching runs on a tiny synthetic copy task. NOT a large-model or safety benchmark.",
              "primary_evaluation_split": "development probe (used for diagnosis); final_probe is separate and optional",
              "seed": seed, "final_probe_enabled": final_eval, "environment": environment(device),
              "config": vars(ModelConfig()), "budgets": budgets, "data_card": data_card(), "source_sha256": source_hashes,
              "stages": [], "limitations": ["Only 8 held-out symbol cases; do not infer general LLM capability.",
              "Held-out templates share symbols/intents with training; this is not unseen-intent OOD.",
              "Wall-clock timing depends on host load. CUDA timings synchronize the device; optimizer bytes omit activations and allocator memory. GPU peaks include resident models and cache, not isolated algorithm memory.",
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

    base = TinyLM().to(device)
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
    chosen_control = copy.deepcopy(full)
    control_steps = budgets["preference"] * 5 // 2  # 5 tokens/pair vs 2 chosen tokens.
    stats, optimizer = train_sft(chosen_control, pairs, control_steps, lr=.0007)
    record("chosen_sft", chosen_control, stats, {"start":"same sft checkpoint", "comparison":"same effective response-token budget as DPO; different update count"}, optimizer)
    for method in ["dpo", "dpo_anchor", "simpo"]:
        model = copy.deepcopy(full)
        stats, optimizer = train_preference(model, reference, pairs, budgets["preference"], method)
        record(method, model, stats, {"start": "same sft checkpoint", "preference": "brief correct over verbose correct", "anchor_weight": .2 if method=='dpo_anchor' else 0}, optimizer)
    for method in ["ppo", "grpo"]:
        model = copy.deepcopy(full)
        stats, optimizer = train_online(model, reference, pairs, budgets["online"], method, seed+100)
        record(method, model, stats, {"start": "same sft checkpoint"}, optimizer)
    teacher_candidates = []
    teacher_name = None
    for name in ["dpo_anchor", "simpo", "sft"]:
        quality = evaluate(models[name], pairs)["accuracy"]
        teacher_candidates.append({"name":name,"training_prompt_accuracy":quality})
        if quality >= .95:
            teacher_name = name
            break
    if teacher_name is None:
        raise RuntimeError("no teacher passes the training-side quality gate; do not fall back to gold answers")
    selected, selection = sample_and_filter(models[teacher_name], pairs, seed+200, teacher_name)
    selection.update({"teacher":teacher_name,"teacher_gate":teacher_candidates,"gate_threshold":.95,"gate_split":"training prompts only"})
    if not selected:
        raise RuntimeError("teacher produced no accepted data; preserve this failure rather than inserting gold answers")
    student = copy.deepcopy(base)
    stats, optimizer = train_sft(student, selected, budgets["distill"])
    record("distill", student, stats, {"selection": selection, "start": "warmup checkpoint; teacher passes a training-side quality gate"}, optimizer)
    for entry in report["stages"]:
        entry["evaluation"] = evaluate(models[entry["name"]], heldout, reference)
        if final_eval:
            entry["final_evaluation"] = evaluate(models[entry["name"]], final_probe(), reference)
    loaded, state = load_checkpoint(output/'sft.pt', device=device)
    ids, _ = collate(heldout, device=device)
    with torch.no_grad():
        max_difference = (loaded(ids)-full(ids)).abs().max().item()
    report["checkpoint_reload_max_logit_difference"] = max_difference
    example_ids, example_labels = collate([train[-1]])
    report["mask_walkthrough"] = [{"position":t,"input":TOKENS[example_ids[0,t]],
        "next_token":TOKENS[example_ids[0,t+1]],"label":example_labels[0,t+1].item(),
        "supervised":example_labels[0,t+1].item()!=-100} for t in range(example_ids.size(1)-1)]
    report["process_peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 if platform.system() != "Darwin" else 1024**2)
    report["raw_outputs"] = "Each stage includes every held-out prompt, expected response and generated response. Recompute correctness from these, not from train loss."
    (output/'data.json').write_text(json.dumps({"train":train,"development":heldout,"final_probe":final_probe()},ensure_ascii=False,indent=2))
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(f'Report: {output / "report.json"}', flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=None)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--final-eval', action='store_true', help='evaluate the separate final template after freezing the recipe')
    parser.add_argument('--device', default='cuda', help='cuda (default), cuda:N, or cpu for explicit compatibility runs')
    parser.add_argument('--check-device', action='store_true', help='check CUDA kernels and print runtime information without training')
    args = parser.parse_args()
    try:
        if args.check_device:
            selected_device = resolve_device(args.device)
            configure_runtime(selected_device, args.seed)
            print(json.dumps(environment(selected_device), ensure_ascii=False, indent=2))
        else:
            output = args.output or f'artifacts/tiny-lm-{args.device.replace(":", "-")}-seed{args.seed}'
            run(output, args.seed, args.final_eval, args.device)
    except (RuntimeError, ValueError) as exc:
        parser.exit(1, f"Training stopped: {exc}\n")
