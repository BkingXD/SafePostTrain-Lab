"""Inspectable training recipes. Explicit losses; no trainer abstraction."""
import copy
import math
import time

import torch
from torch.nn import functional as F

from .data import assess, brief, incorrect, verbose
from .model import ScoreModel, collate, generate, model_device, parameter_counts, sft_loss, token_statistics


def optimizer_bytes(optimizer):
    return sum(v.numel()*v.element_size() for state in optimizer.state.values() for v in state.values() if isinstance(v, torch.Tensor))


def frozen(model):
    return copy.deepcopy(model).requires_grad_(False).eval()


def masked_mean(values, mask):
    return (values * mask).sum() / mask.sum().clamp_min(1)


def exact_kl(stats, reference):
    return masked_mean((stats["log_distribution"].exp() * (stats["log_distribution"] - reference["log_distribution"])).sum(-1), stats["mask"])


def entropy(stats):
    return -masked_mean((stats["log_distribution"].exp() * stats["log_distribution"]).sum(-1), stats["mask"])


def optimize(loss, model, optimizer):
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
    optimizer.step()


def start_measurement(model):
    """Time completed GPU work and reset this device's process memory peaks."""
    device = model_device(model)
    if device.type != "cuda":
        raise ValueError("CUDA GPU training is required")
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    return time.perf_counter()


def finish(model, optimizer, start, steps, tokens, history):
    device = model_device(model)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    return {"parameters": parameter_counts(model), "steps": steps, "effective_training_tokens": tokens,
            "elapsed_seconds": elapsed, "tokens_per_second": tokens / max(elapsed, 1e-9),
            "optimizer_tensor_bytes": optimizer_bytes(optimizer), "history": history,
            "device": str(device),
            "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "gpu_memory_scope": "PyTorch allocator peaks on the model's CUDA device during the timed loop; include allocations already live at stage start and reserved cache, but exclude CUDA context and non-PyTorch allocations."}


def train_sft(model, records, steps, lr=.003):
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.)
    ids, labels = collate(records, device=model_device(model))
    count = labels[:, 1:].ne(-100).sum().item()
    start, history = start_measurement(model), []
    for step in range(steps):
        model.train()
        loss = sft_loss(model, ids, labels)
        optimize(loss, model, optimizer)
        if step == 0 or (step+1) % 10 == 0 or step+1 == steps:
            history.append({"step": step+1, "loss_before_update": loss.item()})
    return finish(model, optimizer, start, steps, count*steps, history), optimizer


def preference_loss(chosen, rejected, reference_chosen=None, reference_rejected=None, method="dpo"):
    if method in {"dpo", "dpo_anchor"}:
        margin = (chosen["sums"] - rejected["sums"]) - (reference_chosen["sums"] - reference_rejected["sums"])
        loss = -F.logsigmoid(.2 * margin).mean()
        if method == "dpo_anchor":
            loss = loss + .2 * (-chosen["logps"].sum()/chosen["mask"].sum())
        return loss
    if method == "simpo":
        margin = chosen["sums"] / chosen["lengths"] - rejected["sums"] / rejected["lengths"]
        return -F.logsigmoid(2. * margin - .5).mean()
    raise ValueError("expected dpo, dpo_anchor or simpo")


def train_preference(model, reference, records, steps, method):
    device = model_device(model)
    if model_device(reference) != device:
        raise ValueError("policy and reference must use the same device")
    chosen_ids, chosen_labels = collate(records, [brief(r) for r in records], device=device)
    rejected_ids, rejected_labels = collate(records, [verbose(r) for r in records], device=device)
    with torch.no_grad():
        ref_w = token_statistics(reference, chosen_ids, chosen_labels)
        ref_l = token_statistics(reference, rejected_ids, rejected_labels)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0007, weight_decay=0.)
    start, history = start_measurement(model), []
    for step in range(steps):
        chosen = token_statistics(model, chosen_ids, chosen_labels)
        rejected = token_statistics(model, rejected_ids, rejected_labels)
        loss = preference_loss(chosen, rejected, ref_w, ref_l, method)
        optimize(loss, model, optimizer)
        if step == 0 or (step+1) % 10 == 0 or step+1 == steps:
            history.append({"step": step+1, "loss_before_update": loss.item(),
                            "chosen_logp": chosen["sums"].mean().item(), "rejected_logp": rejected["sums"].mean().item()})
    count = (chosen_labels != -100).sum().item() + (rejected_labels != -100).sum().item()
    return finish(model, optimizer, start, steps, count*steps, history), optimizer


def reward_accuracy(model, rows):
    device = model_device(model)
    ids_w, _ = collate(rows, [brief(r) for r in rows], device=device)
    ids_l, _ = collate(rows, [incorrect(r) for r in rows], device=device)
    with torch.no_grad():
        probs = (model(ids_w)-model(ids_l)).sigmoid()
    return {"pair_accuracy": (probs > .5).float().mean().item(),
            "brier_score": ((probs-1)**2).mean().item(),
            "mean_preferred_probability": probs.mean().item(),
            "probabilities": probs.tolist()}


def train_reward(backbone, train, heldout, steps):
    model = ScoreModel(copy.deepcopy(backbone))
    device = model_device(model)
    ids_w, labels_w = collate(train, [brief(r) for r in train], device=device)
    ids_l, labels_l = collate(train, [incorrect(r) for r in train], device=device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001, weight_decay=0.)
    before = reward_accuracy(model, heldout)
    start, history = start_measurement(model), []
    for step in range(steps):
        loss = -F.logsigmoid(model(ids_w) - model(ids_l)).mean()
        optimize(loss, model, optimizer)
        if step == 0 or (step+1) % 10 == 0 or step+1 == steps:
            history.append({"step": step+1, "loss_before_update": loss.item()})
    # Full scorer input tokens count; reward models supervise a pair, not token CE.
    tokens = (ids_w.ne(0).sum()+ids_l.ne(0).sum()).item()*steps
    result = finish(model, optimizer, start, steps, tokens, history)
    result.update({"before": before, "train": reward_accuracy(model, train), "heldout": reward_accuracy(model, heldout),
                   "scope": "Correct-vs-wrong response ranking; labels are NOT features. This scorer is NOT the online RL reward."})
    return model, result


def gae_tensor(rewards, values, mask, gamma=1., lam=.95):
    """Each masked answer ends at EOS or the task's maximum-output horizon."""
    advantages = torch.zeros_like(values)
    for row in range(len(rewards)):
        tail, next_value = 0., 0.
        for index in torch.where(mask[row])[0].flip(0):
            delta = rewards[row, index] + gamma * next_value - values[row, index]
            tail = delta + gamma * lam * tail
            advantages[row, index] = tail
            next_value = values[row, index]
    return advantages, advantages + values


def train_online(model, reference, rows, steps, method, seed=107):
    if method not in {"ppo", "grpo"}:
        raise ValueError("expected ppo or grpo")
    device = model_device(model)
    if model_device(reference) != device:
        raise ValueError("policy and reference must use the same device")
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0004, weight_decay=0.)
    critic = ScoreModel(copy.deepcopy(model)) if method == "ppo" else None
    critic_optimizer = torch.optim.AdamW([p for p in critic.parameters() if p.requires_grad], lr=.001) if critic else None
    start, history, training_tokens, generated_tokens = start_measurement(model), [], 0, 0
    group_size = 4
    generator = torch.Generator(device=device).manual_seed(seed)
    for step in range(steps):
        prompts = [rows[(step*4+i) % len(rows)] for i in range(4)]
        batch = [r for r in prompts for _ in range(group_size)]
        answers = [generate(model, r["prompt"], sample=True, generator=generator) for r in batch]
        score = torch.tensor([assess(r, answer)["reward"] for r, answer in zip(batch, answers)],
                             device=device, dtype=next(model.parameters()).dtype)
        ids, labels = collate(batch, answers, device=device)
        with torch.no_grad():
            old = token_statistics(model, ids, labels)
            ref = token_statistics(reference, ids, labels)
            mask = old["mask"]
            if critic is not None:
                values = critic.token_values(ids[:, :-1])
                rewards = torch.zeros_like(values)
                for row in range(len(batch)):
                    rewards[row, torch.where(mask[row])[0][-1]] = score[row]
                advantages, returns = gae_tensor(rewards, values, mask)
                zero_groups = None
            else:
                grouped = score.reshape(-1, group_size)
                group_std = grouped.std(dim=1, unbiased=False)
                relative = ((grouped-grouped.mean(dim=1, keepdim=True))/(group_std[:, None]+1e-8)).reshape(-1)
                advantages = relative[:, None].expand_as(old["logps"])
                zero_groups = (group_std == 0).float().mean().item()
        for _ in range(2):
            stats = token_statistics(model, ids, labels)
            ratio = (stats["logps"] - old["logps"]).exp()
            surrogate = torch.minimum(ratio*advantages, ratio.clamp(.8, 1.2)*advantages)
            policy_loss = -(surrogate*mask).sum(dim=1).div(mask.sum(dim=1)).mean() if method == "grpo" else -masked_mean(surrogate, mask)
            kl = exact_kl(stats, ref)
            loss = policy_loss + .02*kl
            optimize(loss, model, optimizer)
            if critic is not None:
                critic_loss = .5*masked_mean((critic.token_values(ids[:, :-1])-returns)**2, mask)
                optimize(critic_loss, critic, critic_optimizer)
            training_tokens += mask.sum().item()
        generated_tokens += sum(map(len, answers))
        if step == 0 or (step+1) % 5 == 0 or step+1 == steps:
            entry = {"step": step+1, "loss_before_update": loss.item(), "rollout_reward": score.mean().item(),
                     "kl_on_sampled_prefixes": kl.item(), "entropy": entropy(stats).item(),
                     "mean_response_tokens": sum(map(len, answers))/len(answers),
                     "ratio_outside_clip_range": masked_mean((ratio-1).abs().gt(.2).float(),mask).item()}
            if zero_groups is not None:
                entry["zero_variance_group_fraction"] = zero_groups
            if critic is not None:
                entry["critic_loss"] = critic_loss.item()
            history.append(entry)
    result = finish(model, optimizer, start, steps, training_tokens, history)
    result.update({"generated_tokens": generated_tokens, "rollout_group_size": group_size,
                   "sampling_seed": seed,
                   "reward": "exact task verifier: brief correct=1, verbose correct=0.5, wrong=0",
                   "reference": "fixed shared SFT checkpoint", "kl_coefficient": .02,
                   "critic_parameters": parameter_counts(critic) if critic else None,
                   "critic_optimizer_tensor_bytes": optimizer_bytes(critic_optimizer) if critic else 0,
                   "cost_scope": "parameters, effective_training_tokens and optimizer_tensor_bytes describe the actor; wall time includes sampling, actor and critic updates",
                   "total_trainable_parameters": parameter_counts(model)["trainable"] + (parameter_counts(critic)["trainable"] if critic else 0),
                   "total_optimizer_tensor_bytes": optimizer_bytes(optimizer) + (optimizer_bytes(critic_optimizer) if critic else 0)})
    return result, optimizer


def sample_and_filter(teacher, rows, seed=207, teacher_name="verified-teacher"):
    candidates, selected = [], []
    generator = torch.Generator(device=model_device(teacher)).manual_seed(seed)
    for row in rows:
        accepted = []
        for attempt in range(4):
            response = generate(teacher, row["prompt"], sample=True, temperature=1.5, generator=generator)
            valid = assess(row, response)["correct"]
            candidates.append({"prompt_id": row["id"], "attempt": attempt, "response": response, "accepted": valid})
            if valid:
                accepted.append(response)
        if accepted:
            selected.append({**row, "response": accepted[0], "source": f"generated-by-{teacher_name}-and-verified"})
    return selected, {"candidates": candidates, "generated": len(candidates),
                      "accepted": sum(c["accepted"] for c in candidates), "selected": len(selected),
                      "selection_rule": "first correct candidate per training prompt; no gold-response fallback",
                      "temperature": 1.5}
