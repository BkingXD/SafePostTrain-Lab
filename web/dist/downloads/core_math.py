"""Standard-library, CPU-only numerical examples. Not an LLM trainer."""
import math


def softplus(x):
    return max(x, 0) + math.log1p(math.exp(-abs(x)))


def sigmoid(x):
    return 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))


def sft_loss(probabilities, mask):
    if len(probabilities) != len(mask) or not probabilities:
        raise ValueError("probabilities and mask need equal nonzero length")
    if any(not 0 < p <= 1 for p in probabilities) or any(m not in (0, 1) for m in mask):
        raise ValueError("invalid probability or mask")
    if not sum(mask):
        raise ValueError("no supervised tokens: mean loss is undefined")
    return -sum(m * math.log(p) for p, m in zip(probabilities, mask)) / sum(mask)


def dpo_loss(chosen, rejected, ref_chosen, ref_rejected, beta):
    if beta <= 0 or any(not math.isfinite(x) for x in [chosen, rejected, ref_chosen, ref_rejected, beta]):
        raise ValueError("beta must be positive and inputs finite")
    if any(x > 0 for x in [chosen, rejected, ref_chosen, ref_rejected]):
        raise ValueError("log probabilities cannot exceed zero")
    return softplus(-beta * ((chosen - rejected) - (ref_chosen - ref_rejected)))


def ppo_surrogate(ratio, advantage, epsilon=0.2):
    if ratio < 0 or not 0 <= epsilon < 1:
        raise ValueError("invalid ratio or clipping range")
    return min(ratio * advantage, min(1 + epsilon, max(1 - epsilon, ratio)) * advantage)


def group_advantages(rewards, epsilon=1e-8):
    if len(rewards) < 2 or any(not math.isfinite(x) for x in rewards) or epsilon <= 0:
        raise ValueError("need at least two finite rewards and positive epsilon")
    mean = sum(rewards) / len(rewards)
    std = math.sqrt(sum((r - mean) ** 2 for r in rewards) / len(rewards))
    return [(r - mean) / (std + epsilon) for r in rewards]


if __name__ == "__main__":
    print("教学计算，不代表真实模型成绩")
    print(f"SFT loss: {sft_loss([.8, .6, .5, .25], [0, 0, 1, 1]):.6f}")
    print(f"DPO loss: {dpo_loss(-2, -4, -3, -4, .1):.6f}")
    print(f"PPO positive-advantage surrogate: {ppo_surrogate(1.5, 1):.6f}")
    print(f"PPO negative-advantage surrogate: {ppo_surrogate(.5, -1):.6f}")
    print("GRPO advantages:", [round(x, 6) for x in group_advantages([0, 0, 1, 1])])
