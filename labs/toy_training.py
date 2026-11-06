"""Train a one-parameter, two-action policy using SFT/DPO/PPO/GRPO.

This is a one-step bandit for inspecting updates, not language-model training.
Action 1 has reward 1; action 0 has reward 0. All methods start at p(1)=0.5.
The comparison is educational, not a fair real-world algorithm benchmark.
"""
import argparse
import json
import math
import random
from core_math import sigmoid, group_advantages


def train(method, steps=80, seed=7, learning_rate=0.2):
    rng = random.Random(seed)
    theta, value = 0.0, 0.5
    history = []
    for step in range(steps):
        old_p = sigmoid(theta)
        if method == "sft":
            # Cross-entropy derivative for a supervised action-1 target.
            theta -= learning_rate * (old_p - 1)
        elif method == "dpo":
            # log pi(1) - log pi(0) = theta; reference log-odds = 0.
            beta = 0.5
            theta += learning_rate * beta * sigmoid(-beta * theta)
        else:
            actions = [int(rng.random() < old_p) for _ in range(16)]
            rewards = list(actions)
            advantages = group_advantages(rewards) if method == "grpo" else [r - value for r in rewards]
            if method == "ppo":
                value += 0.2 * (sum(rewards) / len(rewards) - value)
            # Two epochs on the fixed rollout, retaining its old probabilities.
            for _ in range(2):
                p = sigmoid(theta)
                gradient = 0.0
                for action, advantage in zip(actions, advantages):
                    ratio = (p / old_p) if action else ((1 - p) / (1 - old_p))
                    clipped = max(.8, min(1.2, ratio))
                    if ratio * advantage <= clipped * advantage:
                        gradient += ratio * advantage * (action - p)
                gradient /= len(actions)
                # Exact KL derivative to a fixed Bernoulli(0.5) reference.
                gradient -= .02 * p * (1 - p) * theta
                theta += learning_rate * gradient
        if step == 0 or (step + 1) % 10 == 0 or step == steps - 1:
            history.append({"step": step + 1, "p_action_1": round(sigmoid(theta), 6)})
    return {"method": method, "seed": seed, "steps": steps, "initial_probability": .5, "final_probability": sigmoid(theta), "history": history}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["sft", "dpo", "ppo", "grpo", "all"], default="all")
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if not 1 <= args.steps <= 10000:
        parser.error("steps must be between 1 and 10000")
    methods = ["sft", "dpo", "ppo", "grpo"] if args.method == "all" else [args.method]
    print(json.dumps({"scope": "teaching-only one-step bandit; not an LLM benchmark", "results": [train(m, args.steps, args.seed) for m in methods]}, indent=2))
