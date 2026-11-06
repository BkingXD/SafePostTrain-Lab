"""Fit a two-feature Bradley-Terry reward model on synthetic preferences.

Feature 1: an oracle correctness label; feature 2: answer length / 10.
This exposes how pairwise learning works. Real LLM reward models do NOT get
an oracle correctness feature and require learned representations and auditing.
"""
import json
import random
from core_math import sigmoid, softplus


def fit(seed=7, epochs=100):
    rng = random.Random(seed)
    pairs = [([1., rng.uniform(.1, 1)], [0., rng.uniform(.1, 1)]) for _ in range(80)]
    weights = [0., 0.]
    for _ in range(epochs):
        rng.shuffle(pairs)
        for chosen, rejected in pairs:
            delta = [a-b for a,b in zip(chosen, rejected)]
            margin = sum(w*x for w,x in zip(weights, delta))
            for i in range(2):
                weights[i] += .03 * sigmoid(-margin) * delta[i]
    # Evaluate a short correct answer against a long incorrect answer.
    margin = weights[0] + weights[1] * (.1 - 1.)
    return {"scope": "synthetic linear reward model, not an LLM", "weights": weights, "short_correct_vs_long_incorrect_probability": sigmoid(margin)}


if __name__ == "__main__":
    print(json.dumps(fit(), indent=2))
