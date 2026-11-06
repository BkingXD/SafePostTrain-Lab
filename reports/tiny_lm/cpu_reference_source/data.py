"""Deterministic, hand-authored copy task; no benchmark or harmful prompts."""
import hashlib
import json
from .model import SYMBOLS


def record(template, symbol, style="brief"):
    prompt = ["<bos>", "<user>", *template, symbol, "<assistant>"]
    answer = (["answer"] if style == "verbose" else []) + [symbol, "<eos>"]
    return {"id": "-".join([*template, symbol, style]), "source": "authored-synthetic-copy-v1",
            "template_family": " ".join(template), "intent_group": symbol,
            "prompt": prompt, "response": answer, "symbol": symbol}


def datasets():
    train = [record(t, symbol, style) for t in [["repeat"], ["copy"], ["please", "copy"]]
             for symbol in SYMBOLS for style in ["brief", "verbose"]]
    heldout = [record(["please", "repeat"], symbol) for symbol in SYMBOLS]
    return train, heldout


def final_probe():
    """A second held-out combination, reserved until recipe/code freeze."""
    return [record(["repeat", "please"], symbol) for symbol in SYMBOLS]


def preference_rows():
    train, _ = datasets()
    return [r for r in train if r["response"][0] != "answer"]


def brief(row):
    return [row["symbol"], "<eos>"]


def verbose(row):
    return ["answer", row["symbol"], "<eos>"]


def incorrect(row):
    return [SYMBOLS[(SYMBOLS.index(row["symbol"]) + 1) % len(SYMBOLS)], "<eos>"]


def assess(row, response):
    correct = response in (brief(row), verbose(row))
    is_brief = response == brief(row)
    return {"correct": correct, "brief": is_brief,
            "reward": 1.0 if is_brief else .5 if correct else 0.0}


def data_card():
    train, heldout = datasets()
    digest = hashlib.sha256(json.dumps({"train": train, "development": heldout, "final_probe": final_probe()}, sort_keys=True).encode()).hexdigest()
    return {"source": "authored-synthetic-copy-v1", "train_rows": len(train), "development_rows": len(heldout), "final_probe_rows":len(final_probe()),
            "sha256": digest, "split": "development: please repeat; final probe: repeat please; SAME symbols/intents appear in training",
            "scope": "Tests template transfer on a tiny copy task, NOT unseen-intent generalization or LLM safety.",
            "development_use": "The initial held-out template was inspected during recipe diagnosis and is explicitly development data.",
            "license": "Authored synthetic data within this project; no third-party dataset. Repository distribution license is not selected.",
            "labels": "Correctness is exact matching; preference favors brief over verbose correct answers.",
            "external_data": False, "personal_data": False}
