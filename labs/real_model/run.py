"""Single-GPU Qwen3 SFT/DPO teaching run on a tiny synthetic dataset.

Requires a locally downloaded, pinned model snapshot. This is a pipeline check,
not evidence of a safe model or benchmark improvement.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import time
from pathlib import Path

from .data_validation import DATA, load_data

MODEL_ID = "Qwen/Qwen3-0.6B"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"


def prepare_compiler():
    """TRL's fused CUDA kernel requires a host compiler for its launcher."""
    compiler = os.environ.get("CC")
    if compiler:
        if not shutil.which(compiler):
            raise RuntimeError("CC does not point to an executable C compiler")
        return compiler
    compiler = shutil.which("gcc") or shutil.which("clang")
    if compiler:
        return compiler
    # Optional, workspace-local toolchain prepared for this machine. It is not
    # included in the source bundle and never changes system configuration.
    local = Path(__file__).resolve().parents[2] / "artifacts/toolchain/bin/cc"
    if local.is_file() and os.access(local, os.X_OK):
        os.environ["CC"] = str(local)
        return str(local)
    raise RuntimeError("TRL CUDA kernels need gcc or clang; install a C compiler or set CC")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_provenance(directory):
    """Check Hub metadata when present; do not invent provenance for a copy."""
    directory = Path(directory)
    files = ("config.json", "tokenizer.json", "model.safetensors")
    missing = [name for name in files if not (directory / name).is_file()]
    if missing:
        raise ValueError(f"model snapshot is incomplete: {missing}")
    metadata = directory / ".cache/huggingface/download"
    revisions = []
    for name in files:
        path = metadata / f"{name}.metadata"
        if path.is_file():
            revisions.append(path.read_text().splitlines()[0])
    if revisions and any(rev != MODEL_REVISION for rev in revisions):
        raise ValueError("local snapshot metadata has a different revision")
    return {"expected_model_id": MODEL_ID, "declared_revision": MODEL_REVISION,
            "revision_from_all_hub_metadata": len(revisions) == len(files),
            "file_sha256": {name: file_hash(directory / name) for name in files}}


def examples(rows, stage):
    result = []
    for row in rows:
        item = {"prompt": [{"role": "user", "content": row["prompt"]}],
                "chat_template_kwargs": {"enable_thinking": False}}
        chosen = [{"role": "assistant", "content": row["chosen"]}]
        if stage == "sft":
            item["completion"] = chosen
        else:
            item["chosen"] = chosen
            item["rejected"] = [{"role": "assistant", "content": row["rejected"]}]
        result.append(item)
    return result


def inspect_sft_mask(trainer, tokenizer, rows):
    """Inspect the actual trainer collator, including truncation and labels."""
    total = 0
    first = None
    for index, row in enumerate(rows):
        batch = trainer.data_collator([trainer.train_dataset[index]])
        ids, labels = batch["input_ids"][0], batch["labels"][0]
        mask = labels != -100
        if not mask.any() or mask.all():
            raise ValueError(f"SFT sample {row['id']} lacks prompt mask or supervised completion")
        decoded = tokenizer.decode(labels[mask].tolist(), skip_special_tokens=True).strip()
        if row["chosen"] not in decoded:
            raise ValueError(f"SFT sample {row['id']} completion missing/truncated: {decoded!r}")
        if not (labels[mask] == ids[mask]).all():
            raise ValueError("supervised labels do not match the token sequence")
        if tokenizer.eos_token_id not in labels[mask].tolist():
            raise ValueError(f"SFT sample {row['id']} is missing supervised EOS")
        total += int(mask.sum().item())
        if first is None:
            first = {"sample_id": row["id"], "prompt_masked_tokens": int((~mask).sum().item()),
                     "supervised_tokens": int(mask.sum().item()), "supervised_text": decoded,
                     "eos_supervised": True}
    return {"examples_checked": len(rows), "supervised_tokens_across_examples": total, "first_sample": first}


def inspect_dpo(trainer, tokenizer, rows):
    """Check paired completion tokens and the frozen copy of the SFT adapter."""
    if len(trainer.train_dataset) != len(rows):
        raise ValueError("some preference prompts were removed or truncated")
    for index, row in enumerate(rows):
        item = trainer.train_dataset[index]
        if not item["prompt_ids"] or not item["chosen_ids"] or not item["rejected_ids"]:
            raise ValueError(f"DPO sample {row['id']} has an empty token segment")
        batch = trainer.data_collator([item])
        for position, column in enumerate(("chosen", "rejected")):
            completion = batch["input_ids"][position][batch["completion_mask"][position].bool()].tolist()
            decoded = tokenizer.decode(completion, skip_special_tokens=True).strip()
            if row[column] not in decoded:
                raise ValueError(f"DPO {column} for {row['id']} is missing or truncated")
            if tokenizer.eos_token_id not in completion:
                raise ValueError(f"DPO {column} for {row['id']} is missing completion EOS")
    model = trainer.model
    if "ref" not in model.peft_config or "default" not in model.peft_config:
        raise ValueError("TRL did not create a frozen reference copy of the SFT adapter")
    paired = []
    for name, param in model.named_parameters():
        if ".default." in name:
            ref = model.get_parameter(name.replace(".default.", ".ref."))
            if ref.requires_grad or not (ref.detach() == param.detach()).all().item():
                raise ValueError("reference adapter is not a frozen copy of the SFT start")
            paired.append((ref, ref.detach().clone()))
    if not paired:
        raise ValueError("no policy/reference adapter parameters found")
    return paired


def run(stage, model_dir, output, steps, adapter=None, seed=7):
    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        raise RuntimeError("An accessible CUDA GPU is required")
    compiler = prepare_compiler() if stage in {"sft", "dpo"} else None
    if stage == "dpo" and not adapter:
        raise ValueError("DPO requires --adapter from the SFT stage")
    if steps < 1:
        raise ValueError("steps must be positive")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory is not empty; choose a new directory")
    data, data_hash = load_data()
    provenance = snapshot_provenance(model_dir)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda:0")
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    output.mkdir(parents=True, exist_ok=True)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype, local_files_only=True,
                                                 trust_remote_code=False).to(device)
    model.config.use_cache = False
    reference = []
    metrics = {}
    adapter_hash = None
    if adapter:
        adapter_dir = Path(adapter)
        adapter_hash = {name: file_hash(adapter_dir / name) for name in
                        ("adapter_config.json", "adapter_model.safetensors")}
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    if stage in {"sft", "dpo"}:
        dataset = Dataset.from_list(examples(data["train"], stage))
        common = dict(output_dir=str(output / "trainer"), max_steps=steps,
                      per_device_train_batch_size=1, gradient_accumulation_steps=1,
                      gradient_checkpointing=False, bf16=dtype == torch.bfloat16,
                      fp16=False, optim="adamw_torch", logging_steps=1,
                      save_strategy="no", report_to="none", seed=seed,
                      disable_tqdm=True, max_length=256)
        if stage == "sft":
            config = SFTConfig(**common, learning_rate=2e-4, completion_only_loss=True)
            lora = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0,
                              target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
            trainer = SFTTrainer(model=model, args=config, train_dataset=dataset,
                                 processing_class=tokenizer, peft_config=lora)
            metrics["mask_check"] = inspect_sft_mask(trainer, tokenizer, data["train"])
        else:
            model = PeftModel.from_pretrained(model, adapter, is_trainable=True)
            config = DPOConfig(**common, learning_rate=1e-5, beta=0.1, loss_type="sigmoid")
            trainer = DPOTrainer(model=model, ref_model=None, args=config, train_dataset=dataset,
                                 processing_class=tokenizer)
            reference = inspect_dpo(trainer, tokenizer, data["train"])
            metrics["preference_pairs_checked"] = len(data["train"])
            metrics["reference"] = "frozen copy of the loaded SFT adapter at DPO start"
        trainable = [(name, p, p.detach().clone()) for name, p in trainer.model.named_parameters() if p.requires_grad]
        if not trainable or any("lora_" not in name or ".ref." in name for name, _, _ in trainable):
            raise ValueError("unexpected trainable parameters; inspect adapter configuration")
        metrics["trainable_parameters"] = sum(p.numel() for _, p, _ in trainable)
        result = trainer.train()
        metrics["updated_parameter_tensors"] = sum(not torch.equal(p.detach(), before) for _, p, before in trainable)
        if not metrics["updated_parameter_tensors"]:
            raise RuntimeError("training completed without changing adapter parameters")
        if reference and any(not torch.equal(p.detach(), before) for p, before in reference):
            raise RuntimeError("reference adapter changed during DPO")
        metrics["reference_unchanged"] = bool(reference) if stage == "dpo" else None
        metrics["global_step"] = int(result.global_step)
        metrics["training_loss"] = float(result.training_loss)
        metrics["training_log"] = trainer.state.log_history
        trainer.model.save_pretrained(output / "adapter", selected_adapters=["default"])
        tokenizer.save_pretrained(output / "adapter")
        metrics["output_adapter_sha256"] = {name: file_hash(output / "adapter" / name) for name in
                                            ("adapter_config.json", "adapter_model.safetensors")}
    elif stage == "generate":
        if adapter:
            model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
        model.eval()
        outputs = []
        for row in data["probe"]:
            messages = [{"role": "user", "content": row["prompt"]}]
            batch = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                                  return_dict=True, return_tensors="pt",
                                                  enable_thinking=False).to(device)
            with torch.inference_mode():
                generated = model.generate(**batch, max_new_tokens=80, do_sample=False,
                                           pad_token_id=tokenizer.pad_token_id)
            new_ids = generated[0, batch["input_ids"].shape[1]:]
            answer = tokenizer.decode(new_ids, skip_special_tokens=True)
            eos_ids = model.generation_config.eos_token_id
            if isinstance(eos_ids, int):
                eos_ids = [eos_ids]
            ended_on_eos = bool(len(new_ids) and eos_ids and int(new_ids[-1]) in eos_ids)
            outputs.append({"id": row["id"], "category": row["category"],
                            "prompt": row["prompt"], "generated": answer,
                            "generated_tokens": len(new_ids), "ended_on_eos": ended_on_eos,
                            "hit_max_new_tokens": len(new_ids) == 80 and not ended_on_eos})
        metrics["outputs"] = outputs
        metrics["decode"] = {"enable_thinking": False, "do_sample": False, "max_new_tokens": 80}
    else:
        raise ValueError("unsupported stage")
    torch.cuda.synchronize(device)
    report = {"scope": "Small synthetic pipeline demonstration, not a safety benchmark or evidence of improvement.",
              "stage": stage, "model": provenance, "model_dir": str(model_dir), "input_adapter": str(adapter) if adapter else None,
              "input_adapter_sha256": adapter_hash,
              "run_config": {"requested_steps": steps if stage != "generate" else None, "max_length": 256,
                             "micro_batch": 1, "gradient_accumulation": 1, "optimizer": "adamw_torch",
                             "learning_rate": 2e-4 if stage == "sft" else (1e-5 if stage == "dpo" else None),
                             "lora": {"r": 8, "alpha": 16, "dropout": 0.0, "target_modules": ["q_proj", "v_proj"]},
                             "beta": 0.1 if stage == "dpo" else None, "enable_thinking": False},
              "dataset_sha256": data_hash, "dataset_counts": {"train": len(data["train"]), "probe": len(data["probe"])},
              "source_sha256": {p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
              "environment": {"python": os.sys.version.split()[0], "gpu": torch.cuda.get_device_name(device),
                              "device": str(device), "precision": str(dtype), "cuda": torch.version.cuda,
                              "c_compiler": ("workspace-local toolchain" if compiler and "/artifacts/toolchain/" in compiler else compiler),
                              **{name: importlib.metadata.version(name) for name in
                                 ("torch", "transformers", "trl", "peft", "datasets", "accelerate")}},
              "seed": seed, "elapsed_seconds_after_model_load": time.perf_counter() - start,
              "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
              "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(device), "metrics": metrics}
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"stage": stage, "report": str(output / "report.json"),
                      "gpu": report["environment"]["gpu"], "metrics": metrics}, ensure_ascii=False))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["sft", "dpo", "generate"])
    parser.add_argument("--model-dir", default="artifacts/models/Qwen3-0.6B")
    parser.add_argument("--output", required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    run(args.stage, args.model_dir, args.output, args.steps, args.adapter, args.seed)


if __name__ == "__main__":
    main()
