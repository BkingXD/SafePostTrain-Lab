# Qwen3 真实模型教学实验

下载包解压后，从包含 `labs/` 的目录执行。首先安装兼容的 CUDA 版 PyTorch，再安装真实模型训练依赖：

```bash
python3.12 -m venv .venv-gpu
source .venv-gpu/bin/activate
python -m pip install -r labs/tiny_lm/requirements-gpu.txt
python -m pip install -r labs/real_model/requirements.txt
python -c 'from huggingface_hub import snapshot_download; snapshot_download(repo_id="Qwen/Qwen3-0.6B", revision="c1899de289a04d12100db370d81485cdf75e47ca", local_dir="artifacts/models/Qwen3-0.6B", allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"])'
```

下载包不含模型权重。训练所需的 Triton CUDA 内核需要可用的 `gcc` 或 `clang`；如不在 PATH 中，可通过 `CC` 指定。准备后运行：

```bash
python -m labs.real_model.run generate --output artifacts/real-model/my-base
python -m labs.real_model.run sft --steps 2 --output artifacts/real-model/my-sft
python -m labs.real_model.run dpo --steps 2 --adapter artifacts/real-model/my-sft/adapter --output artifacts/real-model/my-dpo
python -m labs.real_model.run generate --adapter artifacts/real-model/my-dpo/adapter --output artifacts/real-model/my-dpo-probe
```

`data/demo_pairs.json` 只有 8 条合成训练样本和 3 条探针，专门用来验证接口，不是安全训练数据集或 benchmark。SFT 检查标签位置；DPO 检查参考 adapter 在训练前与 SFT 起点相同且之后保持冻结。报告记录数据/源码/权重哈希、版本、GPU、资源、输出及截断状态。需要遵循模型许可与资源限制，正式效果必须在独立数据、对照和评测协议下验证。

完整说明与后训练学习材料见 [SafePostTrain-Lab](https://github.com/BkingXD/SafePostTrain-Lab)。
