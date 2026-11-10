# 使用开放权重模型：从生成到 SFT、DPO

本指南将已有微型模型机制实验连接到真正的开放权重语言模型。示例使用 RTX 5070、单卡 CUDA 和 [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)；数据只有 8 条合成训练样本和 3 条探针，默认两步训练。它用于理解模型、模板和训练接口，不是安全训练成果或 benchmark。

[下载真实模型教学源码包](../web/dist/downloads/safeposttrain-real-model.zip)，包含脚本、合成数据和依赖清单；不含模型权重。

## 1. 先选对模型和接口

| 模型 | 适用学习 | 接口与注意事项 |
|---|---|---|
| Qwen3-0.6B | 本项目实际运行的生成、LoRA SFT、DPO 流程 | 已后训练文本模型，支持 thinking/non-thinking；模板需明确 `enable_thinking`。模型及数据都很小，不能证明安全性能 |
| [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) | 更大但仍相对紧凑的开放模型；候选教学和小规模研究 | 相同家族不等于完全相同的显存开销。先读取模型卡和精确 revision，再按 GPU 实测最大长度与 batch |
| [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) | 2026 年多模态模型接口与新架构研究 | 官方模型卡列出视觉编码器与文本能力，使用相应 processor 和多模态模型类；不能直接假设本项目文本训练脚本支持它。超长上下文标称能力不等于本机可在该长度训练 |
| [Olmo 3](https://arxiv.org/abs/2512.13961) | 阅读开放模型生命周期、数据与配方 | 论文所述模型规模与本机资源不同；下载、许可证和训练预算需独立确认，本项目未复现其训练结果 |

Qwen3 模型卡说明：聊天模板可通过 `enable_thinking` 控制模式。训练和生成必须使用同一策略；不要先手写一个 `<think>` 再让模板重复添加。thinking 模式、输出长度、采样参数和推理 token 会影响比较结果。Qwen3.5 的 processor/多模态数据接口不同，参见其模型卡与本指南最后的说明。

## 2. 准备环境与固定模型版本

以下命令从项目根目录执行，需要兼容 NVIDIA 驱动与 CUDA 构建；首次获取权重需要网络和足够磁盘。请按 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/)选择适合机器的版本。下面是本机验证过的依赖组合，其他系统应重新检查兼容性。

```bash
python3.12 -m venv .venv-gpu
source .venv-gpu/bin/activate
python -m pip install -r labs/tiny_lm/requirements-gpu.txt
python -m pip install -r labs/real_model/requirements.txt
python -m labs.tiny_lm.run --check-device
```

本机已有 `.venv-gpu` 可直接激活；真实模型示例的完整安装版本见 [依赖锁](../labs/real_model/requirements-lock.txt)。训练权重与输出位于被 Git 忽略的 `artifacts/`。训练前确保 GPU 显存有余量；本示例最大长度为 256，增加长度、batch 或模型规模必须重新测量，不能只按权重文件大小估算。

本次验证使用的 TRL 版本调用 Triton CUDA 内核，其启动代码需要可用的 C 编译器。通常安装 `gcc` 或 `clang` 并确保其在 `PATH` 中；也可以通过 `CC` 指向编译器。本机验证使用已有工具链的项目内包装器，位于忽略的 `artifacts/toolchain/bin/cc`，脚本会在未发现其他编译器时使用它；其他机器需要自行准备工具链。

下载固定的模型 snapshot；本项目的训练脚本只从本地读取，不自动下载：

```bash
python -c 'from huggingface_hub import snapshot_download; snapshot_download(repo_id="Qwen/Qwen3-0.6B", revision="c1899de289a04d12100db370d81485cdf75e47ca", local_dir="artifacts/models/Qwen3-0.6B", allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"])'
```

报告记录本地模型文件 SHA256，以及 Hub 下载元信息中是否存在对应 revision。手工复制、删除元信息或更换文件时，不能仅凭目录名声称来源已核验。

## 3. 固定一个基线

```bash
python -m labs.real_model.run generate --output artifacts/real-model/my-base
```

脚本在 CUDA 上读取模型，使用 `enable_thinking=False`、固定贪心解码和最多 80 个新 token，对 3 条未用于梯度更新的合成探针保存 prompt、生成、环境和配置，并标记是否遇到长度上限。这些探针的主题与训练样本相近；数量、难度和样本来源都不满足独立安全 benchmark 要求。

## 4. 完成 LoRA SFT

```bash
python -m labs.real_model.run sft --steps 2 --output artifacts/real-model/my-sft
```

输入是 prompt/completion 形式的合成样本，TRL 的 `completion_only_loss=True` 只监督完成部分。脚本使用训练器真正的数据整理器检查每条样本：prompt 位置被屏蔽、回答未被截断、有效标签与输入 token 对齐；不通过就停止。LoRA 作用于 Qwen3 的 `q_proj` 和 `v_proj`，具体 rank、alpha、精度、有效标签及训练日志都写入报告。

`artifacts/real-model/my-sft/adapter/` 保存的是 adapter，需要绑定同一底模、tokenizer 和模板，不能单独当作完整模型。路径必须是新目录，避免覆盖已有记录。两步小数据更新只验证流程和显存，不能声称模型变得更安全。

## 5. 从同一个 SFT 起点进行 DPO

```bash
python -m labs.real_model.run dpo --steps 2 --adapter artifacts/real-model/my-sft/adapter --output artifacts/real-model/my-dpo
```

每一对数据由同一个 prompt、chosen 和 rejected 组成。脚本检查成对 token 非空且回答未丢失；在当前验证过的 TRL/PEFT 组合下，DPO 的参考 adapter 是载入的 SFT adapter 的冻结副本。脚本在训练前检查权重相同，训练后检查参考副本未改变。不同版本可能改变这一行为，必须重新核验。

```bash
python -m labs.real_model.run generate --adapter artifacts/real-model/my-sft/adapter --output artifacts/real-model/my-sft-probe
python -m labs.real_model.run generate --adapter artifacts/real-model/my-dpo/adapter --output artifacts/real-model/my-dpo-probe
```

三个生成报告使用同一探针与解码参数，保留原文，适合观察模板和运行流程。样本太少，不能计算可信的安全提升；重复查看这些探针并调参后，它们就是开发数据。真正研究要另设数据分组、独立测试、chosen-only SFT 对照、裁判复核与多 seed。

## 6. 本机实际运行记录

2026-11-10 已在 RTX 5070、CUDA 12.8、PyTorch 2.9.1+cu128、TRL 1.17.1、Transformers 5.22.2 上运行固定版本的 Qwen3-0.6B。SFT 与 DPO 各执行两次更新，8 条 SFT 样本的监督标签和 EOS 已核查，8 对 DPO 样本的实际 collator 输出和冻结参考已核查。两阶段均观察到 adapter 参数更新。

[原模型生成](../reports/real_model/base.json) · [SFT 训练](../reports/real_model/sft.json) · [DPO 训练](../reports/real_model/dpo.json) · [SFT 生成](../reports/real_model/sft_probe.json) · [DPO 生成](../reports/real_model/dpo_probe.json)。SFT 与 DPO 的 PyTorch 峰值已分配显存分别约为 1235.7 MiB 和 1316.7 MiB；这只统计该分配器，不等于设备总显存或实际大模型实验的资源需求。

三组生成各 3 条回答全部达到 80-token 上限，原始文本可能不完整，报告中逐条标有 `hit_max_new_tokens`；生成中的具体事实也未经核验。这些结果只证明当前环境下的流程和资源记录能够运行，**不支持训练改善回答质量或安全性的结论**。

## 7. 真实模型与框架调用时要核对的内容

- 样本结构要满足所用 TRL 版本的数据格式；[SFTTrainer](https://huggingface.co/docs/trl/sft_trainer)、[DPOTrainer](https://huggingface.co/docs/trl/dpo_trainer) 和 [GRPOTrainer](https://huggingface.co/docs/trl/grpo_trainer) 的配置会随版本变化，不要混用不同版本示例。
- SFT 先验证实际 `input_ids`、labels、EOS 和 truncation；DPO 额外核对 prompt 对齐、chosen/rejected 与冻结 reference；在线 RL 要保存策略版本、采样配置、验证结果与各项成本。
- 相同模型在不同量化、精度、上下文长度或推理引擎下可能出现不同输出。导出与合并后重新核验 tokenizer、模板和代表性任务。
- Qwen3.5 官方示例使用 `AutoProcessor` 与多模态模型类或 `image-text-to-text` pipeline。若尝试该模型，应安装与其模型卡兼容的图像依赖、保留图像/文本消息结构并先做单样本预检；本项目未声称已经验证其训练流程。

本项目的原始训练链是 [GPU 微型 Transformer](../labs/tiny_lm/README.md)，可先使用它理解每个 token 的因果位移和损失。完整安全研究流程与对照见 [后训练流程指南](POST_TRAINING_WORKFLOW.md)和[项目大纲](PROJECT_OUTLINE.md)。

## 8. 扩展：Qwen3.5 多模态接口

以下代码依据 [Qwen3.5-0.8B 官方模型卡](https://huggingface.co/Qwen/Qwen3.5-0.8B)和 [Transformers 多模态模板文档](https://huggingface.co/docs/transformers/main/en/chat_templating_multimodal)整理，用于理解多模态推理接口；**本项目尚未运行或验证此示例**。首次运行需要下载模型、图像及兼容的视觉依赖；确认 CUDA 显存和网络条件后再试。下面版本号和 revision 是阅读时固定的示例，不保证适配所有机器。

```python
import torch
from transformers import AutoProcessor, AutoModelForMultimodalLM

model_id = "Qwen/Qwen3.5-0.8B"
revision = "2fc06364715b967f1860aea9cf38778875588b17"
if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
    raise RuntimeError("需要支持 BF16 的 CUDA GPU")
processor = AutoProcessor.from_pretrained(model_id, revision=revision)
model = AutoModelForMultimodalLM.from_pretrained(
    model_id, revision=revision, dtype=torch.bfloat16
).to("cuda").eval()
messages = [{"role": "user", "content": [
    {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"},
    {"type": "text", "text": "请描述图片中能看到的内容。"},
]}]
inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt", enable_thinking=False,
).to("cuda")
with torch.inference_mode():
    outputs = model.generate(**inputs, max_new_tokens=128, do_sample=False)
answer = outputs[0, inputs["input_ids"].shape[-1]:]
print(processor.decode(answer, skip_special_tokens=True))
```

图像预处理依赖及模型输入随库版本变化。先核对 processor 输出键、图像顺序和实际 token 长度；若需要训练，还要选择可训练的视觉/投影/语言模块，并为图文联合场景单独建立数据和安全评测。本例不能代替这些验证。
