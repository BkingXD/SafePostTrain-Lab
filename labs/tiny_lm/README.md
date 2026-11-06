# GPU 训练实战：把后训练串成一次真实运行

本实验使用 **19,281 参数、两层、四个注意力头的因果 Transformer**。它会真正计算 token 概率、反向传播和更新权重。数据只是复制 A～H 的合成任务，没有外部模型、真实用户信息或危险请求。

训练主流程使用 **CUDA GPU、FP32**，帮助理解大模型后训练的机制，**不能代表大模型能力、安全 benchmark、算法排名或大规模训练效率**。浏览器提供操作指南、GPU 实测和独立历史参考，训练本身在 Python 中进行。

已在 cuda:0 / NVIDIA GeForce RTX 5070 上完成全量训练，环境为 Python 3.12.14、PyTorch 2.9.1+cu128、CUDA 12.8、FP32。[gpu_seed7.json](../../reports/tiny_lm/gpu_seed7.json)保留 11 阶段的开发与最终探针、原始回答和资源测量；22 组结果及源码摘要已复核，权重重载 logits 最大差值为 0。历史 CPU `verified_seed7.json` 与 pilot 单列保留。[同环境复跑](../../reports/tiny_lm/gpu_reproducibility.json)在排除耗时、吞吐、RSS 和阶段显存峰值后，全部训练/开发/最终记录一致。21 项 TinyLM 与设备测试通过，其中 4 项真实 CUDA 测试实际执行、没有跳过。

## 先准备 GPU 环境，再做设备预检

本机已在沙箱外识别 RTX 5070 12GB，最初的沙箱访问失败并不表示驱动损坏。原 `.venv-teaching` 是 Python 3.12 / PyTorch 2.9.1+cpu；GPU 主流程已使用独立 `.venv-gpu` 完成全量运行，本机可直接 `source .venv-gpu/bin/activate` 复用，并在可以访问 GPU 的执行环境中运行。

建议使用独立环境。下面给出本机选定的安装组合；其他机器先按 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/)选择系统、Pip、Python 和与机器兼容的 CUDA 构建，再执行相应命令。不要使用 CPU 锁文件安装 GPU 主流程：

```bash
python3.12 -m venv .venv-gpu
source .venv-gpu/bin/activate
python -m pip install torch==2.9.1 numpy==2.2.6 --index-url https://download.pytorch.org/whl/cu128
```

基础依赖见 [requirements-gpu.txt](requirements-gpu.txt)，完整已安装版本见 [requirements-gpu-lock.txt](requirements-gpu-lock.txt)。同类 Linux 环境可在激活后使用 `python -m pip install -r labs/tiny_lm/requirements-gpu-lock.txt` 安装锁定版本。这里采用的 PyTorch 2.9.1 / CUDA 12.8 构建来自[官方版本安装表](https://pytorch.org/get-started/previous-versions/)。其他机器应先确认驱动和硬件兼容性，再选择对应构建；不要把本机组合当作所有 GPU 的通用要求。

安装完成后，从项目根目录运行：

```bash
nvidia-smi
python -c 'import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())'
python -m labs.tiny_lm.run --check-device
```

`nvidia-smi` 检查系统驱动访问，PyTorch 输出检查当前解释器及 CUDA 构建，项目预检负责确认训练可使用所选设备。仅“安装成功”或“看见显卡名称”都不代替成功的 CUDA 执行；[PyTorch 官方说明](https://pytorch.org/get-started/locally/)也建议使用 `torch.cuda.is_available()` 检查可用性。

训练入口默认 `--device cuda`。CUDA 不可用时直接失败，不静默使用 CPU；预检失败时先解决环境，不把失败改记为训练完成。

## 在 GPU 上执行完整流程

```bash
python -m labs.tiny_lm.run --device cuda --final-eval --output artifacts/tiny-lm-cuda-seed7 --seed 7
```

使用新的输出目录，避免覆盖结果。当前目标精度为 FP32；没有宣称 BF16、QLoRA、FSDP 或多卡训练已经实现。GPU 报告应保留设备、PyTorch/CUDA 版本、精度、预算、原始回答及测量记录；实际设备和运行日志才是 GPU 完成证据。

验收顺序：设备预检通过 → 关键测试 → 完整训练 → 检查报告设备/精度 → 重算原始回答指标 → 检查保存/加载与实际资源开销。测试和训练命令必须使用同一个已激活的 GPU 环境。模型太小、任务太短，GPU 耗时可能由启动与调度主导，不能据此概括真实大模型的加速比。

显存字段来自 `torch.cuda.max_memory_allocated` / `max_memory_reserved`，仅覆盖所选设备的 PyTorch 分配器。阶段峰值包含开始时已驻留的模型和缓存，不含 CUDA 上下文及非 PyTorch 外部库显存；不能解释为全进程总显存或该算法独占的开销。

## 历史 CPU 参考与兼容性验证

`requirements.txt` 和 `requirements-lock.txt` 中的 `torch==2.9.1+cpu` 只描述此前 CPU 环境，不能作为 GPU 安装入口。历史报告的源码摘要对应历史代码；GPU 迁移后不能继续声称它们匹配当前源码。

确需验证 CPU 兼容性时必须显式指定：

```bash
python -m labs.tiny_lm.run --device cpu --output artifacts/tiny-lm-cpu-compat-seed7 --seed 7
```

这条运行只证明兼容路径，不完成 GPU 实测。标准库数学练习仍可独立运行；本项目训练主线保持 GPU 优先。

审计当前 GPU 报告，以及对照冻结源码审计历史 CPU 报告：

```bash
python labs/evaluate_measured.py
python labs/evaluate_measured.py --report reports/tiny_lm/verified_seed7.json --source-dir reports/tiny_lm/cpu_reference_source
```

两条命令都只复算已保存的原始输出和核验对应源码，不会重新训练。GPU 默认报告为 `gpu_seed7.json`；CPU 历史源在 `reports/tiny_lm/cpu_reference_source/`。

## 任务、数据和切分

例子：`<bos> <user> repeat A <assistant>`，允许回答 `A <eos>` 或 `answer A <eos>`。偏好阶段在两种都正确的回答里偏好简短版本。

| 集合 | 内容 | 用途 |
|---|---|---|
| 训练 | repeat、copy、please copy 三种模板 × 8 个符号 × 两种回答风格，共 48 行 | 更新参数 |
| 开发探针 | please repeat × 8 个符号 | 已用于诊断，不能冒充独立测试 |
| 最终探针 | repeat please × 8 个符号 | 配方冻结后通过 `--final-eval` 额外评测 |

符号和任务意图在各集合重叠；这里仅测试模板组合迁移，不是未见意图泛化。8 个样本不足以支持算法优劣结论。若根据最终探针结果再调参，应将它视为开发数据并另留测试。

来源、标签、切分和 SHA256 由 `data_card()` 写入报告。代码不会把验证失败的生成替换为标准答案。

## 逐段看代码

| 环节 | 读哪里 | 观察什么 |
|---|---|---|
| 因果模型 | `model.py` 的 `CausalBlock`、`TinyLM` | Q/K/V、未来位置 mask、padding 与残差 |
| SFT | `collate`、`token_statistics`、`train_sft` | 只 shift 一次，prompt 不计 loss，首个回答 token 和 EOS 计入 |
| LoRA | `LoRALinear`、`apply_lora` | B 初始为零；底模、embedding 和 norm 冻结；只更新 A/B |
| 奖励模型 | `ScoreModel`、`train_reward` | 用真实隐藏状态学习正确/错误回答的偏好；没有 oracle 正确性输入特征 |
| 偏好优化 | `preference_loss` | DPO 参考差分、SimPO 长度平均、辅助 SFT 项 |
| 在线训练 | `train_online`、`gae_tensor` | 当前策略 rollout、独立采样随机流、旧 log-prob、reference、critic 与组优势 |
| 生成筛选 | `sample_and_filter` | 记录全部候选，仅保留实际通过验证的回答 |
| 评测与保存 | `run.py`、`evaluate_measured.py`、`checkpoint` | 原始生成、开发/最终探针、源文件哈希、保存/加载一致性 |

SFT 的输入和 label 对齐表会保存为 `mask_walkthrough`。比如输入 `<assistant>` 的位置预测第一个回答 token；用户 token 虽然不贡献监督 loss，仍参与上下文计算。

## 对照设计与成本

- 全参数 SFT 与 LoRA 从同一个热身 checkpoint 开始，使用相同数据、更新次数与有效回答 token 预算。学习率不同，因此这是一个受控教学比较，不是充分调参后的算法优劣证明。
- DPO、DPO+辅助 SFT、SimPO、PPO、GRPO 都从同一个 SFT checkpoint 分支。
- chosen-only SFT 使用与 DPO 相同的有效回答 token 预算；其更新次数更多。不能同时把“等 token”和“等更新次数”混为一谈。
- 在线训练的奖励来自精确规则：简短正确=1，较长正确=0.5，错误=0。训练出来的奖励模型没有被偷偷当作同一个验证器；本例并未宣称完整 RM 驱动的 RLHF。
- PPO 的基础参数数目、有效训练 token、optimizer 字段描述 actor；报告另列 critic 与总优化器状态。耗时包含采样和更新。优化器字节不是总峰值显存。

PPO 使用一个独立 critic；每个回答的奖励放在末尾，GAE 在回答 token 上倒序计算。EOS 和最大输出长度都按本教学任务的终止 horizon 处理。它不是支持任意环境截断的通用 trainer。

## 从实际 GPU 运行读懂失败案例

当前 GPU 报告中，DPO 的开发与最终探针均为 0/8，而 DPO + 辅助 SFT 的最终探针为 8/8。结合日志和逐条回答检查：偏好目标改善不保证任务完成，不能只看 loss。该负结果也曾在 CPU 配方中出现；各设备的数据分别保存，不要求所有阶段结果相同。仅 8 个样例不能支持算法排名。

这个负结果被完整保留。增加辅助 chosen-SFT 项，是一个用于演示的受控修复：`L = L_DPO + 0.2 × L_chosen_SFT`，并非声称发明了新算法，也不保证消除全部退化。SimPO 和等 token 的 chosen-only SFT 都保留为比较对象。

初次失败教师也降低了后续蒸馏的质量。当前流程在**训练侧**检查教师质量，按事先固定的候选顺序选择通过 95% 正确率门槛的教师，再生成、验证、筛选和训练学生。门槛不使用开发或最终探针结果。这里是同架构模型的序列蒸馏，没有展示模型压缩。

## 保存、恢复和复现的边界

真实 CUDA 测试已对全参数与 LoRA 模型检查 checkpoint：恢复模型与 AdamW 状态、CPU/CUDA 两种 RNG，核对张量设备和 logits，并验证下一次更新一致。这是限定范围的 actor 恢复证据，不覆盖整个在线训练系统。

**没有实现完整在线 PPO 中断续训**：critic、其优化器和整个 rollout 队列未作为统一恢复状态保存。不要把 actor 保存/加载测试扩展为整套分布式 RL 的恢复保证。

报告保留 Python/PyTorch 版本、设备与精度、种子、预算、数据摘要、源码哈希、训练轨迹、每条探针回答、生成筛选记录和重新加载误差。不同设备/软件环境的数值与耗时可能不同，不要求 GPU 与历史 CPU 逐位相同；先在同一 GPU 环境复跑并说明允许的数值差异。

## 建议完成的练习

1. 打印一条样本，指出第一次被监督的位置，解释为何不能再次 shift。
2. 只改用户 padding，确认有效回答 loss 与梯度不变。
3. 在同一 GPU 环境比较 full SFT 与 LoRA 的可训练参数、优化器状态、耗时、显存与正确率，解释哪些成本没有计入；不要把历史 CPU 耗时填入 GPU 表格。
4. 从 DPO 日志同时读取 chosen/rejected 的变化，再查看真实生成，找出“目标改善但输出退化”。
5. 查看 GRPO 的零方差组比例，解释为什么不能把所有 prompt 都当作有用的更新信号。
6. 只依据生成记录重新计算准确率，不读取预先汇总的分数。
7. 区分开发与最终探针，并说明为什么不能用这组结果给真实大模型算法排名。

## 对照的一手资料

- [PyTorch 交叉熵接口](https://docs.pytorch.org/docs/2.9/generated/torch.nn.functional.cross_entropy.html)：核对 logits、target 和 ignore_index 的含义。
- [DPO](https://arxiv.org/abs/2305.18290)、[SimPO](https://arxiv.org/abs/2405.14734)：比较目标函数与长度归一化。
- [PPO](https://arxiv.org/abs/1707.06347)、[GAE](https://arxiv.org/abs/1506.02438)、[DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300)：将代码映射到原始方法，并指出本教学实现的简化。
