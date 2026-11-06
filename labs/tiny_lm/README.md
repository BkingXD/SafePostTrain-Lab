# CPU 微型语言模型：把后训练串成一次真实运行

本实验使用 **19,281 参数、两层、四个注意力头的因果 Transformer**。它会真正计算 token 概率、反向传播和更新权重。数据只是复制 A～H 的合成任务，没有外部模型、真实用户信息或危险请求。

它帮助理解大模型后训练的机制，**不能代表大模型能力、安全 benchmark、算法排名或大规模训练效率**。浏览器展示保存的真实运行记录；训练本身在 Python 中进行。

## 先运行，再逐段阅读

本项目工作区已经准备好 Python 3.12 / PyTorch 2.9.1+cpu 环境：

```bash
.venv-teaching/bin/python -m labs.tiny_lm.run --output artifacts/my-tiny-lm-run --seed 7
.venv-teaching/bin/python -m unittest discover -s tests -p 'test_tiny_lm.py' -v
python3 labs/evaluate_measured.py
```

输出目录必须是一个尚未存在 `report.json` 的新目录，避免覆盖实验。最后一条命令不需要 PyTorch，会从项目内保存的原始回答重新计算指标，并核对源文件哈希。

下载实战包到另一台 Linux 机器后，先准备 Python 3.12，再在解压根目录运行：

```bash
python3.12 -m venv .venv-teaching
.venv-teaching/bin/python -m pip install --index-url https://download.pytorch.org/whl/cpu -r labs/tiny_lm/requirements-lock.txt
.venv-teaching/bin/python -m labs.tiny_lm.run --output artifacts/my-run --seed 7
```

已验证的是 Linux CPU、单 PyTorch 线程。未声明其他硬件上的耗时或逐位相同。依赖需要首次联网安装；模型权重与数据不需要下载。标准库入门实验仍可在没有 PyTorch 的环境运行。

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

## 必须读懂的失败案例

初次配方中，DPO loss 显著下降，但开发探针准确率为 0/8：模型经常直接输出 EOS。训练日志显示，chosen 的 log-prob 也下降了，只是 rejected 下降得更多。因此相对偏好目标改善，不等于回答质量改善。

这个负结果被完整保留。增加辅助 chosen-SFT 项，是一个用于演示的受控修复：`L = L_DPO + 0.2 × L_chosen_SFT`，并非声称发明了新算法，也不保证消除全部退化。SimPO 和等 token 的 chosen-only SFT 都保留为比较对象。

初次失败教师也降低了后续蒸馏的质量。当前流程在**训练侧**检查教师质量，按事先固定的候选顺序选择通过 95% 正确率门槛的教师，再生成、验证、筛选和训练学生。门槛不使用开发或最终探针结果。这里是同架构模型的序列蒸馏，没有展示模型压缩。

## 保存、恢复和复现的边界

模型 checkpoint 保存模型配置、权重、actor 优化器状态和 torch RNG。测试会显式恢复这些状态，并验证下一次 AdamW 更新一致；LoRA 也有独立恢复测试。

**没有实现完整在线 PPO 中断续训**：critic、其优化器和整个 rollout 队列未作为统一恢复状态保存。不要把 actor 保存/加载测试扩展为整套分布式 RL 的恢复保证。

报告保留 Python/PyTorch 版本、种子、预算、数据摘要、源码哈希、训练轨迹、每条探针回答、生成筛选记录和重新加载误差。同一环境下复跑时，计时会变化，数值与生成记录才是确定性核对对象。

## 建议完成的练习

1. 打印一条样本，指出第一次被监督的位置，解释为何不能再次 shift。
2. 只改用户 padding，确认有效回答 loss 与梯度不变。
3. 比较 full SFT 与 LoRA 的可训练参数、优化器状态、耗时与正确率，解释哪些成本没有计入。
4. 从 DPO 日志同时读取 chosen/rejected 的变化，再查看真实生成，找出“目标改善但输出退化”。
5. 查看 GRPO 的零方差组比例，解释为什么不能把所有 prompt 都当作有用的更新信号。
6. 只依据生成记录重新计算准确率，不读取预先汇总的分数。
7. 区分开发与最终探针，并说明为什么不能用这组结果给真实大模型算法排名。

## 对照的一手资料

- [PyTorch 交叉熵接口](https://docs.pytorch.org/docs/2.9/generated/torch.nn.functional.cross_entropy.html)：核对 logits、target 和 ignore_index 的含义。
- [DPO](https://arxiv.org/abs/2305.18290)、[SimPO](https://arxiv.org/abs/2405.14734)：比较目标函数与长度归一化。
- [PPO](https://arxiv.org/abs/1707.06347)、[GAE](https://arxiv.org/abs/1506.02438)、[DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300)：将代码映射到原始方法，并指出本教学实现的简化。
