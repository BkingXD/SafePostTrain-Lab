# 后训练论文与模型研究地图

核查日期：2026-11-10。下表链接原始论文、模型卡和官方文档。预印本与技术报告的结论取决于论文设置；列出不表示本项目已经复现，也不是包揽所有新工作。每次复现前重新核对许可、版本、代码与评测协议。

## 先建立一张主线

| 主题 | 原始资料 | 阅读时要回答的问题 |
|---|---|---|
| 基础与全景 | [Transformer](https://arxiv.org/abs/1706.03762)、[后训练综述（2025）](https://arxiv.org/abs/2503.06072) | 因果注意力、下一 token、预训练与后训练边界分别是什么？综述将哪些方向放在同一框架？ |
| 完整开放配方 | [Tülu 3（2024）](https://arxiv.org/abs/2411.15124)、[Olmo 3（2025）](https://arxiv.org/abs/2512.13961) | 数据、SFT、偏好、可验证奖励与评测各产生什么产物？开放权重与开放完整数据/配方有什么区别？ |
| 真实模型与模式 | [Qwen3 技术报告（2025）](https://arxiv.org/abs/2505.09388)、[Qwen3-0.6B 模型卡](https://huggingface.co/Qwen/Qwen3-0.6B)、[Qwen3.5-0.8B 模型卡（2026）](https://huggingface.co/Qwen/Qwen3.5-0.8B) | Base/后训练模型、thinking 模式、模板和多模态 processor 如何选择？本机 GPU 在短上下文下能运行什么？ |
| 数据与模板 | [Datasheets for Datasets](https://arxiv.org/abs/1803.09010)、[TRL 数据格式](https://huggingface.co/docs/trl/dataset_formats)、[Transformers Chat Templates](https://huggingface.co/docs/transformers/chat_templating) | 如何追溯来源/许可/分组？训练与推理的模板、标签和特殊 token 怎样一致？ |
| SFT 与适配 | [TRL SFTTrainer](https://huggingface.co/docs/trl/sft_trainer)、[LoRA](https://arxiv.org/abs/2106.09685)、[QLoRA](https://arxiv.org/abs/2305.14314) | 哪些标签参与 loss？为什么低秩矩阵有 `r(d_in+d_out)` 参数？量化如何影响显存和加载？ |
| 奖励模型 | [Learning to Summarize from Human Feedback](https://arxiv.org/abs/2009.01325) | 成对排序学到的是哪个分数差？长度偏差与分布外失效如何检测？ |
| 离线偏好 | [DPO](https://arxiv.org/abs/2305.18290)、[SimPO](https://arxiv.org/abs/2405.14734)、[TRL DPOTrainer](https://huggingface.co/docs/trl/dpo_trainer) | reference、beta 与序列概率怎样进入目标？长度归一化或参考策略变更会怎样改变对比？ |
| 在线策略 | [PPO](https://arxiv.org/abs/1707.06347)、[GAE](https://arxiv.org/abs/1506.02438)、[DeepSeekMath](https://arxiv.org/abs/2402.03300) | actor/old/reference/critic 的职责是什么？GAE 的终止条件及 GRPO 零方差组如何处理？ |
| 推理与开放 RL | [DeepSeek-R1（2025）](https://arxiv.org/abs/2501.12948)、[DAPO（2025）](https://arxiv.org/abs/2503.14476)、[TRL GRPOTrainer](https://huggingface.co/docs/trl/grpo_trainer) | 冷启动、rollout、奖励、裁剪和生成预算分别解决什么问题？DAPO 与基础 GRPO 哪些设计不同？ |
| 奖励与探索新研究 | [Demystifying RL Post-Training（2026，预印本）](https://arxiv.org/abs/2608.24949)、[LLMs Gaming Verifiers（2026，预印本）](https://arxiv.org/abs/2604.15149) | 初始策略和任务分布如何影响探索？验证器可能检查错什么？论文的受控结论能否迁移到开放安全任务？ |
| 安全深度与退化 | [Safe RLHF](https://arxiv.org/abs/2310.12773)、[Shallow Safety Alignment（2024）](https://arxiv.org/abs/2406.05946)、[Beyond Shallow Alignment（2026）](https://arxiv.org/abs/2609.03887) | 帮助性与安全如何区分？后续微调、预填充和模型结构会怎样影响拒绝？论文的模型/攻击范围是什么？ |
| 评测与报告 | [HarmBench](https://github.com/centerforaisafety/HarmBench)、[XSTest](https://arxiv.org/abs/2308.01263)、[StrongREJECT](https://arxiv.org/abs/2402.10260)、[Model Cards](https://arxiv.org/abs/1810.03993) | 攻击预算、分母、误拒绝、裁判误差及能力保持如何记录？哪些结论不能从小样本推广？ |

## 论文精读模板

1. **问题与适用范围**：研究对象、起始模型、数据与威胁模型是什么？哪些是作者声明而不是独立复验？
2. **技术机制**：输入、loss/奖励、参考模型、归一化、更新步骤及产物分别是什么？写出一个可手算的例子。
3. **对照和预算**：与什么比较；是否控制训练 token、生成与裁判成本、调参预算？哪些重要对照缺失？
4. **证据与局限**：样本数、seed、分布外测试、裁判可靠性和失败案例；论文报告的结果能否支持作者的主张？
5. **本机实验**：本机 RTX 5070 适合验证哪个机制，哪些设置需要更大资源？写一个可能推翻预期的对照。

建议先读 Tülu 3 的整体配方，再分别精读 SFT/LoRA、DPO、PPO/GRPO 和安全评测；之后阅读 Qwen3/Olmo 3 与 2026 年研究，比较它们实际公开的代码与数据。2026 年预印本作为研究线索，需要留意后续版本及独立复现。

## 高 Star 项目：学习组织与工程参考

以下 Star 数为 **2026-11-10 检查时的近似值**，会随时间变化；均超过 1,000。前两项侧重学习教程，后两项是训练框架。这里参考其目录组织与实践入口，本站正文、图解和例题由本项目独立编写；使用其代码或材料时仍须遵守各自许可证。

| 项目 | 检查时约 Star | 借鉴方式与适合查阅的问题 |
|---|---:|---|
| [Datawhale Happy-LLM](https://github.com/datawhalechina/happy-llm) | 34.9k | 按基础、模型架构、训练与实践逐层组织；参考其图文和从原理到代码的学习路线。涉及其具体材料时核对 CC BY-NC-SA 许可。 |
| [Datawhale Self-LLM](https://github.com/datawhalechina/self-llm) | 32.7k | 以具体模型组织环境、部署与微调说明；借鉴模型版本、硬件及执行步骤并列展示的方式。 |
| [Hugging Face TRL](https://github.com/huggingface/trl) | 19.7k | 官方训练框架；查阅 SFT、DPO、GRPO 的数据格式、配置、参考策略及版本差异。本项目真实模型教学示例使用固定版本。 |
| [LlamaFactory](https://github.com/hiyouga/LlamaFactory) | 76.0k | 训练工程框架；参考其按模型和训练方式组织 YAML、命令和示例的结构。使用前核对实际版本、数据定义和硬件。 |

这些项目的 Star 是筛选参考来源的条件，并不单独证明技术结论正确。算法定义优先回到论文与相应版本的官方文档，运行效果仍应由自己的日志和独立评测支持。
