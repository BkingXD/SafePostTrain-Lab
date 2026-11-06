# GPU 实验验收记录

日期：2026-11-07（Asia/Shanghai）。模型训练仅支持 CUDA GPU。

## 运行环境

- NVIDIA GeForce RTX 5070，驱动 595.99.02。
- `.venv-gpu`：Python 3.12.14、PyTorch 2.9.1+cu128、CUDA runtime 12.8、FP32、cuda:0。
- 依赖与完整版本锁分别保存在 `labs/tiny_lm/requirements-gpu.txt` 和 `labs/tiny_lm/requirements-gpu-lock.txt`。
- 训练入口只接受 `cuda` / `cuda:N`。缺少可用 CUDA 构建、设备或内核时，在创建训练产物前退出。

## 训练、测量与恢复

模型、LoRA、奖励/价值头、输入、策略参考及采样 Generator 都跟随目标 GPU。检查点保存模型、优化器及主机/GPU 随机状态，支持设备映射和限定范围的下一步恢复。

GPU 计时在边界同步。已分配与保留显存峰值来自 PyTorch 分配器，包含阶段开始驻留对象和缓存；不含 CUDA 上下文与外部库分配，也不等于某算法独占显存。

当前报告由以下完整运行生成：

```bash
.venv-gpu/bin/python -m labs.tiny_lm.run --device cuda --final-eval --output artifacts/tiny-lm-gpu-only-seed7 --seed 7
.venv-gpu/bin/python -m labs.tiny_lm.run --device cuda --final-eval --output artifacts/tiny-lm-gpu-only-replay-seed7 --seed 7
```

[主报告](tiny_lm/gpu_seed7.json)、[复跑报告](tiny_lm/gpu_replay_seed7.json)、[独立比对摘要](tiny_lm/gpu_reproducibility.json)已保存。两份完整记录在排除耗时、吞吐、RSS 和显存峰值后完全一致，包含所有训练、开发与最终生成记录。此结论仅适用于当前硬件、软件与同 seed。

11 个模型阶段均保留开发与最终探针，每次运行 22 组指标可由原始回答复算。源文件 SHA256 与当前实现一致，权重重载 logits 差值为 0。DPO 负结果保留，不用训练 loss 下降代替生成质量评测。

## 检查结果

共 56 项检查通过：

- 21 项模型不变量、训练路径、设备拒绝和预检契约；模型测试在实际 CUDA 设备执行，检查因果性、shift、padding 梯度、LoRA、DPO、GAE、采样、全参/LoRA 的优化器和随机状态恢复。
- 19 项公式、下载包和报告审计；包含原始回答、标签与汇总篡改检测、非 CUDA 证据拒绝、源文件核验、两份复跑记录及摘要哈希核验。
- 16 项课程、数值交互和生成一致性检查。

实验下载包包含 17 个文件，涵盖 GPU 源码、依赖、测试以及三份完整实验记录/摘要；解压后的报告审计通过。课程、实战页面与下载项统一使用 GPU 流程。

## 实验边界

当前为 19,281 参数的合成符号任务，不代表开放权重大模型能力、安全 benchmark 或算法排名。BF16、QLoRA、多卡和正式安全评测属于后续扩展。完整在线 PPO 续训还需要统一保存 critic、其优化器及 rollout 状态，不能由 actor 恢复测试代替。

## 页面与发布

本地浏览器已核对 GPU 实战界面及下载入口，见 [验证截图](gpu-practice.png)。发布包已核对仅含当前资源。

[私人学习站](https://safeposttrain-learning-lab.cappedcone1.chatgpt.site)保持原私有访问，发布结果为 `succeeded`。线上访问需要站点所有者登录；交互检查使用与发布一致的本地静态构建。

- Sites 源码提交：`de8763508b7a62d976a2a3a0db1624a77329b7a2`
- 版本：`appgprj_6aeda54c30e881918ae0a8e64e7ed7e1~appgver_4be3ffb5e988819187f77d796a60755e`
- 部署：`appgdep_6aee1fadf78081919f948b879d779b48`
