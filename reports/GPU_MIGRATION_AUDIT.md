# GPU 迁移与实测验收

日期：2026-11-07（Asia/Shanghai）。本记录补充此前 CPU 第一阶段验收；当前训练主流程为 CUDA，CPU 仅作显式兼容路径。

## 实际环境与实现

- 本机 NVIDIA GeForce RTX 5070；驱动 595.99.02。最初沙箱内 `nvidia-smi` 失败，但宿主机检查及真实 CUDA 运算成功，未修改系统驱动。
- 独立 `.venv-gpu`：Python 3.12.14、PyTorch 2.9.1+cu128、CUDA runtime 12.8、FP32、cuda:0；完整依赖锁在 `labs/tiny_lm/requirements-gpu-lock.txt`。
- 默认 CUDA，设备/内核预检失败即退出；不会静默回退 CPU。CPU 必须显式指定 `--device cpu`。
- 模型、LoRA、奖励/价值头、输入、采样 Generator、参考策略均跟随目标设备。检查点可恢复 CPU/CUDA RNG，加载可映射设备。
- GPU 计时在边界同步。峰值已分配/保留显存仅来自 PyTorch 分配器，包含阶段开始的驻留对象和缓存，不包含 CUDA 上下文或非 PyTorch 库显存，不能当作算法独占显存。

## 实测与复现

最终源码对应的完整运行：

```bash
.venv-gpu/bin/python -m labs.tiny_lm.run --device cuda --final-eval --output artifacts/tiny-lm-cuda-verified-seed7 --seed 7
.venv-gpu/bin/python -m labs.tiny_lm.run --device cuda --final-eval --output artifacts/tiny-lm-cuda-verified-replay-seed7 --seed 7
```

已保留 [首次 GPU 报告](tiny_lm/gpu_seed7.json)、[GPU 复跑报告](tiny_lm/gpu_replay_seed7.json)和[比对摘要](tiny_lm/gpu_reproducibility.json)。两份报告的训练记录、开发与最终生成输出完全一致；比对排除耗时、吞吐、RSS 与显存峰值。此结论只覆盖当前软件/硬件与同 seed，不要求 CPU 和 GPU 逐位相同。

报告包含 11 个模型阶段、奖励模型、逐条原始回答、配置和 6 个源文件 SHA256。22 组指标逐条复算通过，SFT/LoRA 和 chosen-SFT/DPO 的 token 预算匹配。权重重载 logits 最大差值为 0。

| 阶段 | 开发探针 | 最终探针 |
|---|---|---|
| random | 0/8 | 0/8 |
| warmup | 0/8 | 2/8 |
| sft | 8/8 | 6/8 |
| lora | 8/8 | 5/8 |
| chosen_sft | 7/8 | 6/8 |
| dpo | 0/8 | 0/8 |
| dpo_anchor | 7/8 | 8/8 |
| simpo | 8/8 | 6/8 |
| ppo | 7/8 | 5/8 |
| grpo | 6/8 | 5/8 |
| distill | 7/8 | 6/8 |

DPO 的失败结果保留，不能用训练 loss 下降代替生成质量评测。每个集合只有 8 条，任务是合成符号复制，不是大模型安全 benchmark，也不能用来排名算法。

历史 CPU 报告保持原样，冻结源码保存在 `tiny_lm/cpu_reference_source/`，其摘要仍可严格核验。当前源码与历史源码不混用。

## 验证记录

- 21 项模型/设备测试通过，其中 4 项真实 CUDA 测试确实执行，没有跳过；包含各算法短路径、全参/LoRA 的优化器和 CPU/CUDA RNG 恢复、相同下一步更新、同步与显存 API 调用。
- 13 项公式、下载包、指标与来源完整性检查通过。
- 16 项 Node 课程、数值交互及生成一致性检查通过。合计 50 项。
- 最后一次训练源码变更仅细化显存统计范围文字；之后用最终源码重新生成两份 GPU 完整报告并复核来源和复跑一致性。
- 下载实验包包含当前 GPU 源码、依赖、GPU 报告/复跑以及 CPU 历史源码；解压后的标准库报告审计成功。
- 本地浏览器确认 GPU 标签、RTX 5070、FP32、阶段/探针切换、PPO actor+critic 成本及显存展示；未记录浏览器 error。见 [截图](gpu-practice.png)。

## 保留的边界

当前为 19,281 参数教学模型，真实开放权重大模型安全 SFT/DPO、正式 benchmark、多种子研究属于后续阶段。BF16、QLoRA 和多卡仍是课程及扩展设计。完整在线 PPO 续训尚未统一保存 critic、其优化器和 rollout 队列，不能由 actor 恢复测试代替。

## 私人站点发布

GPU 版已成功发布至 [私人学习站](https://safeposttrain-learning-lab.cappedcone1.chatgpt.site)。保持原创建者私有访问，未上传 GitHub。

- Sites 源码提交：`175627ff5aa08bf6af2f92e866bb31de598e4aa8`
- 版本：`appgprj_6aeda54c30e881918ae0a8e64e7ed7e1~appgver_2608252d509c81919de8409158c170ba`
- 部署：`appgdep_6aee1da9aa848191b3719cb12711e99a`
- 原生部署工具返回 `succeeded`。交互验证在本地同源静态构建完成；线上私有访问仍要求站点所有者登录。
