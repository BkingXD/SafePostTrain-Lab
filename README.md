# SafePostTrain-Lab

系统学习大模型后训练，并逐步构建可复现的安全对齐项目。

第一阶段提供中文课程和交互学习前端：**17 章、34 道理解检查、12 个交互实验、56 个术语**。课程覆盖基础、数据、SFT、LoRA/QLoRA、奖励建模、DPO、PPO、GAE、GRPO、反馈设计、蒸馏、安全、工程、评测与专项任务。每章都有学习目标、白话解释、数值或场景示例、误区、练习和一手资料。

[打开私人学习站](https://safeposttrain-learning-lab.cappedcone1.chatgpt.site) · [当前交付核对](reports/STAGE1_FINAL_AUDIT.md)

新增 [完整学习指南](docs/STUDY_GUIDE.md)、[17 模块覆盖表](docs/STAGE1_COVERAGE.md)及 [CPU 微型模型实战](labs/tiny_lm/README.md)。本地前端包含“学习指南”和“训练实战”入口，直接展示推导、案例和真实运行记录。

新增学习指南与 TinyLM 实战已按确认同步到上述私人站点。访问需要使用拥有该站点的 ChatGPT 账号登录；本地项目也可独立启动。

在线站点采用创建者私有访问。按 2026-11-07 的要求，本阶段交付本地项目与学习前端，暂不上传 GitHub；GitHub 同步不作为本阶段完成条件。

当前范围：知识学习与教学实验。参数计算器使用教学输入；“训练实战”展示 19,281 参数因果 Transformer 的真实 CPU 运行。两者均**不是大模型安全训练或 benchmark 成绩**。后续安全主项目的研究设计见 [项目大纲](docs/PROJECT_OUTLINE.md)。

## 直接打开本地学习站

需要 Python 3.10+；网页资源已经生成，无需安装前端依赖、GPU 或 API Key。

```bash
python3 -m http.server 4173 --bind 127.0.0.1 --directory web/dist
```

浏览器打开 `http://127.0.0.1:4173/`。请使用 HTTP 服务，不要直接双击 HTML 文件。网站是静态资源，可托管在支持静态文件的服务中；哈希路由无需额外的路径重写。

## 建议学习方式

1. 按 01～17 章学习，或根据每章前置知识选择路径。
2. 先读直觉，再亲手计算一个例子，最后打开相关交互实验。
3. 完成本章理解检查，在笔记中用自己的话解释一个概念。
4. 运行 Python 实验，核对浏览器与代码计算。
5. 完成 [毕业练习与验收清单](docs/LEARNING_CHECKLIST.md)，再准备真实模型训练。

答对一章全部检查题后可以标记完成。笔记、收藏、错题和进度只保存在当前浏览器的 localStorage 中，不会上传。更换设备或清理数据前，请在“我的学习”导出 JSON；网站支持导入与合并。不同域名和浏览器拥有独立进度。

## 课程目录

| 章节 | 主题 |
|---|---|
| [01](docs/curriculum/01.md) | 概率、token、梯度与后训练地图 |
| [02](docs/curriculum/02.md) | 目标、代理奖励、KL 与分布迁移 |
| [03](docs/curriculum/03.md) | 数据构建、标注、切分与污染控制 |
| [04](docs/curriculum/04.md) | SFT、mask、模板、packing 与训练诊断 |
| [05](docs/curriculum/05.md) | LoRA、QLoRA、参数与显存 |
| [06](docs/curriculum/06.md) | 奖励模型、偏好概率与校准 |
| [07](docs/curriculum/07.md) | DPO 推导与偏好方法家族 |
| [08](docs/curriculum/08.md) | 策略梯度、PPO、GAE 与四个模型角色 |
| [09](docs/curriculum/09.md) | GRPO、组内优势与在线采样 |
| [10](docs/curriculum/10.md) | RLHF、RLAIF、RLVR 与奖励投机 |
| [11](docs/curriculum/11.md) | 推理后训练、蒸馏与自改进 |
| [12](docs/curriculum/12.md) | 安全边界、威胁模型与双向偏好 |
| [13](docs/curriculum/13.md) | 诚实、隐私、公平、遗忘与安全退化 |
| [14](docs/curriculum/14.md) | 训练系统、分布式、吞吐与恢复 |
| [15](docs/curriculum/15.md) | benchmark、裁判、消融与统计 |
| [16](docs/curriculum/16.md) | 长上下文、多语言、工具、Agent 与多模态 |
| [17](docs/curriculum/17.md) | 可复现项目与第二阶段准备 |

## 无 GPU 教学实验

所有脚本仅用 Python 标准库，不会下载模型或调用外部服务。

```bash
python3 labs/core_math.py
python3 labs/data_audit.py
python3 labs/reward_model.py
python3 labs/toy_training.py --method all --steps 80 --seed 7
```

- `core_math.py`：CE、DPO、PPO surrogate 和 GRPO 优势的核心计算。
- `data_audit.py`：合成数据精确去重和按原始问题簇切分；不包含真实危险请求。
- `reward_model.py`：两特征线性奖励模型，学习成对偏好；其中正确性特征由教学环境提供，真实奖励模型没有这个 oracle。
- `toy_training.py`：用 SFT、DPO、PPO、GRPO 更新一个单参数、双动作策略，包含实际采样与梯度更新。这是一步 bandit，不是语言模型训练，不用于算法排名。

详细假设和观察任务见 [实验指南](labs/README.md)。

## Token 级训练实战

在同一个微型因果 Transformer 上运行 SFT、LoRA、神经奖励建模、DPO/SimPO、PPO/GRPO 和生成—验证—筛选—蒸馏。保留 chosen-only SFT、辅助 SFT 对照、失败结果、开发与最终探针、原始回答及成本。

本工作区已安装隔离环境；不需要 GPU 或外部模型权重：

```bash
.venv-teaching/bin/python -m labs.tiny_lm.run --output artifacts/my-run --seed 7
.venv-teaching/bin/python -m unittest discover -s tests -p 'test_tiny_lm.py' -v
python3 labs/evaluate_measured.py
```

第三条命令不需要 PyTorch，可从保存的原始回答重新计算指标。其他机器的安装方法、配方假设、失败诊断和恢复边界见 [实战 README](labs/tiny_lm/README.md)。本例只有符号复制任务；真实安全策略训练属于下一阶段。

## 修改内容与验证

构建需要 Node.js 20+ 与 Python 3.10+，无需 `npm install`。`web/course.mjs` 是课程内容的唯一维护入口；生成的 JSON 与 Markdown 不手工修改。构建同时生成可下载、保留目录结构的 Python 教学实验 ZIP。

```bash
node scripts/build.mjs
node --test tests/*.test.mjs
python3 -m unittest discover -s tests -p 'test_*.py'
node --check web/dist/app.mjs
```

构建同步生成浏览器数据与 GitHub 可读课程。数值测试验证手算例子、不变性与边界；课程检查验证章节、引用、前置关系和生成一致性。这些自动检查不等于教学质量的全部证明，还需人工审阅与浏览器交互验证。

默认 Python 未安装 PyTorch 时，11 项 TinyLM 测试会明确跳过。完整验证应再用 `.venv-teaching/bin/python -m unittest discover -s tests -p 'test_*.py'` 执行全部 Python 测试。网页记录来自 `reports/tiny_lm/verified_seed7.json`，历史 pilot 单独保留。

## 文件结构

```text
docs/PROJECT_OUTLINE.md      完整项目研究大纲
docs/curriculum/             生成的中文课程
docs/LEARNING_CHECKLIST.md   学习验收和毕业练习
labs/                       Python 教学实验与说明
web/course.mjs              课程与术语源数据
web/dist/                   可直接托管的静态学习站
scripts/build.mjs           课程构建脚本
tests/                      数值与内容一致性检查
reports/                    实际验证记录与已知局限
```

第二阶段将基于真实资源选择主模型，实施安全 SFT、偏好优化、数据迭代与独立 benchmark。只有实际完成并经验证的结果才进入项目报告和简历。
