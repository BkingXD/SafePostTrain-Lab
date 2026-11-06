# SafePostTrain-Lab

系统学习大模型后训练，并逐步构建可复现的安全对齐项目。

第一阶段提供中文课程和交互学习前端：**17 章、34 道理解检查、12 个交互实验、56 个术语**。课程覆盖基础、数据、SFT、LoRA/QLoRA、奖励建模、DPO、PPO、GAE、GRPO、反馈设计、蒸馏、安全、工程、评测与专项任务。每章都有学习目标、白话解释、数值或场景示例、误区、练习和一手资料。

[打开私人学习站](https://safeposttrain-learning-lab.cappedcone1.chatgpt.site) · [第一阶段验收记录](reports/STAGE1_AUDIT.md)

在线站点采用创建者私有访问；GitHub 仓库源码与站点托管分别管理。当前工作区尚未关联用户仓库，不能将站点源码推送成功误认为已同步到 GitHub。

当前范围：知识学习与教学实验。浏览器中的数字和 Python 的小模型是教学示例，**不是已完成的真实大模型安全训练或 benchmark 成绩**。后续安全主项目的研究设计见 [项目大纲](docs/PROJECT_OUTLINE.md)。

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

## 修改内容与验证

构建需要 Node.js 20+ 与 Python 3.10+，无需 `npm install`。`web/course.mjs` 是课程内容的唯一维护入口；生成的 JSON 与 Markdown 不手工修改。构建同时生成可下载、保留目录结构的 Python 教学实验 ZIP。

```bash
node scripts/build.mjs
node --test tests/*.test.mjs
python3 -m unittest discover -s tests -p 'test_*.py'
node --check web/dist/app.mjs
```

构建同步生成浏览器数据与 GitHub 可读课程。数值测试验证手算例子、不变性与边界；课程检查验证章节、引用、前置关系和生成一致性。这些自动检查不等于教学质量的全部证明，还需人工审阅与浏览器交互验证。

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
