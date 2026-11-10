// Canonical chapter names mirror section 3 of docs/PROJECT_OUTLINE.md.
// The ids, prior exercises and quiz indices remain unchanged.
export const moduleSpecs = [
 {id:'01',title:'前置基础与训练阶段',topics:['Transformer 与因果语言模型','tokenizer、logits、概率和交叉熵','梯度、优化器及有效 token','预训练、持续预训练与后训练的边界','训练与推理的输入、输出和状态']},
 {id:'02',title:'后训练目标与问题建模',topics:['帮助性、安全性、成本与任务分布','约束目标、代理奖励与 KL','分布迁移和灾难性遗忘','训练/开发/最终测试的职责','可否证假设、基线和预算']},
 {id:'03',title:'数据工程',topics:['指令、对话、偏好及轨迹格式','合成数据、标注规则与质量审查','采样、混配和有效 token','按来源/意图分组、去重与污染控制','数据卡、许可和完整追溯']},
 {id:'04',title:'监督微调 SFT',topics:['teacher forcing 与 token 级交叉熵','chat template、角色边界及 EOS','assistant-only loss 与多轮标签','shift、截断、padding 和 packing','训练/推理模板一致性与排错']},
 {id:'05',title:'参数高效微调',topics:['全参数、冻结层与 adapter 的取舍','LoRA 低秩矩阵、rank 与目标模块','QLoRA、量化与计算精度','可训练参数、显存及吞吐','适配器保存、加载、合并与回归']},
 {id:'06',title:'偏好数据与奖励模型',topics:['偏好标注与 pairwise 数据','Bradley–Terry 与排序损失','奖励模型、代价模型、ORM/PRM','准确率、校准、长度偏差和噪声','分布外失效与奖励投机']},
 {id:'07',title:'离线偏好优化',topics:['DPO 目标、四个 log-prob 与 beta','参考策略、隐式奖励及序列归一化','chosen/rejected、模板与截断','IPO、KTO、ORPO 和 SimPO 的差别','chosen-only SFT 基线与失败诊断']},
 {id:'08',title:'强化学习基础与 PPO',topics:['MDP、轨迹与策略梯度','优势、价值函数与 GAE','actor、critic、old、reference 与 reward','importance ratio、PPO clip 和 KL','rollout、终止边界与恢复状态']},
 {id:'09',title:'GRPO 与在线训练',topics:['同 prompt 多采样与组相对优势','均值、方差、零方差组和奖励尺度','token/回答归一化及长度偏差','DAPO 等变体与版本差异','rollout 策略延迟、生成成本与日志']},
 {id:'10',title:'反馈来源与奖励设计',topics:['RLHF、RLAIF、RLVR 的不同维度','人类、模型、规则及环境反馈','结果/过程奖励与多目标聚合','验证器漏洞和 reward hacking','奖励审计与独立评价']},
 {id:'11',title:'推理后训练与自改进',topics:['冷启动、长思考与任务分布','结果监督、过程监督与验证器','best-of-N、拒绝采样和计算预算','序列/软标签蒸馏与自训练','迭代筛选、覆盖率与数据退化']},
 {id:'12',title:'安全对齐原理',topics:['安全政策、风险类别与威胁模型','拒绝、合法敏感与部分安全回答','双向偏好和正常能力回放','红队、越狱、多轮与对抗训练','有害完成、误拒绝和约束权衡']},
 {id:'13',title:'广义安全专题',topics:['诚实性、事实性与校准','公平性、偏见与分组评价','隐私、记忆、遗忘和对手权限','投毒、后门和后续微调退化','不同安全属性的独立指标']},
 {id:'14',title:'训练系统与效率',topics:['权重、梯度、优化器与激活显存','BF16/FP16、梯度累积和 checkpointing','ZeRO/FSDP 与数据/张量/流水线并行','rollout、采样同步及推理系统','GPU 峰值、吞吐和故障恢复']},
 {id:'15',title:'评测与实验方法',topics:['基线、测试协议、固定解码与原始输出','安全、误拒绝、能力与成本矩阵','攻击预算、分母和统计置信区间','裁判校准、人工抽检与污染控制','消融、负结果和可复算报告']},
 {id:'16',title:'专项能力后训练',topics:['领域适应与持续预训练','长上下文的长度分布及位置评测','多语言数据与语义复核','工具调用、Agent 轨迹和环境反馈','多模态 processor、模型组件及风险']},
 {id:'17',title:'交付与持续改进',topics:['配置、依赖、模型/数据版本与追溯','checkpoint、adapter 和模型卡','导出、量化、部署和安全回归','线上反馈、版本化与受控迭代','复现材料、失败分析和面试表达']}
];
export const modulesById = Object.fromEntries(moduleSpecs.map(module=>[module.id,module]));
