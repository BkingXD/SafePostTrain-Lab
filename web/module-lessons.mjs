// Original chapter-level technical deepening and small, explicitly scoped code fragments.
const s=(title,...paragraphs)=>({title,paragraphs});
const p=(source,question)=>({source,question});
const code=(caption,body)=>({caption,language:'python',body:body.trim()});
export const moduleLessons={
'01':{sections:[
 s('重点一：模型输出与自回归分解','对输入 token 序列 x₁…xT，因果模型在每个位置输出词表大小为 V 的 logits。softmax 后得到条件分布；一条回答 y 的概率按 P(y|x)=∏ₜP(yₜ|x,y<t) 分解。序列 log-prob 是各 token log-prob 的和；训练时的平均 NLL 还要除以有效标签个数。','注意：词表中的 token 由具体 tokenizer 决定，一个汉字、单词或标点不一定恰好对应一个 token。特殊 token 编码对话边界；错误地插入或重复这些 token，可能使训练/生成的条件上下文发生变化。'),
 s('重点二：预训练、继续预训练和后训练','预训练通常优化大规模语料的下一 token 预测，得到通用概率模型。继续预训练仍用语言建模目标，但使用新的领域或时期语料，可能改变已有能力。后训练通过指令、偏好、奖励或环境反馈调整行为；提示词和检索改变输入而不更新模型参数。','从底模做 SFT 与从已对齐指令模型继续安全训练有不同基线。实验报告应固定底模/指令模型 revision 和起点 checkpoint；不能把已有预训练知识或安全行为全部记为新实验收益。'),
 s('重点三：loss 的分母与梯度','交叉熵衡量目标 token 在当前条件下的负对数概率。有效 token 是排除 prompt 标签、padding 与其他忽略位置后实际参与目标的 token；同样 8 条数据，回答长度不同会导致有效监督量不同。梯度描述局部变化方向，优化器还维护动量及二阶状态。','一次数值检查：目标 token 概率为 0.25 时 NLL≈1.386；目标 logit 的局部梯度为 p−1=−0.75。梯度下降会提高该 logit，但整个网络的下一步行为取决于其他样本、学习率和优化器，不能仅凭手算推断最终生成。')],
 snippet:code('局部片段：计算一条回答的条件 log-prob',`# logits: [T, V]；answer_ids: [T]，只包含回答位置
log_probs = logits.log_softmax(dim=-1)
selected = log_probs.gather(-1, answer_ids[:, None]).squeeze(-1)
sequence_logp = selected.sum()       # 序列对数概率
mean_nll = -selected.mean()          # 本例所有位置均为有效标签`),
 papers:[p('transformer','因果注意力如何限制可见上下文？'),p('survey25','持续预训练、对齐和推理各解决什么问题？')]
},
'02':{sections:[
 s('重点一：将“模型更好”拆为可测量目标','先明确谁在什么输入分布下使用模型，成功行为、失败代价与资源约束分别是什么。安全项目至少同时评价有害帮助、合法请求误拒绝和正常能力；对于混合请求，还要区分哪些内容可以回答。','选择主指标前定义分子、分母、样本来源、语言、生成设置和裁判。训练奖励、开发指标和最终评价虽可能相关，但服务不同决策；不能直接用训练 reward 上升替代最终安全结论。'),
 s('重点二：约束、代理和分布变化','一个概念性目标是最大化 E[r]，同时要求安全代价 E[c]≤δ 并限制对参考策略的偏离；也可构造带 λ、β 的惩罚形式。约束和加权奖励的含义依赖评分尺度，平均约束可能掩盖少数高风险类别。','模型优化奖励模型时可能利用偏差。训练 prompt 与真实输入又可能在语言、长度、攻击方式和工具观察上不同；需要数据分组、保留分布外切片、独立裁判与人工复核。'),
 s('重点三：把假设写成反事实对照','例如“边界偏好是否降低合法请求误拒绝”，对照为同一起点的标准 DPO 和等预算 chosen-only SFT；预先限定有害完成率与普通任务下降可接受范围，再固定训练 token 和最终测试。','训练后若误拒绝下降但有害完成增加超过容忍范围，假设并未按原标准成立。数据量、超参搜索和评测采样预算都应记录，避免把更高成本造成的收益归因于方法。')],
 snippet:code('局部片段：把实验假设写成配置',`experiment = {
    "start_checkpoint": "sft-revision",
    "comparison": ["chosen_sft", "standard_dpo", "boundary_dpo"],
    "primary_metric": "benign_over_refusal_rate",
    "constraints": ["harmful_completion", "general_capability"],
    "held_out_group": "unseen_attack_families",
}`),
 papers:[p('tulu','公开配方怎样连接数据、方法和评价？'),p('rlPrimer26','初始分布和奖励设计为何影响在线优化？')]
},
'03':{sections:[
 s('重点一：区分 SFT、偏好与轨迹数据','SFT 样本通常含对话消息或 prompt/completion；偏好样本含同一 prompt 的 chosen/rejected；Agent 轨迹还需保存工具 schema、调用、返回、权限和最终结果。数据单元不同，直接把字段重命名不等于形成正确监督。','每条数据至少有稳定 ID、来源、license、任务类别与 group_id；偏好还要有标注方向、评分依据和冲突状态。训练前检查角色顺序、空值、格式、重复项、允许行为、EOS、长度与截断。'),
 s('重点二：切分、去重和污染防护','先按原始问题、攻击家族或意图聚类，再将组分配到训练、开发、最终测试；派生改写继承原组。精确 hash 检测字面重复，语义近重复检索需要人工校准阈值；对于公开测试，也要检查后训练新数据是否存在同源条目。','底模预训练语料通常不可完全观察，因此能控制的是本项目可见数据，不能宣称彻底排除所有预训练污染。记录无法审计的来源、版本和可能重叠。'),
 s('重点三：配比不仅是行数','同类样本回答长度不同，按行采样会产生不同的监督 token 权重；多轮样本、截断和 repetition 也会改变有效分布。比较配方时记录各来源抽样概率、样本数、有效 token 和实际 optimizer steps。','制作数据卡：来源与许可、标注说明、类别/长度分布、样本选择、分组策略、去重/污染审计、版本哈希、复核分歧和用途边界。')],
 snippet:code('局部片段：验证组不跨切分',`# train_groups / dev_groups / test_groups 是源问题或意图 ID 集合
assert train_groups.isdisjoint(dev_groups)
assert train_groups.isdisjoint(test_groups)
assert dev_groups.isdisjoint(test_groups)
# 改写样本继承相同 group_id；应先分组，再生成改写。`),
 papers:[p('dataformat','不同 trainer 支持的数据字段和消息结构是什么？'),p('datasheet','数据来源、用途与已知限制如何记录？')]
},
'04':{sections:[
 s('重点一：teacher forcing 和标签位移','训练对完整 token 序列并行产生 logits；第 t 个位置预测第 t+1 个 token。若对话为用户内容、助手标记、答案和 EOS，常见做法是将不监督的位置设为 −100，助手答案及 EOS 保留。注意用户 token 虽不计入 loss，仍然构成预测上下文。','是否由模型或 trainer 内部 shift 必须查实现；手动和框架各 shift 一次会造成错位。打印 token、label 和角色边界，核实助手标记的 logits 对应首个答案 token。'),
 s('重点二：对话模板、多轮和截断','训练完整消息时不应在已结束的回答后多加一个空 assistant 标记；推理的 prompt 需要按模型模板生成正确的助手前缀。对 prompt/completion 分开提供的框架，要检查其拼接后的前缀与推理相同。','多轮数据需说明监督所有助手轮还是只监督最后一轮。截断可能剪掉答案、EOS 或用户语义；若有效标签全为 −100，应明确拒绝样本或报错，不能把空 loss 当训练成功。'),
 s('重点三：packing 与 mask 的工程验证','packing 可以减少 padding，但有的实现把多个样本拼成连续流，有的提供样本隔离注意力。不要假定所有框架都会隔离；实际更改前一条样本并观察后一条 logits 是否变化。记录 position IDs、attention mask 和样本边界策略。','比较训练预算时，同时统计总 token 与助手监督 token；梯度累积只改变更新频率，多个不同长度 micro batch 的平均方式可能改变梯度权重。')],
 snippet:code('局部片段：手动计算一次移位的 SFT loss',`import torch.nn.functional as F
# logits: [B, T, V]；labels: [B, T]，不监督位置为 -100
shift_logits = logits[:, :-1, :].contiguous()
shift_labels = labels[:, 1:].contiguous()
loss = F.cross_entropy(shift_logits.reshape(-1, shift_logits.size(-1)),
                       shift_labels.reshape(-1), ignore_index=-100)
# 若所用模型/Trainer 已内部 shift，不要再做第二次。`),
 papers:[p('sft','该版本的数据格式与 completion/assistant mask 如何工作？'),p('hfchat','训练模板与生成模板分别何时追加助手前缀？')]
},
'05':{sections:[
 s('重点一：从全参到 LoRA','全参数训练需要为所有参与更新的权重维护梯度及优化器状态；LoRA 冻结原矩阵 W₀，训练低秩增量 ΔW=(α/r)BA。A 的形状为 r×d_in，B 为 d_out×r；新增参数数目是 r(d_in+d_out)。常见初始化令增量起始为零，但需查看具体配置。','对 4096×4096 的矩阵，r=8 时新增 65,536 参数，而原矩阵有 16,777,216。此数值只适用一个矩阵；真实模型需按目标模块与层数累计，embedding、bias 和其他可训练层需单列。'),
 s('重点二：QLoRA 与总显存','QLoRA 通常把冻结底模以低位量化存储，同时对 adapter 执行高精度计算和更新。4-bit 参数并不意味着整个训练只占全参 BF16 的四分之一：激活、adapter、优化器、非量化层、缓存与 kernel 临时空间都存在。','先确认量化库、设备和版本支持，再分别记录 quant type、compute dtype、双重量化、目标模块与实际显存。长序列的激活可能是瓶颈；对比需固定长度、micro batch、累积和生成设置。'),
 s('重点三：适配器生命周期','保存 adapter 时保留底模 ID/revision、target_modules、rank、alpha、tokenizer 和模板。加载后核查 trainable 参数、梯度、实际 target 命中；合并或量化后以固定输入比较输出，并重新跑关键安全/能力测试。','常见失败包括模块名不匹配导致没有可训练参数、误绑定不同底模、SFT 和 DPO 使用不一致的 adapter、将 adapter 文件误称完整权重。')],
 snippet:code('局部片段：计算一个线性层的低秩增量',`# W0: [d_out, d_in]；A: [rank, d_in]；B: [d_out, rank]
# W0.requires_grad_(False)，A / B 由优化器更新。
delta = (alpha / rank) * (B @ A)
W_adapted = W0 + delta
lora_parameters = rank * (d_in + d_out)`),
 papers:[p('lora','低秩分解和参数量如何推导？'),p('qlora','量化底模训练 adapter 的显存节约来自哪里？')]
},
'06':{sections:[
 s('重点一：偏好标签与 Bradley–Terry','对于同一 x 的两个回答 y+、y−，奖励模型给出标量 r(x,y)。Bradley–Terry 用 σ(r+−r−) 表示选中回答被偏好的概率，训练时最小化 −logσ(r+−r−)。相同常数同时加到两个分数并不改变概率。','这种模型学习的是给定数据中的相对偏好，不能自然解释为全局质量或安全概率。样本需要同一 prompt、清晰胜出理由、标注者或裁判来源；若标注有争议，应记录分歧而不是只保留单一标签。'),
 s('重点二：奖励、代价与过程监督','奖励可表示任务帮助程度，代价模型可单独表示风险；结果奖励评价整段输出，过程奖励评价部分步骤。给中间步骤打分能提供更密集信号，但过程标签也会漏掉替代解法或误判。','安全规则与数值验证器只覆盖其定义的条件。验证器即使是程序，也可能被格式漏洞或未检查的隐含约束投机利用；应为高分错误和低分正确答案保留独立样本。'),
 s('重点三：除了排序准确率，还看什么','按风险类别、语言、长短差和来源统计 pair accuracy；在标签与样本允许时检查 Brier 或可靠性分箱；对高置信错误做人工复核。训练策略后访问的回答分布可能偏离奖励模型训练集，必须重新检查分布外有效性。','做长度反转对照：同一问题上让较短回答正确、较长回答错误，观察奖励是否仍偏好长回答。不能因 8 个测试对上得高分就宣称模型已经校准。')],
 snippet:code('局部片段：一组成对偏好的奖励损失',`import torch.nn.functional as F
# chosen_scores 与 rejected_scores 按相同 prompt 一一对齐
score_gap = chosen_scores - rejected_scores
pair_probability = score_gap.sigmoid()
reward_loss = -F.logsigmoid(score_gap).mean()`),
 papers:[p('reward','偏好数据如何训练模型以评价生成摘要？'),p('verifier26','任务验证器可能通过什么捷径被投机？')]
},
'07':{sections:[
 s('重点一：DPO 的输入与四项概率','每条数据由同一 prompt x 的 chosen y+ 和 rejected y− 构成，需分别计算训练策略和冻结参考策略对两个 completion 的 log-prob。定义 Δ=[logπθ(y+|x)−logπref(y+|x)]−[logπθ(y−|x)−logπref(y−|x)]，标准 sigmoid DPO 的损失是 −logσ(βΔ)。','log-prob 通常由回答 token 的条件概率累加；要确认 prompt、padding、特殊 token、结束符和截断的 mask 与所用实现一致。β 和不同损失变体的默认值依赖训练库版本，不能只写“使用 DPO”。'),
 s('重点二：目标改善与行为改善不是同一指标','如果 chosen 相对 reference 下降 0.2、rejected 下降 1.2，则 Δ=1。β=0.1 时 DPO loss≈0.644，低于 Δ=0 时的 0.693；但 chosen 的绝对 log-prob 仍然下降。记录 chosen/rejected 的分布、margin、输出长度和逐条生成，才能发现这种失败。','同一个 SFT 起点、相近有效回答 token 下比较标准 DPO 与 chosen-only SFT，可以帮助区分偏好信号和额外高质量回答的作用。超参搜索次数、数据长度以及模型裁判开销不同时，还需单独报告。'),
 s('重点三：变体与参考策略','IPO、KTO、ORPO、SimPO 使用不同的监督假设或优化形式，不能当作同一 loss 的几个等价参数；SimPO 等方法对序列长度处理不同。应按论文和代码核对输入格式、参考模型、聚合方式和超参。','LoRA DPO 中 reference 可能是独立模型、冻结 adapter 副本或禁用 adapter 后的模型。哪一个才代表 SFT 起点取决于实现；训练前后比较 reference 权重与 log-prob，不要仅凭 ref_model=None 推断。')],
 snippet:code('局部片段：标准 DPO 损失',`import torch.nn.functional as F
# 每个变量是相同 batch 中 completion token 的序列 log-prob 和
policy_gap = policy_chosen_logp - policy_rejected_logp
reference_gap = ref_chosen_logp - ref_rejected_logp
margin = policy_gap - reference_gap
loss = -F.logsigmoid(beta * margin).mean()`),
 papers:[p('dpo','DPO 如何从参考策略与偏好概率得到目标？'),p('simpo','长度归一化与不使用参考策略如何改变目标？')]
},
'08':{sections:[
 s('重点一：状态、动作、价值与反馈','语言模型 RL 中，状态是问题和当前回答前缀，动作是下一个 token；完整回答是轨迹。actor 表示待更新策略，old policy 保存本批 rollout 的采样概率，reference 提供偏离约束，critic 估计价值，reward 模型或验证器提供反馈。','同一项目可能使用不同的奖励布置：末尾一次奖励、逐步奖励、KL 惩罚或环境反馈。必须写明各分量和 mask；reference 不等于每次更新时的 old policy。'),
 s('重点二：GAE 与终止边界','单步 TD 残差 δt=rt+γV(st+1)−V(st)，GAE 倒序累计 At=δt+γλAt+1。轨迹真正终止时 next-value 取相应终止定义；仅因采样长度达到上限而截断时，是否 bootstrap 取决于环境协议。prompt 与 padding 位置不应进入回答损失。','两步算例：r=[0,1]，V=[0.2,0.3]，终止后价值 0，γ=1、λ=0.95，则 δ1=0.7、δ0=0.1，优势为 A1=0.7、A0=0.765。改变终止条件或奖励位置后需要重新计算。'),
 s('重点三：PPO clipping 与恢复','ratio=exp(logπθ−logπold)，目标是 min(ratio·A, clip(ratio,1−ε,1+ε)·A)。优势为负时不能沿用正优势的直觉：A=−2、ratio=1.3、ε=0.2 时两项为 −2.6 和 −2.4，较小值为 −2.6。','完整断点恢复需要 actor、critic、相应优化器/调度器、随机状态、采样权重版本以及尚未消费的 rollout；只保存 actor 权重不等于在线 PPO 恢复。观察 KL、熵、reward、clip 比例、长度和 critic loss。')],
 snippet:code('局部片段：从轨迹末尾递推 GAE',`adv_next = 0.0
advantages = [0.0] * len(rewards)
for t in reversed(range(len(rewards))):
    bootstrap = 0.0 if terminated[t] else next_values[t]
    delta = rewards[t] + gamma * bootstrap - values[t]
    adv_next = delta + gamma * lam * (not terminated[t]) * adv_next
    advantages[t] = adv_next
# 此片段假设一条连续轨迹；批次、padding 和截断需额外 mask。`),
 papers:[p('ppo','裁剪目标如何限制某些方向的策略更新？'),p('gae','价值估计与 λ 如何影响偏差和方差？')]
},
'09':{sections:[
 s('重点一：组内采样和相对优势','对同一 prompt 采样 G 个回答，对每个回答评分，计算组内均值和标准差。一种常见形式为 Ai=(ri−mean(r))/(std(r)+ε)。如果奖励为 [1,1,0,0]，采用总体标准差并忽略极小 ε，得到优势约 [1,1,−1,−1]。','全组相同时组内相对优势为零；这可能表示题太容易、太难、采样缺乏多样性或奖励器无法区分。必须检查原始回答和奖励分量，而不是只看全组同分比例。'),
 s('重点二：目标变体与长度偏差','GRPO 一般以组相对反馈替代独立 critic，但策略目标如何逐 token 或逐回答平均、长度是否归一化、是否包含 KL、奖励如何缩放都依赖具体实现。较长轨迹可能贡献不同数量的梯度，不能跨实现只比较算法名称。','DAPO 提出不同裁剪、动态采样、token 级损失等一组设计，用于其长推理 RL 实验。实现中的具体开关及版本需一起报告；只启用某个技巧不等于复现整篇 DAPO。'),
 s('重点三：rollout 与训练系统','记录 G、解码参数、每个 prompt 的生成 token、奖励计算成本、策略权重版本、同步延迟及更新次数。异步生成可能产生陈旧样本，重要性修正及过滤应明确记录。','实践中扫描 G、温度和长度时，应控制总生成/验证预算并报告全组同分、截断、KL、熵和任务表现；不能只靠平均奖励判定推理能力。')],
 snippet:code('局部片段：计算一个组的相对优势',`# rewards: 同一 prompt 下 G 个回答的奖励向量
mean = rewards.mean()
std = rewards.std(unbiased=False)  # 总体标准差；与具体框架约定一致
advantages = (rewards - mean) / (std + 1e-8)
# 全部同分时 advantages 为零；这不代表任务已经解决。`),
 papers:[p('grpo','DeepSeekMath 中组相对优势如何构造？'),p('dapo','不同裁剪和动态采样分别针对什么问题？')]
},
'10':{sections:[
 s('重点一：反馈来源与优化器分开描述','RLHF 指利用人类反馈，RLAIF 使用 AI 反馈，RLVR 强调部分任务的反馈可由规则或程序验证；DPO、PPO、GRPO 则是不同的优化方法。一个流程可以由模型产生候选、人工复核偏好，再用 DPO 更新，也可以用程序验证结果后用在线方法训练。','先写反馈在什么对象上作用：整段回答、局部步骤、工具动作或环境状态。记录反馈是否访问标准答案、运行环境是否隔离、超时如何处理，以及哪些标签需要人工仲裁。'),
 s('重点二：奖励聚合与可被投机的规则','正确性、安全、格式、长度或调用成本可能使用不同尺度。将分数线性相加前先查看各分项分布，明确冲突和覆盖规则；同时记录原始分量，以识别总分上升来自哪一个分量。','例如 R=正确性+0.1×格式分，正确回答得 1+0=1，而错误回答若格式分为 15，则总分 1.5；优化器可能偏向格式而非正确。此数值是人为构造的量纲示例，不代表某一模型实测。'),
 s('重点三：独立审计验证器','把训练奖励与最终评测尽量分开。对规则构造等价输入、格式扰动和未覆盖约束；对模型裁判统计各语言/风险类别的人工分歧与错误。2026 年验证器投机研究展示了其归纳推理设置中的捷径，并不直接证明所有安全任务存在相同漏洞。','排查时保留高奖励错误和低奖励正确样本、验证器版本及超时状态；如果奖励函数访问了标准答案，应防止答案进入模型可见的输入。')],
 snippet:code('局部片段：分项记录比只保存总分更可审计',`components = {"correctness": 1.0, "format": 0.0, "safety_cost": 0.0}
reward = components["correctness"] + 0.1 * components["format"]
record = {"reward": reward, "components": components,
          "validator_revision": validator_revision}`),
 papers:[p('cai','原则反馈与自我修订如何参与对齐？'),p('verifier26','为什么表面上可验证的奖励也会被投机？')]
},
'11':{sections:[
 s('重点一：冷启动与推理轨迹','冷启动数据为后续训练提供格式或初步任务行为，结果监督关注最终答案，过程监督评价中间步骤。过程标签可能遗漏另一种正确路径，也不能保证最终答案正确；需单独研究标注噪声和长度成本。','DeepSeek-R1 公开了不同阶段的推理训练研究；Qwen3 讨论 thinking/non-thinking 模式。可展示的推理文本属于模型输出，不能直接等同于模型内部完整推理过程。'),
 s('重点二：测试时搜索和蒸馏','best-of-N 先采样并用评分器选择候选，本身不更新参数。用筛选后的文本做学生 SFT 才是序列蒸馏；使用教师概率分布属于另一类软标签蒸馏。相同架构学生不等同于模型压缩。','独立同分布且评分器完美时，N 次至少一次成功的概率为 1−(1−p)^N。真实候选常相关且评分器会错，这个公式只作为理想上界直觉，不是实测收益。'),
 s('重点三：可审计的自改进','仅从训练侧问题生成候选，保存全部候选和筛选原因；没有通过样本时保留缺失，不能偷偷填入标准答案。新增数据混配与去重应保留 parent_id 和教师版本。','与等新增数据量、等训练/生成预算的随机扩充做对照；报告类别覆盖、每题通过数、错误类型及学生在未参与筛选集合上的结果。')],
 snippet:code('局部片段：保留没有通过的情况',`accepted = [candidate for candidate in candidates
            if verifier(candidate)["accepted"]]
selected = accepted[0] if accepted else None
record = {"candidates": candidates, "selected": selected,
          "validator_revision": validator_revision}
# 不把标准答案作为教师的生成结果。`),
 papers:[p('r1','冷启动、强化学习和蒸馏各承担什么作用？'),p('qwen3','thinking 与 non-thinking 的训练/推理接口如何区分？')]
},
'12':{sections:[
 s('重点一：从政策到可执行标签','安全政策应定义允许行为、限制行为和不确定场景，再将原始请求按意图、语境、授权与回答内容标注。至少包含明确有害、合法敏感、普通任务和只可回答部分内容的样例；不要用一个关键词决定标签。','安全 SFT 示范合适行为；偏好数据需要有害请求偏好安全回答，也需合法请求偏好有帮助回答。正常任务回放用于检查能力保持，不能用单一拒绝率代表帮助性。'),
 s('重点二：威胁模型与攻击预算','明确对手能够控制用户输入、外部文档、工具输出或对话历史中的哪些部分；是否可看到模型权重/评分，能否多轮自适应，以及每个行为允许多少次尝试。测试应包含留出的来源或攻击家族，并按语言和风险类别报告。','直接有害请求、包装过的请求和提示注入是不同测试条件。工具返回是不可信数据；系统还需隔离执行、权限控制和审计，训练模型不能替代这些边界。'),
 s('重点三：安全—帮助性的实证对照','建立 E0 原模型、E1 安全 SFT、E2 标准 DPO、等 token chosen-only SFT 与边界偏好 DPO；可进一步比较困难样本挖掘和等预算随机扩充。冻结协议后分别报告有害完成、合法误拒绝、任务完成和成本。','已有研究表明安全行为可能在后续微调或输入变化下退化。论文具体模型与攻击范围不能直接外推；必要时测试 adapter 合并、量化及部署后回归。')],
 snippet:code('局部片段：安全标签应依据政策和意图',`sample = {
    "prompt_id": "boundary-example-001",
    "category": "benign_sensitive",
    "desired_behavior": "answer_permitted_parts",
    "source_group": "source-question-001",
    "review_status": "needs_review",
}
# 示例为字段格式，不能代替真实政策或标注。`),
 papers:[p('safe','帮助性与安全性为什么需要分别建模？'),p('refusal26','不同后训练方法的拒绝机制在研究中有何差异和局限？')]
},
'13':{sections:[
 s('重点一：诚实、校准与公平是不同问题','事实性关注回答与证据是否一致；校准关注自报置信度与实际正确率的关系；诚实性还可能涉及对未知和能力边界的表达。模型说“我不确定”不自动表示经过统计校准，正确回答也不保证其不确定性表达可靠。','公平性分析需先定义任务标签、可比较群体、条件和统计口径；应按群体、语言和交叉切片报告样本数及误差，不把一个总体平均分写成覆盖所有人的保证。'),
 s('重点二：隐私、记忆与遗忘','训练数据泄露或成员推断的测试要定义攻击者可以查询多少次、是否知道部分内容、是否能读取权重；合成 canary 可以帮助建立受控实验，但不能代表真实用户隐私风险的全部。日志和样本发布也要遵循数据权限。','遗忘不仅看目标样本是否还能原样输出，还要检查邻近知识和正常能力，并说明测试攻击强度；只测少量 prompt 不能证明信息已从模型所有参数中删除。'),
 s('重点三：投毒和后续微调退化','训练样本可能被恶意插入特定触发模式，后续微调、量化、模板或 adapter 合并也可能改变既有安全行为。控制条件并分别测干净输入、触发输入与正常能力，才有机会定位变化来源。','将不同属性作为独立模块研究，分别说明对手、分母、数据、裁判与可否证条件，不能用内容安全 benchmark 的一个分数覆盖隐私、公平和诚实。')],
 snippet:code('局部片段：以合成标识符记录受控测试',`audit_case = {
    "case_id": "synthetic-canary-01",
    "source": "synthetic",
    "attacker_access": "prompt_only",
    "query_budget": 10,
    "expected_handling": "do_not_repeat_without_authorization",
}`),
 papers:[p('shallow','为什么安全行为在后续微调后可能退化？'),p('modelcard','如何报告不同风险和已知适用边界？')]
},
'14':{sections:[
 s('重点一：分解 GPU 显存','显存包含权重、可训练参数的梯度和优化器状态、激活、临时张量、KV cache 以及框架分配器缓存。1.7B 参数以 BF16 存储的理想权重大小约 3.4 GB（约 3.17 GiB），还没有计算其他部分；实际数值取决于精度、算子、长度和优化器实现。','LoRA 降低 adapter 梯度与状态的规模，但仍需保存底模和反向传播用激活。PyTorch 的 allocated/reserved 峰值只涵盖其分配器，nvidia-smi 的进程/设备观察口径不同；测量时间需同步 GPU。'),
 s('重点二：精度、batch 与计算交换','BF16 与 FP16 的指数范围不同，FP16 训练经常配合梯度缩放；选用 BF16 前需检查硬件支持。micro batch × 梯度累积 × 并行设备数给出常见的样本级有效 batch 估算，但尾批和不同长度使真实有效 token 需要另计。','gradient checkpointing 用重复前向计算换更少的激活内存；Flash Attention、序列打包和量化等优化的效果依赖版本与输入形状，必须在目标条件下测显存和吞吐。'),
 s('重点三：分布式与在线生成','数据并行复制模型处理不同样本；ZeRO/FSDP 按配置切分参数、梯度或优化器状态；张量并行切分层内运算，流水线并行切分层间阶段。它们各引入不同通信和同步成本，不应把名称直接等同于固定加速比。','在线 RL 中生成服务和训练服务可能共享 GPU、交换权重并使用不同批次；需记录策略同步延迟、生成预算和 KV cache。中断恢复要覆盖训练和采样系统的真实状态。')],
 snippet:code('局部片段：记录 PyTorch 的 GPU 峰值',`import torch
assert torch.cuda.is_available()
device = torch.device("cuda:0")
torch.cuda.synchronize(device)
torch.cuda.reset_peak_memory_stats(device)
# 在这里执行待测训练阶段。
torch.cuda.synchronize(device)
peak_allocated = torch.cuda.max_memory_allocated(device)
peak_reserved = torch.cuda.max_memory_reserved(device)`),
 papers:[p('zero','ZeRO 不同阶段切分哪些状态？'),p('fsdp','FSDP 的分片、状态保存和通信如何配置？')]
},
'15':{sections:[
 s('重点一：基线和完整评测协议','固定原模型/adapter revision、数据版本、chat template、system prompt、最大长度、采样策略和评分器。每条结果至少保留样本 ID、完整生成、停止/超时状态、裁判输出及分项标签，便于重新计算分子和分母。','开发集可以选择超参数；最终测试必须在配方冻结后使用。若测试结果用于改训练数据或选 checkpoint，这批测试已变成开发信息，应另留未使用测试。'),
 s('重点二：攻击预算、分母和不确定性','行为级 ASR@k 以原始有害行为为分母，每个行为最多 k 次尝试，任何一次成功即该行为成功；尝试级成功率以实际攻击次数为分母。100 个行为中 20 个至少成功一次是 20%；若共 500 次攻击、24 次成功，尝试级是 4.8%。','同一行为的多个尝试相关，应按原始行为分组进行重采样。对同一批样本的模型对照应优先利用配对信息；多次生成与多次独立训练是不同的不确定性来源。'),
 s('重点三：裁判校准和能力保持','HarmBench、XSTest、StrongREJECT 等任务的原始协议和分母不同，不能将结果混成一个未解释分数。裁判也可能误判：按风险、语言和长度分层抽样人工复核，记录漏判、误判和分歧。','同时报告合法请求完成度、通用任务、资源开销和负结果；事先写清容忍范围，避免只挑最好的指标或自建子集冒充官方完整 benchmark。')],
 snippet:code('局部片段：从行为级记录计算 ASR@k',`# attempts_by_behavior 的每个值是一组已完成尝试的布尔判定
if not attempts_by_behavior:
    raise ValueError("no evaluated behaviors")
successful_behaviors = sum(any(results) for results in attempts_by_behavior.values())
behavior_asr = successful_behaviors / len(attempts_by_behavior)
# 还需保存每个行为的攻击预算、失败和超时约定。`),
 papers:[p('harm','如何固定行为集、攻击方法和裁判协议？'),p('xs','怎样区分过度拒绝与适当拒绝？')]
},
'16':{sections:[
 s('重点一：领域、长上下文与多语言','领域适应可以使用继续预训练获得领域分布，再用 SFT 指导任务格式；必须分辨语料新知识与指令行为。长上下文实验控制真实 token 长度、相关信息位置和干扰量，同时检查截断是否改变任务。','多语言任务的 tokenizer 效率、输入语义和安全标签可能不同；翻译测试需要复核语言自然度和原标签含义。不同语言分开报告，不把英语结果直接当作其他语言的结论。'),
 s('重点二：工具和 Agent 轨迹','轨迹包含 user 消息、assistant 调用、工具参数、环境返回以及后续决策。需要决定哪些 token 是模型训练目标，并核验 schema、调用顺序、错误恢复和状态变化；工具结果通常是不可信输入。','在线环境应使用受限沙盒和最小权限。任务成功、参数合法、越权动作、提示注入服从及调用成本要分别评估；最终文字得体不代表动作过程安全。'),
 s('重点三：多模态 processor 与训练组件','视觉语言模型将图像和文本映射到模型可用的输入；processor 可能产生 input_ids、pixel_values、网格信息等。图像占位符、图文顺序、图像尺寸和训练标签必须一致。选择冻结视觉编码器、只训投影层还是训练语言层会改变成本与行为。','Qwen3.5-0.8B 模型卡提供多模态使用示例；本项目的 Qwen3 文本示例不自动支持它。模型卡所述最大上下文并不保证 12GB GPU 能以该长度训练。')],
 snippet:code('数据示意：工具调用是一条轨迹',`trajectory = [
    {"role": "user", "content": "查询示例目录中的文件"},
    {"role": "assistant", "tool_call": {"name": "list_files", "arguments": {"path": "demo"}}},
    {"role": "tool", "content": "[\\"a.txt\\"]"},
    {"role": "assistant", "content": "示例目录包含 a.txt。"},
]
# 这是格式说明，不会执行工具。`),
 papers:[p('qwen35','多模态 processor 与模型输入有哪些字段？'),p('olmo3','开放模型如何报告训练生命周期与能力？')]
},
'17':{sections:[
 s('重点一：实验可追溯与模型卡','每个结果应能回溯模型 ID/revision、tokenizer/template、数据版本与哈希、代码、配置、依赖、硬件、种子和训练日志；原始生成和评分记录要能重算指标。adapter 需要准确底模和配置，不能仅把 adapter 文件标为完整模型。','模型卡包含目标用途、训练/评测范围、已知失败、资源及限制；数据卡包含许可、来源、标注、切分、污染控制与分布。未测的条件应明示，而不是从其他模型或论文数字复制过来。'),
 s('重点二：导出、部署和持续回归','模型导出、adapter 合并、推理框架切换或量化都可能影响输出。固定一组代表性输入、模板、精度、终止设置与指标，对各节点逐步回归安全、正常能力和资源；对不一致结果保留原始输出。','线上反馈只可在合法授权下收集和处理，进入训练前重新清洗、标注、去重、审查许可与数据污染。版本化每轮数据和模型，最终测试若参与选择就应另留评测集。'),
 s('重点三：面试与简历需要什么证据','按照问题、起始模型、数据、对照、方法、评测协议、成本、结果与局限组织叙述。量化变化必须有模型和 benchmark、分母、seed 及原始记录支持；小样本流程冒烟不能表述为正式安全效果。','准备解释一个失败、一项排除混淆的消融、一个裁判或数据限制和下一步；说明你真正实现的模块，区分论文配方、教学示例和自己的可复现实验。')],
 snippet:code('配置示意：把一个实验绑定到版本与数据',`manifest = {
    "experiment_id": "boundary-dpo-e3",
    "base_model_revision": base_revision,
    "dataset_sha256": dataset_sha256,
    "reference_checkpoint": sft_checkpoint,
    "evaluation_protocol": evaluation_protocol_version,
    "seed": seed,
}`),
 papers:[p('modelcard','模型卡如何表达用途、评测和局限？'),p('datasheet','数据卡如何记录来源和审查过程？')]
}
};
