// Original vector diagrams authored for this project. Text remains editable.
import {mkdir,writeFile} from 'node:fs/promises';
import path from 'node:path';
const flow=(title,nodes,note,caption)=>({kind:'flow',title,nodes,note,caption});
export const moduleFigures={
 '01':flow('从数据到助手：训练阶段与产物',[
  ['原始语料','预测下一个 token'],['预训练','得到通用底模'],['继续预训练','适应领域分布'],['SFT / 对齐','学习任务与偏好'],['评测与部署','检验行为和成本']],
  '流程是常见路径；具体项目可跳过或交替阶段。', '数据、底模、后训练和评测之间的关系；阶段顺序需按具体模型确认。'),
 '02':flow('把“更好”写成可检验目标',[
  ['使用场景','人群与输入分布'],['策略目标','有用且可执行'],['安全约束','限制有害完成'],['能力约束','保留正常任务'],['独立评测','比较基线与预算']],
  '帮助性、安全性、能力和成本分别记录，不能只看单一奖励。', '目标建模：输入分布、目标、约束和独立评测。'),
 '03':flow('数据工程：先分组，再派生样本',[
  ['来源与许可','source / license'],['按意图分组','同源问题一组'],['划分集合','训练 / 开发 / 测试'],['构造样本','对话 / 偏好 / 轨迹'],['审计与清单','去重 / 截断 / hash']],
  '先划分来源组再生成改写，降低同意图跨集合泄漏。', '数据处理流程：追溯来源、分组切分、格式化与审计。'),
 '04':{kind:'mask',title:'SFT 标签与一次因果位移',caption:'同一序列中，用户与角色位置不监督；助手回答及 EOS 参与监督。'},
 '05':{kind:'lora',title:'LoRA：冻结原矩阵，只训练低秩增量',caption:'矩阵形状、低秩增量和新增参数的计算。'},
 '06':flow('偏好标注如何训练奖励模型',[
  ['同一问题 x','固定 prompt'],['两个回答','chosen / rejected'],['评分模型','r(x,y+) / r(x,y−)'],['分数差','σ(r+ − r−)'],['排序损失','−log P(chosen)']],
  '得分排序不是安全真值；还要检查校准、长度偏差和分布外样本。', '同一问题的两个回答经评分模型与成对排序损失形成训练信号。'),
 '07':{kind:'dpo',title:'DPO：四个 log-prob 进入同一个偏好损失',caption:'当前策略与冻结参考对 chosen/rejected 的四项概率，共同决定 DPO 相对目标。'},
 '08':flow('PPO 一次在线更新的状态流',[
  ['Actor 采样','保存 old log-prob'],['Reward 打分','任务或偏好反馈'],['Critic / GAE','估计每步优势'],['PPO 裁剪','比较当前与 old'],['更新与监控','reference KL / 熵']],
  '终止、截断及 padding 需要分别处理；reference 不等于 old policy。', 'PPO 在线更新：采样、反馈、优势、裁剪与参考约束。'),
 '09':{kind:'group',title:'GRPO：同一问题的组内相对优势',caption:'四个奖励经组均值与总体标准差归一化，形成正负优势。'},
 '10':flow('反馈来源与优化方法是不同维度',[
  ['反馈来源','人类 / 模型 / 规则'],['评分与检查','结果 / 过程 / 环境'],['构造信号','奖励 / 代价 / 偏好'],['优化方法','DPO / PPO / GRPO'],['独立评价','裁判复核与反例']],
  'RLHF、RLAIF、RLVR 描述反馈；算法选择是另一维度。', '从反馈来源到训练信号、优化方法和独立评测的关系。'),
 '11':flow('自改进：生成、验证、训练和留出测试',[
  ['训练问题','固定来源分组'],['教师采样 N 次','记录全部候选'],['验证与筛选','保留失败原因'],['训练学生','更新模型权重'],['独立评价','检查泛化与成本']],
  '只做 best-of-N 选择不会更新权重；筛选后训练才产生模型更新。', '教师采样、验证筛选、学生更新与独立评测的闭环。'),
 '12':{kind:'safety',title:'安全对齐：按意图选择边界',caption:'合法、有害及混合请求需要不同处理，并共同评估帮助性与安全性。'},
 '13':flow('不同安全问题需要不同评测',[
  ['诚实与校准','事实 / 不确定性'],['隐私与记忆','泄露 / 查询权限'],['公平与偏见','人群 / 标签切片'],['投毒与后门','触发 / 干净对照'],['复核与报告','攻击假设 / 局限']],
  '每个属性有独立威胁模型；一个拒绝率不能覆盖全部安全维度。', '诚实、隐私、公平和投毒需要各自的威胁模型与指标。'),
 '14':flow('训练显存与资源测量',[
  ['模型权重','底模 / adapter'],['梯度与优化器','只作用于可训练项'],['激活与缓存','随长度和 batch 变化'],['临时与通信','算子 / 分片 / rollout'],['实际测量','已分配 / 保留 / 总量']],
  'PyTorch 分配器峰值与设备总占用口径不同；参数量不等于总显存。', '显存由权重、梯度、激活、临时张量和缓存共同构成。'),
 '15':{kind:'eval',title:'评测分母：行为级 ASR 与尝试级成功率',caption:'同一行为多次攻击时，行为级 ASR 和尝试级成功率使用不同分母。'},
 '16':flow('Agent 轨迹和多模态数据接口',[
  ['用户与输入','文本 / 图像 / 上下文'],['模型决策','选择工具和参数'],['工具或环境','受限执行与返回'],['继续推理','处理不可信观察'],['轨迹评价','完成 / 越权 / 成本']],
  '多模态需保持 processor、图片顺序与模板；工具执行需隔离。', '多模态输入、模型动作、工具反馈和轨迹评测的连接方式。'),
 '17':flow('从实验到可复现交付',[
  ['实验卡','问题 / 对照 / 预算'],['训练制品','模型 / 数据 / 配置'],['原始输出','样本 / 裁判 / 日志'],['复算与复核','指标 / 消融 / 局限'],['部署与模型卡','量化回归 / 版本']],
  '结果需能回溯模型、代码、数据和协议；负结果也应保存。', '从实验卡、训练与评测记录到部署和模型卡的证据链。')
};
const esc=s=>String(s).replace(/[&<>"']/g,ch=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&apos;'}[ch]));
const text=(x,y,value,size=18,fill='#244538',extra='')=>`<text x="${x}" y="${y}" fill="${fill}" font-size="${size}" ${extra}>${esc(value)}</text>`;
const box=(x,y,w,h,fill='#eef4e8',stroke='#ceddc6')=>`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="13" fill="${fill}" stroke="${stroke}"/>`;
const arrow=(x1,y1,x2,y2)=>`<path d="M ${x1} ${y1} L ${x2} ${y2}" fill="none" stroke="#7c947c" stroke-width="2.2" marker-end="url(#arrow)"/>`;
function frame(spec,body,height=250){
 const desc=spec.caption;
 return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 ${height}" role="img" aria-labelledby="title desc"><title id="title">${esc(spec.title)}</title><desc id="desc">${esc(desc)}</desc><defs><marker id="arrow" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0 0 L9 4.5 L0 9 z" fill="#7c947c"/></marker></defs><rect width="1000" height="${height}" rx="20" fill="#f8faf5" stroke="#dce6d7"/><g font-family="Noto Sans CJK SC, Noto Sans SC, Microsoft YaHei, sans-serif">${text(36,43,spec.title,24,'#193d30','font-weight="700"')}${body}</g></svg>\n`;
}
function renderFlow(spec){
 const w=160,step=196,x0=28,y=78;
 let b='';spec.nodes.forEach(([title,note],i)=>{const x=x0+i*step;b+=box(x,y,w,82);b+=text(x+15,y+34,title,19,'#234735','font-weight="700"');b+=text(x+15,y+60,note,14,'#526b58');if(i<spec.nodes.length-1)b+=arrow(x+w+5,y+41,x+step-7,y+41);});
 b+=text(36,210,spec.note,17,'#526b58');return frame(spec,b);
}
function renderMask(spec){
 const tokens=['<bos>','<user>','问题','<assistant>','答案₁','答案₂','<eos>'];
 const labels=['−100','−100','−100','−100','答案₁','答案₂','<eos>'];
 let b=text(37,102,'token',19,'#526b58','font-weight="700"')+text(37,172,'label',19,'#526b58','font-weight="700"');
 tokens.forEach((token,i)=>{const x=150+i*118;b+=box(x,67,108,50,'#ffffff','#d7e2d4')+text(x+54,98,token,16,'#244538','text-anchor="middle"');b+=box(x,137,108,50,i<4?'#f0f0ec':'#ddeddb',i<4?'#d9ddd5':'#b8d1b3')+text(x+54,168,labels[i],17,i<4?'#7a817b':'#245d3a','text-anchor="middle"');});
 b+=`<path d="M 563 122 C 590 112 635 112 664 122" fill="none" stroke="#d28c47" stroke-width="3" marker-end="url(#arrow)"/>`;
 b+=text(38,230,'assistant 位置的 logits 预测下一位置答案₁；标签对齐后只 shift 一次。',17,'#526b58');return frame(spec,b,265);
}
function renderLora(spec){
 let b=box(40,85,190,90,'#e6ede6')+text(135,119,'冻结底模 W₀',21,'#234735','text-anchor="middle" font-weight="700"')+text(135,149,'d_out × d_in',17,'#526b58','text-anchor="middle"');
 b+=text(256,143,'+',32,'#526b58')+box(310,85,145,90,'#dcebd7')+text(382,123,'B',26,'#234735','text-anchor="middle" font-weight="700"')+text(382,150,'d_out × r',16,'#526b58','text-anchor="middle"');
 b+=text(470,143,'×',30,'#526b58')+box(520,105,180,50,'#dcebd7')+text(610,137,'A : r × d_in',18,'#234735','text-anchor="middle" font-weight="700"');
 b+=arrow(714,130,765,130)+box(775,90,190,80,'#e9f0e0')+text(870,123,'W = W₀ + (α/r)BA',17,'#234735','text-anchor="middle" font-weight="700"')+text(870,149,'只更新 A 和 B',16,'#526b58','text-anchor="middle"');
 b+=text(38,219,'新增参数：r × (d_in + d_out)；显存还包括激活、底模和优化器。',17,'#526b58');return frame(spec,b);
}
function renderDpo(spec){
 let b=box(35,75,270,53,'#e7f2e5')+text(170,108,'当前 log πθ(chosen)',18,'#234735','text-anchor="middle"');
 b+=box(35,145,270,53,'#f0f0ec')+text(170,178,'参考 log πref(chosen)',18,'#526b58','text-anchor="middle"');
 b+=box(365,75,270,53,'#e7f2e5')+text(500,108,'当前 log πθ(rejected)',18,'#234735','text-anchor="middle"');
 b+=box(365,145,270,53,'#f0f0ec')+text(500,178,'参考 log πref(rejected)',18,'#526b58','text-anchor="middle"');
 b+=text(170,141,'−',20,'#526b58','text-anchor="middle"')+text(500,141,'−',20,'#526b58','text-anchor="middle"')+arrow(650,136,705,136)+box(720,94,242,85,'#e5efe0')+text(841,126,'Δ = 选中增量 − 拒绝增量',15,'#234735','text-anchor="middle" font-weight="700"')+text(841,155,'L = −log σ(βΔ)',19,'#234735','text-anchor="middle"');
 b+=text(37,235,'两侧的策略增量分别为：当前 log-prob 减参考 log-prob。',17,'#526b58');return frame(spec,b,270);
}
function renderGroup(spec){
 const rewards=['1','1','0','0'],advantages=['+1','+1','−1','−1'];let b='';
 rewards.forEach((v,i)=>{const x=70+i*155;b+=box(x,77,125,53,i<2?'#e0efdc':'#f0f0eb')+text(x+62,109,'奖励  '+v,19,'#244538','text-anchor="middle"');b+=arrow(x+62,135,x+62,160);b+=text(x+62,192,'优势  '+advantages[i],18,i<2?'#275d3b':'#946a4d','text-anchor="middle" font-weight="700"');});
 b+=box(705,80,245,108,'#fff4e3','#ead7b8')+text(827,112,'均值 = 0.5',19,'#654d32','text-anchor="middle"')+text(827,141,'总体标准差 = 0.5',17,'#654d32','text-anchor="middle"')+text(827,169,'全同分 → 相对优势为零',16,'#654d32','text-anchor="middle"');
 b+=text(38,235,'同一 prompt 的 4 个回答；示例忽略极小 ε，不同实现的归一化可能不同。',17,'#526b58');return frame(spec,b,270);
}
function renderSafety(spec){
 let b=box(35,78,285,62,'#e6f1e2')+text(177,115,'合法敏感请求',20,'#244538','text-anchor="middle" font-weight="700"');b+=arrow(326,109,382,109)+box(395,78,250,62,'#f2f7ed')+text(520,115,'提供合适的帮助',18,'#244538','text-anchor="middle"');
 b+=box(35,151,285,62,'#fff1e8','#edd4c4')+text(177,188,'明确有害请求',20,'#74503f','text-anchor="middle" font-weight="700"');b+=arrow(326,182,382,182)+box(395,151,250,62,'#fff7ee','#eddcc8')+text(520,188,'限制有害帮助',18,'#74503f','text-anchor="middle"');
 b+=box(700,94,260,106,'#eaf0e4')+text(830,127,'独立评测',20,'#244538','text-anchor="middle" font-weight="700"')+text(830,153,'有害完成 / 误拒绝',16,'#526b58','text-anchor="middle"')+text(830,178,'普通能力 / 成本',16,'#526b58','text-anchor="middle"');
 b+=text(38,253,'混合请求需区分允许与限制部分；按内容和意图判断，不只看关键词。',17,'#526b58');return frame(spec,b,285);
}
function renderEval(spec){
 let b=box(40,80,275,105,'#e7f1e3')+text(177,116,'100 个行为',23,'#244538','text-anchor="middle" font-weight="700"')+text(177,151,'每个行为最多 5 次攻击',16,'#526b58','text-anchor="middle"');
 b+=arrow(325,132,373,132)+box(388,80,265,105,'#eff4e8')+text(520,114,'20 个行为至少成功一次',17,'#244538','text-anchor="middle"')+text(520,150,'行为级 ASR@5 = 20%',19,'#244538','text-anchor="middle" font-weight="700"');
 b+=arrow(663,132,711,132)+box(725,80,235,105,'#fff3e4','#ead8bd')+text(842,114,'24 / 500 次尝试成功',16,'#654d32','text-anchor="middle"')+text(842,150,'尝试级 = 4.8%',19,'#654d32','text-anchor="middle" font-weight="700"');
 b+=text(38,230,'以上为手算示例：两种指标分母不同；提前停止时使用真实尝试数。',17,'#526b58');return frame(spec,b,265);
}
export function renderModuleFigure(id){
 const spec=moduleFigures[id];if(!spec)throw new Error(`Missing figure ${id}`);
 const renderer={flow:renderFlow,mask:renderMask,lora:renderLora,dpo:renderDpo,group:renderGroup,safety:renderSafety,eval:renderEval}[spec.kind];
 return renderer(spec);
}
export async function writeModuleFigures(root){
 const output=path.join(root,'dist','figures');await mkdir(output,{recursive:true});
 for(const id of Object.keys(moduleFigures))await writeFile(path.join(output,`${id}.svg`),renderModuleFigure(id));
}
