const finite = x => { if (!Number.isFinite(x)) throw new Error('请输入有限数值'); return x; };
export const sigmoid = x => x >= 0 ? 1 / (1 + Math.exp(-x)) : Math.exp(x) / (1 + Math.exp(x));
export const softplus = x => Math.max(x, 0) + Math.log1p(Math.exp(-Math.abs(x)));
export function distribution(logits, temperature) {
  if (!Array.isArray(logits) || !logits.length) throw new Error('至少需要一个候选');
  logits.forEach(finite); finite(temperature);
  if (temperature <= 0) throw new Error('温度必须大于零；贪心解码需单独处理');
  const z = logits.map(x => x / temperature), max = Math.max(...z), exp = z.map(x => Math.exp(x-max)), sum = exp.reduce((a,b)=>a+b,0);
  const p=exp.map(x=>x/sum); return { p, entropy:-p.reduce((a,x)=>a+(x>0?x*Math.log(x):0),0) };
}
export function sftLoss(probabilities, mask) {
  if (!probabilities.length || probabilities.length !== mask.length) throw new Error('概率与 mask 长度需一致');
  probabilities.forEach(x=>{finite(x); if(x<=0||x>1) throw new Error('正确 token 概率需在 (0,1] 内');});
  if (mask.some(x=>x!==0&&x!==1)) throw new Error('mask 只能取 0 或 1');
  const count=mask.reduce((a,b)=>a+b,0), terms=probabilities.map((x,i)=>-Math.log(x)*mask[i]);
  return {count, terms, loss:count?terms.reduce((a,b)=>a+b,0)/count:null};
}
export function loraCount(input, output, rank) {
  [input,output,rank].forEach(x=>{finite(x);if(!Number.isInteger(x)||x<1)throw new Error('维度与 rank 必须为正整数');});
  const base=input*output, adapter=rank*(input+output);return {base,adapter,percent:100*adapter/base};
}
export function rewardPair(chosen,rejected) { finite(chosen);finite(rejected);const margin=chosen-rejected;return{margin,probability:sigmoid(margin),loss:softplus(-margin)}; }
export function dpoLoss(chosen,rejected,refChosen,refRejected,beta) {
  [chosen,rejected,refChosen,refRejected,beta].forEach(finite);if(beta<=0)throw new Error('beta 必须大于零');
  if([chosen,rejected,refChosen,refRejected].some(x=>x>0))throw new Error('序列 log-prob 不应大于零');
  const margin=(chosen-rejected)-(refChosen-refRejected),z=beta*margin;return{margin,z,probability:sigmoid(z),loss:softplus(-z)};
}
export function ppoSurrogate(ratio,advantage,epsilon) {
  [ratio,advantage,epsilon].forEach(finite);if(ratio<0||epsilon<0||epsilon>=1)throw new Error('ratio 需非负，epsilon 需在 [0,1)');
  const clippedRatio=Math.max(1-epsilon,Math.min(1+epsilon,ratio)),raw=ratio*advantage,clipped=clippedRatio*advantage,surrogate=Math.min(raw,clipped);return{raw,clipped,clippedRatio,surrogate,loss:-surrogate};
}
export function groupAdvantages(rewards,epsilon=1e-8) {
  if(!Array.isArray(rewards)||rewards.length<2||rewards.length>32)throw new Error('请输入 2～32 个奖励'); rewards.forEach(finite);finite(epsilon);if(epsilon<=0)throw new Error('epsilon 必须大于零');
  const mean=rewards.reduce((a,b)=>a+b,0)/rewards.length, std=Math.sqrt(rewards.reduce((a,b)=>a+(b-mean)**2,0)/rewards.length);
  if(!Number.isFinite(mean)||!Number.isFinite(std))throw new Error('奖励数值过大，请缩小到可稳定计算的范围');
  return{mean,std,advantages:rewards.map(x=>(x-mean)/(std+epsilon))};
}
export function safetyMetrics(harmTotal,harmful,benignTotal,refused) {
  [harmTotal,harmful,benignTotal,refused].forEach(x=>{finite(x);if(!Number.isInteger(x)||x<0)throw new Error('样本数需为非负整数');});
  if(harmful>harmTotal||refused>benignTotal)throw new Error('失败数不能超过对应样本总数');
  return{harmRate:harmTotal?harmful/harmTotal:null,refusalRate:benignTotal?refused/benignTotal:null};
}
export function samplingProbability(p,n) {finite(p);finite(n);if(p<0||p>1||!Number.isInteger(n)||n<1)throw new Error('p 需在 [0,1]，N 需为正整数');return 1-(1-p)**n;}
export function memoryEstimate(billions,workers,activationGB) {
  [billions,workers,activationGB].forEach(finite);if(billions<=0||!Number.isInteger(workers)||workers<1||activationGB<0)throw new Error('参数量需正数，卡数需正整数，激活估计需非负');
  return{weights:billions*2,gradients:billions*2,master:billions*4,optimizer:billions*8,perWorker:billions*16/workers+activationGB};
}
