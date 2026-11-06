import test from 'node:test';
import assert from 'node:assert/strict';
import * as m from '../web/dist/math.mjs';
const close=(a,b,tolerance=1e-8)=>assert.ok(Math.abs(a-b)<tolerance,`${a} ≠ ${b}`);
test('SFT example, masked positions, empty supervision and invalid probabilities',()=>{
 close(m.sftLoss([.8,.6,.5,.25],[0,0,1,1]).loss,1.0397207708399179);
 close(m.sftLoss([.01,.99,.5,.25],[0,0,1,1]).loss,1.0397207708399179);
 assert.equal(m.sftLoss([.5],[0]).loss,null);
 assert.throws(()=>m.sftLoss([0],[1]));assert.throws(()=>m.sftLoss([.5],[2]));
});
test('softmax normalization, shift invariance, temperature entropy and stable extremes',()=>{
 const a=m.distribution([2,1,0,-1],1),b=m.distribution([1002,1001,1000,999],1);
 close(a.p.reduce((x,y)=>x+y),1);a.p.forEach((v,i)=>close(v,b.p[i]));
 assert.ok(m.distribution([2,1,0,-1],3).entropy>a.entropy);
 assert.ok(Number.isFinite(m.distribution([1000,-1000],.1).entropy));assert.throws(()=>m.distribution([1,0],0));
});
test('LoRA arithmetic uses both matrices and validates dimensions',()=>{
 assert.deepEqual(m.loraCount(100,100,4),{base:10000,adapter:800,percent:8});assert.throws(()=>m.loraCount(1,4,0));
});
test('reward difference invariant to offset; loss is stable',()=>{
 close(m.rewardPair(2,0).probability,.8807970779778823);
 close(m.rewardPair(2,0).probability,m.rewardPair(102,100).probability);
 assert.equal(m.rewardPair(-1000,1000).loss,2000);
});
test('DPO equal-policy baseline, worked example and common shifts',()=>{
 close(m.dpoLoss(-2,-4,-2,-4,.1).loss,Math.log(2));
 const r=m.dpoLoss(-2,-4,-3,-4,.1);close(r.margin,1);close(r.loss,.6443966600735709);
 close(m.dpoLoss(-1,-3,-3,-4,.1).loss,r.loss);
 assert.ok(m.dpoLoss(-1,-4,-3,-4,.1).loss<r.loss);
 assert.throws(()=>m.dpoLoss(1,-1,-1,-1,.1));
});
test('PPO clipping sign cases and zero advantage',()=>{
 close(m.ppoSurrogate(1.5,1,.2).surrogate,1.2);
 close(m.ppoSurrogate(.5,-1,.2).surrogate,-.8);
 close(m.ppoSurrogate(1.5,-1,.2).surrogate,-1.5);
 close(m.ppoSurrogate(.5,1,.2).surrogate,.5);
 close(m.ppoSurrogate(1.5,0,.2).surrogate,0);
});
test('GRPO signal, equal rewards and affine invariance',()=>{
 const a=m.groupAdvantages([0,0,1,1]);close(a.mean,.5);close(a.std,.5);
 a.advantages.forEach((x,i)=>close(x,i<2?-1:1,1e-6));
 assert.deepEqual(m.groupAdvantages([1,1,1,1]).advantages,[0,0,0,0]);
 const b=m.groupAdvantages([10,10,11,11]);a.advantages.forEach((x,i)=>close(x,b.advantages[i]));
 assert.throws(()=>m.groupAdvantages([1,NaN]));assert.throws(()=>m.groupAdvantages([1]));
});
test('safety denominators cannot silently become zero or negative',()=>{
 assert.deepEqual(m.safetyMetrics(100,0,100,100),{harmRate:0,refusalRate:1});
 assert.deepEqual(m.safetyMetrics(0,0,0,0),{harmRate:null,refusalRate:null});
 assert.throws(()=>m.safetyMetrics(2,3,4,1));assert.throws(()=>m.safetyMetrics(2.5,1,4,1));
});
test('sampling edge probabilities and memory assumptions',()=>{
 close(m.samplingProbability(.5,2),.75);assert.equal(m.samplingProbability(0,3),0);assert.equal(m.samplingProbability(1,3),1);
 close(m.memoryEstimate(7,4,4).perWorker,32);assert.throws(()=>m.memoryEstimate(7,0,4));
});
