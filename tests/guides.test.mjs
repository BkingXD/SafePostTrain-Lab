import test from 'node:test';
import assert from 'node:assert/strict';
import {renderGuide,inline} from '../web/render-guides.mjs';
import {readFile} from 'node:fs/promises';

test('guide renderer preserves teaching structure and refuses executable HTML/URLs',()=>{
 const rendered=renderGuide('# 标题\n\n## 公式\n\n用户 token `<user>` 不计 loss。\n\n| 名称 | 值 |\n|---|---|\n| NLL | **1.04** |\n\n```python\nprint("<bos>")\n```\n');
 assert.equal(rendered.title,'标题');assert.ok(rendered.html.includes('<h2'));assert.ok(rendered.html.includes('<table'));assert.ok(rendered.html.includes('&lt;user&gt;'));assert.ok(rendered.html.includes('<strong>1.04</strong>'));
 assert.ok(!inline('<script>alert(1)</script>').includes('<script>'));
 assert.ok(!inline('[bad](javascript:alert)').includes('href='));
 assert.ok(inline('[课程](curriculum/04.md)').includes('#/lesson/04'));
});

test('all seven guide views are generated from current source',async()=>{
 const generated=JSON.parse(await readFile(new URL('../web/dist/content/guides.json',import.meta.url),'utf8'));
 for(const [key,file] of Object.entries({study:'STUDY_GUIDE.md',workflow:'POST_TRAINING_WORKFLOW.md',research:'RESEARCH_FRONTIER.md',models:'REAL_MODEL_USAGE.md',coverage:'STAGE1_COVERAGE.md',checklist:'LEARNING_CHECKLIST.md',outline:'PROJECT_OUTLINE.md'})){
  const source=await readFile(new URL('../docs/'+file,import.meta.url),'utf8');assert.deepEqual(generated[key],renderGuide(source));
 }
 assert.ok(generated.study.html.includes('推导卡'));assert.ok(generated.study.html.includes('案例 A'));
 assert.ok(generated.workflow.html.includes('DPO'));
 assert.ok(generated.research.html.includes('2604.15149'));
 assert.ok(generated.models.html.includes('Qwen3-0.6B'));
});
