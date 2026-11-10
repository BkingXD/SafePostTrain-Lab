import test from 'node:test';
import assert from 'node:assert/strict';
import {course,glossary,sources,stages} from '../web/course.mjs';
import {deepDive} from '../web/deep-dive.mjs';
import {appliedPractice} from '../web/applied-practice.mjs';
import {moduleSpecs} from '../web/module-spec.mjs';
import {moduleFigures,renderModuleFigure} from '../web/module-figures.mjs';
import {moduleLessons} from '../web/module-lessons.mjs';
import {moduleCases} from '../web/module-cases.mjs';
import {implementationNotes} from '../web/implementation-notes.mjs';
import {execFileSync} from 'node:child_process';
import {readFile} from 'node:fs/promises';
test('curriculum covers the original 17 modules with valid references and learning scaffolds',()=>{
 assert.equal(course.length,17);assert.equal(new Set(course.map(c=>c.id)).size,17);
 const ids=new Set(course.map(c=>c.id));
 for(const c of course){
  assert.ok(c.sections.length>=4,c.id);assert.ok(c.objectives.length>=3,c.id);
  assert.ok(c.sections.flatMap(s=>s.paragraphs).join('').length>=800,`chapter ${c.id} needs substantial prose`);
  assert.ok(stages[c.stage]);c.prerequisites.forEach(id=>assert.ok(ids.has(id)&&id!==c.id));
  assert.ok(c.references.length>=2);c.references.forEach(ref=>assert.ok(sources[ref]?.[1].startsWith('https://')));
  assert.ok(c.quizzes.length>=2);for(const q of c.quizzes){assert.ok(q.options.length>=3);assert.ok(q.correct>=0&&q.correct<q.options.length);assert.ok(q.explanation.length>15);}
 }
 for(const t of glossary)assert.ok(ids.has(t.chapter));
});
test('the prerequisite graph is acyclic',()=>{
 const byId=new Map(course.map(c=>[c.id,c])),visited=new Set();
 function visit(id,path=new Set()){assert.ok(!path.has(id),`prerequisite cycle at ${id}`);if(visited.has(id))return;const next=new Set([...path,id]);byId.get(id).prerequisites.forEach(p=>visit(p,next));visited.add(id);}
 course.forEach(c=>visit(c.id));
});
test('existing stored quiz answers keep their meaning after chapter renaming',async()=>{
 const baseline=JSON.parse(await readFile(new URL('./fixtures/progress-v1.json',import.meta.url),'utf8'));
 const current=course.map(c=>({id:c.id,quizzes:c.quizzes.map(q=>({options:q.options,correct:q.correct}))}));
 assert.deepEqual(current,baseline);
});
test('all chapters expose technical expansions, worked examples and sourced research',()=>{
 assert.deepEqual(Object.keys(deepDive).sort(),course.map(c=>c.id).sort());
 assert.deepEqual(Object.keys(appliedPractice).sort(),course.map(c=>c.id).sort());
 for(const c of course){
  assert.ok(deepDive[c.id].sections.length>=3,`${c.id}: technical coverage`);
  assert.ok(appliedPractice[c.id].length>=2,`${c.id}: examples and diagnostics`);
  assert.ok(deepDive[c.id].references.every(key=>sources[key]?.[1].startsWith('https://')&&c.references.some(kept=>sources[kept][1]===sources[key][1])));
 }
 assert.ok(sources.verifier26[1].includes('2604.15149'));
 assert.ok(sources.refusal26[1].includes('2609.03887'));
});
test('generated browser content and GitHub-readable chapters match the source',async()=>{
 const generated=JSON.parse(await readFile(new URL('../web/dist/content/course.json',import.meta.url),'utf8'));
 assert.deepEqual(generated.course,course);
 for(const c of course){const doc=await readFile(new URL(`../docs/curriculum/${c.id}.md`,import.meta.url),'utf8');assert.ok(doc.includes(c.title));for(const s of c.sections)for(const p of s.paragraphs)assert.ok(doc.includes(p));}
});
test('chapter names match the outline, and every chapter has a usable original diagram',async()=>{
 const outline=await readFile(new URL('../docs/PROJECT_OUTLINE.md',import.meta.url),'utf8');
 const table=outline.split('## 3. 后训练知识模块地图')[1].split('每个模块统一产出')[0];
 const rows=[...table.matchAll(/^\| (\d{2}) \| ([^|]+) \|/gm)].map(m=>({id:m[1],title:m[2].trim()}));
 assert.deepEqual(course.map(({id,title})=>({id,title})),rows);
 assert.deepEqual(moduleSpecs.map(({id,title})=>({id,title})),rows);
 for(const c of course){
  assert.ok(c.topics.length>=5,`${c.id} topics`);
  assert.ok(c.paperReadings.length>=2&&c.snippet.body,`${c.id} research and example`);
  assert.ok(moduleFigures[c.id].caption.length>12&&c.figure.alt.length>12,`${c.id} accessible caption`);
  assert.ok(moduleLessons[c.id].sections.length>=3,`${c.id} technical coverage`);
  assert.ok(implementationNotes[c.id].length>=2&&implementationNotes[c.id].every(note=>c.sections.some(section=>section.title===note.title)),`${c.id} implementation details`);
  assert.ok(c.sections.some(section=>section.title===moduleCases[c.id].title&&section.paragraphs.length>=2),`${c.id} concrete case`);
  const rendered=renderModuleFigure(c.id);
  assert.equal(await readFile(new URL(`../web/dist/figures/${c.id}.svg`,import.meta.url),'utf8'),rendered);
  assert.ok(rendered.includes('<title')&&rendered.includes('<desc')&&!/<script|<foreignObject/i.test(rendered));
  const chapter=await readFile(new URL(`../docs/curriculum/${c.id}.md`,import.meta.url),'utf8');
  assert.ok(chapter.includes(`../../web/dist/figures/${c.id}.svg`));
 }
 const snippets=course.map(c=>({id:c.id,body:c.snippet.body}));
 const script='import ast,json,sys\nfor s in json.load(sys.stdin): ast.parse(s["body"], filename=s["id"])\n';
 execFileSync('python3',['-c',script],{input:JSON.stringify(snippets)});
});
