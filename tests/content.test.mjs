import test from 'node:test';
import assert from 'node:assert/strict';
import {course,glossary,sources,stages} from '../web/course.mjs';
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
test('generated browser content and GitHub-readable chapters match the source',async()=>{
 const generated=JSON.parse(await readFile(new URL('../web/dist/content/course.json',import.meta.url),'utf8'));
 assert.deepEqual(generated.course,course);
 for(const c of course){const doc=await readFile(new URL(`../docs/curriculum/${c.id}.md`,import.meta.url),'utf8');assert.ok(doc.includes(c.title));for(const s of c.sections)for(const p of s.paragraphs)assert.ok(doc.includes(p));}
});
