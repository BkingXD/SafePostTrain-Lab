import {mkdir,writeFile,copyFile} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
import {execFileSync} from 'node:child_process';
import {course,glossary,sources,stages} from '../web/course.mjs';
const root=path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const output=path.join(root,'web/dist/content');
await mkdir(output,{recursive:true});
await mkdir(path.join(root,'docs/curriculum'),{recursive:true});
await writeFile(path.join(output,'course.json'),JSON.stringify({version:1,course,glossary,sources,stages}));
for(const chapter of course){
 const refs=chapter.references.map(key=>`- [${sources[key][0]}](${sources[key][1]})`).join('\n');
 const text=`# ${chapter.id} · ${chapter.title}\n\n${chapter.subtitle}\n\n建议学习时间：${chapter.minutes} 分钟（含阅读与练习，按个人基础调整）。\n\n前置章节：${chapter.prerequisites.join('、')||'无'}。\n\n## 学习目标\n\n${chapter.objectives.map(x=>'- '+x).join('\n')}\n\n${chapter.sections.map(s=>'## '+s.title+'\n\n'+s.paragraphs.join('\n\n')).join('\n\n')}\n\n## 常见误区\n\n${chapter.pitfall}\n\n## 检查理解\n\n${chapter.quizzes.map((q,i)=>`${i+1}. ${q.question}\n\n${q.options.map((x,j)=>`   - ${String.fromCharCode(65+j)}. ${x}`).join('\n')}\n\n   答案：${String.fromCharCode(65+q.correct)}。${q.explanation}`).join('\n\n')}\n\n## 一手资料\n\n${refs}\n\n本章为原创教学讲解，数值示例不代表真实模型训练结果。\n`;
 await writeFile(path.join(root,'docs/curriculum',chapter.id+'.md'),text);
}
console.log(`Built ${course.length} chapters, ${course.reduce((n,c)=>n+c.quizzes.length,0)} checks, ${glossary.length} glossary terms.`);
const downloads=path.join(root,'web/dist/downloads');
await mkdir(downloads,{recursive:true});
for(const file of ['core_math.py','toy_training.py','reward_model.py','data_audit.py','README.md'])await copyFile(path.join(root,'labs',file),path.join(downloads,file));
await copyFile(path.join(root,'docs/LEARNING_CHECKLIST.md'),path.join(downloads,'LEARNING_CHECKLIST.md'));
execFileSync('python3',[path.join(root,'scripts/package_labs.py')],{stdio:'inherit'});
