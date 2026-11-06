import {mkdir,writeFile} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
import {course,glossary,sources,stages} from './course.mjs';
const root=path.dirname(fileURLToPath(import.meta.url));
await mkdir(path.join(root,'dist/content'),{recursive:true});
await writeFile(path.join(root,'dist/content/course.json'),JSON.stringify({version:1,course,glossary,sources,stages}));
console.log(`Website data: ${course.length} chapters.`);
