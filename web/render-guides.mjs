// Deliberately small Markdown renderer for this repository's authored guides.
// Raw HTML is escaped. Unknown/unsafe URLs are rendered as text.
const escape = value => String(value).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function destination(href){
 if(/^https?:\/\//.test(href))return href;
 const chapter=href.match(/curriculum\/(\d{2})\.md/);if(chapter)return '#/lesson/'+chapter[1];
 if(href.includes('STUDY_GUIDE.md'))return '#/guide';
 if(href.includes('STAGE1_COVERAGE.md'))return '#/guide/coverage';
 if(href.includes('LEARNING_CHECKLIST.md'))return '#/guide/checklist';
 if(href.includes('PROJECT_OUTLINE.md'))return '#/guide/outline';
 if(href.includes('tiny_lm/README.md')||href.endsWith('tiny_lm/'))return '#/practice';
 if(href.includes('verified_seed7.json'))return './content/tiny-lm-report.json';
 if(href.includes('pilot_seed7.json'))return './content/tiny-lm-pilot.json';
 if(href.match(/tiny_lm\/[a-z_-]+\.(py|txt)$/))return './downloads/tiny-lm/'+href.split('/').at(-1);
 if(href.match(/labs\/[a-z_]+\.py$/))return './downloads/'+href.split('/').at(-1);
 if(href.endsWith('labs/README.md'))return './downloads/README.md';
 return null;
}
export function inline(text){
 return text.split(/(`[^`]+`|\[[^\]]+\]\([^)]+\)|\*\*[^*]+\*\*)/g).map(part=>{
  if(part.startsWith('`')&&part.endsWith('`'))return '<code>'+escape(part.slice(1,-1))+'</code>';
  const link=part.match(/^\[([^\]]+)\]\(([^)]+)\)$/);
  if(link){const href=destination(link[2]);return href?`<a href="${escape(href)}" ${href.startsWith('http')?'target="_blank" rel="noopener noreferrer"':''}>${escape(link[1])}</a>`:`${escape(link[1])}<small>（见源码）</small>`;}
  if(part.startsWith('**')&&part.endsWith('**'))return '<strong>'+escape(part.slice(2,-2))+'</strong>';
  return escape(part);
 }).join('');
}
export function renderGuide(markdown){
 const lines=markdown.replace(/\r/g,'').split('\n');let html=[],paragraph=[],list=[],listType=null,table=[],code=null,language='',title='',headings=[];
 const flush=()=>{
  if(paragraph.length){html.push('<p>'+inline(paragraph.join(' '))+'</p>');paragraph=[];}
  if(list.length){html.push(`<${listType}>`+list.map(x=>'<li>'+inline(x)+'</li>').join('')+`</${listType}>`);list=[];listType=null;}
  if(table.length){const rows=table.filter(row=>!/^\|?\s*:?-+:?\s*\|/.test(row));html.push('<div class="table-scroll" tabindex="0" role="region" aria-label="参考表格"><table class="table"><thead><tr>'+rows[0].split('|').slice(1,-1).map(c=>'<th>'+inline(c.trim())+'</th>').join('')+'</tr></thead><tbody>'+rows.slice(1).map(row=>'<tr>'+row.split('|').slice(1,-1).map(c=>'<td>'+inline(c.trim())+'</td>').join('')+'</tr>').join('')+'</tbody></table></div>');table=[];}
 };
 for(const line of lines){
  if(line.trim().startsWith('```')){if(code===null){flush();code=[];language=line.trim().slice(3);}else{html.push(`<pre aria-label="${escape(language||'代码示例')}"><code>`+escape(code.join('\n'))+'</code></pre>');code=null;}continue;}
  if(code!==null){code.push(line);continue;}
  const heading=line.match(/^(#{1,6})\s+(.*)$/);
  if(heading){flush();if(heading[1]==='#'&&!title){title=heading[2];continue;}const level=Math.min(4,Math.max(2,heading[1].length)),id='guide-heading-'+headings.length;headings.push({id,title:heading[2],level});html.push(`<h${level} id="${id}">${inline(heading[2])}</h${level}>`);continue;}
  if(line.trim().startsWith('|')){if(paragraph.length||list.length)flush();table.push(line.trim());continue;}else if(table.length)flush();
  const item=line.match(/^\s*(?:([-*])|\d+\.)\s+(.*)$/);
  if(item){const kind=item[1]?'ul':'ol';if(paragraph.length||listType&&listType!==kind)flush();listType=kind;list.push(item[2]);continue;}else if(list.length)flush();
  if(!line.trim()){flush();continue;}
  if(line.startsWith('>')){flush();html.push('<blockquote>'+inline(line.replace(/^>\s?/,''))+'</blockquote>');continue;}
  paragraph.push(line.trim());
 }
 if(code!==null)throw new Error('Unclosed code fence in guide');flush();return {title,html:html.join('\n'),headings};
}
