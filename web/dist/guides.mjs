let cached;
const escape=value=>String(value).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const names={study:'从这里开始',coverage:'大纲覆盖',checklist:'学习验收',outline:'项目全景'};
export function guidePage(id='study'){
 return `<div class="eyebrow">A GUIDE FOR LEARNING</div><h1 class="lesson-title">把知识，串成一条路。</h1><p class="lesson-intro">有顺序、有例子、有参考答案。知道自己正在学什么，也知道下一步怎么做。</p><nav class="filters" aria-label="学习指南导航">${Object.entries(names).map(([key,name])=>`<a class="filter ${id===key?'selected':''}" href="#/guide/${key}">${name}</a>`).join('')}</nav><div class="guide-document" id="guide-content" data-guide="${escape(id)}"><p role="status">正在打开学习指南…</p></div>`;
}
export async function mountGuide(){
 const root=document.querySelector('#guide-content');if(!root)return;
 try{if(!cached){const response=await fetch('./content/guides.json');if(!response.ok)throw Error('指南加载失败');cached=await response.json();}if(!root.isConnected)return;const guide=cached[root.dataset.guide];if(!guide){root.innerHTML='<p>没有这份指南，请选择上面的入口。</p>';return;}root.innerHTML=`<h2 class="guide-title">${escape(guide.title)}</h2>${guide.html}`;}
 catch{if(root.isConnected)root.innerHTML='<p role="alert">暂时无法打开指南。请刷新页面，或在源码包的 docs 目录阅读 Markdown。</p>';}
}
