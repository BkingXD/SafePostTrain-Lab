// Curated selection avoids repeating the same formula and examples in multiple
// inherited sections. Indices refer to module-lessons, deep-dive and applied-practice.
export const moduleSelection = {
 '01':{lesson:[2],deep:[0,1,2],practice:[0,1]},
 '02':{lesson:[2],deep:[2],practice:[0,1]},
 '03':{lesson:[0,1,2],deep:[0],practice:[0,1]},
 '04':{lesson:[1,2],deep:[2],practice:[0,1]},
 '05':{lesson:[0,1,2],deep:[],practice:[1]},
 '06':{lesson:[0,1,2],deep:[],practice:[0,1]},
 '07':{lesson:[2],deep:[2],practice:[0,1]},
 '08':{lesson:[2],deep:[2],practice:[0,1]},
 '09':{lesson:[1,2],deep:[],practice:[1]},
 '10':{lesson:[1],deep:[1],practice:[1]},
 '11':{lesson:[0],deep:[1],practice:[1]},
 '12':{lesson:[2],deep:[2],practice:[0,1]},
 '13':{lesson:[2],deep:[2],practice:[0,1]},
 '14':{lesson:[1,2],deep:[],practice:[0,1]},
 '15':{lesson:[0],deep:[1],practice:[1]},
 '16':{lesson:[2],deep:[0],practice:[0,1]},
 '17':{lesson:[0],deep:[2],practice:[0,1]}
};
export function selectedSections(id,lesson,deep,practice){
 const selection=moduleSelection[id];if(!selection)throw Error(`Missing selection ${id}`);
 const pick=(items,indices,label)=>indices.map(index=>{
  if(!items[index])throw Error(`Invalid ${label} selection ${id}/${index}`);
  const item=items[index];return {...item,title:item.title.replace(/^重点[一二三]：/,'技术重点：')};
 });
 return [...pick(lesson,selection.lesson,'lesson'),...pick(deep,selection.deep,'deep'),...pick(practice,selection.practice,'practice')];
}
