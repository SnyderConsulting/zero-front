import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
import {MODEL_FORMAT,OBSERVATION_SPEC,TENSOR_SHAPES,validateModel,createPolicyController} from '../dist/policy.js';

function policy(){
  return {format:MODEL_FORMAT,version:1,name:'Test model',createdAt:'2026-11-07T00:00:00Z',architecture:'cnn-v1',
    observation:{...OBSERVATION_SPEC,scalars:[...OBSERVATION_SPEC.scalars]},actionSpace:'discrete-v1',actionRepeat:6,
    weights:Object.fromEntries(Object.entries(TENSOR_SHAPES).map(([key,shape])=>[key,{shape:[...shape],data:Array(shape.reduce((a,b)=>a*b,1)).fill(0)}]))};
}
function setup(bridge){
  class Element{
    value='';listeners=new Map();children=[];disabled=false;hidden=false;textContent='';
    addEventListener(name,callback){this.listeners.set(name,callback);}
    replaceChildren(...children){this.children=children;}
    append(child){this.children.push(child);}
    click(){this.clicked=true;}
    dispatch(name){return this.listeners.get(name)?.();}
  }
  const elements=new Map();
  const element=id=>{if(!elements.has(id))elements.set(id,new Element());return elements.get(id);};
  element('blue-model').value=element('red-model').value='scripted';
  const context=vm.createContext({window:{arenaModels:bridge},document:{getElementById:element,createElement:()=>new Element()},createPolicyController,validateModel});
  const source=readFileSync(new URL('../dist/round-models.js',import.meta.url),'utf8').replace(/^import .+;\r?\n/gm,'').replace('export function createRoundModels','function createRoundModels');
  vm.runInContext(source,context);
  const models=vm.runInContext('createRoundModels()',context);
  return {models,element};
}

test('selected models preload before play and only selected policy data is requested',async()=>{
  const model=policy();let resolveRead,reads=0;
  const {models,element}=setup({list:async()=>({models:[{id:'test.json',name:model.name}],errors:[]}),read:()=>{reads++;return new Promise(resolve=>{resolveRead=resolve;});}});
  await models.refresh();assert.equal(reads,0);assert.equal(models.ready,true);
  element('red-model').value='test.json';const loading=element('red-model').dispatch('change');
  assert.equal(models.ready,false);assert.equal(element('red-model').disabled,true);
  assert.throws(()=>models.controllers(),/Wait for/);
  resolveRead(model);await loading;
  assert.equal(models.ready,true);assert.equal(element('red-model').disabled,false);
  assert.equal(models.controllers()[0],null);assert.equal(typeof models.controllers()[1],'function');
  element('blue-model').value='test.json';await element('blue-model').dispatch('change');
  assert.equal(reads,1);assert.equal(typeof models.controllers()[0],'function');
});

test('native first load selects and preloads the bundled policy for both teams',async()=>{
  const model=policy();let resolveRead,startedRead;const reads=[];
  const reading=new Promise(resolve=>{startedRead=resolve;});
  const {models,element}=setup({list:async()=>({models:[{id:'pixel-selfplay.json',name:model.name}],errors:[]}),
    read:id=>{reads.push(id);startedRead();return new Promise(resolve=>{resolveRead=resolve;});}});
  const loading=models.refresh();await reading;
  assert.equal(element('blue-model').value,'pixel-selfplay.json');
  assert.equal(element('red-model').value,'pixel-selfplay.json');
  assert.equal(models.ready,false);assert.equal(element('blue-model').disabled,true);
  assert.deepEqual(reads,['pixel-selfplay.json']);
  resolveRead(model);await loading;
  assert.equal(models.ready,true);
  for(const controller of models.controllers())assert.equal(typeof controller,'function');
});

test('native refresh preserves explicit choices after applying the bundled default',async()=>{
  const model=policy();
  const {models,element}=setup({list:async()=>({models:[{id:'pixel-selfplay.json',name:model.name},{id:'other.json',name:'Other'}],errors:[]}),read:async()=>model});
  await models.refresh();
  element('blue-model').value='scripted';await element('blue-model').dispatch('change');
  element('red-model').value='other.json';await element('red-model').dispatch('change');
  await models.refresh();
  assert.equal(element('blue-model').value,'scripted');assert.equal(element('red-model').value,'other.json');
  assert.equal(models.ready,true);assert.equal(models.controllers()[0],null);
  assert.equal(typeof models.controllers()[1],'function');
});

test('a bundled default rejected during read stays selected and blocks play',async()=>{
  const {models,element}=setup({list:async()=>({models:[{id:'pixel-selfplay.json',name:'Pixel self-play'}],errors:[]}),read:async()=>({format:'invalid'})});
  await models.refresh();
  assert.equal(element('blue-model').value,'pixel-selfplay.json');assert.equal(element('red-model').value,'pixel-selfplay.json');
  assert.equal(models.ready,false);assert.throws(()=>models.controllers(),/Wait for/);
  assert.match(element('model-status').textContent,/Zero Front pixel policy/);
});

test('a corrupt bundled file reported by the catalog blocks the initial default',async()=>{
  const {models,element}=setup({list:async()=>({models:[],errors:[{file:'pixel-selfplay.json',message:'Model contains invalid JSON.'}]})});
  await models.refresh();
  assert.equal(element('blue-model').value,'pixel-selfplay.json');assert.equal(element('red-model').value,'pixel-selfplay.json');
  assert.equal(models.ready,false);assert.match(element('model-status').textContent,/no longer available/);
  for(const id of ['blue-model','red-model']){element(id).value='scripted';await element(id).dispatch('change');}
  assert.equal(models.ready,true);
});

test('a rejected selected model blocks play until a valid choice is made, without silently substituting a bot',async()=>{
  const {models,element}=setup({list:async()=>({models:[{id:'broken.json',name:'Broken'}],errors:[]}),read:async()=>({format:'not-an-arena-model'})});
  await models.refresh();element('red-model').value='broken.json';await element('red-model').dispatch('change');
  assert.equal(models.ready,false);assert.equal(element('red-model').value,'broken.json');
  assert.equal(element('model-status').hidden,false);assert.match(element('model-status').textContent,/Zero Front pixel policy/);
  element('red-model').value='scripted';await element('red-model').dispatch('change');
  assert.equal(models.ready,true);assert.equal(element('model-status').hidden,true);
});

test('native import cancellation preserves selections and valid imports refresh the catalog',async()=>{
  let entries=[],result=null;
  const {models,element}=setup({list:async()=>({models:entries,errors:[]}),import:async()=>result});
  await models.refresh();await element('import-model').dispatch('click');
  assert.equal(models.ready,true);assert.equal(element('red-model').value,'scripted');
  result={id:'new.json',name:'New model'};entries=[result];await element('import-model').dispatch('click');
  assert.equal(models.ready,true);assert.ok(element('red-model').children.some(option=>option.value==='new.json'));
});

test('a selected model removed from the catalog remains visibly unavailable until explicitly changed',async()=>{
  const model=policy();let entries=[{id:'test.json',name:model.name}];
  const {models,element}=setup({list:async()=>({models:entries,errors:[]}),read:async()=>model});
  await models.refresh();element('red-model').value='test.json';await element('red-model').dispatch('change');
  assert.equal(models.ready,true);
  entries=[];await models.refresh();
  assert.equal(models.ready,false);assert.equal(element('red-model').value,'test.json');
  const missing=element('red-model').children.find(option=>option.value==='test.json');
  assert.equal(missing.disabled,true);assert.match(missing.textContent,/unavailable/);
  assert.match(element('model-status').textContent,/no longer available/);
  element('red-model').value='scripted';await element('red-model').dispatch('change');
  assert.equal(models.ready,true);
});

test('browser JSON import validates the model and makes it selectable for this session',async()=>{
  const {models,element}=setup();await models.refresh();
  await element('import-model').dispatch('click');assert.equal(element('model-file').clicked,true);
  const model=policy();element('model-file').files=[{size:200000,text:async()=>JSON.stringify(model)}];
  await element('model-file').dispatch('change');
  const option=element('red-model').children.find(value=>value.value!=='scripted');
  assert.equal(option.textContent,model.name);
  element('red-model').value=option.value;await element('red-model').dispatch('change');
  assert.equal(models.ready,true);assert.equal(typeof models.controllers()[1],'function');
});
