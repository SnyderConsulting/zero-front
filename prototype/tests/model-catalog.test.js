import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,writeFile,readFile,readdir,rm,mkdir} from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {createModelCatalog,MAX_MODEL_BYTES} from '../desktop/model-catalog.mjs';
import {MODEL_FORMAT,OBSERVATION_SPEC,TENSOR_SHAPES,validateModel,inferPolicy} from '../dist/policy.js';

const validate=input=>{
  if(input?.format!=='test-policy'||typeof input.name!=='string')throw new TypeError('Unsupported model.');
  return {format:input.format,name:input.name,createdAt:input.createdAt||'2026-11-07T00:00:00.000Z',training:{method:'test'}};
};
async function fixture(t,validator=validate){
  const root=await mkdtemp(path.join(os.tmpdir(),'arena-model-catalog-'));
  t.after(()=>rm(root,{recursive:true,force:true}));
  const directory=path.join(root,'models'),catalog=createModelCatalog({directory,validate:validator});
  return {root,directory,catalog};
}

test('catalog imports canonical plain JSON once, lists it, and reads it again',async t=>{
  const {root,directory,catalog}=await fixture(t);
  assert.deepEqual(await catalog.list(),{models:[],errors:[]});
  const source=path.join(root,'external.json');
  await writeFile(source,JSON.stringify({format:'test-policy',name:'Learning agent',ignored:'removed by shared validation'}));
  const first=await catalog.importFile(source),second=await catalog.importFile(source);
  assert.deepEqual(first,second);
  assert.match(first.id,/^model-[a-f0-9]{24}\.json$/);
  assert.equal((await catalog.list()).models.length,1);
  assert.equal((await catalog.read(first.id)).name,'Learning agent');
  assert.equal(JSON.parse(await readFile(path.join(directory,first.id),'utf8')).ignored,undefined);
  assert.deepEqual(await readdir(directory),[first.id]);
});

test('catalog rejects traversal identifiers and keeps valid models beside bad files',async t=>{
  const {directory,catalog}=await fixture(t);
  await catalog.list();
  await writeFile(path.join(directory,'good.json'),JSON.stringify({format:'test-policy',name:'Good'}));
  await writeFile(path.join(directory,'bad.json'),'{"invalid":');
  await writeFile(path.join(directory,'wrong.json'),JSON.stringify({format:'arbitrary-script',name:'Wrong'}));
  await mkdir(path.join(directory,'directory.json'));
  for(const id of ['../outside.json','..\\outside.json','C:\\outside.json','model.json:stream','/outside.json',{},'']){
    await assert.rejects(catalog.read(id),/Invalid model identifier/);
  }
  const result=await catalog.list();
  assert.deepEqual(result.models.map(model=>model.id),['good.json']);
  assert.deepEqual(result.errors.map(error=>error.file),['bad.json','directory.json','wrong.json']);
});

test('oversized and invalid imports never create catalog files',async t=>{
  const {root,directory,catalog}=await fixture(t);
  await catalog.list();
  const source=path.join(root,'too-large.json');
  await writeFile(source,Buffer.alloc(MAX_MODEL_BYTES+1,32));
  await assert.rejects(catalog.importFile(source),/8 MiB/);
  await writeFile(source,'{"format":"wrong"}');
  await assert.rejects(catalog.importFile(source),/Unsupported model/);
  assert.deepEqual(await readdir(directory),[]);
});

test('read revalidates a model that changed after it was listed',async t=>{
  const {directory,catalog}=await fixture(t);
  await catalog.list();
  const filename=path.join(directory,'changing.json');
  await writeFile(filename,JSON.stringify({format:'test-policy',name:'Initially valid'}));
  assert.equal((await catalog.list()).models.length,1);
  await writeFile(filename,'{"format":"wrong"}');
  await assert.rejects(catalog.read('changing.json'),/Unsupported model/);
});

test('real CNN model survives native catalog import and inference; unsupported networks are rejected',async t=>{
  const {root,catalog}=await fixture(t,validateModel);
  const source=path.join(root,'pixel-policy.json');
  const model={format:MODEL_FORMAT,version:1,name:'Pixel policy integration',createdAt:'2026-11-07T00:00:00Z',architecture:'cnn-v1',
    observation:{...OBSERVATION_SPEC,scalars:[...OBSERVATION_SPEC.scalars]},actionSpace:'discrete-v1',actionRepeat:6,
    weights:Object.fromEntries(Object.entries(TENSOR_SHAPES).map(([key,shape])=>[key,{shape:[...shape],data:Array(shape.reduce((a,b)=>a*b,1)).fill(0)}])),
    training:{method:'integration-test'}};
  await writeFile(source,JSON.stringify(model));
  const imported=await catalog.importFile(source),loaded=await catalog.read(imported.id);
  assert.equal(imported.name,model.name);
  assert.deepEqual(loaded,validateModel(model));
  assert.deepEqual(inferPolicy(loaded,{pixels:new Uint8Array(64*36*3),scalars:new Float32Array(6)}).action,
    {forward:-1,strafe:-1,turn:-3.6,pitch:-1.2,fire:false,reload:false});
  assert.deepEqual((await catalog.list()).models.map(entry=>entry.id),[imported.id]);
  model.architecture='untrusted-code';
  await writeFile(source,JSON.stringify(model));
  await assert.rejects(catalog.importFile(source),/Expected an Arena pixel policy/);
  model.architecture='cnn-v1';model.weights.conv1.data[0]=null;
  await writeFile(source,JSON.stringify(model));
  await assert.rejects(catalog.importFile(source),/Invalid conv1 weights/);
});
