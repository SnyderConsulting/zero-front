import test from 'node:test';
import assert from 'node:assert/strict';
import {ACTION_HEADS,actionLabels,collectSamples} from '../training/bridge.mjs';
import {assertHeldOut} from '../training/provenance.mjs';

test('teacher actions quantize into the deployed policy heads, including retreat and reload',()=>{
  assert.deepEqual(actionLabels({forward:-.35,strafe:.55,turn:2.9,pitch:-.9,fire:true,reload:false}),[0,2,4,0,1,0]);
  assert.deepEqual(actionLabels({}),[1,1,2,1,0,0]);
});

test('demonstrations are deterministic pixel/status packets with bounded per-seed sampling',()=>{
  const options={seeds:[42,43],teamSize:2,seconds:1,actorsPerFrame:2,maxSamples:8};
  const first=[],second=[];
  collectSamples(options,packet=>first.push(packet));
  collectSamples(options,packet=>second.push(packet));
  assert.deepEqual(first,second);
  const complete=first.at(-1);assert.equal(complete.type,'complete');assert.equal(complete.samples,8);
  assert.deepEqual(complete.seeds.map(seed=>seed.samples),[4,4]);
  for(const packet of first.filter(packet=>packet.type==='samples')){
    assert.deepEqual(Object.keys(packet).sort(),['count','labels','pixels','scalars','type']);
    assert.equal(Buffer.from(packet.pixels,'base64').length,packet.count*64*36*3);
    assert.equal(packet.scalars.length,packet.count*6);assert.equal(packet.labels.length,packet.count*6);
    for(let i=0;i<packet.labels.length;i++)assert.ok(packet.labels[i]>=0&&packet.labels[i]<ACTION_HEADS[i%6].values.length);
    for(let i=0;i<packet.scalars.length;i++)assert.ok(Number.isFinite(packet.scalars[i]));
  }
});

test('collection validates bounds before producing demonstration data',()=>{
  const emit=()=>assert.fail('Invalid collection emitted data.');
  for(const options of [{seeds:[]},{seeds:[-1]},{seeds:[42],teamSize:1},{seeds:[42],seconds:181},
    {seeds:[42],actorsPerFrame:100},{seeds:[42,43],maxSamples:1}])assert.throws(()=>collectSamples(options,emit),TypeError);
});

test('gameplay evaluation rejects current training, ancestral training and validation seeds',()=>{
  const training={trainSeeds:[10,11],priorTrainingSeeds:[2,3],validationSeeds:[20]};
  for(const seed of [2,3,10,11,20])assert.throws(()=>assertHeldOut([seed],training),/Choose unseen seeds/);
  assert.doesNotThrow(()=>assertHeldOut([30,31],training));
  assert.throws(()=>assertHeldOut([30],{trainSeeds:['10']}),TypeError);
});
