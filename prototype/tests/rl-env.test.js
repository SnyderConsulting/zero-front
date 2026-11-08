import test from 'node:test';
import assert from 'node:assert/strict';
import {TeamVectorEnv} from '../training/rl-env.mjs';

const idle=[1,1,4,1,0,0];
test('team vector environments expose only pixel/status policy observations and alternate teams',()=>{
  const env=new TeamVectorEnv({environments:2,teamSize:2,seconds:1});
  const initial=env.observe();
  assert.equal(initial.count,4);assert.equal(Buffer.from(initial.pixels,'base64').length,4*36*64*3);
  assert.equal(initial.scalars.length,24);assert.deepEqual(initial.alive,[1,1,1,1]);
  assert.deepEqual(env.envs.map(entry=>entry.team),[0,1]);
  assert.deepEqual(Object.keys(initial).sort(),['alive','count','done','episodes','pixels','rewards','scalars','type']);
});
test('autoreset returns terminal masks and separate completed episode metadata',()=>{
  const env=new TeamVectorEnv({environments:2,teamSize:2,seconds:1});
  let result;
  for(let i=0;i<10;i++)result=env.step(Array(4).fill(idle).flat());
  assert.deepEqual(result.done,[true,true,true,true]);assert.equal(result.episodes.length,2);
  assert.ok(result.episodes.every(episode=>episode.complete===false&&episode.seconds===1));
  assert.ok(env.envs.every(entry=>entry.sim.tick===0));
  assert.deepEqual(env.envs.map(entry=>entry.team),[1,0]);
  assert.deepEqual(result.scalars.filter((_,index)=>index%6===0),[1,1,1,1]);
});
test('invalid actions fail before any vector environment advances',()=>{
  const env=new TeamVectorEnv({environments:2,teamSize:2});
  const labels=Array(4).fill(idle).flat();labels[labels.length-1]=99;
  assert.throws(()=>env.step(labels),TypeError);assert.ok(env.envs.every(entry=>entry.sim.tick===0));
});
