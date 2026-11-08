import test from 'node:test';
import assert from 'node:assert/strict';
import {createSimulation,scriptedAction} from '../dist/simulation.js';
import {createExpert,pixelTargets,labels,EXPERT_HEADS,collectSamples,quantize} from '../training/expert.mjs';
import {ACTION_HEADS_V2} from '../dist/policy.js';

test('expert labels use exactly the deployed v2 action definitions',()=>{
  assert.equal(EXPERT_HEADS,ACTION_HEADS_V2);
  assert.deepEqual(labels({forward:1,strafe:-1,turn:.3,pitch:0,fire:true,reload:false}),[2,0,5,1,1,0]);
  assert.deepEqual(labels({forward:-.35,strafe:.55,turn:0,pitch:0,fire:false,reload:false}),[0,2,4,1,0,0]);
});

test('built-in demonstration teacher uses scripted labels without advancing the simulation',()=>{
  const sim=createSimulation({seed:12,teamSize:4}),teacher=createExpert({scriptedTeacher:true});
  const before=JSON.stringify(sim.player._brain),rng=sim.rngState;
  assert.deepEqual(teacher(sim,sim.player),quantize(scriptedAction(sim,sim.player.id)));
  assert.equal(JSON.stringify(sim.player._brain),before);assert.equal(sim.rngState,rng);
});

test('collection honors requested team/sample size and rejects invalid native DAgger',async()=>{
  const packets=[];
  await collectSamples({seeds:[12,13],seconds:1,teamSize:2,actorsPerFrame:1,maxSamples:10},packet=>packets.push(packet));
  assert.equal(packets.at(-1).samples,10);assert.equal(packets.at(-1).teamSize,2);
  assert.equal(packets.at(-1).actorsPerFrame,1);
  assert.equal(packets.filter(packet=>packet.type==='progress').length,2);
  await assert.rejects(collectSamples({seeds:[12],scriptedTeacher:true,model:'unused'},()=>{}),/demonstration-only/);
  await assert.rejects(collectSamples({seeds:[],maxSamples:0},()=>{}),/Invalid seeds/);
  await assert.rejects(collectSamples({seeds:[12],teamSize:2,actorsPerFrame:3},()=>{}),/sample size/);
});

test('enemy pixel components detect visible red pixels without treating allies as enemies',()=>{
  const pixels=new Uint8Array(64*36*3);
  for(let y=12;y<20;y++)for(let x=30;x<33;x++)pixels.set([240,88,70],(y*64+x)*3);
  for(let y=10;y<22;y++)for(let x=40;x<45;x++)pixels.set([55,180,240],(y*64+x)*3);
  const targets=pixelTargets(pixels);
  assert.equal(targets.length,1);assert.equal(targets[0].count,24);assert.equal(targets[0].x,31);
});

test('scan-and-engage demonstration labels depend only on supplied pixels and own status',()=>{
  const a=createSimulation({seed:12,teamSize:4}),b=createSimulation({seed:600,teamSize:4});
  const observation={pixels:new Uint8Array(64*36*3),scalars:new Float32Array(6)};
  Object.assign(a.player,{health:100,ammo:10,pitch:0});Object.assign(b.player,{health:100,ammo:10,pitch:0});
  const expert=createExpert({camp:true});
  assert.deepEqual(expert(a,a.player,observation),expert(b,b.player,observation));
  assert.equal(expert(a,a.player,observation).turn,.3);
  assert.equal(createExpert({camp:true,scanRate:3.6})(a,a.player,observation).turn,3.6);
  assert.throws(()=>createExpert({camp:true,scanRate:Infinity}),TypeError);
  for(let y=12;y<25;y++)for(let x=36;x<40;x++)observation.pixels.set([240,88,70],(y*64+x)*3);
  assert.deepEqual(expert(a,a.player,observation),expert(b,b.player,observation));
  assert.equal(expert(a,a.player,observation).forward,1);
});
