import test from 'node:test';
import assert from 'node:assert/strict';
import {createSimulation,reset,stepAgents,setController,scriptedAction} from '../dist/simulation.js';
import {MODEL_FORMAT,OBSERVATION_SPEC,TENSOR_SHAPES,TENSOR_SHAPES_V2,validateModel,policyObservation,inferPolicy,createPolicyController} from '../dist/policy.js';
import {combatFixture} from './fixtures.js';

function model(){
  return {format:MODEL_FORMAT,version:1,name:'Test policy',createdAt:'2026-11-07T00:00:00Z',architecture:'cnn-v1',
    observation:{...OBSERVATION_SPEC,scalars:[...OBSERVATION_SPEC.scalars]},actionSpace:'discrete-v1',actionRepeat:6,
    weights:Object.fromEntries(Object.entries(TENSOR_SHAPES).map(([key,shape])=>[key,{shape:[...shape],data:Array(shape.reduce((a,b)=>a*b,1)).fill(0)}])),training:{method:'test'}};
}
function healthPolicy(){
  const result=model();
  result.weights.fc.data[1568]=1;
  result.weights.policy.data[2*64]=1;
  result.weights.policyBias.data[1]=.5;
  for(const index of [4,8,12,14,16])result.weights.policyBias.data[index]=1;
  return result;
}

test('model validation rejects incompatible, non-finite and malformed weights before inference',()=>{
  const source=model(),copy=validateModel(source);
  source.weights.fc.data[0]=2;assert.equal(copy.weights.fc.data[0],0);
  for(const mutate of [value=>value.version=2,value=>value.actionRepeat=1,value=>value.observation.width=320,
    value=>value.weights.conv1.shape[0]=9,value=>value.weights.fc.data.pop(),
    value=>value.weights.policy.data[0]=Infinity,value=>value.weights.policy.data[0]=101,
    value=>value.weights.extra={},value=>value.training={code:()=>{}}]){
    const invalid=model();mutate(invalid);assert.throws(()=>validateModel(invalid),TypeError);
  }
});

test('actor pixel observations respect wall occlusion and relative team colors without mutating the world',()=>{
  const sim=combatFixture({teamSize:1}),blue=sim.player,red=sim.actors[1];
  Object.assign(blue,{x:5.5,y:5.5,yaw:0,pitch:0});
  Object.assign(red,{x:10.5,y:5.5,yaw:Math.PI,pitch:0});
  const before=JSON.stringify(sim),hidden=policyObservation(sim,blue.id);
  assert.equal(JSON.stringify(sim),before);
  red.x=13;assert.deepEqual(policyObservation(sim,blue.id).pixels,hidden.pixels);
  Object.assign(blue,{x:5.5,y:11.5});Object.assign(red,{x:8.5,y:11.5});
  const fromBlue=policyObservation(sim,blue.id),fromRed=policyObservation(sim,red.id);
  const countEnemy=pixels=>{let count=0;for(let i=0;i<pixels.length;i+=3)if(pixels[i]===240&&pixels[i+1]===88&&pixels[i+2]===70)count++;return count;};
  assert.ok(countEnemy(fromBlue.pixels)>0);assert.ok(countEnemy(fromRed.pixels)>0);
  assert.deepEqual(Object.keys(fromRed).sort(),['pixels','scalars']);
  assert.equal(fromRed.pixels.length,64*36*3);assert.equal(fromRed.scalars.length,6);
});

test('CNN inference uses own-status inputs and controllers hold independent actions for six ticks',()=>{
  const policy=healthPolicy(),sim=createSimulation({teamSize:1}),blue=sim.player,red=sim.actors[1];
  const controller=createPolicyController(policy);
  sim.tick=1;assert.equal(controller(sim,blue).forward,1);
  blue.health=0;sim.tick=2;assert.equal(controller(sim,blue).forward,1);
  red.health=0;assert.equal(controller(sim,red).forward,0);
  sim.tick=7;assert.equal(controller(sim,blue).forward,0);
  blue.health=100;blue.deaths++;sim.tick=8;assert.equal(controller(sim,blue).forward,1);
  const result=inferPolicy(policy,policyObservation(sim,blue.id));
  assert.equal(result.logits.length,18);assert.deepEqual(result.action,{forward:1,strafe:0,turn:0,pitch:0,fire:false,reload:false});
});

test('v2 uses finer aim bins while preserving v1 tensor and action contracts',()=>{
  const policy=model();policy.architecture='cnn-v2';policy.actionSpace='discrete-v2';
  for(const key of ['policy','policyBias']){
    const shape=TENSOR_SHAPES_V2[key];policy.weights[key]={shape:[...shape],data:Array(shape.reduce((a,b)=>a*b,1)).fill(0)};
  }
  for(const index of [1,4,11,16,18,20])policy.weights.policyBias.data[index]=1;
  const sim=createSimulation({teamSize:1});
  const result=inferPolicy(validateModel(policy),policyObservation(sim,'player'));
  assert.equal(result.logits.length,22);
  assert.deepEqual(result.action,{forward:0,strafe:0,turn:.3,pitch:0,fire:false,reload:false});
  policy.actionSpace='discrete-v1';assert.throws(()=>validateModel(policy),/action space/);
});

test('scripted demonstration labels do not advance RNG or bot memory',()=>{
  const sim=createSimulation({seed:28,teamSize:2}),other=createSimulation({seed:28,teamSize:2});
  const before=JSON.stringify(sim);scriptedAction(sim,'red-1');scriptedAction(sim,'player');
  assert.equal(JSON.stringify(sim),before);
  stepAgents(sim,{},120);stepAgents(other,{},120);assert.equal(JSON.stringify(sim),JSON.stringify(other));
});

test('explicit actions override attached policies, and reset clears all controllers',()=>{
  const sim=combatFixture({teamSize:1}),red=sim.actors[1];
  let calls=0;setController(sim,red.id,()=>{calls++;return {forward:1};});
  const start=red.x;stepAgents(sim,{player:{},[red.id]:{}},2);
  assert.equal(red.x,start);assert.equal(calls,0);
  stepAgents(sim,{player:{}},2);assert.ok(red.x<start);assert.equal(calls,2);
  assert.throws(()=>setController(sim,'missing',()=>({})),TypeError);
  assert.throws(()=>setController(sim,red.id,'unknown'),TypeError);
  reset(sim);assert.equal(sim.controllers.size,0);
});

test('scripted-role and externally controlled bots use identical weapon spread',()=>{
  const attached=combatFixture({seed:7,teamSize:1}),explicit=combatFixture({seed:7,teamSize:1});
  for(const sim of [attached,explicit]){
    sim.player.x=16;sim.actors[1].x=19;sim.actors.forEach(actor=>actor.spawnProtection=0);
  }
  setController(attached,'red-1',()=>({fire:true}));
  stepAgents(attached,{player:{}},35);stepAgents(explicit,{player:{},'red-1':{fire:true}},35);
  assert.equal(attached.rngState,explicit.rngState);
  assert.deepEqual(attached.events,explicit.events);assert.equal(attached.player.health,explicit.player.health);
});
