import test from 'node:test';
import assert from 'node:assert/strict';
import {createSimulation,reset,step,stepAgents,observe,DT,TICK_RATE} from '../dist/simulation.js';
import {createReplay,sampleReplay,buildRoundSummary} from '../dist/replay.js';
import {combatFixture} from './fixtures.js';

const idle=sim=>Object.fromEntries(sim.actors.map(actor=>[actor.id,{}]));
const near=(a,b,tolerance=1e-4)=>assert.ok(Math.abs(a-b)<tolerance,`${a} should equal ${b}`);
function recordedDuel(options={}){
  const sim=combatFixture({teamSize:1,scoreLimit:100,...options});
  Object.assign(sim.player,{x:3.5,y:11.5,yaw:0,spawnProtection:0});
  Object.assign(sim.entities[0],{x:5.5,y:11.5,yaw:Math.PI,spawnProtection:0});
  sim.recordReplay=true;sim.replay=createReplay(sim);
  return sim;
}
function shoot(sim,actor){
  actor.cooldown=0;
  const actions=idle(sim);actions[actor.id]={fire:true};stepAgents(sim,actions);
}

test('recording is opt-in, bounded, and does not change simulation outcomes or observations',()=>{
  const plain=createSimulation({seed:52,teamSize:8,scoreLimit:100});
  const recorded=createSimulation({seed:52,teamSize:8,scoreLimit:100,recordReplay:true});
  assert.equal(plain.replay,null);
  assert.equal(recorded.replay.frameCount,1);
  assert.ok(recorded.replay.frames.byteLength+recorded.replay.scores.byteLength<10*1024*1024);
  for(let i=0;i<4;i++){
    step(plain,{fire:true,turn:.3},600);step(recorded,{fire:true,turn:.3},600);
    assert.deepEqual(observe(recorded),observe(plain));
  }
  assert.equal(recorded.rngState,plain.rngState);
  assert.deepEqual(recorded.actors,plain.actors);
  assert.equal(recorded.replay.frameCount,recorded.tick+1);
  assert.equal(recorded.replay.duration,recorded.time);
  assert.equal(Object.hasOwn(observe(recorded),'replay'),false);
  assert.throws(()=>reset(recorded,{recordReplay:'yes'}),TypeError);
});

test('batched actions retain every intermediate frame and event exactly like individual ticks',()=>{
  const batched=createSimulation({seed:2,teamSize:2,scoreLimit:100,recordReplay:true});
  const single=createSimulation({seed:2,teamSize:2,scoreLimit:100,recordReplay:true});
  step(batched,{forward:.5,turn:.5,fire:true},600);
  for(let i=0;i<600;i++)step(single,{forward:.5,turn:.5,fire:true});
  const length=batched.replay.frameCount*batched.replay.actorCount*batched.replay.stride;
  assert.deepEqual(batched.replay.frames.subarray(0,length),single.replay.frames.subarray(0,length));
  assert.deepEqual(batched.replay.events,single.replay.events);
  assert.equal(batched.replay.frameCount,601);
  for(const tick of [0,1,137,599,600])assert.equal(sampleReplay(batched.replay,tick*DT).tick,tick);
});

test('the entire shot history survives the live event ring and includes exact ray origins',()=>{
  const sim=createSimulation({seed:42,teamSize:8,scoreLimit:100,recordReplay:true});
  for(let i=0;i<3;i++)step(sim,{},600);
  assert.ok(sim.replay.events.length>128);
  assert.equal(sim.events.length,128);
  const first=sim.replay.events[0];
  assert.ok(first.time<sim.events[0].time);
  const shot=sim.replay.events.find(e=>e.type==='shot');
  for(const field of ['fromX','fromY','x','y'])assert.ok(Number.isFinite(shot[field]));
  assert.ok(sampleReplay(sim.replay,shot.time).events.some(e=>e===shot));
  assert.ok(sampleReplay(sim.replay,shot.time-DT/2).events.every(e=>e.time<=shot.time-DT/2));
  const last=sampleReplay(sim.replay,sim.time);
  const back=sampleReplay(sim.replay,0);
  assert.equal(back.tick,0);assert.deepEqual(back.events,[]);
  assert.equal(last.tick,sim.tick);
});

test('sampling interpolates facing across angle wrap without interpolating death or respawn teleports',()=>{
  const sim=recordedDuel(),victim=sim.player,shooter=sim.entities[0];
  for(let i=0;i<7;i++)shoot(sim,shooter);
  const deathTime=sim.time;
  assert.equal(sampleReplay(sim.replay,deathTime).actors[0].active,false);
  assert.equal(sampleReplay(sim.replay,deathTime-DT/2).actors[0].active,true);
  stepAgents(sim,idle(sim),120);
  const respawn=sim.replay.events.find(e=>e.type==='respawn'&&e.id==='player');
  assert.ok(respawn);
  const before=sampleReplay(sim.replay,respawn.time-DT/2).actors[0];
  assert.equal(before.active,false);near(before.x,3.5);near(before.y,11.5);
  const after=sampleReplay(sim.replay,respawn.time).actors[0];
  assert.equal(after.active,true);near(after.x,victim.x);near(after.y,victim.y);

  const turn=recordedDuel();turn.player.yaw=Math.PI-.02;
  // Restart the recording after arranging the initial orientation.
  turn.replay=createReplay(turn);
  stepAgents(turn,{...idle(turn),player:{turn:3}});
  const middle=sampleReplay(turn.replay,DT/2).actors[0];
  near(Math.cos(middle.yaw),Math.cos(Math.PI+.005));
  near(Math.sin(middle.yaw),Math.sin(Math.PI+.005));
});

test('terminal frames and detached summary preserve the final result after reset',()=>{
  const sim=recordedDuel({scoreLimit:1});
  stepAgents(sim,{...idle(sim),player:{fire:true}},600);
  assert.equal(sim.terminated,true);
  const recording=sim.replay,last=sampleReplay(recording,recording.duration);
  assert.equal(recording.frameCount,sim.tick+1);
  assert.deepEqual(last.scores,[1,0]);assert.equal(last.actors[1].active,false);
  assert.equal(recording.summary.winner,0);assert.equal(recording.summary.reason,'score');
  assert.equal(recording.summary.players[0].kills,1);
  assert.equal(recording.summary.players[0].shots,7);
  assert.equal(recording.summary.players[0].accuracy,100);
  assert.deepEqual(recording.summary,buildRoundSummary(sim));
  const frameCount=recording.frameCount;step(sim,{},60);assert.equal(recording.frameCount,frameCount);
  const initial=sampleReplay(recording,0),firstMapRow=recording.map[0];
  reset(sim,{seed:813});
  assert.equal(sim.recordReplay,true);assert.notEqual(sim.replay,recording);assert.equal(sim.replay.frameCount,1);
  sim.player.x=999;sim.map[0]='changed';
  assert.deepEqual(sampleReplay(recording,0),initial);assert.equal(recording.map[0],firstMapRow);
  assert.deepEqual(recording.summary.scores,[1,0]);
  reset(sim,{recordReplay:false});assert.equal(sim.replay,null);
  assert.throws(()=>sampleReplay(recording,NaN),TypeError);
});

test('time-limit draws record the exact endpoint and clamp scrubbing to round bounds',()=>{
  const sim=createSimulation({teamSize:1,recordReplay:true});
  for(let i=0;i<18;i++)stepAgents(sim,idle(sim),600);
  assert.equal(sim.replay.frameCount,180*TICK_RATE+1);
  assert.equal(sim.replay.summary.reason,'time');assert.equal(sim.replay.summary.winner,'draw');
  assert.equal(sampleReplay(sim.replay,-20).time,0);
  assert.equal(sampleReplay(sim.replay,200).time,180);
  assert.equal(sampleReplay(sim.replay,180).tick,10800);
});
