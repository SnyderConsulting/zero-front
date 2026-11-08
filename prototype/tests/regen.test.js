import test from 'node:test';
import assert from 'node:assert/strict';
import {reset,stepAgents,TICK_RATE,DT,HEALTH_REGEN_DELAY,HEALTH_REGEN_RATE} from '../dist/simulation.js';
import {combatFixture} from './fixtures.js';

const idleAll=sim=>Object.fromEntries(sim.actors.map(actor=>[actor.id,{}]));
const idle=(sim,ticks)=>stepAgents(sim,idleAll(sim),ticks);
const near=(actual,expected)=>assert.ok(Math.abs(actual-expected)<1e-8,`${actual} should equal ${expected}`);
const delayTicks=HEALTH_REGEN_DELAY*TICK_RATE;
function duel(options={}){
  const sim=combatFixture({teamSize:1,...options});
  Object.assign(sim.player,{x:3.5,y:11.5,yaw:0,spawnProtection:0});
  Object.assign(sim.actors[1],{x:5.5,y:11.5,yaw:Math.PI,spawnProtection:0});
  return sim;
}
function fire(sim,actor){
  actor.cooldown=0;
  const actions=idleAll(sim);actions[actor.id]={fire:true};
  stepAgents(sim,actions);
}

test('human and bots regenerate after the full delay, at the same rate, capped at max health',()=>{
  for(const victimIndex of [0,1]){
    const sim=duel(),victim=sim.actors[victimIndex],shooter=sim.actors[1-victimIndex];
    for(let i=0;i<5;i++)fire(sim,shooter);
    assert.equal(victim.health,25);
    idle(sim,delayTicks);
    assert.equal(victim.health,25,'No healing during the three-second delay');
    idle(sim,1);near(victim.health,25+HEALTH_REGEN_RATE*DT);
    idle(sim,TICK_RATE-1);near(victim.health,75);
    idle(sim,TICK_RATE*2);assert.equal(victim.health,victim.maxHealth);
  }
});

test('taking another hit immediately interrupts healing and restarts the complete delay',()=>{
  const sim=duel(),victim=sim.player,shooter=sim.actors[1];
  fire(sim,shooter);idle(sim,delayTicks+12);
  near(victim.health,95);
  fire(sim,shooter);near(victim.health,80);
  assert.equal(victim.lastDamageTick,sim.tick);
  idle(sim,delayTicks);near(victim.health,80);
  idle(sim,1);near(victim.health,80+HEALTH_REGEN_RATE*DT);
});

test('dead actors do not regenerate; respawn and reset clear previous damage timing',()=>{
  const sim=duel(),victim=sim.player,shooter=sim.actors[1];
  for(let i=0;i<7;i++)fire(sim,shooter);
  assert.equal(victim.active,false);assert.equal(victim.health,0);
  idle(sim,TICK_RATE);assert.equal(victim.health,0);assert.equal(victim.active,false);
  idle(sim,TICK_RATE);assert.equal(victim.active,true);assert.equal(victim.health,100);
  assert.equal(victim.lastDamageTick,null);
  Object.assign(victim,{x:3.5,y:11.5,spawnProtection:0});
  fire(sim,shooter);assert.equal(victim.health,85);
  reset(sim);assert.ok(sim.actors.every(actor=>actor.health===100&&actor.lastDamageTick===null));
});

test('completed matches cannot advance regeneration',()=>{
  const sim=duel({scoreLimit:1});fire(sim,sim.actors[1]);
  for(let i=0;i<7;i++)fire(sim,sim.player);
  assert.equal(sim.terminated,true);assert.equal(sim.player.health,85);
  const endTick=sim.tick;idle(sim,600);
  assert.equal(sim.tick,endTick);assert.equal(sim.player.health,85);
});

test('spawn-protected and friendly hits do not postpone regeneration',()=>{
  const protectedSim=duel(),red=protectedSim.actors[1];
  fire(protectedSim,protectedSim.player);const damagedTick=red.lastDamageTick;
  idle(protectedSim,delayTicks);red.spawnProtection=1;
  fire(protectedSim,protectedSim.player);
  assert.equal(red.lastDamageTick,damagedTick);near(red.health,85+HEALTH_REGEN_RATE*DT);

  const sim=combatFixture({teamSize:2});
  const [player,ally,enemy]=sim.actors;
  Object.assign(player,{x:3.5,y:11.5,yaw:0,spawnProtection:0});
  Object.assign(ally,{x:4.5,y:11.5,spawnProtection:0});
  Object.assign(enemy,{x:6.5,y:11.5,yaw:Math.PI,spawnProtection:0});
  fire(sim,enemy);assert.equal(ally.health,85);
  const allyDamageTick=ally.lastDamageTick;idle(sim,delayTicks);
  fire(sim,player);
  assert.equal(ally.lastDamageTick,allyDamageTick);near(ally.health,85+HEALTH_REGEN_RATE*DT);
});
