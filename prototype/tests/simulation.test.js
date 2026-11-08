import test from 'node:test';
import assert from 'node:assert/strict';
import {createSimulation,reset,step,stepAgents,observe,raycast,wallAt} from '../dist/simulation.js';
import {combatFixture} from './fixtures.js';
const idleAll=sim=>Object.fromEntries(sim.actors.map(a=>[a.id,{}]));

for(const teamSize of [1,4,8])test(`${teamSize}v${teamSize} has correct actors and readable symmetric layout`,()=>{
  const sim=createSimulation({teamSize});
  assert.equal(sim.actors.length,teamSize*2);assert.equal(sim.entities.length,teamSize*2-1);
  assert.equal(sim.player,sim.actors[0]);assert.equal(sim.player.id,'player');
  for(const team of [0,1])assert.equal(sim.actors.filter(a=>a.team===team).length,teamSize);
  for(const a of sim.actors)assert.equal(wallAt(sim,a.x,a.y),'.');
  for(let y=0;y<sim.height;y++)for(let x=0;x<sim.width;x++){
    const otherX=sim.symmetry==='mirror-y'?x:sim.width-1-x;
    const otherY=sim.symmetry==='mirror-x'?y:sim.height-1-y;
    assert.equal(sim.map[y][x]==='.',sim.map[otherY][otherX]==='.');
  }
});

test('seeded bot matches are deterministic and reset keeps the same simulation object',()=>{
  const a=createSimulation({seed:813}),b=createSimulation({seed:813});
  for(let i=0;i<15;i++){const action={strafe:i%3===0?.5:0,turn:i%4===0?.2:0,fire:i%2===0};step(a,action,60);step(b,action,60);}
  assert.deepEqual(a,b);
  assert.equal(reset(a),a);assert.deepEqual(a,createSimulation({seed:813}));
});

test('bots move, shoot, and score without human input',()=>{
  const sim=createSimulation({seed:42});
  for(let i=0;i<6;i++)step(sim,{},600);
  assert.ok(sim.scores[0]+sim.scores[1]>=3,`Expected a progressing match; scores ${sim.scores}`);
  assert.ok(sim.actors.some(a=>a.shots>10));
});

test('human death awards the enemy team a point, then respawns with protection',()=>{
  const sim=combatFixture({teamSize:1});const red=sim.entities[0];
  Object.assign(red,{x:5.5,y:11.5,yaw:Math.PI,spawnProtection:0});sim.player.spawnProtection=0;
  stepAgents(sim,{player:{},[red.id]:{fire:true}},75);
  assert.equal(sim.player.active,false);assert.equal(sim.player.deaths,1);assert.deepEqual(sim.scores,[0,1]);
  stepAgents(sim,idleAll(sim),140);
  assert.equal(sim.player.active,true);assert.equal(sim.player.health,100);assert.ok(sim.player.spawnProtection>0);
  assert.equal(sim.player.ammo,20);
});

test('actual hitscan respects walls and allies, with no friendly fire',()=>{
  const sim=combatFixture({teamSize:2});const ally=sim.actors[1],red=sim.actors[2];
  for(const a of sim.actors)a.spawnProtection=0;
  Object.assign(ally,{x:4.5,y:11.5});Object.assign(red,{x:6.5,y:11.5});
  let actions=idleAll(sim);actions.player={fire:true};stepAgents(sim,actions,40);
  assert.equal(ally.health,100);assert.equal(red.health,100);assert.equal(sim.player.hits,0);
  Object.assign(sim.player,{x:5.5,y:5.5,yaw:0,cooldown:0});Object.assign(red,{x:10.5,y:5.5});
  assert.equal(raycast(sim,5.5,5.5,0).distance,1.5);
  stepAgents(sim,actions,40);assert.equal(red.health,100);
});

test('first team to score limit wins, and completed matches stop advancing',()=>{
  const sim=combatFixture({teamSize:1,scoreLimit:1});const red=sim.entities[0];
  Object.assign(red,{x:5.5,y:11.5,spawnProtection:0});
  const result=stepAgents(sim,{player:{fire:true},[red.id]:{}},90);
  assert.equal(result.terminated,true);assert.equal(sim.winner,0);assert.deepEqual(sim.scores,[1,0]);
  assert.equal(sim.player.kills,1);assert.equal(sim.player.hits,7);assert.equal(sim.player.shots,7);
  const hits=result.events.filter(event=>event.type==='hit');
  assert.equal(hits.length,7);assert.equal(hits.at(-1).damage,15);assert.equal(hits.at(-1).actualDamage,10);
  assert.equal(hits.reduce((sum,event)=>sum+event.actualDamage,0),100);
  const tick=sim.tick;step(sim,{forward:1},60);assert.equal(sim.tick,tick);
});

test('spawn protection blocks damage and time limit produces a draw for idle teams',()=>{
  const sim=combatFixture({teamSize:1});const red=sim.entities[0];Object.assign(red,{x:5.5,y:11.5});
  stepAgents(sim,{player:{fire:true},[red.id]:{}},40);assert.equal(red.health,100);
  reset(sim);for(let i=0;i<18;i++)stepAgents(sim,idleAll(sim),600);
  assert.equal(sim.time,180);assert.equal(sim.truncated,true);assert.equal(sim.terminated,false);assert.equal(sim.winner,'draw');
});

test('wall collision, timed reload, per-actor observation, and pre-mutation validation',()=>{
  const sim=combatFixture({teamSize:1});let actions=idleAll(sim);actions.player={forward:-1};stepAgents(sim,actions,600);
  assert.ok(sim.player.x>=1.34);assert.equal(wallAt(sim,sim.player.x,sim.player.y),'.');
  sim.player.ammo=7;actions.player={reload:true};stepAgents(sim,actions);assert.equal(sim.player.ammo,7);
  actions.player={};stepAgents(sim,actions,67);assert.equal(sim.player.ammo,20);
  const before=JSON.stringify(sim);
  for(const a of[{forward:2},{turn:NaN},{fire:1},{unknown:true},null])assert.throws(()=>step(sim,a),TypeError);
  for(const n of[0,601,1.5,Infinity])assert.throws(()=>step(sim,{},n),TypeError);
  assert.throws(()=>stepAgents(sim,{missing:{}}),TypeError);assert.equal(JSON.stringify(sim),before);
  for(const options of[{teamSize:0},{teamSize:9},{scoreLimit:101},{scenario:'range'},{seed:1.2}])assert.throws(()=>reset(sim,options),TypeError);
  const obs=observe(sim,sim.entities[0].id);assert.equal(obs.team,1);assert.ok(!('map'in obs));assert.ok(!('entities'in obs));assert.ok(!('x'in obs));
});
