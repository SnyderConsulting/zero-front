import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createSimulation,step} from '../dist/simulation.js';
import {ARENA_LAYOUTS} from '../dist/mapgen.js';

function trackMatch(sim,seconds){
  const previous=new Map(sim.entities.map(actor=>[actor.id,{
    x:actor.x,y:actor.y,shots:actor.shots,deaths:actor.deaths,stationarySeconds:0,
  }]));
  let longestStall=0,firstShot=null,firstKill=null;
  for(let second=0;second<seconds;second++){
    const {events}=step(sim,{},60);
    for(const event of events){
      if(event.type==='shot'&&firstShot===null)firstShot=event.time;
      if(event.type==='elimination'&&firstKill===null)firstKill=event.time;
    }
    for(const actor of sim.entities){
      const prior=previous.get(actor.id);
      const notFighting=actor.active&&actor.deaths===prior.deaths&&actor.shots===prior.shots;
      prior.stationarySeconds=notFighting&&Math.hypot(actor.x-prior.x,actor.y-prior.y)<.12
        ?prior.stationarySeconds+1:0;
      longestStall=Math.max(longestStall,prior.stationarySeconds);
      Object.assign(prior,{x:actor.x,y:actor.y,shots:actor.shots,deaths:actor.deaths});
    }
  }
  return {longestStall,firstShot,firstKill};
}

test('bots keep navigating when enemies flicker into sight around doorway corners',()=>{
  // Preserve the geometry that trapped bots oscillating between path steering
  // and combat strafing, independently of future generator changes.
  const sim=createSimulation({seed:5,teamSize:8,scoreLimit:100});
  const {map,spawns}=JSON.parse(readFileSync(new URL('./fixtures/doorway.json',import.meta.url),'utf8'));
  Object.assign(sim,{map,spawns,width:map[0].length,height:map.length,rooms:[]});
  for(const actor of sim.actors)Object.assign(actor,spawns[actor.team][actor.slot]);
  const {longestStall}=trackMatch(sim,30);
  assert.ok(sim.scores[0]+sim.scores[1]>0,'Bots must reach one another and fight');
  assert.ok(longestStall<=3,`A bot stopped moving and firing for ${longestStall} seconds`);
});

for(const [seed,layout] of ARENA_LAYOUTS.entries())test(`${layout} routes support prompt combat without navigation stalls`,()=>{
  for(const teamSize of [4,8]){
    const sim=createSimulation({seed,teamSize,scoreLimit:100});
    const {longestStall,firstShot,firstKill}=trackMatch(sim,25);
    assert.ok(firstShot!==null&&firstShot<=15,`${teamSize}v${teamSize}: bots must reach firing distance promptly`);
    assert.ok(firstKill!==null&&firstKill<=20,`${teamSize}v${teamSize}: the layout must produce actual eliminations`);
    assert.ok(longestStall<=3,`${teamSize}v${teamSize}: a bot stopped moving and firing for ${longestStall} seconds`);
  }
});
