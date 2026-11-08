/** Held-out, real-match comparison. A single selected actor uses the tested policy. */
import {readFile,writeFile,mkdir} from 'node:fs/promises';
import path from 'node:path';
import {createSimulation,setController,stepAgents} from '../dist/simulation.js';
import {createPolicyController,validateModel} from '../dist/policy.js';
import {ACTION_HEADS} from './bridge.mjs';
import {assertHeldOut} from './provenance.mjs';

const args=process.argv.slice(2),value=(name,fallback)=>{const i=args.indexOf(name);return i<0?fallback:args[i+1];};
if(args.includes('--help')){
  console.log('node training/evaluate.mjs --model models/pixel-baseline.json [--seeds 3000:3005] [--team-size 4] [--seconds 180] [--output training/runs/evaluation.json]');
  process.exit(0);
}
function parseSeeds(raw){
  const values=[];
  for(const item of raw.split(',')){
    if(item.includes(':')){const [a,b]=item.split(':').map(Number);if(!Number.isInteger(a)||!Number.isInteger(b)||b<=a||b-a>100)throw new Error('Invalid seed range.');for(let i=a;i<b;i++)values.push(i);}
    else values.push(Number(item));
  }
  if(!values.length||values.length>100||values.some(seed=>!Number.isInteger(seed)||seed<0||seed>4294967295))throw new Error('Provide 1–100 unsigned 32-bit seeds.');
  return [...new Set(values)];
}
const modelPath=value('--model');if(!modelPath)throw new Error('--model is required.');
const model=validateModel(JSON.parse(await readFile(modelPath,'utf8')));
const seeds=parseSeeds(value('--seeds','3000:3005'));
assertHeldOut(seeds,model.training);
const teamSize=Number(value('--team-size','4')),seconds=Number(value('--seconds','180'));
if(!Number.isInteger(teamSize)||teamSize<2||teamSize>8)throw new Error('--team-size must be 2–8.');
if(!Number.isInteger(seconds)||seconds<1||seconds>180)throw new Error('--seconds must be 1–180.');
function randomController(seed){
  let state=(seed||1)>>>0,last=-Infinity,cached={};
  const random=()=>{state^=state<<13;state^=state>>>17;state^=state<<5;return(state>>>0)/4294967296;};
  return sim=>{
    if(sim.tick-last>=6){last=sim.tick;cached=Object.fromEntries(ACTION_HEADS.map(head=>[head.name,head.values[Math.floor(random()*head.values.length)]]));}
    return cached;
  };
}
const matches=[];
for(const kind of ['model','random','scripted'])for(const seed of seeds)for(const team of [0,1]){
  const sim=createSimulation({seed,teamSize,scoreLimit:25,recordReplay:false});
  setController(sim,'player','scripted');
  const actor=sim.actors.find(actor=>actor.team===team&&actor.id!=='player');
  if(kind==='model')setController(sim,actor.id,createPolicyController(model));
  if(kind==='random')setController(sim,actor.id,randomController(seed^((team+1)*0x9e3779b9)));
  while(!sim.terminated&&!sim.truncated&&sim.time<seconds)stepAgents(sim,{},Math.min(6,Math.max(1,Math.round((seconds-sim.time)*60))));
  const row={kind,seed,team,actorId:actor.id,seconds:sim.time,complete:sim.terminated||sim.truncated,
    winner:sim.winner,win:sim.winner===team,draw:sim.winner==='draw',scores:sim.scores,
    kills:actor.kills,deaths:actor.deaths,shots:actor.shots,hits:actor.hits,damageDealt:actor.damageDealt};
  matches.push(row);console.log(JSON.stringify({type:'match',...row}));
}
const comparisons={};
for(const kind of ['model','random','scripted']){
  const rows=matches.filter(row=>row.kind===kind),sum=key=>rows.reduce((total,row)=>total+row[key],0);
  comparisons[kind]={matches:rows.length,completed:rows.filter(row=>row.complete).length,
    wins:rows.filter(row=>row.win).length,draws:rows.filter(row=>row.draw).length,
    winRate:rows.filter(row=>row.complete).length?rows.filter(row=>row.win).length/rows.filter(row=>row.complete).length:null,
    kills:sum('kills'),deaths:sum('deaths'),killsPerMinute:sum('kills')/Math.max(1e-6,sum('seconds')/60),
    shots:sum('shots'),hits:sum('hits'),accuracy:sum('shots')?sum('hits')/sum('shots'):0,
    damageDealt:sum('damageDealt')};
}
const result={model:model.name,createdAt:new Date().toISOString(),seeds,teamSize,seconds,
  note:'One tested actor per match; all other actors are scripted. Team win rate includes scripted teammates. Partial matches have no win result.',comparisons,matches};
const output=value('--output');if(output){await mkdir(path.dirname(output),{recursive:true});await writeFile(output,JSON.stringify(result,null,2)+'\n');}
console.log(JSON.stringify({type:'evaluation',...result}));
