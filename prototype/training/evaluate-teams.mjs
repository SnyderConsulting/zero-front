/** Full learned team versus the unchanged scripted team, with paired spawn sides. */
import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {pathToFileURL} from 'node:url';
import path from 'node:path';
import {createSimulation,setController,stepAgents} from '../dist/simulation.js';
import {createPolicyController,validateModel} from '../dist/policy.js';
import {assertHeldOut} from './provenance.mjs';

const wrap=value=>((value+Math.PI)%(Math.PI*2)+Math.PI*2)%(Math.PI*2)-Math.PI;
const integer=(value,name,min,max)=>{
  if(!Number.isInteger(value)||value<min||value>max)throw new TypeError(`${name} must be an integer from ${min} to ${max}.`);
  return value;
};

export function parseSeeds(raw){
  if(typeof raw!=='string')throw new TypeError('A seed list is required.');
  const seeds=[];
  for(const part of raw.split(',')){
    if(part.includes(':')){
      const values=part.split(':');
      if(values.length!==2||values.some(value=>!value.trim()))throw new TypeError('Invalid seed range.');
      const [start,end]=values.map(Number);
      integer(start,'seed',0,4294967295);integer(end,'range end',1,4294967296);
      if(end<=start||end-start>100)throw new TypeError('A seed range must contain 1–100 seeds.');
      for(let seed=start;seed<end;seed++)seeds.push(seed);
    }else{if(!part.trim())throw new TypeError('Empty seed.');seeds.push(integer(Number(part),'seed',0,4294967295));}
  }
  const unique=[...new Set(seeds)];
  if(!unique.length||unique.length>100)throw new TypeError('Provide 1–100 seeds.');
  return unique;
}

/** Every role is explicitly assigned, including the slot normally reserved for a human. */
export function configureTeams(sim,team,controller){
  integer(team,'model team',0,1);
  if(typeof controller!=='function')throw new TypeError('A model controller is required.');
  for(const actor of sim.actors)setController(sim,actor.id,actor.team===team?controller:'scripted');
}

export function runTeamMatch(model,{seed,team,teamSize=4,scoreLimit=25,maxSeconds=180}={}){
  integer(seed,'seed',0,4294967295);integer(team,'model team',0,1);
  integer(teamSize,'team size',1,8);integer(scoreLimit,'score limit',1,100);integer(maxSeconds,'time limit',1,180);
  const sim=createSimulation({seed,teamSize,scoreLimit,recordReplay:false});
  configureTeams(sim,team,createPolicyController(model));
  const learned=sim.actors.filter(actor=>actor.team===team);
  const motion=new Map(learned.map(actor=>[actor.id,{distance:0,stationarySeconds:0,activeSeconds:0,rotationRadians:0}]));
  while(!sim.terminated&&!sim.truncated&&sim.time<maxSeconds){
    const before=learned.map(actor=>({x:actor.x,y:actor.y,yaw:actor.yaw,active:actor.active,deaths:actor.deaths}));
    const start=sim.time;
    stepAgents(sim,{},Math.min(6,Math.max(1,Math.round((maxSeconds-sim.time)*60))));
    const seconds=sim.time-start;
    learned.forEach((actor,index)=>{
      const previous=before[index],entry=motion.get(actor.id);
      if(!previous.active||!actor.active||previous.deaths!==actor.deaths)return;
      const distance=Math.hypot(actor.x-previous.x,actor.y-previous.y);
      entry.distance+=distance;entry.activeSeconds+=seconds;
      if(distance<.02)entry.stationarySeconds+=seconds;
      entry.rotationRadians+=Math.abs(wrap(actor.yaw-previous.yaw));
    });
  }
  const players=learned.map(actor=>({id:actor.id,kills:actor.kills,deaths:actor.deaths,shots:actor.shots,hits:actor.hits,
    damageDealt:actor.damageDealt,...motion.get(actor.id)}));
  const sum=key=>players.reduce((total,actor)=>total+actor[key],0);
  const complete=sim.terminated||sim.truncated;
  return {seed,layout:sim.layout,team,teamSize,scoreLimit,seconds:sim.time,complete,
    winner:sim.winner,win:complete&&sim.winner===team,draw:complete&&sim.winner==='draw',
    scores:[...sim.scores],modelScore:sim.scores[team],scriptedScore:sim.scores[1-team],
    kills:sum('kills'),deaths:sum('deaths'),shots:sum('shots'),hits:sum('hits'),damageDealt:sum('damageDealt'),
    distance:sum('distance'),stationarySeconds:sum('stationarySeconds'),activeSeconds:sum('activeSeconds'),
    rotationRadians:sum('rotationRadians'),players};
}

export function summarizeMatches(rows){
  const completed=rows.filter(row=>row.complete),wins=completed.filter(row=>row.win).length;
  const draws=completed.filter(row=>row.draw).length,total=completed.length;
  const rate=total?wins/total:null,z=1.959963984540054;
  let interval=null;
  if(total){
    const denominator=1+z*z/total,center=(rate+z*z/(2*total))/denominator;
    const half=z*Math.sqrt(rate*(1-rate)/total+z*z/(4*total*total))/denominator;
    interval=[Math.max(0,center-half),Math.min(1,center+half)];
  }
  const sum=key=>rows.reduce((value,row)=>value+(row[key]||0),0);
  return {matches:rows.length,completed:total,wins,draws,losses:total-wins-draws,winRate:rate,
    winRateWilson95:interval,modelScore:sum('modelScore'),scriptedScore:sum('scriptedScore'),
    kills:sum('kills'),deaths:sum('deaths'),shots:sum('shots'),hits:sum('hits'),
    accuracy:sum('shots')?sum('hits')/sum('shots'):0,damageDealt:sum('damageDealt'),
    stationaryFraction:sum('activeSeconds')?sum('stationarySeconds')/sum('activeSeconds'):null,
    distancePerActiveMinute:sum('activeSeconds')?sum('distance')*60/sum('activeSeconds'):null,
    rotationTurnsPerActiveMinute:sum('activeSeconds')?sum('rotationRadians')*60/(2*Math.PI*sum('activeSeconds')):null};
}

export function evaluationReport(model,modelHash,matches,{seeds,teamSize,purpose='tuning',targetWinRate=.8,minMatches=40,environmentSha256=null}={}){
  const overall=summarizeMatches(matches),bySide={};
  for(const team of [0,1])bySide[team===0?'blue':'red']=summarizeMatches(matches.filter(row=>row.team===team));
  const byLayout={};
  for(const layout of [...new Set(matches.map(row=>row.layout))].sort())byLayout[layout]=summarizeMatches(matches.filter(row=>row.layout===layout));
  const paired=seeds.every(seed=>[0,1].every(team=>matches.filter(row=>row.seed===seed&&row.team===team&&row.complete).length===1));
  const targetMet=overall.completed>=minMatches&&overall.completed===matches.length&&paired&&overall.winRate>=targetWinRate;
  return {model:model.name,modelSha256:modelHash,environmentSha256,createdAt:new Date().toISOString(),purpose,seeds,teamSize,
    rules:{scoreLimit:25,timeLimit:180,opponent:'unchanged-scripted',learnedRoles:'entire team including player slot',
      weaponSpread:'identical for all bot roles',pairedSides:true,drawsCountAsWins:false},
    criterion:{targetWinRate,minMatches,pairedSidesRequired:true,targetMet},
    note:purpose==='final'?'Unseen lockbox result; no optimization or checkpoint selection should use these matches.':
      'Tuning result. These maps may guide checkpoint selection and must not be reused as the final lockbox.',
    provenance:model.training,overall,bySide,byLayout,matches};
}

export function assertEvaluationSeeds(seeds,training,purpose){
  if(!['tuning','final'].includes(purpose))throw new TypeError('Purpose must be tuning or final.');
  if(purpose==='final')assertHeldOut(seeds,training);
  else assertHeldOut(seeds,{...training,validationSeeds:[],selectionSeeds:[],validationSeedRanges:[],selectionSeedRanges:[]});
}

async function main(){
  const args=process.argv.slice(2),value=(name,fallback)=>{const i=args.indexOf(name);return i<0?fallback:args[i+1];};
  if(args.includes('--help')){
    console.log('node training/evaluate-teams.mjs --model models/pixel-baseline.json [--seeds 4000:4005] [--purpose tuning|final] [--team-size 4] [--output training/runs/teams.json]');return;
  }
  const modelPath=value('--model');if(!modelPath)throw new TypeError('--model is required.');
  const bytes=await readFile(modelPath),model=validateModel(JSON.parse(bytes.toString('utf8')));
  const modelHash=createHash('sha256').update(bytes).digest('hex');
  const environmentHash=createHash('sha256');
  for(const filename of ['simulation.js','renderer.js','mapgen.js','policy.js']){
    environmentHash.update(filename);environmentHash.update(await readFile(new URL(`../dist/${filename}`,import.meta.url)));
  }
  const seeds=parseSeeds(value('--seeds','4000:4005')),purpose=value('--purpose','tuning');
  assertEvaluationSeeds(seeds,model.training,purpose);
  const teamSize=integer(Number(value('--team-size','4')),'team size',1,8),matches=[];
  for(const seed of seeds)for(const team of [0,1]){
    const match=runTeamMatch(model,{seed,team,teamSize});matches.push(match);
    console.log(JSON.stringify({type:'match',seed,team,layout:match.layout,seconds:match.seconds,
      win:match.win,draw:match.draw,scores:match.scores,kills:match.kills,deaths:match.deaths,
      accuracy:match.shots?match.hits/match.shots:0}));
  }
  const report=evaluationReport(model,modelHash,matches,{seeds,teamSize,purpose,environmentSha256:environmentHash.digest('hex')});
  const output=value('--output');if(output){await mkdir(path.dirname(output),{recursive:true});await writeFile(output,JSON.stringify(report,null,2)+'\n');}
  console.log(JSON.stringify({type:'evaluation',...report,matches:undefined,provenance:undefined}));
}
if(process.argv[1]&&import.meta.url===pathToFileURL(process.argv[1]).href)main().catch(error=>{console.error(error);process.exitCode=1;});
