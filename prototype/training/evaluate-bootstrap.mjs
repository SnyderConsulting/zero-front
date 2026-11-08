/** Teacher/student tuning diagnostics: full teams against Scripted and identical-policy self matches. */
import {readFile,writeFile,mkdir,rename,lstat} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {Worker,isMainThread,parentPort,workerData} from 'node:worker_threads';
import {pathToFileURL} from 'node:url';
import path from 'node:path';
import {createSimulation,stepAgents} from '../dist/simulation.js';
import {createPolicyController,validateModel,MODEL_MAX_BYTES} from '../dist/policy.js';
import {configureBenchmarkTeams,policyHash,summarizeBenchmarkMatches} from './benchmark-models.mjs';
import {parseSeeds,assertEvaluationSeeds} from './evaluate-teams.mjs';

const integer=(value,name,min,max)=>{
  if(!Number.isInteger(value)||value<min||value>max)throw new TypeError(`${name} must be ${min}–${max}.`);
  return value;
};
const hash=value=>createHash('sha256').update(value).digest('hex');
const canonical=value=>process.platform==='win32'?path.resolve(value).toLowerCase():path.resolve(value);

/** Motion is measured from displacement, excluding death/respawn intervals. It is never a policy input. */
export function runBootstrapMatch(model,{seed,team=0,opponent='scripted',teamSize=4,maxSeconds=180}={}){
  integer(seed,'seed',0,4294967295);integer(team,'team',0,1);
  integer(teamSize,'team size',1,8);integer(maxSeconds,'seconds',1,180);
  if(!['scripted','self'].includes(opponent))throw new TypeError('Opponent must be scripted or self.');
  const sim=createSimulation({seed,teamSize,scoreLimit:25,recordReplay:false});
  configureBenchmarkTeams(sim,team,createPolicyController(model),opponent==='self'?createPolicyController(model):'scripted');
  const actors=sim.actors.filter(actor=>opponent==='self'||actor.team===team);
  const motion=new Map(actors.map(actor=>[actor.id,{distance:0,activeSeconds:0,stationarySeconds:0}]));
  let firstHitSeconds=null,firstKillSeconds=null;
  while(!sim.terminated&&!sim.truncated&&sim.time<maxSeconds){
    const before=actors.map(actor=>({x:actor.x,y:actor.y,active:actor.active,deaths:actor.deaths})),start=sim.time;
    stepAgents(sim,{},Math.min(6,Math.max(1,Math.round((maxSeconds-sim.time)*60))));
    const duration=sim.time-start;
    for(const [index,actor] of actors.entries()){
      const previous=before[index],entry=motion.get(actor.id);
      if(!previous.active||!actor.active||previous.deaths!==actor.deaths)continue;
      const distance=Math.hypot(actor.x-previous.x,actor.y-previous.y);
      entry.distance+=distance;entry.activeSeconds+=duration;
      if(distance<.02)entry.stationarySeconds+=duration;
    }
    if(firstHitSeconds===null&&actors.some(actor=>actor.hits>0))firstHitSeconds=sim.time;
    if(firstKillSeconds===null&&actors.some(actor=>actor.kills>0))firstKillSeconds=sim.time;
  }
  const players=actors.map(actor=>({id:actor.id,team:actor.team,kills:actor.kills,deaths:actor.deaths,
    shots:actor.shots,hits:actor.hits,damageDealt:actor.damageDealt,...motion.get(actor.id)}));
  const sum=key=>players.reduce((total,actor)=>total+actor[key],0),complete=maxSeconds===180&&(sim.terminated||sim.truncated);
  return {seed,layout:sim.layout,opponent,team:opponent==='self'?null:team,teamSize,seconds:sim.time,
    complete,partial:!complete,winner:complete?sim.winner:null,
    win:complete&&opponent==='scripted'?sim.winner===team:null,draw:complete?sim.winner==='draw':null,
    scores:[...sim.scores],modelScore:opponent==='self'?sim.scores[0]+sim.scores[1]:sim.scores[team],
    opponentScore:opponent==='self'?null:sim.scores[1-team],
    kills:sum('kills'),deaths:sum('deaths'),shots:sum('shots'),hits:sum('hits'),damageDealt:sum('damageDealt'),
    distance:sum('distance'),activeSeconds:sum('activeSeconds'),stationarySeconds:sum('stationarySeconds'),
    firstHitSeconds,firstKillSeconds,players};
}

export function summarizeBootstrapMatches(matches,{self=false}={}){
  const summary=summarizeBenchmarkMatches(matches),sum=key=>matches.reduce((total,row)=>total+(row[key]||0),0);
  const active=sum('activeSeconds'),stationary=sum('stationarySeconds');
  const meanFirst=key=>{
    const values=matches.map(row=>row[key]).filter(value=>typeof value==='number');
    return values.length?values.reduce((total,value)=>total+value,0)/values.length:null;
  };
  return {...summary,...(self?{wins:null,losses:null,winRate:null,opponentScore:null}:{}),
    contactMatches:matches.filter(row=>row.hits>0).length,killMatches:matches.filter(row=>row.kills>0).length,
    zeroContactMatches:matches.filter(row=>row.hits===0).length,
    stationaryFraction:active?stationary/active:null,movingFraction:active?1-stationary/active:null,
    distancePerActiveMinute:active?sum('distance')*60/active:null,
    meanFirstHitSecondsAmongContactMatches:meanFirst('firstHitSeconds'),
    meanFirstKillSecondsAmongKillMatches:meanFirst('firstKillSeconds')};
}

export function bootstrapJobs(seeds,{teamSize=4,maxSeconds=180}={}){
  return seeds.flatMap(seed=>[
    ...[0,1].map(team=>({seed,team,opponent:'scripted',teamSize,maxSeconds})),
    // A second side assignment would reproduce the same self match and double-count it.
    {seed,team:0,opponent:'self',teamSize,maxSeconds}
  ]);
}

/** Earlier imitation validation/selection maps remain unavailable to a final test. */
export function assertBootstrapSeeds(seeds,training={},purpose='tuning'){
  if(!training||typeof training!=='object'||Array.isArray(training))throw new TypeError('Invalid training provenance.');
  const sources=[training];
  if(training.initialTraining!==undefined&&training.initialTraining!==null){
    if(typeof training.initialTraining!=='object'||Array.isArray(training.initialTraining))throw new TypeError('Invalid initial training provenance.');
    sources.push(training.initialTraining);
  }
  const ancestry={};
  for(const [target,keys] of Object.entries({trainSeeds:['trainSeeds','priorTrainingSeeds'],
    trainingSeedRanges:['trainingSeedRanges','priorTrainingSeedRanges'],
    validationSeeds:['validationSeeds','priorValidationSeeds'],
    validationSeedRanges:['validationSeedRanges','priorValidationSeedRanges'],
    selectionSeeds:['selectionSeeds','priorSelectionSeeds'],
    selectionSeedRanges:['selectionSeedRanges','priorSelectionSeedRanges']})){
    ancestry[target]=sources.flatMap(source=>keys.flatMap(key=>{
      const values=source[key]??[];
      if(!Array.isArray(values))throw new TypeError(`Invalid ${key} provenance.`);
      return values;
    }));
  }
  assertEvaluationSeeds(seeds,ancestry,purpose);
}

async function runJobs(model,jobs,workers,onProgress){
  if(workers===1)return jobs.map(job=>{const result=runBootstrapMatch(model,job);onProgress(result);return result;});
  const pool=[],results=new Array(jobs.length);let cursor=0,completed=0;
  try{
    await new Promise((resolve,reject)=>{
      for(let index=0;index<Math.min(workers,jobs.length);index++){
        const worker=new Worker(new URL(import.meta.url),{workerData:{arenaBootstrap:true,model}});pool.push(worker);
        const send=()=>{if(cursor<jobs.length){const index=cursor++;worker.postMessage({index,options:jobs[index]});}};
        worker.on('message',message=>{
          if(message.error){reject(new Error(message.error));return;}
          results[message.index]=message.result;completed++;onProgress(message.result);
          if(completed===jobs.length)resolve();else send();
        });
        worker.on('error',reject);
        worker.on('exit',code=>{if(completed<jobs.length)reject(new Error(`Bootstrap evaluator exited early (${code}).`));});
        send();
      }
    });
  }finally{await Promise.allSettled(pool.map(worker=>worker.terminate()));}
  return results;
}

export async function evaluateBootstrap({modelPath,seeds=parseSeeds('610000:610005'),purpose='tuning',teamSize=4,maxSeconds=180,workers=4,onProgress=()=>{}}){
  integer(teamSize,'team size',1,8);integer(maxSeconds,'seconds',1,180);integer(workers,'workers',1,16);
  if(!['tuning','final'].includes(purpose))throw new TypeError('Purpose must be tuning or final.');
  if(purpose==='final'&&maxSeconds!==180)throw new TypeError('Final evaluation requires complete 180-second-limit matches.');
  if(!Array.isArray(seeds)||!seeds.length||seeds.length>100||new Set(seeds).size!==seeds.length)throw new TypeError('Provide 1–100 unique seeds.');
  for(const seed of seeds)integer(seed,'seed',0,4294967295);
  const stat=await lstat(modelPath);if(!stat.isFile()||stat.size>MODEL_MAX_BYTES)throw new TypeError('Model must be a regular JSON file at most 8 MiB.');
  const bytes=await readFile(modelPath);if(bytes.length>MODEL_MAX_BYTES)throw new RangeError('Model exceeds 8 MiB.');
  const model=validateModel(JSON.parse(bytes.toString('utf8')));assertBootstrapSeeds(seeds,model.training,purpose);
  const environment=createHash('sha256');
  for(const name of ['simulation.js','renderer.js','mapgen.js','policy.js']){environment.update(name);environment.update(await readFile(new URL(`../dist/${name}`,import.meta.url)));}
  const createdAt=new Date().toISOString(),matches=await runJobs(model,bootstrapJobs(seeds,{teamSize,maxSeconds}),workers,onProgress);
  const groups=Object.fromEntries(['scripted','self'].map(opponent=>{
    const rows=matches.filter(row=>row.opponent===opponent),options={self:opponent==='self'};
    return [opponent,{overall:summarizeBootstrapMatches(rows,options),
      byLayout:Object.fromEntries([...new Set(rows.map(row=>row.layout))].sort().map(layout=>[layout,summarizeBootstrapMatches(rows.filter(row=>row.layout===layout),options)])),
      matches:rows}];
  }));
  return {format:'arena-bootstrap-evaluation',version:1,createdAt,finishedAt:new Date().toISOString(),
    purpose:maxSeconds===180?purpose:'integration-smoke',checkpointSelectionAllowed:purpose==='tuning',
    candidate:{name:model.name,sha256:hash(bytes),policySha256:policyHash(model),provenance:model.training},
    environmentSha256:environment.digest('hex'),seeds,teamSize,workers,
    rules:{scoreLimit:25,timeLimit:180,requestedSeconds:maxSeconds,scriptedPairedSides:true,selfMatchesPerSeed:1,
      roles:'entire teams including human slot',weaponSpread:'identical for every bot role',pixelActions:'deployed argmax; six-tick cadence'},
    note:(purpose==='final'?'Held-out final evaluation of this frozen candidate. Do not use these results for further optimization or checkpoint selection. ':'Development diagnostics, not an untouched final test. Record these seeds as selection maps if results guide candidate selection. ')+
      'Self-match totals include both teams; self win rate is undefined. Motion alone does not establish useful navigation.',
    ...groups};
}

async function main(){
  const args=process.argv.slice(2),values=new Map(),allowed=new Set(['--model','--seeds','--purpose','--team-size','--seconds','--workers','--output']);
  if(args.includes('--help')){console.log('node training/evaluate-bootstrap.mjs --model <candidate.json> [--seeds 610000:610005] [--purpose tuning|final] [--workers 4] [--output report.json]');return;}
  for(let index=0;index<args.length;index+=2){const flag=args[index],value=args[index+1];if(!allowed.has(flag)||!value||value.startsWith('--'))throw new TypeError(`Unknown option or missing value: ${flag}`);values.set(flag,value);}
  const modelPath=values.get('--model');if(!modelPath)throw new TypeError('--model is required.');
  const output=values.get('--output');
  if(output&&(canonical(output)===canonical(modelPath)||canonical(path.dirname(output))===canonical('models')))throw new TypeError('Report must not overwrite a model or be saved in the game model library.');
  const report=await evaluateBootstrap({modelPath,seeds:parseSeeds(values.get('--seeds')??'610000:610005'),purpose:values.get('--purpose')??'tuning',teamSize:Number(values.get('--team-size')??4),
    maxSeconds:Number(values.get('--seconds')??180),workers:Number(values.get('--workers')??4),
    onProgress:row=>console.log(JSON.stringify({type:'bootstrap-match',seed:row.seed,opponent:row.opponent,team:row.team,
      complete:row.complete,scores:row.scores,hits:row.hits,kills:row.kills}))});
  if(output){await mkdir(path.dirname(output),{recursive:true});const temporary=`${output}.${process.pid}.partial`;await writeFile(temporary,JSON.stringify(report,null,2)+'\n');await rename(temporary,output);}
  console.log(JSON.stringify({type:'bootstrap-evaluation',purpose:report.purpose,candidate:report.candidate.name,
    sha256:report.candidate.sha256,output:output??null,scripted:report.scripted.overall,self:report.self.overall}));
}
if(!isMainThread&&workerData?.arenaBootstrap){
  parentPort.on('message',job=>{try{parentPort.postMessage({index:job.index,result:runBootstrapMatch(workerData.model,job.options)});}catch(error){parentPort.postMessage({index:job.index,error:error.stack||error.message});}});
}else if(isMainThread&&process.argv[1]&&import.meta.url===pathToFileURL(process.argv[1]).href)main().catch(error=>{console.error(error);process.exitCode=1;});
