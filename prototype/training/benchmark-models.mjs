/** Read-only, paired full-team benchmarks. Results never enter a training update or select a checkpoint. */
import {readFile,writeFile,readdir,lstat,mkdir,rename} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {pathToFileURL} from 'node:url';
import path from 'node:path';
import {Worker,isMainThread,parentPort,workerData} from 'node:worker_threads';
import {createSimulation,setController,stepAgents} from '../dist/simulation.js';
import {createPolicyController,validateModel,MODEL_MAX_BYTES} from '../dist/policy.js';
import {parseSeeds} from './evaluate-teams.mjs';
import {assertHeldOut} from './provenance.mjs';

const hash=bytes=>createHash('sha256').update(bytes).digest('hex');
const canonicalPath=filename=>process.platform==='win32'?path.resolve(filename).toLowerCase():path.resolve(filename);
const integer=(value,name,min,max)=>{
  if(!Number.isInteger(value)||value<min||value>max)throw new TypeError(`${name} must be ${min}–${max}.`);
  return value;
};

/** Excludes metadata so renamed or reformatted copies of the candidate are still excluded. */
export function policyHash(model){
  return hash(JSON.stringify({architecture:model.architecture,observation:model.observation,actionSpace:model.actionSpace,
    actionRepeat:model.actionRepeat,weights:model.weights}));
}

async function loadModel(filename){
  const stat=await lstat(filename);
  if(!stat.isFile())throw new TypeError('Model must be a regular file.');
  if(stat.size>MODEL_MAX_BYTES)throw new RangeError('Model exceeds 8 MiB.');
  const bytes=await readFile(filename);
  if(bytes.length>MODEL_MAX_BYTES)throw new RangeError('Model exceeds 8 MiB.');
  const model=validateModel(JSON.parse(bytes.toString('utf8')));
  return {id:path.basename(filename),name:model.name,path:path.resolve(filename),sha256:hash(bytes),policySha256:policyHash(model),model};
}

/** All model bytes are loaded once before matches; subsequent training exports cannot change this benchmark. */
export async function loadBenchmarkInputs(modelPath,{modelsDirectory='models',seeds=[600000],excludeModels=[]}={}){
  const candidate=await loadModel(modelPath);
  assertHeldOut(seeds,candidate.model.training);
  const exclusions=new Set([modelPath,...excludeModels].map(canonicalPath));
  const opponents=[{id:'scripted',name:'Scripted',kind:'scripted',model:null}],errors=[],skipped=[];
  let files;
  try{files=await readdir(modelsDirectory,{withFileTypes:true});}
  catch(error){errors.push({file:path.resolve(modelsDirectory),message:error.message});files=[];}
  for(const file of files.filter(entry=>entry.name.toLowerCase().endsWith('.json')).sort((a,b)=>a.name.localeCompare(b.name))){
    const filename=path.join(modelsDirectory,file.name);
    if(exclusions.has(canonicalPath(filename))){skipped.push({file:file.name,reason:'Explicitly excluded candidate or live model path.'});continue;}
    try{
      const opponent=await loadModel(filename);
      if(opponent.policySha256===candidate.policySha256){skipped.push({file:file.name,reason:'Identical candidate policy weights.'});continue;}
      assertHeldOut(seeds,opponent.model.training);
      opponents.push({...opponent,kind:'pixel'});
    }catch(error){errors.push({file:file.name,message:error.message});}
  }
  return {candidate,opponents,errors,skipped};
}

/** Explicit assignment includes the human slot and gives every actor the same bot weapon spread. */
export function configureBenchmarkTeams(sim,team,candidateController,opponentController){
  integer(team,'candidate team',0,1);
  if(typeof candidateController!=='function'||!(opponentController==='scripted'||typeof opponentController==='function'))
    throw new TypeError('Both team controllers must be explicit.');
  for(const actor of sim.actors)setController(sim,actor.id,actor.team===team?candidateController:opponentController);
}

export function runBenchmarkMatch(candidate,opponent,{seed,team,teamSize=4,maxSeconds=180}={}){
  integer(seed,'seed',0,4294967295);integer(team,'candidate team',0,1);
  integer(teamSize,'team size',1,8);integer(maxSeconds,'time limit',1,180);
  const sim=createSimulation({seed,teamSize,scoreLimit:25,recordReplay:false});
  configureBenchmarkTeams(sim,team,createPolicyController(candidate),opponent?createPolicyController(opponent):'scripted');
  while(!sim.terminated&&!sim.truncated&&sim.time<maxSeconds)
    stepAgents(sim,{},Math.min(6,Math.max(1,Math.round((maxSeconds-sim.time)*60))));
  const players=sim.actors.filter(actor=>actor.team===team).map(actor=>({id:actor.id,kills:actor.kills,deaths:actor.deaths,
    shots:actor.shots,hits:actor.hits,damageDealt:actor.damageDealt}));
  const sum=key=>players.reduce((total,actor)=>total+actor[key],0);
  // Short runs are integration smoke checks, even if a round happened to end early.
  const complete=maxSeconds===180&&(sim.terminated||sim.truncated);
  return {seed,layout:sim.layout,team,teamSize,seconds:sim.time,complete,partial:!complete,
    winner:complete?sim.winner:null,win:complete?sim.winner===team:null,draw:complete?sim.winner==='draw':null,
    scores:[...sim.scores],modelScore:sim.scores[team],opponentScore:sim.scores[1-team],
    kills:sum('kills'),deaths:sum('deaths'),shots:sum('shots'),hits:sum('hits'),damageDealt:sum('damageDealt'),players};
}

export function summarizeBenchmarkMatches(matches){
  const completed=matches.filter(row=>row.complete),wins=completed.filter(row=>row.win).length,
    draws=completed.filter(row=>row.draw).length;
  const sum=key=>matches.reduce((total,row)=>total+(row[key]||0),0),shots=sum('shots'),hits=sum('hits');
  return {matches:matches.length,completed:completed.length,partial:matches.length-completed.length,wins,draws,
    losses:completed.length-wins-draws,winRate:completed.length?wins/completed.length:null,
    modelScore:sum('modelScore'),opponentScore:sum('opponentScore'),kills:sum('kills'),deaths:sum('deaths'),
    shots,hits,accuracy:shots?hits/shots:null,damageDealt:sum('damageDealt')};
}

function entryMetadata(entry){
  return {id:entry.id,name:entry.name,kind:entry.kind??'pixel',sha256:entry.sha256??null,
    policySha256:entry.policySha256??null,provenance:entry.model?.training??null};
}

async function runMatches(candidate,opponents,jobs,workers,onProgress){
  if(workers===1)return jobs.map(job=>{
    const result=runBenchmarkMatch(candidate,opponents[job.opponent].model,job.options);
    onProgress(job,result);return result;
  });
  const pool=[],results=new Array(jobs.length);
  let cursor=0,completed=0;
  try{
    await new Promise((resolve,reject)=>{
      for(let i=0;i<Math.min(workers,jobs.length);i++){
        const worker=new Worker(new URL(import.meta.url),{workerData:{arenaBenchmark:true,candidate,
          opponents:opponents.map(opponent=>opponent.model)}});
        pool.push(worker);
        const send=()=>{
          if(cursor<jobs.length){const index=cursor++;worker.postMessage({index,...jobs[index]});}
        };
        worker.on('message',message=>{
          if(message.error){reject(new Error(message.error));return;}
          results[message.index]=message.result;completed++;
          onProgress(jobs[message.index],message.result);
          if(completed===jobs.length)resolve();else send();
        });
        worker.on('error',reject);
        worker.on('exit',code=>{if(completed<jobs.length)reject(new Error(`Benchmark worker exited early (${code}).`));});
        send();
      }
    });
  }finally{await Promise.allSettled(pool.map(worker=>worker.terminate()));}
  return results;
}

export async function runBenchmark({modelPath,modelsDirectory='models',excludeModels=[],seeds=parseSeeds('600000:600005'),
  teamSize=4,maxSeconds=180,workers=4,onProgress=()=>{}}){
  integer(teamSize,'team size',1,8);integer(maxSeconds,'time limit',1,180);
  integer(workers,'workers',1,16);
  const loaded=await loadBenchmarkInputs(modelPath,{modelsDirectory,seeds,excludeModels});
  const environmentHash=createHash('sha256');
  for(const filename of ['simulation.js','renderer.js','mapgen.js','policy.js']){
    environmentHash.update(filename);environmentHash.update(await readFile(new URL(`../dist/${filename}`,import.meta.url)));
  }
  const report={format:'arena-selfplay-benchmark',version:1,createdAt:new Date().toISOString(),finishedAt:null,
    purpose:maxSeconds===180?'periodic-evaluation-only':'integration-smoke',
    trainingFeedback:false,checkpointSelection:false,candidate:entryMetadata(loaded.candidate),
    environmentSha256:environmentHash.digest('hex'),seeds,teamSize,workers,
    rules:{scoreLimit:25,timeLimit:180,requestedSeconds:maxSeconds,pairedSides:true,roles:'entire teams including human slot',
      weaponSpread:'identical for every bot role',pixelActions:'deployed argmax policy; six-tick cadence'},
    note:'Monitoring only. These repeated benchmark maps are not an untouched final test. No benchmark trajectories, rewards, opponent weights, or results are used for training or checkpoint selection.',
    errors:loaded.errors,skipped:loaded.skipped,opponents:[]};
  const jobs=[];
  for(let opponent=0;opponent<loaded.opponents.length;opponent++)for(const seed of seeds)for(const team of [0,1])
    jobs.push({opponent,options:{seed,team,teamSize,maxSeconds}});
  const results=await runMatches(loaded.candidate.model,loaded.opponents,jobs,workers,(job,match)=>
    onProgress({type:'benchmark-match',opponent:loaded.opponents[job.opponent].name,seed:match.seed,team:match.team,
      complete:match.complete,win:match.win,scores:match.scores}));
  for(const [opponentIndex,opponent] of loaded.opponents.entries()){
    const matches=results.filter((row,index)=>jobs[index].opponent===opponentIndex);
    const bySide=Object.fromEntries([0,1].map(team=>[team?'red':'blue',summarizeBenchmarkMatches(matches.filter(row=>row.team===team))]));
    const byLayout=Object.fromEntries([...new Set(matches.map(row=>row.layout))].sort().map(layout=>
      [layout,summarizeBenchmarkMatches(matches.filter(row=>row.layout===layout))]));
    report.opponents.push({...entryMetadata(opponent),overall:summarizeBenchmarkMatches(matches),bySide,byLayout,matches});
  }
  report.finishedAt=new Date().toISOString();
  return report;
}

async function main(){
  const args=process.argv.slice(2),values=new Map(),allowed=new Set(['--model','--models-dir','--exclude-model','--seeds','--team-size','--seconds','--workers','--output']);
  if(args.includes('--help')){console.log('node training/benchmark-models.mjs --model <frozen.json> [--models-dir models] [--exclude-model <live.json>] [--seeds 600000:600005] [--team-size 4] [--seconds 180] [--workers 4] [--output report.json]');return;}
  for(let index=0;index<args.length;index+=2){
    const flag=args[index],value=args[index+1];
    if(!allowed.has(flag)||!value||value.startsWith('--'))throw new TypeError(`Unknown option or missing value: ${flag}`);
    if(flag==='--exclude-model')values.set(flag,[...(values.get(flag)||[]),value]);else values.set(flag,value);
  }
  const modelPath=values.get('--model');if(!modelPath)throw new TypeError('--model is required.');
  const report=await runBenchmark({modelPath,modelsDirectory:values.get('--models-dir')??'models',
    excludeModels:values.get('--exclude-model')??[],seeds:parseSeeds(values.get('--seeds')??'600000:600005'),
    teamSize:Number(values.get('--team-size')??4),maxSeconds:Number(values.get('--seconds')??180),workers:Number(values.get('--workers')??4),
    onProgress:row=>console.log(JSON.stringify(row))});
  const output=values.get('--output');
  if(output){
    const protectedPaths=[modelPath,...(values.get('--exclude-model')??[])].map(canonicalPath);
    if(protectedPaths.includes(canonicalPath(output))||canonicalPath(path.dirname(output))===canonicalPath(values.get('--models-dir')??'models'))
      throw new TypeError('Benchmark reports must not overwrite a model or be written inside the model library.');
    await mkdir(path.dirname(output),{recursive:true});
    const temporary=`${output}.${process.pid}.partial`;
    await writeFile(temporary,JSON.stringify(report,null,2)+'\n');await rename(temporary,output);
  }
  console.log(JSON.stringify({type:'benchmark',purpose:report.purpose,candidate:report.candidate.name,
    candidateSha256:report.candidate.sha256,output:output??null,errors:report.errors,skipped:report.skipped,
    opponents:report.opponents.map(entry=>({name:entry.name,...entry.overall}))}));
}
if(!isMainThread&&workerData?.arenaBenchmark){
  parentPort.on('message',job=>{
    try{parentPort.postMessage({index:job.index,result:runBenchmarkMatch(workerData.candidate,workerData.opponents[job.opponent],job.options)});}
    catch(error){parentPort.postMessage({index:job.index,error:error.stack||error.message});}
  });
}else if(isMainThread&&process.argv[1]&&import.meta.url===pathToFileURL(process.argv[1]).href)
  main().catch(error=>{console.error(error);process.exitCode=1;});
