/** Small derivative-free policy search, scored only on declared tuning maps. */
import {Worker,isMainThread,parentPort,workerData} from 'node:worker_threads';
import {readFile,writeFile,mkdir} from 'node:fs/promises';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {validateModel} from '../dist/policy.js';
import {runTeamMatch,summarizeMatches,parseSeeds,assertEvaluationSeeds} from './evaluate-teams.mjs';

function adjusted(model,offsets){
  const copy=structuredClone(model),[forward,neutral,fire]=offsets;
  copy.weights.policyBias.data[2]+=forward;
  copy.weights.policyBias.data[10]+=neutral;
  copy.weights.policyBias.data[19]+=fire;
  return copy;
}
if(!isMainThread){
  try{
    const model=validateModel(JSON.parse(await readFile(workerData.model,'utf8')));
    const candidate=adjusted(model,workerData.offsets),matches=[];
    for(const seed of workerData.seeds)for(const team of [0,1])matches.push(runTeamMatch(candidate,{seed,team}));
    parentPort.postMessage({id:workerData.id,offsets:workerData.offsets,summary:summarizeMatches(matches)});
  }catch(error){throw error;}
}else{
  const args=process.argv.slice(2),value=(name,fallback)=>{const index=args.indexOf(name);return index<0?fallback:args[index+1];};
  const input=value('--model'),output=value('--output','training/runs/bias-v2.json');
  if(!input)throw new Error('--model is required.');
  const bytes=await readFile(input),model=validateModel(JSON.parse(bytes));
  if(model.architecture!=='cnn-v2')throw new Error('Bias search requires cnn-v2.');
  const seeds=parseSeeds(value('--seeds','4000:4005'));assertEvaluationSeeds(seeds,model.training,'tuning');
  const concurrency=Number(value('--workers','4'));
  if(!Number.isInteger(concurrency)||concurrency<1||concurrency>8)throw new Error('--workers must be 1–8.');
  const candidates=[[0,0,0],[.5,0,0],[1,0,0],[2,0,0],[1,.5,0],[1,-.5,0],[2,-.5,0],[.5,0,.5],[1,0,.5],[2,0,.5]];
  const results=[];let cursor=0;
  async function work(){
    while(cursor<candidates.length){
      const id=cursor++,offsets=candidates[id];
      const result=await new Promise((resolve,reject)=>{
        const worker=new Worker(new URL(import.meta.url),{workerData:{id,offsets,seeds,model:path.resolve(input)}});
        worker.once('message',resolve);worker.once('error',reject);
        worker.once('exit',code=>{if(code!==0)reject(new Error(`Worker exited ${code}`));});
      });
      results.push(result);console.log(JSON.stringify(result));
    }
  }
  await Promise.all(Array.from({length:concurrency},work));
  const ranked=[...results].sort((a,b)=>b.summary.wins-a.summary.wins||
    (b.summary.modelScore-b.summary.scriptedScore)-(a.summary.modelScore-a.summary.scriptedScore));
  const best=ranked[0],winner=adjusted(model,best.offsets);
  winner.name='Pixel policy · bias tuned';winner.createdAt=new Date().toISOString();
  winner.training={...winner.training,selectionSeeds:[...new Set([...(model.training.selectionSeeds||[]),...seeds])],
    policyBiasSearch:{method:'grid search on full team wins, then score margin',
      parentSha256:createHash('sha256').update(bytes).digest('hex'),offsets:best.offsets,
      candidateCount:candidates.length,tuningWins:best.summary.wins,tuningMatches:best.summary.matches}};
  validateModel(winner);await mkdir(path.dirname(output),{recursive:true});
  await writeFile(output,JSON.stringify(winner));
  await writeFile(output.replace(/\.json$/,'.search.json'),JSON.stringify({seeds,best,results},null,2));
  console.log(JSON.stringify({output,best}));
}
