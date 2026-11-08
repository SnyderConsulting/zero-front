/** Line-delimited local data collection bridge. stdout is protocol only. */
import readline from 'node:readline';
import {pathToFileURL} from 'node:url';
import {createSimulation,stepAgents,setController,scriptedAction} from '../dist/simulation.js';
import {policyObservation,ACTION_HEADS} from '../dist/policy.js';
export {ACTION_HEADS} from '../dist/policy.js';
const integer=(value,name,min,max)=>{
  if(!Number.isInteger(value)||value<min||value>max)throw new TypeError(`${name} must be an integer from ${min} to ${max}.`);
  return value;
};
export function actionLabels(action){
  return ACTION_HEADS.map(({name,values})=>{
    const value=action[name]??0;
    if(typeof values[0]==='boolean')return value?1:0;
    // Preserve deliberate backing away: scripted forward can be -0.35.
    if(name==='forward'||name==='strafe')return value<-.2?0:value>.2?2:1;
    let best=0;
    for(let i=1;i<values.length;i++)if(Math.abs(values[i]-value)<Math.abs(values[best]-value))best=i;
    return best;
  });
}

/** Collect the observation before applying its teacher label. No hidden state is emitted. */
export function collectSamples(options,emit){
  const seeds=options.seeds;
  if(!Array.isArray(seeds)||!seeds.length||seeds.length>1000)throw new TypeError('seeds must contain 1–1000 seeds.');
  seeds.forEach(seed=>integer(seed,'seed',0,4294967295));
  const teamSize=integer(options.teamSize??4,'teamSize',2,8);
  const seconds=integer(options.seconds??45,'seconds',1,180);
  const actorsPerFrame=integer(options.actorsPerFrame??4,'actorsPerFrame',1,teamSize*2-1);
  const maxSamples=integer(options.maxSamples??100000,'maxSamples',seeds.length,1000000);
  const quota=Math.floor(maxSamples/seeds.length),batchSize=128;
  let pixels=[],scalars=[],labels=[],batchCount=0,total=0;
  const flush=()=>{
    if(!batchCount)return;
    emit({type:'samples',count:batchCount,pixels:Buffer.concat(pixels).toString('base64'),scalars,labels});
    pixels=[];scalars=[];labels=[];batchCount=0;
  };
  const counts=[];
  for(const seed of seeds){
    const sim=createSimulation({seed,teamSize,scoreLimit:100,recordReplay:false});
    setController(sim,'player','scripted');
    const actors=sim.actors.filter(actor=>actor.id!=='player');
    let collected=0,decision=0;
    while(!sim.terminated&&!sim.truncated&&sim.time<seconds&&collected<quota){
      for(let slot=0;slot<actorsPerFrame&&collected<quota;slot++){
        const actor=actors[(decision*actorsPerFrame+slot)%actors.length];
        if(!actor.active)continue;
        const input=policyObservation(sim,actor.id);
        const target=actionLabels(scriptedAction(sim,actor.id));
        pixels.push(Buffer.from(input.pixels));scalars.push(...input.scalars);labels.push(...target);
        collected++;total++;batchCount++;
        if(batchCount>=batchSize)flush();
      }
      stepAgents(sim,{},6);decision++;
    }
    flush();counts.push({seed,samples:collected,seconds:sim.time});
    emit({type:'progress',seed,samples:collected,total});
  }
  emit({type:'complete',samples:total,seeds:counts,width:64,height:36,actionRepeat:6});
  return {samples:total,seeds:counts};
}

async function main(){
  const lines=readline.createInterface({input:process.stdin,crlfDelay:Infinity});
  const emit=value=>process.stdout.write(JSON.stringify(value)+'\n');
  for await(const line of lines){
    if(!line.trim())continue;
    try{
      const command=JSON.parse(line);
      if(command.command==='collect')collectSamples(command,emit);
      else if(command.command==='close')break;
      else throw new TypeError('Unknown bridge command.');
    }catch(error){emit({type:'error',error:error.message});}
  }
}
if(process.argv[1]&&import.meta.url===pathToFileURL(process.argv[1]).href)main().catch(error=>{console.error(error);process.exitCode=1;});
