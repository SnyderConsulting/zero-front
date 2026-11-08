/** Privileged demonstration teacher. This module is never used by game policies. */
import {createSimulation,stepAgents,setController,wallAt,scriptedAction} from '../dist/simulation.js';
import {policyObservation,createPolicyController,validateModel,ACTION_HEADS_V2} from '../dist/policy.js';
import {readFile} from 'node:fs/promises';
import {pathToFileURL} from 'node:url';
import readline from 'node:readline';

export const EXPERT_HEADS=ACTION_HEADS_V2;
const wrap=value=>((value+Math.PI)%(Math.PI*2)+Math.PI*2)%(Math.PI*2)-Math.PI;
const clamp=(value,min,max)=>Math.max(min,Math.min(max,value));
export function labels(action){return EXPERT_HEADS.map(({name,values})=>{
  if(name==='forward'||name==='strafe'){const value=Number(action[name]||0);return value<-.2?0:value>.2?2:1;}
  let best=0;for(let i=1;i<values.length;i++)if(Math.abs(Number(values[i])-Number(action[name]||0))<Math.abs(Number(values[best])-Number(action[name]||0)))best=i;
  return best;
});}
export function quantize(action){return Object.fromEntries(EXPERT_HEADS.map((head,index)=>[head.name,head.values[labels(action)[index]]]));}

/** Detect visible enemy components from the image alone, without world coordinates. */
export function pixelTargets(pixels){
  const visited=new Uint8Array(64*36),targets=[];
  const enemy=index=>pixels[index*3]===240&&pixels[index*3+1]===88&&pixels[index*3+2]===70;
  for(let index=0;index<64*36;index++){
    if(visited[index]||!enemy(index))continue;
    let count=0,sumX=0,sumY=0,minX=64,maxX=0,minY=36,maxY=0;
    const queue=[index];visited[index]=1;
    for(let cursor=0;cursor<queue.length;cursor++){
      const next=queue[cursor],x=next%64,y=Math.floor(next/64);
      count++;sumX+=x;sumY+=y;minX=Math.min(minX,x);maxX=Math.max(maxX,x);minY=Math.min(minY,y);maxY=Math.max(maxY,y);
      for(const [nx,ny]of[[x-1,y],[x+1,y],[x,y-1],[x,y+1]]){
        const item=ny*64+nx;
        if(nx>=0&&nx<64&&ny>=0&&ny<36&&!visited[item]&&enemy(item)){visited[item]=1;queue.push(item);}
      }
    }
    if(count>=2)targets.push({count,x:sumX/count,y:sumY/count,minX,maxX,minY,maxY});
  }
  return targets.sort((a,b)=>(b.count/(1+Math.abs(b.x-31.5)*.04))-(a.count/(1+Math.abs(a.x-31.5)*.04)));
}

function pathTo(sim,actor,target){
  const start=Math.floor(actor.y)*sim.width+Math.floor(actor.x),end=Math.floor(target.y)*sim.width+Math.floor(target.x);
  const previous=new Map([[start,-1]]),queue=[start];
  for(let i=0;i<queue.length&&!previous.has(end);i++){
    const id=queue[i],x=id%sim.width,y=Math.floor(id/sim.width);
    for(const[nx,ny]of[[x+1,y],[x-1,y],[x,y+1],[x,y-1]]){
      const next=ny*sim.width+nx;if(wallAt(sim,nx,ny)!=='.'||previous.has(next))continue;
      previous.set(next,id);queue.push(next);
    }
  }
  if(!previous.has(end))return[];
  const path=[];for(let id=end;id!==start;id=previous.get(id))path.push({x:id%sim.width+.5,y:Math.floor(id/sim.width)+.5});
  return path.reverse();
}

/** Returns labels from precise visible aim, with privileged navigation only as a teacher. */
export function createExpert({rush=true,strafe=true,reactiveNav=false,camp=false,scriptedTeacher=false,scanRate=.3}={}){
  if(!Number.isFinite(scanRate)||scanRate<0||scanRate>3.6)throw new TypeError('scanRate must be between 0 and 3.6 radians per second.');
  if([reactiveNav,camp,scriptedTeacher].filter(Boolean).length>1)throw new TypeError('Choose one teacher behavior.');
  if(scriptedTeacher)return(sim,actor)=>quantize(scriptedAction(sim,actor.id));
  const states=new WeakMap();
  return (sim,actor,observation=null)=>{
    let state=states.get(actor);
    if(!state||state.deaths!==actor.deaths){state={path:[],pathTick:-Infinity,deaths:actor.deaths};states.set(actor,state);}
    const input=observation||policyObservation(sim,actor.id),target=pixelTargets(input.pixels)[0];
    if(target){
      const error=Math.atan((target.x-31.5)/32);
      const extent=target.maxX-target.minX+1;
      const close=target.maxY-target.minY>19;
      const action={forward:rush&&!close?1:0,strafe:strafe?(error<-.07?-1:1):0,
        turn:clamp(error*10,-3.6,3.6),pitch:clamp(-actor.pitch*10,-1.2,1.2),
        fire:Math.abs(error)<Math.max(.1,extent/64),reload:actor.ammo===0};
      return quantize(action);
    }
    if(camp)return quantize({forward:0,strafe:0,turn:actor.health<100?3.6:scanRate,pitch:-actor.pitch*10,fire:false,reload:actor.ammo<20});
    if(reactiveNav){
      const floorScores=[];
      for(let x=0;x<64;x++){
        let floor=0;
        for(let y=24;y<36;y++){
          const index=(y*64+x)*3;
          if(input.pixels[index]===60&&input.pixels[index+1]===60&&input.pixels[index+2]===70)floor++;
        }
        floorScores.push(floor);
      }
      const average=(start,end)=>floorScores.slice(start,end).reduce((a,b)=>a+b,0)/(end-start);
      const front=average(25,39),left=average(2,19),right=average(45,62);
      const turn=front>=7?0:left>right+.5?-3.6:3.6;
      return quantize({forward:front>=4?1:0,strafe:0,turn:actor.health<85&&front>=7?3.6:turn,
        pitch:clamp(-actor.pitch*10,-1.2,1.2),fire:false,reload:actor.ammo<12});
    }
    const enemies=sim.actors.filter(other=>other.active&&other.team!==actor.team).sort((a,b)=>Math.hypot(a.x-actor.x,a.y-actor.y)-Math.hypot(b.x-actor.x,b.y-actor.y));
    const closest=enemies[0];
    if(!closest)return {forward:0,strafe:0,turn:.3,pitch:0,fire:false,reload:actor.ammo<20};
    if(sim.tick-state.pathTick>=30||!state.path.length){state.path=pathTo(sim,actor,closest);state.pathTick=sim.tick;}
    while(state.path.length&&Math.hypot(state.path[0].x-actor.x,state.path[0].y-actor.y)<.3)state.path.shift();
    const point=state.path[0]||closest,error=wrap(Math.atan2(point.y-actor.y,point.x-actor.x)-actor.yaw);
    return quantize({forward:Math.abs(error)<.45?1:0,strafe:0,turn:clamp(error*10,-3.6,3.6),pitch:clamp(-actor.pitch*10,-1.2,1.2),fire:false,reload:actor.ammo<12});
  };
}
export function expertController(options){
  const expert=createExpert(options),states=new WeakMap();
  return(sim,actor)=>{const state=states.get(actor);if(state&&sim.tick-state.tick<6&&state.deaths===actor.deaths)return state.action;
    const action=expert(sim,actor);states.set(actor,{tick:sim.tick,deaths:actor.deaths,action});return action;};
}

export async function collectSamples(command,emit){
  const seeds=command.seeds??[1000],seconds=command.seconds??90,maxSamples=command.maxSamples??200000;
  const teamSize=command.teamSize??4,actorsPerFrame=command.actorsPerFrame??teamSize;
  if(!Array.isArray(seeds)||!seeds.length||seeds.some(seed=>!Number.isInteger(seed)||seed<0||seed>4294967295)||seeds.length>128)throw new Error('Invalid seeds.');
  if(!Number.isInteger(seconds)||seconds<1||seconds>180||!Number.isInteger(maxSamples)||maxSamples<seeds.length||maxSamples>1000000)throw new Error('Invalid collection limits.');
  if(!Number.isInteger(teamSize)||teamSize<2||teamSize>8||!Number.isInteger(actorsPerFrame)||actorsPerFrame<1||actorsPerFrame>teamSize)throw new Error('Invalid collection team/sample size.');
  // Diagnostic scripted labels cannot update reaction/path timers on student-controlled bots.
  if(command.scriptedTeacher&&command.model)throw new Error('Built-in scripted teacher is demonstration-only; use the stateful expert for learner corrections.');
  const learner=command.model?createPolicyController(validateModel(JSON.parse(await readFile(command.model,'utf8')))):null;
  let parts=[],scalars=[],targets=[],count=0,total=0;
  const flush=()=>{if(count){emit({type:'samples',count,pixels:Buffer.concat(parts).toString('base64'),scalars,labels:targets});parts=[];scalars=[];targets=[];count=0;}};
  const quota=Math.floor(maxSamples/seeds.length);
  for(const seed of seeds){
    const sim=createSimulation({seed,teamSize,scoreLimit:100}),teacher=createExpert(command);
    for(const actor of sim.actors)setController(sim,actor.id,'scripted');
    const trainedTeam=seed%2,actors=sim.actors.filter(actor=>actor.team===trainedTeam);
    const controller=learner||(command.scriptedTeacher?'scripted':expertController(command));
    for(const actor of actors)setController(sim,actor.id,controller);
    let local=0,decision=0;
    while(!sim.terminated&&!sim.truncated&&sim.time<seconds&&local<quota){
      for(let slot=0;slot<actorsPerFrame;slot++){
        const actor=actors[(decision*actorsPerFrame+slot)%actors.length];
        if(!actor.active||local>=quota)continue;
        const input=policyObservation(sim,actor.id),action=teacher(sim,actor,input);
        parts.push(Buffer.from(input.pixels));scalars.push(...input.scalars);targets.push(...labels(action));count++;local++;total++;
        if(count===128)flush();
      }
      stepAgents(sim,{},6);decision++;
    }
    flush();emit({type:'progress',seed,samples:local,total});
  }
  emit({type:'complete',samples:total,teamSize,actorsPerFrame,behavior:learner?'learner':'teacher'});
}
async function evaluate(command,emit){
  let wins=0,kills=0,deaths=0;
  for(const seed of command.seeds||[4000,4001,4002,4003,4004])for(const team of [0,1]){
    const sim=createSimulation({seed,teamSize:4,scoreLimit:25}),controller=command.scriptedTeacher?'scripted':expertController(command);
    for(const actor of sim.actors)setController(sim,actor.id,actor.team===team?controller:'scripted');
    while(!sim.terminated&&!sim.truncated)stepAgents(sim,{},600);
    wins+=Number(sim.winner===team);kills+=sim.scores[team];deaths+=sim.scores[1-team];
    emit({type:'match',seed,team,winner:sim.winner,scores:sim.scores,time:sim.time});
  }
  emit({type:'complete',wins,kills,deaths});
}
async function main(){
  const lines=readline.createInterface({input:process.stdin,crlfDelay:Infinity});
  const emit=value=>process.stdout.write(JSON.stringify(value)+'\n');
  for await(const line of lines){if(!line.trim())continue;try{const command=JSON.parse(line);
    if(command.command==='collect')await collectSamples(command,emit);else if(command.command==='evaluate')await evaluate(command,emit);else if(command.command==='close')break;else throw new Error('Unknown command.');
  }catch(error){emit({type:'error',error:error.stack});}}
}
if(process.argv[1]&&import.meta.url===pathToFileURL(process.argv[1]).href)main();
