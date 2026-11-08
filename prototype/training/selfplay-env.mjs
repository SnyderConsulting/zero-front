/** Strict self-play bridge. Every action on both teams comes from an external pixel policy. */
import readline from 'node:readline';
import {pathToFileURL} from 'node:url';
import {createSimulation,stepAgents,setController,TICK_RATE} from '../dist/simulation.js';
import {policyObservation,ACTION_HEADS_V2} from '../dist/policy.js';
import {SEMANTIC_PALETTE} from '../dist/renderer.js';

export const REWARD_COMPONENTS=Object.freeze(['aim','damageDealt','damageTaken','kills','deaths','teamScore','outcome']);
const SHAPED_V1=Object.freeze({profile:'shaped-v1',damageDealtPerHP:.01,damageTakenPerHP:-.01,personalKill:1,personalDeath:-1,
  teamScoreDifferential:.25,win:5,loss:-5,draw:0,aimPerSecond:.02,aimCapPerLife:.2,
  aimPixels:'nearest visible enemy RGB pixel to image center (31.5,17.5); quality 1 within central 2x2, linear falloff to 0 at radius 4',
  aimEligibility:'own actor alive and unprotected throughout decision; enemy protection is not inferred from pixels'});
export const REWARD_CONFIGS=Object.freeze({
  'sparse-v1':Object.freeze({profile:'sparse-v1',damageDealtPerHP:0,damageTakenPerHP:0,personalKill:0,personalDeath:0,
    teamScoreDifferential:1,win:5,loss:-5,draw:0,aimPerSecond:0,aimCapPerLife:0,aimPixels:'none'}),
  'shaped-v1':SHAPED_V1,
  'shaped-v2':Object.freeze({...SHAPED_V1,profile:'shaped-v2',draw:-.5})
});
const componentArrays=count=>Object.fromEntries(REWARD_COMPONENTS.map(key=>[key,new Array(count).fill(0)]));
/** Reads only the image seen by the policy. Allies, dead actors and occluded surfaces do not have enemy RGB. */
export function enemyCrosshairQuality(pixels){
  if(!pixels||pixels.length!==64*36*3)throw new TypeError('Expected a 64x36 RGB observation.');
  const enemy=SEMANTIC_PALETTE.enemy;
  let nearest=4;
  for(let y=14;y<=21;y++)for(let x=28;x<=35;x++){
    const offset=(y*64+x)*3;
    if(pixels[offset]===enemy[0]&&pixels[offset+1]===enemy[1]&&pixels[offset+2]===enemy[2])nearest=Math.min(nearest,Math.hypot(x-31.5,y-17.5));
  }
  return Math.max(0,Math.min(1,(4-nearest)/(4-Math.SQRT1_2)));
}

const integer=(value,name,min,max)=>{
  if(!Number.isInteger(value)||value<min||value>max)throw new TypeError(`${name} must be ${min}–${max}.`);
  return value;
};
const neutralAction=Object.freeze({forward:0,strafe:0,turn:0,pitch:0,fire:false,reload:false});
// A safety fallback only: step() requires all actions before advancing any environment.
// Attaching it to the human slot also gives every actor the same bot weapon spread.
const neutralController=()=>neutralAction;

export class SelfPlayVectorEnv{
  constructor(options={}){
    this.count=integer(options.environments??8,'environments',1,64);
    this.teamSize=integer(options.teamSize??4,'teamSize',1,8);
    this.seedStart=integer(options.seedStart??1000000,'seedStart',0,4294967295);
    this.seedCount=integer(options.seedCount??10000,'seedCount',1,1000000);
    if(this.seedStart+this.seedCount>4294967296)throw new TypeError('Seed range exceeds unsigned32.');
    this.seconds=integer(options.seconds??180,'seconds',1,180);
    this.scoreLimit=integer(options.scoreLimit??25,'scoreLimit',1,100);
    this.rewardProfile=options.rewardProfile??'sparse-v1';
    if(!Object.hasOwn(REWARD_CONFIGS,this.rewardProfile))throw new TypeError('Unknown self-play reward profile.');
    this.rewardSpec=REWARD_CONFIGS[this.rewardProfile];
    this.repeat=6;this.cursor=0;this.heads=ACTION_HEADS_V2;
    if(options.heads&&JSON.stringify(options.heads)!==JSON.stringify(this.heads))throw new TypeError('Expected the discrete-v2 action heads.');
    if(options.navigationReward||options.explorationReward)throw new TypeError('Strict self-play does not support navigation or exploration rewards.');
    this.actorsPerEnvironment=this.teamSize*2;
    this.actorCount=this.count*this.actorsPerEnvironment;
    this.envs=Array.from({length:this.count},(_,index)=>this.fresh(index,0));
  }
  fresh(index,episode){
    const seed=this.seedStart+(this.cursor++%this.seedCount);
    const sim=createSimulation({seed,teamSize:this.teamSize,scoreLimit:this.scoreLimit,recordReplay:false});
    for(const actor of sim.actors)setController(sim,actor.id,neutralController);
    return {sim,actors:sim.actors,episode,returns:sim.actors.map(()=>0),aimSpent:sim.actors.map(()=>0),
      rewardComponents:componentArrays(sim.actors.length),observations:null};
  }
  observe(rewards=new Array(this.actorCount).fill(0),done=new Array(this.actorCount).fill(false),episodes=[],deaths=new Array(this.actorCount).fill(0),rewardComponents=componentArrays(this.actorCount)){
    const images=[],scalars=[],alive=[],teams=[],environments=[],episodeIds=[],actorIds=[];
    for(let e=0;e<this.envs.length;e++){
      const env=this.envs[e];
      env.observations??=env.actors.map(actor=>policyObservation(env.sim,actor.id));
      for(let a=0;a<env.actors.length;a++){
        const actor=env.actors[a],observation=env.observations[a];
        images.push(Buffer.from(observation.pixels));scalars.push(...observation.scalars);alive.push(Number(actor.active));
        // Routing metadata is separate from the only model inputs: pixels and scalars.
        teams.push(actor.team);environments.push(e);episodeIds.push(env.episode);actorIds.push(actor.id);
      }
    }
    return {type:'transition',count:images.length,pixels:Buffer.concat(images).toString('base64'),scalars,alive,
      rewards,done,deaths,episodes,teams,environments,episodeIds,actorIds,rewardComponents,
      rewardProfile:this.rewardProfile,rewardSpec:this.rewardSpec};
  }
  step(labels){
    if(!Array.isArray(labels)||labels.length!==this.actorCount*6)throw new TypeError(`Expected ${this.actorCount*6} action indices for every actor on both teams.`);
    // Validate the entire vector before advancing any simulation.
    for(let i=0;i<labels.length;i++)integer(labels[i],'action index',0,this.heads[i%6].values.length-1);
    const rewards=[],done=[],deaths=[],episodes=[],rewardComponents=componentArrays(this.actorCount),spec=this.rewardSpec;
    for(let e=0;e<this.count;e++){
      const env=this.envs[e],{sim,actors}=env,actions={};
      const previousScores=[...sim.scores],previousDeaths=actors.map(actor=>actor.deaths),previousKills=actors.map(actor=>actor.kills);
      const previousActive=actors.map(actor=>actor.active),previousProtection=actors.map(actor=>actor.spawnProtection),previousTick=sim.tick;
      for(let a=0;a<actors.length;a++)actions[actors[a].id]=Object.fromEntries(this.heads.map((head,h)=>
        [head.name,head.values[labels[(e*this.actorsPerEnvironment+a)*6+h]]]));
      const transition=stepAgents(sim,actions,Math.min(this.repeat,Math.max(1,this.seconds*TICK_RATE-sim.tick)));
      const finished=sim.terminated||sim.truncated;
      const terminal=finished||sim.tick>=this.seconds*TICK_RATE;
      const blueDelta=(sim.scores[0]-previousScores[0])-(sim.scores[1]-previousScores[1]);
      const damageDealt=actors.map(()=>0),damageTaken=actors.map(()=>0),indices=new Map(actors.map((actor,index)=>[actor.id,index]));
      const respawned=new Set();
      for(const event of transition.events){
        if(event.type==='respawn')respawned.add(event.id);
        if(event.type!=='hit'||this.rewardProfile==='sparse-v1')continue;
        const shooter=indices.get(event.actor),victim=indices.get(event.victim);
        if(shooter===undefined||victim===undefined||actors[shooter].team===actors[victim].team)throw new Error('Invalid enemy hit telemetry.');
        // Actual HP removed is captured at impact: no overkill credit and no healing penalty.
        if(!Number.isFinite(event.actualDamage)||event.actualDamage<=0||event.actualDamage>event.damage)throw new Error('Shaped rewards require actualDamage hit telemetry.');
        damageDealt[shooter]+=event.actualDamage;damageTaken[victim]+=event.actualDamage;
      }
      // Reuse this same image for the policy's next observation; aim has no extra scene access.
      env.observations=actors.map(actor=>policyObservation(sim,actor.id));
      const elapsed=(sim.tick-previousTick)/TICK_RATE;
      for(let a=0;a<actors.length;a++){
        const actor=actors[a],died=actor.deaths-previousDeaths[a],index=e*this.actorsPerEnvironment+a;
        if(died||respawned.has(actor.id))env.aimSpent[a]=0;
        let aim=0;
        if(spec.aimPerSecond&&previousActive[a]&&actor.active&&!died&&!respawned.has(actor.id)&&previousProtection[a]<=0&&actor.spawnProtection<=0){
          const quality=enemyCrosshairQuality(env.observations[a].pixels);
          aim=Math.max(0,Math.min(spec.aimCapPerLife-env.aimSpent[a],spec.aimPerSecond*elapsed*quality));
          env.aimSpent[a]+=aim;
        }
        const components={aim,damageDealt:damageDealt[a]*spec.damageDealtPerHP,damageTaken:damageTaken[a]?damageTaken[a]*spec.damageTakenPerHP:0,
          kills:(actor.kills-previousKills[a])*spec.personalKill,deaths:died?died*spec.personalDeath:0,
          teamScore:blueDelta===0?0:(actor.team===0?blueDelta:-blueDelta)*spec.teamScoreDifferential,
          outcome:finished?(sim.winner==='draw'?spec.draw:sim.winner===actor.team?spec.win:spec.loss):0};
        let reward=0;
        for(const key of REWARD_COMPONENTS){rewardComponents[key][index]=components[key];env.rewardComponents[key][a]+=components[key];reward+=components[key];}
        rewards.push(reward);done.push(terminal);deaths.push(actor.deaths-previousDeaths[a]);env.returns[a]+=reward;
      }
      if(terminal){
        const teams=[0,1].map(team=>{
          const members=actors.filter(actor=>actor.team===team);
          return {team,kills:members.reduce((sum,a)=>sum+a.kills,0),deaths:members.reduce((sum,a)=>sum+a.deaths,0),
            hits:members.reduce((sum,a)=>sum+a.hits,0),shots:members.reduce((sum,a)=>sum+a.shots,0),
            returns:actors.reduce((sum,a,index)=>sum+(a.team===team?env.returns[index]:0),0)/this.teamSize,
            rewardComponents:Object.fromEntries(REWARD_COMPONENTS.map(key=>[key,actors.reduce((sum,a,index)=>sum+(a.team===team?env.rewardComponents[key][index]:0),0)]))};
        });
        episodes.push({environment:e,episode:env.episode,seed:sim.seed,seconds:sim.time,complete:finished,
          winner:sim.winner,scores:[...sim.scores],teams,rewardProfile:this.rewardProfile,
          rewardComponents:Object.fromEntries(REWARD_COMPONENTS.map(key=>[key,env.rewardComponents[key].reduce((sum,value)=>sum+value,0)]))});
        this.envs[e]=this.fresh(e,env.episode+1);
      }
    }
    // On done, these observations are the next round. done masks its bootstrap value.
    // A death is not a round terminal: the same player respawns into the same match.
    return this.observe(rewards,done,episodes,deaths,rewardComponents);
  }
}

async function main(){
  const lines=readline.createInterface({input:process.stdin,crlfDelay:Infinity});let env;
  for await(const line of lines){
    if(!line.trim())continue;
    try{
      const command=JSON.parse(line);let response;
      if(command.command==='init'){env=new SelfPlayVectorEnv(command);response=env.observe();}
      else if(command.command==='step'){if(!env)throw new Error('Initialize environments first.');response=env.step(command.actions);}
      else if(command.command==='close'){lines.close();process.stdin.pause();break;}
      else throw new TypeError('Unknown environment command.');
      process.stdout.write(JSON.stringify(response)+'\n');
    }catch(error){process.stdout.write(JSON.stringify({type:'error',error:error.message})+'\n');}
  }
}
if(process.argv[1]&&import.meta.url===pathToFileURL(process.argv[1]).href)main().catch(error=>{console.error(error);process.exitCode=1;});
