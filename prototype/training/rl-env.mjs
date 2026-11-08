/** Persistent vectorized team environments; policy inputs contain pixels and own status only. */
import readline from 'node:readline';
import {pathToFileURL} from 'node:url';
import {createSimulation,stepAgents,setController} from '../dist/simulation.js';
import {policyObservation,ACTION_HEADS_V2} from '../dist/policy.js';

const integer=(value,name,min,max)=>{
  if(!Number.isInteger(value)||value<min||value>max)throw new TypeError(`${name} must be ${min}–${max}.`);
  return value;
};
const cell=(sim,actor)=>Math.floor(actor.y)*sim.width+Math.floor(actor.x);

/** This distance field is reward shaping only. It is never part of an observation. */
function enemyDistances(sim,team){
  const distance=new Int16Array(sim.width*sim.height).fill(-1),queue=new Int32Array(distance.length);
  let head=0,tail=0;
  for(const actor of sim.actors)if(actor.active&&actor.team!==team){const c=cell(sim,actor);if(distance[c]===-1){distance[c]=0;queue[tail++]=c;}}
  while(head<tail){
    const c=queue[head++],x=c%sim.width,y=Math.floor(c/sim.width);
    for(const [nx,ny] of [[x-1,y],[x+1,y],[x,y-1],[x,y+1]]){
      if(nx<0||ny<0||nx>=sim.width||ny>=sim.height||sim.map[ny][nx]!=='.')continue;
      const n=ny*sim.width+nx;if(distance[n]!==-1)continue;
      distance[n]=distance[c]+1;queue[tail++]=n;
    }
  }
  return distance;
}

export class TeamVectorEnv{
  constructor(options={}){
    this.count=integer(options.environments??8,'environments',1,64);
    this.teamSize=integer(options.teamSize??4,'teamSize',1,8);
    this.seedStart=integer(options.seedStart??10000,'seedStart',0,4294967295);
    this.seedCount=integer(options.seedCount??100,'seedCount',1,10000);
    if(this.seedStart+this.seedCount>4294967296)throw new TypeError('Seed range exceeds unsigned32.');
    this.seconds=integer(options.seconds??180,'seconds',1,180);
    this.scoreLimit=integer(options.scoreLimit??25,'scoreLimit',1,100);
    this.repeat=6;this.cursor=0;
    // Values arrive as the declared model contract; this bridge does not infer hidden inputs.
    this.heads=ACTION_HEADS_V2;
    if(options.heads&&JSON.stringify(options.heads)!==JSON.stringify(this.heads))throw new TypeError('Expected the discrete-v2 action heads.');
    this.navigation=Number(options.navigationReward??.025);
    this.exploration=Number(options.explorationReward??.02);
    if(!Number.isFinite(this.navigation)||this.navigation<0||this.navigation>1||!Number.isFinite(this.exploration)||this.exploration<0||this.exploration>1)throw new TypeError('Invalid shaping scale.');
    this.envs=Array.from({length:this.count},(_,index)=>this.fresh(index,0));
  }
  fresh(index,episode){
    const seed=this.seedStart+(this.cursor++%this.seedCount),team=(index+episode)%2;
    const sim=createSimulation({seed,teamSize:this.teamSize,scoreLimit:this.scoreLimit,recordReplay:false});
    setController(sim,'player','scripted'); // All actors use the identical bot weapon-spread rule.
    const actors=sim.actors.filter(actor=>actor.team===team);
    return {sim,team,actors,episode,visited:actors.map(actor=>new Set([cell(sim,actor)])),returns:actors.map(()=>0)};
  }
  observe(rewards=new Array(this.count*this.teamSize).fill(0),done=new Array(this.count*this.teamSize).fill(false),episodes=[]){
    const images=[],scalars=[],alive=[];
    for(const env of this.envs)for(const actor of env.actors){
      const observation=policyObservation(env.sim,actor.id);
      images.push(Buffer.from(observation.pixels));scalars.push(...observation.scalars);alive.push(Number(actor.active));
    }
    return {type:'transition',count:images.length,pixels:Buffer.concat(images).toString('base64'),scalars,alive,rewards,done,episodes};
  }
  step(labels){
    const total=this.count*this.teamSize;
    if(!Array.isArray(labels)||labels.length!==total*6)throw new TypeError(`Expected ${total*6} action indices.`);
    for(let i=0;i<labels.length;i++)integer(labels[i],'action index',0,this.heads[i%6].values.length-1);
    const rewards=[],done=[],episodes=[];
    for(let e=0;e<this.count;e++){
      const env=this.envs[e],{sim,actors,team}=env,actions={};
      const field=this.navigation?enemyDistances(sim,team):null;
      const before=actors.map(actor=>({x:actor.x,y:actor.y,active:actor.active,hits:actor.hits,kills:actor.kills,deaths:actor.deaths,
        shots:actor.shots,distance:field?field[cell(sim,actor)]:-1}));
      const oldScores=[...sim.scores];
      for(let a=0;a<actors.length;a++)actions[actors[a].id]=Object.fromEntries(this.heads.map((head,h)=>[head.name,head.values[labels[(e*this.teamSize+a)*6+h]]]));
      stepAgents(sim,actions,Math.min(this.repeat,Math.max(1,Math.round((this.seconds-sim.time)*60))));
      const terminal=sim.terminated||sim.truncated||sim.time>=this.seconds;
      const finished=sim.terminated||sim.truncated;
      const teamDelta=(sim.scores[team]-oldScores[team])-(sim.scores[1-team]-oldScores[1-team]);
      for(let a=0;a<actors.length;a++){
        const actor=actors[a],previous=before[a];
        let reward=(actor.hits-previous.hits)*.15+(actor.kills-previous.kills)*1.0-(actor.deaths-previous.deaths)*1.2+teamDelta*.2;
        reward-=Math.max(0,actor.shots-previous.shots-(actor.hits-previous.hits))*.002;
        if(previous.active&&actor.active&&actor.deaths===previous.deaths){
          const c=cell(sim,actor),newDistance=field?field[c]:-1;
          if(previous.distance>=0&&newDistance>=0)reward+=Math.max(-2,Math.min(2,previous.distance-newDistance))*this.navigation;
          if(!env.visited[a].has(c)){env.visited[a].add(c);reward+=this.exploration;}
        }
        if(finished&&sim.winner!=='draw')reward+=sim.winner===team?6:-6;
        rewards.push(reward);done.push(terminal);env.returns[a]+=reward;
      }
      if(terminal){
        episodes.push({environment:e,seed:sim.seed,team,seconds:sim.time,complete:finished,winner:sim.winner,win:sim.winner===team,
          scores:[...sim.scores],kills:actors.reduce((sum,a)=>sum+a.kills,0),deaths:actors.reduce((sum,a)=>sum+a.deaths,0),
          hits:actors.reduce((sum,a)=>sum+a.hits,0),shots:actors.reduce((sum,a)=>sum+a.shots,0),
          returns:env.returns.reduce((sum,value)=>sum+value,0)/actors.length});
        this.envs[e]=this.fresh(e,env.episode+1);
      }
    }
    return this.observe(rewards,done,episodes);
  }
}

async function main(){
  const lines=readline.createInterface({input:process.stdin,crlfDelay:Infinity});let env;
  for await(const line of lines){
    if(!line.trim())continue;
    try{
      const command=JSON.parse(line);let response;
      if(command.command==='init'){env=new TeamVectorEnv(command);response=env.observe();}
      else if(command.command==='step'){if(!env)throw new Error('Initialize environments first.');response=env.step(command.actions);}
      else if(command.command==='close'){lines.close();process.stdin.pause();break;}
      else throw new TypeError('Unknown environment command.');
      process.stdout.write(JSON.stringify(response)+'\n');
    }catch(error){process.stdout.write(JSON.stringify({type:'error',error:error.message})+'\n');}
  }
}
if(process.argv[1]&&import.meta.url===pathToFileURL(process.argv[1]).href)main().catch(error=>{console.error(error);process.exitCode=1;});
