/** Seeded, renderer-independent team deathmatch. Meters, radians, 60 Hz. */
import {generateArena} from './mapgen.js';
import {createReplay,captureReplayFrame,recordReplayEvent} from './replay.js';
export const TICK_RATE = 60;
export const DT = 1 / TICK_RATE;
export const SCENARIOS = ['tdm'];
export const HEALTH_REGEN_DELAY = 3;
export const HEALTH_REGEN_RATE = 50;
const MAGAZINE=20, MATCH_SECONDS=180, DAMAGE=15;
const clamp=(x,lo,hi)=>Math.max(lo,Math.min(hi,x));
const wrap=x=>((x+Math.PI)%(Math.PI*2)+Math.PI*2)%(Math.PI*2)-Math.PI;

function finite(value,name,lo,hi) {
  if(typeof value!=='number'||!Number.isFinite(value)||value<lo||value>hi) throw new TypeError(`${name} must be a finite number from ${lo} to ${hi}.`);
  return value;
}
function integer(value,name,lo,hi) {
  finite(value,name,lo,hi);
  if(!Number.isInteger(value)) throw new TypeError(`${name} must be an integer.`);
  return value;
}
function random(sim) {
  let n=sim.rngState;n^=n<<13;n^=n>>>17;n^=n<<5;sim.rngState=n>>>0;
  return sim.rngState/4294967296;
}

export function wallAt(sim,x,y) {
  const ix=Math.floor(x),iy=Math.floor(y);
  return ix<0||iy<0||ix>=sim.width||iy>=sim.height?'#':sim.map[iy][ix];
}
/** Returns Euclidean ray distance, exact impact coordinates, wall cell and side (0=x, 1=y). */
export function raycast(sim,x,y,angle,maxDistance=50) {
  finite(x,'x',-1e6,1e6);finite(y,'y',-1e6,1e6);finite(angle,'angle',-1e6,1e6);finite(maxDistance,'maxDistance',0,1e6);
  const dx=Math.cos(angle),dy=Math.sin(angle);
  let mx=Math.floor(x),my=Math.floor(y);
  if(wallAt(sim,x,y)!=='.')return{distance:0,x,y,cell:wallAt(sim,x,y),side:0};
  const ddx=Math.abs(dx)<1e-12?Infinity:Math.abs(1/dx),ddy=Math.abs(dy)<1e-12?Infinity:Math.abs(1/dy);
  const sx=dx<0?-1:1,sy=dy<0?-1:1;
  let tx=(dx<0?x-mx:mx+1-x)*ddx,ty=(dy<0?y-my:my+1-y)*ddy,distance=0,side=0;
  while(distance<=maxDistance){
    if(tx<ty){distance=tx;tx+=ddx;mx+=sx;side=0;}else{distance=ty;ty+=ddy;my+=sy;side=1;}
    if(distance>maxDistance)break;
    const cell=mx<0||my<0||mx>=sim.width||my>=sim.height?'#':sim.map[my][mx];
    if(cell!=='.')return{distance,x:x+dx*distance,y:y+dy*distance,cell,side};
  }
  return{distance:maxDistance,x:x+dx*maxDistance,y:y+dy*maxDistance,cell:'.',side};
}
function canStand(sim,x,y,radius) {
  for(let iy=Math.floor(y-radius);iy<=Math.floor(y+radius);iy++)for(let ix=Math.floor(x-radius);ix<=Math.floor(x+radius);ix++){
    if(wallAt(sim,ix,iy)==='.')continue;
    const nx=clamp(x,ix,ix+1),ny=clamp(y,iy,iy+1);
    if((x-nx)**2+(y-ny)**2<radius**2)return false;
  }
  return true;
}
function move(sim,actor,dx,dy) {
  if(canStand(sim,actor.x+dx,actor.y,actor.radius))actor.x+=dx;
  if(canStand(sim,actor.x,actor.y+dy,actor.radius))actor.y+=dy;
}
function event(sim,type,details={}) {
  const e={type,tick:sim.tick,time:sim.time,...details};
  sim.events.push(e);if(sim.events.length>128)sim.events.shift();sim._stepEvents.push(e);
  recordReplayEvent(sim.replay,e);
}
function spawn(sim,actor,initial=false) {
  const points=sim.spawns[actor.team];
  const spot=points[initial?actor.slot:Math.floor(random(sim)*points.length)];
  Object.assign(actor,{x:spot.x+(initial?0:(random(sim)-.5)*.2),y:spot.y+(initial?0:(random(sim)-.5)*.2),
    yaw:spot.yaw??(actor.team===0?0:Math.PI),pitch:0,health:actor.maxHealth,lastDamageTick:null,ammo:MAGAZINE,cooldown:0,reloadTimer:0,
    active:true,respawnTimer:0,spawnProtection:1.2,hitFlash:0,muzzleFlash:0,damageFlash:0,hitMarker:0,
    _brain:{targetId:null,reaction:0,path:[],pathTimer:0,direction:actor.slot%2?-1:1,directionTimer:1+random(sim)*2}});
  if(!initial)event(sim,'respawn',{id:actor.id,team:actor.team});
}
function makeActor(sim,team,slot) {
  const id=team===0&&slot===0?'player':`${team===0?'blue':'red'}-${slot+1}`;
  const actor={id,type:'enemy',team,slot,name:id==='player'?'YOU':`${team===0?'BLUE':'RED'} ${slot+1}`,
    radius:.34,height:1.6,eyeHeight:1,maxHealth:100,kills:0,deaths:0,shots:0,hits:0,damageTaken:0,damageDealt:0};
  spawn(sim,actor,true);return actor;
}
export function createSimulation(options={}) {return reset({},options);}
/** Reset mutates the supplied object. Omitted fields retain prior match settings. */
export function reset(sim,options={}) {
  if(!options||typeof options!=='object'||Array.isArray(options))throw new TypeError('Reset options must be an object.');
  const seed=integer(options.seed??sim.seed??42,'seed',0,4294967295);
  const teamSize=integer(options.teamSize??sim.teamSize??4,'teamSize',1,8);
  const scoreLimit=integer(options.scoreLimit??sim.scoreLimit??25,'scoreLimit',1,100);
  const recordReplay=options.recordReplay??sim.recordReplay??false;
  if(typeof recordReplay!=='boolean')throw new TypeError('recordReplay must be boolean.');
  if(options.scenario!==undefined&&options.scenario!=='tdm')throw new TypeError('Only team deathmatch (tdm) is supported.');
  const {map,spawns,rooms,layout,symmetry}=generateArena(seed);
  Object.assign(sim,{seed,rngState:seed||0x9e3779b9,teamSize,scoreLimit,scenario:'tdm',map,width:map[0].length,height:map.length,
    spawns,rooms,layout,symmetry,recordReplay,replay:null,controllers:new Map(),
    actors:[],entities:[],player:null,tick:0,time:0,scores:[0,0],winner:null,terminated:false,truncated:false,success:false,
    kills:0,shots:0,hits:0,damageTaken:0,events:[],_stepEvents:[],lastReward:0,totalReward:0,_rewards:{}});
  for(let team=0;team<2;team++)for(let slot=0;slot<teamSize;slot++)sim.actors.push(makeActor(sim,team,slot));
  sim.player=sim.actors[0];sim.entities=sim.actors.slice(1);
  if(recordReplay)sim.replay=createReplay(sim,TICK_RATE,MATCH_SECONDS);
  return sim;
}
function normalizeAction(action) {
  if(!action||typeof action!=='object'||Array.isArray(action))throw new TypeError('action must be an object.');
  for(const name of Object.keys(action))if(!['forward','strafe','turn','pitch','fire','reload'].includes(name))throw new TypeError(`Unknown action field: ${name}`);
  const a={};
  for(const [key,lo,hi]of[['forward',-1,1],['strafe',-1,1],['turn',-12,12],['pitch',-6,6]])a[key]=finite(action[key]??0,key,lo,hi);
  for(const key of['fire','reload']){a[key]=action[key]??false;if(typeof a[key]!=='boolean')throw new TypeError(`${key} must be boolean.`);}
  return a;
}
function pathTo(sim,actor,target) {
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
function botAction(sim,actor) {
  const brain=actor._brain;
  const candidates=sim.actors.filter(other=>other.active&&other.team!==actor.team)
    .map(other=>({other,distance:Math.hypot(other.x-actor.x,other.y-actor.y)})).sort((a,b)=>a.distance-b.distance);
  if(!candidates.length)return normalizeAction({});
  let choice=candidates.find(({other,distance})=>distance<18&&raycast(sim,actor.x,actor.y,Math.atan2(other.y-actor.y,other.x-actor.x),distance).distance>=distance-.05);
  choice=choice||candidates[0];
  const target=choice.other,distance=choice.distance;
  const bearing=Math.atan2(target.y-actor.y,target.x-actor.x);
  const visible=raycast(sim,actor.x,actor.y,bearing,Math.max(.01,distance)).distance>=distance-.05;
  if(brain.targetId!==target.id){brain.targetId=target.id;brain.reaction=0;brain.pathTimer=0;}
  brain.reaction=visible?brain.reaction+DT:0;brain.pathTimer-=DT;brain.directionTimer-=DT;
  if(brain.directionTimer<=0){brain.direction*=-1;brain.directionTimer=1.2+random(sim)*2;}
  let desired=bearing,forward=0,strafe=0;
  // Keep following the route through doorways until a target stays in sight.
  if(visible&&brain.reaction>.18){
    forward=distance>8?1:distance<4?-.35:0;
    strafe=distance<15?brain.direction*.55:0;
    // Local separation prevents a whole team from occupying one firing line.
    for(const mate of sim.actors){
      if(mate===actor||!mate.active||mate.team!==actor.team)continue;
      const mx=actor.x-mate.x,my=actor.y-mate.y,length=Math.hypot(mx,my);
      if(length>.01&&length<1.0)strafe+=(-Math.sin(actor.yaw)*mx+Math.cos(actor.yaw)*my)/length*.5;
    }
  }else{
    if(brain.pathTimer<=0){brain.path=pathTo(sim,actor,target);brain.pathTimer=.7;}
    while(brain.path.length&&Math.hypot(brain.path[0].x-actor.x,brain.path[0].y-actor.y)<.2)brain.path.shift();
    if(brain.path.length)desired=Math.atan2(brain.path[0].y-actor.y,brain.path[0].x-actor.x);
    forward=Math.abs(wrap(desired-actor.yaw))<.5?1:0;
  }
  const error=wrap(desired-actor.yaw);
  const pitchGoal=visible?Math.atan2(target.height*.55-actor.eyeHeight,Math.max(distance,.1)):0;
  return {forward,strafe:clamp(strafe,-1,1),turn:clamp(error*6,-3.6,3.6),pitch:clamp((pitchGoal-actor.pitch)*6,-3,3),
    fire:visible&&distance<18&&brain.reaction>.28&&Math.abs(wrap(bearing-actor.yaw))<.13,
    reload:actor.ammo===0||(actor.ammo<5&&!visible)};
}
/** Attach a local controller. Reset clears controllers; explicit step actions win. */
export function setController(sim,actorId,controller) {
  if(!sim.actors.some(actor=>actor.id===actorId))throw new TypeError(`Unknown actor: ${actorId}`);
  if(controller!==null&&controller!=='scripted'&&typeof controller!=='function')throw new TypeError('Controller must be a function, scripted, or null.');
  if(controller===null)sim.controllers.delete(actorId);else sim.controllers.set(actorId,controller);
}
/** Diagnostic teacher label; peeking must not advance the bot brain or RNG. */
export function scriptedAction(sim,actorId) {
  const actor=sim.actors.find(value=>value.id===actorId);
  if(!actor)throw new TypeError(`Unknown actor: ${actorId}`);
  if(!actor.active)return normalizeAction({});
  const brain=actor._brain,rng=sim.rngState;
  actor._brain={...brain,path:[...brain.path]};
  try{return botAction(sim,actor);}finally{actor._brain=brain;sim.rngState=rng;}
}
function finish(sim,truncated=false) {
  sim.truncated=truncated;sim.terminated=!truncated;
  sim.winner=sim.scores[0]===sim.scores[1]?'draw':sim.scores[0]>sim.scores[1]?0:1;
  sim.success=sim.winner===0;
  event(sim,'match-end',{winner:sim.winner,scores:[...sim.scores],reason:truncated?'time':'score'});
}
function shoot(sim,actor,isBot) {
  if(actor.cooldown>0||actor.reloadTimer>0||actor.ammo===0)return;
  actor.ammo--;actor.shots++;actor.cooldown=.18;actor.muzzleFlash=.065;
  const spread=isBot?(random(sim)-.5)*.08:0;
  const yaw=actor.yaw+spread,dx=Math.cos(yaw),dy=Math.sin(yaw);
  let nearest=raycast(sim,actor.x,actor.y,yaw,18).distance,victim=null;
  for(const other of sim.actors){
    if(other===actor||!other.active)continue;
    const ex=other.x-actor.x,ey=other.y-actor.y,along=ex*dx+ey*dy,across=ex*dy-ey*dx;
    if(along<0||Math.abs(across)>other.radius)continue;
    const t=Math.max(0,along-Math.sqrt(other.radius**2-across**2));
    const height=actor.eyeHeight+Math.tan(actor.pitch)*t;
    if(t<nearest&&height>=0&&height<=other.height){nearest=t;victim=other;}
  }
  event(sim,'shot',{actor:actor.id,team:actor.team,fromX:actor.x,fromY:actor.y,x:actor.x+dx*nearest,y:actor.y+dy*nearest});
  if(!victim||victim.team===actor.team||victim.spawnProtection>0)return;
  const actualDamage=Math.min(victim.health,DAMAGE);
  victim.health=Math.max(0,victim.health-DAMAGE);victim.damageTaken+=DAMAGE;victim.damageFlash=.22;victim.hitFlash=.12;
  victim.lastDamageTick=sim.tick;
  actor.damageDealt+=DAMAGE;
  actor.hits++;actor.hitMarker=.14;sim._rewards[actor.id]=(sim._rewards[actor.id]||0)+.1;
  event(sim,'hit',{actor:actor.id,victim:victim.id,team:actor.team,damage:DAMAGE,actualDamage});
  if(victim.health===0){
    victim.active=false;victim.respawnTimer=2;victim.deaths++;victim.reloadTimer=0;
    actor.kills++;sim.scores[actor.team]++;
    sim._rewards[actor.id]=(sim._rewards[actor.id]||0)+1;
    sim._rewards[victim.id]=(sim._rewards[victim.id]||0)-1;
    event(sim,'elimination',{actor:actor.id,killer:actor.id,victim:victim.id,team:actor.team,scores:[...sim.scores]});
    if(sim.scores[actor.team]>=sim.scoreLimit)finish(sim);
  }
}
function tick(sim,explicit) {
  sim.tick++;sim.time=sim.tick*DT;
  const actions=new Map();
  for(const actor of sim.actors){
    for(const key of['cooldown','muzzleFlash','damageFlash','hitMarker','hitFlash','spawnProtection'])actor[key]=Math.max(0,actor[key]-DT);
    if(!actor.active){actor.respawnTimer=Math.max(0,actor.respawnTimer-DT);if(actor.respawnTimer<1e-9)spawn(sim,actor);continue;}
    if(actor.reloadTimer>0){actor.reloadTimer=Math.max(0,actor.reloadTimer-DT);if(actor.reloadTimer<1e-9){actor.reloadTimer=0;actor.ammo=MAGAZINE;event(sim,'reloaded',{actor:actor.id});}}
    const manual=explicit.has(actor.id),controller=sim.controllers.get(actor.id);
    const a=manual?explicit.get(actor.id):typeof controller==='function'?normalizeAction(controller(sim,actor)):
      controller==='scripted'||actor!==sim.player?botAction(sim,actor):normalizeAction({});
    // A bot's weapon spread is identical for scripted, learned and external control.
    actions.set(actor.id,{action:a,isBot:actor!==sim.player||sim.controllers.has(actor.id)});
    if(a.reload&&actor.reloadTimer===0&&actor.ammo<MAGAZINE){actor.reloadTimer=1.1;event(sim,'reload',{actor:actor.id});}
    actor.yaw=wrap(actor.yaw+a.turn*DT);actor.pitch=clamp(actor.pitch+a.pitch*DT,-.45,.45);
    const norm=Math.max(1,Math.hypot(a.forward,a.strafe)),forward=a.forward/norm,strafe=a.strafe/norm;
    move(sim,actor,(Math.cos(actor.yaw)*forward-Math.sin(actor.yaw)*strafe)*3.4*DT,
      (Math.sin(actor.yaw)*forward+Math.cos(actor.yaw)*strafe)*3.4*DT);
  }
  // Rotate firing order each tick to avoid a permanent team-order advantage.
  for(let i=0;i<sim.actors.length&&!sim.terminated;i++){
    const actor=sim.actors[(sim.tick+i)%sim.actors.length],entry=actions.get(actor.id);
    if(actor.active&&entry?.action.fire)shoot(sim,actor,entry.isBot);
  }
  // Resolve damage first: a fresh hit interrupts healing on this very tick.
  if(!sim.terminated)for(const actor of sim.actors){
    if(actor.active&&actor.health>0&&actor.health<actor.maxHealth&&actor.lastDamageTick!==null&&
        sim.tick-actor.lastDamageTick>HEALTH_REGEN_DELAY*TICK_RATE){
      actor.health=Math.min(actor.maxHealth,actor.health+HEALTH_REGEN_RATE*DT);
    }
  }
  if(!sim.terminated&&sim.tick>=MATCH_SECONDS*TICK_RATE)finish(sim,true);
  sim.kills=sim.player.kills;sim.shots=sim.player.shots;sim.hits=sim.player.hits;sim.damageTaken=sim.player.damageTaken;
  captureReplayFrame(sim.replay,sim);
}
/** Control selected actors; omitted bots use scripted AI, omitted human idles. */
export function stepAgents(sim,actionsById={},ticks=1) {
  if(!actionsById||typeof actionsById!=='object'||Array.isArray(actionsById))throw new TypeError('actionsById must be an object.');
  const known=new Set(sim.actors.map(a=>a.id)),explicit=new Map();
  for(const[id,action]of Object.entries(actionsById)){
    if(!known.has(id))throw new TypeError(`Unknown actor: ${id}`);
    explicit.set(id,normalizeAction(action));
  }
  integer(ticks,'ticks',1,600);
  sim._stepEvents=[];sim._rewards={};sim.lastReward=0;
  for(let i=0;i<ticks&&!sim.terminated&&!sim.truncated;i++)tick(sim,explicit);
  sim.lastReward=sim._rewards.player||0;sim.totalReward+=sim.lastReward;
  return{reward:sim.lastReward,rewards:{...sim._rewards},terminated:sim.terminated,truncated:sim.truncated,events:[...sim._stepEvents]};
}
export function step(sim,action={},ticks=1){return stepAgents(sim,{player:action},ticks);}
/** Public personal status and scoreboard. No hidden map or actor coordinates. */
export function observe(sim,actorId='player') {
  const actor=sim.actors.find(a=>a.id===actorId);if(!actor)throw new TypeError(`Unknown actor: ${actorId}`);
  return{actorId:actor.id,team:actor.team,health:actor.health,ammo:actor.ammo,ammoMax:MAGAZINE,magazine:MAGAZINE,
    reloading:actor.reloadTimer>0,reloadProgress:actor.reloadTimer>0?1-actor.reloadTimer/1.1:0,
    alive:actor.active,respawnTimer:actor.respawnTimer,spawnProtection:actor.spawnProtection,
    kills:actor.kills,deaths:actor.deaths,shots:actor.shots,hits:actor.hits,scores:[...sim.scores],teamSize:sim.teamSize,
    scoreLimit:sim.scoreLimit,winner:sim.winner,time:sim.time,tick:sim.tick,timeLimit:MATCH_SECONDS,
    objective:`First team to ${sim.scoreLimit}`,terminated:sim.terminated,truncated:sim.truncated,
    success:sim.winner===actor.team,scenario:'tdm',seed:sim.seed,reward:sim._rewards[actor.id]||0,totalReward:actor===sim.player?sim.totalReward:undefined};
}
