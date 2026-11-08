/** Full-round, simulation-time replay. No DOM or dependency on the live simulation. */
export const REPLAY_FIELDS=Object.freeze([
  'x','y','yaw','pitch','health','ammo','active','kills','deaths','shots','hits',
  'reloadTimer','spawnProtection','respawnTimer',
]);
const STRIDE=REPLAY_FIELDS.length;
const clamp=(value,min,max)=>Math.max(min,Math.min(max,value));
const wrap=value=>((value+Math.PI)%(Math.PI*2)+Math.PI*2)%(Math.PI*2)-Math.PI;

/** Allocate a bounded buffer once; 16 actors × 180 seconds uses <10 MiB for frames. */
export function createReplay(sim,tickRate=60,maxDuration=180){
  const capacity=Math.ceil(tickRate*maxDuration)+1;
  const roster=Object.freeze(sim.actors.map(actor=>Object.freeze({
    id:actor.id,name:actor.name,team:actor.team,slot:actor.slot,radius:actor.radius,
  })));
  const recording={
    version:1,map:Object.freeze([...sim.map]),width:sim.width,height:sim.height,
    seed:sim.seed,layout:sim.layout,teamSize:sim.teamSize,scoreLimit:sim.scoreLimit,
    tickRate,capacity,roster,actorCount:roster.length,stride:STRIDE,
    frames:new Float32Array(capacity*roster.length*STRIDE),scores:new Uint16Array(capacity*2),
    frameCount:0,duration:0,events:[],summary:null,
  };
  captureReplayFrame(recording,sim);
  return recording;
}

/** Each call copies scalar values; moving live actors never alters old frames. */
export function captureReplayFrame(recording,sim){
  if(!recording)return;
  if(recording.summary)return;
  if(sim.tick!==recording.frameCount)throw new Error('Replay frames must be captured once per simulation tick.');
  if(recording.frameCount>=recording.capacity)throw new RangeError('Replay exceeds the round time limit.');
  let offset=recording.frameCount*recording.actorCount*STRIDE;
  for(const actor of sim.actors){
    for(const field of REPLAY_FIELDS)recording.frames[offset++]=field==='active'?Number(actor.active):actor[field]??0;
  }
  recording.scores[recording.frameCount*2]=sim.scores[0];
  recording.scores[recording.frameCount*2+1]=sim.scores[1];
  recording.frameCount++;recording.duration=sim.tick/recording.tickRate;
  if(sim.terminated||sim.truncated)recording.summary=buildRoundSummary(sim);
}

/** Separate from the small live-event ring, so early-round actions remain available. */
export function recordReplayEvent(recording,event){
  if(!recording||recording.summary)return;
  const copy={...event};
  if(event.scores)copy.scores=Object.freeze([...event.scores]);
  recording.events.push(Object.freeze(copy));
}

const accuracy=(hits,shots)=>shots?hits/shots*100:0;
export function buildRoundSummary(sim){
  const players=Object.freeze(sim.actors.map(actor=>Object.freeze({
    id:actor.id,name:actor.name,team:actor.team,slot:actor.slot,
    kills:actor.kills,deaths:actor.deaths,shots:actor.shots,hits:actor.hits,
    accuracy:accuracy(actor.hits,actor.shots),damageDealt:actor.damageDealt??0,damageTaken:actor.damageTaken,
  })));
  const teams=Object.freeze([0,1].map(team=>{
    const total={team,kills:0,deaths:0,shots:0,hits:0};
    for(const player of players.filter(p=>p.team===team))for(const key of ['kills','deaths','shots','hits'])total[key]+=player[key];
    return Object.freeze({...total,accuracy:accuracy(total.hits,total.shots)});
  }));
  return Object.freeze({winner:sim.winner,scores:Object.freeze([...sim.scores]),duration:sim.time,
    reason:sim.truncated?'time':sim.terminated?'score':'in-progress',seed:sim.seed,layout:sim.layout,
    teamSize:sim.teamSize,scoreLimit:sim.scoreLimit,playerId:sim.player.id,players,teams});
}

function upperEventBound(events,time){
  let low=0,high=events.length;
  while(low<high){const middle=(low+high)>>>1;if(events[middle].time<=time)low=middle+1;else high=middle;}
  return low;
}

/** Seek independently of playback direction; interpolate motion, never death/respawn jumps. */
export function sampleReplay(recording,requestedTime=0){
  if(!recording||!recording.frameCount)throw new TypeError('A recorded round is required.');
  if(typeof requestedTime!=='number'||!Number.isFinite(requestedTime))throw new TypeError('Replay time must be finite.');
  const time=clamp(requestedTime,0,recording.duration),position=time*recording.tickRate;
  const tick=Math.min(recording.frameCount-1,Math.floor(position+1e-9));
  const next=Math.min(recording.frameCount-1,tick+1),fraction=clamp(position-tick,0,1);
  const frameStride=recording.actorCount*STRIDE;
  const actors=recording.roster.map((roster,index)=>{
    const offset=tick*frameStride+index*STRIDE,nextOffset=next*frameStride+index*STRIDE;
    const actor={...roster};
    for(let field=0;field<STRIDE;field++)actor[REPLAY_FIELDS[field]]=recording.frames[offset+field];
    actor.active=Boolean(actor.active);actor.reloading=actor.reloadTimer>0;
    if(fraction>0&&actor.active&&recording.frames[nextOffset+6]&&actor.deaths===recording.frames[nextOffset+8]){
      for(const field of [0,1,3])actor[REPLAY_FIELDS[field]]+=(recording.frames[nextOffset+field]-actor[REPLAY_FIELDS[field]])*fraction;
      actor.yaw=wrap(actor.yaw+wrap(recording.frames[nextOffset+2]-actor.yaw)*fraction);
    }
    return actor;
  });
  const end=upperEventBound(recording.events,time),start=upperEventBound(recording.events,time-.35);
  return {time,tick,scores:[recording.scores[tick*2],recording.scores[tick*2+1]],actors,
    events:recording.events.slice(start,end)};
}
