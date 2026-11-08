import test from 'node:test';
import assert from 'node:assert/strict';
import {SelfPlayVectorEnv,REWARD_CONFIGS,REWARD_COMPONENTS,enemyCrosshairQuality} from '../training/selfplay-env.mjs';
import {stepAgents} from '../dist/simulation.js';
import {policyObservation} from '../dist/policy.js';
import {SEMANTIC_PALETTE} from '../dist/renderer.js';

const idle=[1,1,4,1,0,0];
const idleActions=env=>Array.from({length:env.actorCount},()=>idle).flat();
function combatPositions(env){
  const sim=env.envs[0].sim;
  sim.width=10;sim.height=10;
  sim.map=Array.from({length:10},(_,y)=>Array.from({length:10},(_,x)=>x===0||x===9||y===0||y===9?'#':'.'));
  for(const actor of sim.actors)Object.assign(actor,{x:actor.team===0?2.5:5.5,y:2.5+actor.slot*3,yaw:actor.team===0?0:Math.PI,pitch:0,spawnProtection:0});
  return sim;
}

test('self-play exposes every actor with separate routing and only pixel/status policy inputs',()=>{
  const env=new SelfPlayVectorEnv({environments:2,teamSize:2});
  const initial=env.observe();
  assert.equal(initial.count,8);assert.equal(Buffer.from(initial.pixels,'base64').length,8*36*64*3);
  assert.equal(initial.scalars.length,48);assert.deepEqual(initial.alive,Array(8).fill(1));
  assert.deepEqual(initial.teams,[0,0,1,1,0,0,1,1]);
  assert.deepEqual(initial.environments,[0,0,0,0,1,1,1,1]);
  assert.deepEqual(initial.episodeIds,Array(8).fill(0));
  assert.deepEqual(initial.actorIds,['player','blue-2','red-1','red-2','player','blue-2','red-1','red-2']);
  const observation=policyObservation(env.envs[0].sim,'player');
  assert.deepEqual(Object.keys(observation).sort(),['pixels','scalars']);
  assert.deepEqual(Buffer.from(initial.pixels,'base64').subarray(0,36*64*3),Buffer.from(observation.pixels));
});

test('both teams have neutral fallback and idle self-play never calls a scripted brain',()=>{
  const env=new SelfPlayVectorEnv({environments:2,teamSize:2});
  const before=env.envs.map(entry=>entry.actors.map(actor=>[actor.x,actor.y,actor.yaw,actor.pitch]));
  for(const entry of env.envs)for(const actor of entry.actors){
    assert.equal(typeof entry.sim.controllers.get(actor.id),'function');
    Object.defineProperty(actor,'_brain',{configurable:true,get(){throw new Error('Scripted bot brain was accessed.');}});
  }
  for(let i=0;i<12;i++)env.step(idleActions(env));
  // Even a caller bypassing the vector API cannot accidentally engage scripted AI.
  for(const entry of env.envs)stepAgents(entry.sim,{},6);
  env.envs.forEach((entry,e)=>entry.actors.forEach((actor,a)=>{
    const after=[actor.x,actor.y,actor.yaw,actor.pitch];
    after.forEach((value,index)=>assert.ok(Math.abs(value-before[e][a][index])<1e-12));
  }));
  assert.ok(env.envs.every(entry=>entry.actors.every(actor=>actor.shots===0)));
});

test('shared score and complete-outcome rewards are zero-sum across teams',()=>{
  const env=new SelfPlayVectorEnv({environments:1,teamSize:2,scoreLimit:1});
  const sim=combatPositions(env);sim.actors[2].health=15;
  const actions=idleActions(env);actions[4]=1;
  const result=env.step(actions);
  assert.deepEqual(result.rewards,[6,6,-6,-6]);
  assert.equal(result.rewards.reduce((sum,value)=>sum+value,0),0);
  assert.deepEqual(result.done,[true,true,true,true]);
  assert.deepEqual(result.deaths,[0,0,1,0]);
  assert.equal(result.episodes[0].winner,0);assert.equal(result.episodes[0].complete,true);
  assert.deepEqual(result.episodes[0].scores,[1,0]);
  assert.deepEqual(result.episodes[0].teams.map(team=>team.returns),[6,-6]);
  assert.deepEqual(result.episodeIds,[1,1,1,1]);
  assert.equal(env.envs[0].sim.tick,0);
});

test('deaths do not terminate a continuing round and empty movement has no shaping reward',()=>{
  const env=new SelfPlayVectorEnv({environments:1,teamSize:1});
  const sim=combatPositions(env);sim.actors[1].health=15;
  const actions=idleActions(env);actions[4]=1;
  const result=env.step(actions);
  assert.deepEqual(result.rewards,[1,-1]);assert.deepEqual(result.done,[false,false]);
  assert.deepEqual(result.deaths,[0,1]);assert.deepEqual(result.alive,[1,0]);assert.deepEqual(result.episodes,[]);
  const move=idleActions(env);move[0]=2;
  assert.deepEqual(env.step(move).rewards,[0,0]);
});

test('the human actor receives the same randomized weapon spread as bots',()=>{
  const env=new SelfPlayVectorEnv({environments:1,teamSize:1});
  const sim=env.envs[0].sim,oldRng=sim.rngState;
  const actions=idleActions(env);actions[4]=1;
  env.step(actions);
  assert.equal(sim.player.shots,1);assert.notEqual(sim.rngState,oldRng);
});

test('autoreset marks all actors and preserves routing while reporting partial smoke rounds',()=>{
  const env=new SelfPlayVectorEnv({environments:2,teamSize:2,seconds:1,seedStart:1000000,seedCount:3});
  let result;
  for(let i=0;i<10;i++)result=env.step(idleActions(env));
  assert.deepEqual(result.done,Array(8).fill(true));assert.equal(result.episodes.length,2);
  assert.ok(result.episodes.every(episode=>episode.complete===false&&episode.seconds===1&&episode.winner===null));
  assert.ok(env.envs.every(entry=>entry.sim.tick===0));assert.deepEqual(env.envs.map(entry=>entry.sim.seed),[1000002,1000000]);
  assert.deepEqual(result.teams,[0,0,1,1,0,0,1,1]);assert.deepEqual(result.episodeIds,Array(8).fill(1));
  assert.deepEqual(result.rewards,Array(8).fill(0));assert.deepEqual(result.scalars.filter((_,index)=>index%6===0),Array(8).fill(1));
  assert.ok(env.envs.every(entry=>entry.actors.every(actor=>typeof entry.sim.controllers.get(actor.id)==='function')));
});

test('incomplete or invalid actions fail before any environment advances',()=>{
  const env=new SelfPlayVectorEnv({environments:2,teamSize:2});
  const labels=idleActions(env);
  for(const input of [labels.slice(0,-6),labels.concat(idle),null,{},labels.map((value,index)=>index===labels.length-1?99:value)]){
    assert.throws(()=>env.step(input),TypeError);assert.ok(env.envs.every(entry=>entry.sim.tick===0));
  }
  assert.throws(()=>new SelfPlayVectorEnv({navigationReward:.1}),/Strict self-play/);
  assert.throws(()=>new SelfPlayVectorEnv({explorationReward:.1}),/Strict self-play/);
  assert.throws(()=>new SelfPlayVectorEnv({seedStart:4294967295,seedCount:2}),/Seed range/);
  assert.throws(()=>new SelfPlayVectorEnv({rewardProfile:'other'}),/reward profile/);
});

const near=(actual,expected)=>assert.ok(Math.abs(actual-expected)<1e-10,`${actual} != ${expected}`);
const shaped=options=>new SelfPlayVectorEnv({environments:1,teamSize:1,rewardProfile:'shaped-v1',...options});
test('reward profiles are explicit and transition components sum to each actor reward',()=>{
  const env=shaped();combatPositions(env);
  const initial=env.observe();assert.equal(initial.rewardProfile,'shaped-v1');assert.deepEqual(initial.rewardSpec,REWARD_CONFIGS['shaped-v1']);
  assert.deepEqual(Object.keys(initial.rewardComponents),REWARD_COMPONENTS);
  const result=env.step(idleActions(env));
  for(let a=0;a<result.count;a++)near(result.rewards[a],REWARD_COMPONENTS.reduce((sum,key)=>sum+result.rewardComponents[key][a],0));
  assert.ok(result.rewardComponents.aim.every(value=>value>0));
});

test('crosshair quality uses nearest visible enemy pixel with a bounded proximity gradient',()=>{
  const pixels=new Uint8Array(64*36*3);
  const put=(x,y,color)=>pixels.set(color,(y*64+x)*3);
  assert.equal(enemyCrosshairQuality(pixels),0);
  put(31,17,SEMANTIC_PALETTE.ally);assert.equal(enemyCrosshairQuality(pixels),0);
  put(35,17,SEMANTIC_PALETTE.enemy);
  const edge=enemyCrosshairQuality(pixels);assert.ok(edge>0&&edge<1);
  put(34,17,SEMANTIC_PALETTE.enemy);assert.ok(enemyCrosshairQuality(pixels)>edge);
  put(31,17,SEMANTIC_PALETTE.enemy);assert.equal(enemyCrosshairQuality(pixels),1);
  put(32,18,SEMANTIC_PALETTE.enemy);assert.equal(enemyCrosshairQuality(pixels),1);
  pixels.fill(0);put(27,17,SEMANTIC_PALETTE.enemy);assert.equal(enemyCrosshairQuality(pixels),0);
});

test('aim feedback respects wall and ally occlusion and excludes dead or protected learner states',()=>{
  const env=shaped({teamSize:2});const sim=combatPositions(env);
  sim.map[2][4]='#';let result=env.step(idleActions(env));assert.equal(result.rewardComponents.aim[0],0);
  sim.map[2][4]='.';Object.assign(sim.actors[1],{x:4,y:2.5});
  result=env.step(idleActions(env));assert.equal(result.rewardComponents.aim[0],0);
  Object.assign(sim.actors[1],{x:2.5,y:5.5});sim.player.spawnProtection=.05;
  result=env.step(idleActions(env));assert.equal(result.rewardComponents.aim[0],0);
  Object.assign(sim.player,{active:false,health:0,respawnTimer:2});
  result=env.step(idleActions(env));assert.equal(result.rewardComponents.aim[0],0);
});

test('aim rewards use actual elapsed time, cap each life, and reset after death or respawn',()=>{
  const env=shaped();const sim=combatPositions(env);
  let result=env.step(idleActions(env));near(result.rewardComponents.aim[0],.002);
  let accumulated=result.rewardComponents.aim[0];
  for(let i=0;i<120;i++){result=env.step(idleActions(env));accumulated+=result.rewardComponents.aim[0];}
  near(accumulated,.2);assert.equal(result.rewardComponents.aim[0],0);
  sim.player.deaths++;Object.assign(sim.player,{active:false,health:0,respawnTimer:.01});
  result=env.step(idleActions(env));assert.equal(result.rewardComponents.aim[0],0);assert.equal(env.envs[0].aimSpent[0],0);
  // Returning to combat after the new life's protection expires has a new aim budget.
  combatPositions(env);result=env.step(idleActions(env));near(result.rewardComponents.aim[0],.002);
  const short=shaped({seconds:1});const shortSim=combatPositions(short);shortSim.tick=59;shortSim.time=59/60;
  result=short.step(idleActions(short));near(result.rewardComponents.aim[0],.02/60);assert.equal(result.done[0],true);
});

test('valid hits reward actual enemy health removed and penalize health loss symmetrically',()=>{
  const env=shaped();const sim=combatPositions(env);const labels=idleActions(env);labels[4]=1;
  const result=env.step(labels);
  assert.equal(sim.actors[1].health,85);
  near(result.rewardComponents.damageDealt[0],.15);near(result.rewardComponents.damageTaken[1],-.15);
  assert.deepEqual(result.rewardComponents.kills,[0,0]);assert.deepEqual(result.rewardComponents.deaths,[0,0]);
  near(result.rewardComponents.damageDealt.reduce((a,b)=>a+b,0)+result.rewardComponents.damageTaken.reduce((a,b)=>a+b,0),0);
});

test('overkill is clipped and kills, deaths, team score, and wins have distinct rewards',()=>{
  const env=shaped({scoreLimit:1});const sim=combatPositions(env);sim.actors[1].health=4.5;
  const labels=idleActions(env);labels[4]=1;const result=env.step(labels),c=result.rewardComponents;
  near(c.damageDealt[0],.045);near(c.damageTaken[1],-.045);
  assert.deepEqual(c.kills,[1,0]);assert.deepEqual(c.deaths,[0,-1]);
  assert.deepEqual(c.teamScore,[.25,-.25]);assert.deepEqual(c.outcome,[5,-5]);
  near(result.rewards[0],6.295);near(result.rewards[1],-6.295);
  assert.equal(result.episodes[0].rewardProfile,'shaped-v1');
  assert.equal(result.episodes[0].teams[0].rewardComponents.kills,1);
  assert.equal(result.episodes[0].teams[1].rewardComponents.deaths,-1);
  near(result.episodes[0].rewardComponents.damageDealt,.045);
});

test('misses, friendly obstruction, and protected targets earn no damage reward; regeneration is not damage',()=>{
  for(const obstacle of ['wall','ally','protected']){
    const env=shaped({teamSize:2});const sim=combatPositions(env);const labels=idleActions(env);labels[4]=1;
    if(obstacle==='wall')sim.map[2][4]='#';
    if(obstacle==='ally')Object.assign(sim.actors[1],{x:4,y:2.5});
    if(obstacle==='protected')sim.actors[2].spawnProtection=1;
    const result=env.step(labels);
    assert.ok(result.rewardComponents.damageDealt.every(value=>value===0));
    assert.ok(result.rewardComponents.damageTaken.every(value=>value===0));
  }
  const env=shaped();const sim=combatPositions(env);sim.player.health=40;sim.player.lastDamageTick=-300;
  const result=env.step(idleActions(env));assert.ok(sim.player.health>40);
  assert.deepEqual(result.rewardComponents.damageDealt,[0,0]);assert.deepEqual(result.rewardComponents.damageTaken,[0,0]);
});

test('shaped-v2 changes only the immutable draw reward and preserves earlier profiles',()=>{
  assert.deepEqual(REWARD_CONFIGS['shaped-v2'],{...REWARD_CONFIGS['shaped-v1'],profile:'shaped-v2',draw:-.5});
  assert.ok(Object.isFrozen(REWARD_CONFIGS['shaped-v2']));
  assert.equal(REWARD_CONFIGS['shaped-v1'].draw,0);assert.equal(REWARD_CONFIGS['sparse-v1'].draw,0);
});

test('a complete timed draw penalizes every player exactly once under shaped-v2',()=>{
  const env=shaped({rewardProfile:'shaped-v2',teamSize:2});const sim=env.envs[0].sim;
  sim.tick=10794;sim.time=sim.tick/60;
  const result=env.step(idleActions(env));
  assert.equal(result.episodes[0].complete,true);assert.equal(result.episodes[0].winner,'draw');assert.equal(result.episodes[0].seconds,180);
  assert.deepEqual(result.done,[true,true,true,true]);
  assert.deepEqual(result.rewardComponents.outcome,[-.5,-.5,-.5,-.5]);
  assert.deepEqual(result.rewards,[-.5,-.5,-.5,-.5]);
  assert.equal(result.episodes[0].rewardComponents.outcome,-2);
  assert.deepEqual(env.step(idleActions(env)).rewardComponents.outcome,[0,0,0,0]);
});

test('earlier-profile draws and partial smoke cutoffs retain zero outcome reward',()=>{
  for(const rewardProfile of ['sparse-v1','shaped-v1']){
    const env=shaped({rewardProfile});const sim=env.envs[0].sim;sim.tick=10794;sim.time=sim.tick/60;
    const result=env.step(idleActions(env));assert.equal(result.episodes[0].winner,'draw');
    assert.deepEqual(result.rewardComponents.outcome,[0,0]);
  }
  const env=shaped({rewardProfile:'shaped-v2',seconds:1});const sim=env.envs[0].sim;sim.tick=54;sim.time=.9;
  const result=env.step(idleActions(env));assert.equal(result.episodes[0].complete,false);assert.equal(result.episodes[0].winner,null);
  assert.deepEqual(result.rewardComponents.outcome,[0,0]);
});

test('shaped-v2 decisive wins and losses still pay plus or minus five',()=>{
  for(const score of [[1,0],[0,1]]){
    const env=shaped({rewardProfile:'shaped-v2'});const sim=env.envs[0].sim;
    sim.tick=10794;sim.time=sim.tick/60;sim.scores=score;
    const result=env.step(idleActions(env));
    assert.deepEqual(result.rewardComponents.outcome,score[0]>score[1]?[5,-5]:[-5,5]);
  }
});
