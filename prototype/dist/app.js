import {createSimulation,reset,step,observe,setController} from './simulation.js';
import {renderFrame} from './renderer.js';
import {ARENA_LAYOUTS} from './mapgen.js';
import {createPostRound} from './post-round.js';
import {createRoundModels} from './round-models.js';

const $=id=>document.getElementById(id);
const canvas=$('viewport'),ctx=canvas.getContext('2d',{alpha:false});
const sim=createSimulation({seed:42,teamSize:4,scoreLimit:25,recordReplay:true});
const postRound=createPostRound({onNewMatch:()=>configureRound()});
const keys=new Set(),touch=new Set();
const desktop=new URLSearchParams(window.location.search).get('desktop')==='1';
let mode='ready',mx=0,my=0,trigger=false,fireQueued=false,dragging=false,lastTouch=null,aimPointer=null;
let resumeOnFocus=false,capturePending=false;
let previous=0,accumulator=0,lastDraw=0,lastHUD=0,drawnTick=-1,startedAt=0;
let renderWidth=320,renderHeight=180,audio=null;
const clamp=(n,a,b)=>Math.max(a,Math.min(b,n));
const roundModels=createRoundModels({onStateChange:()=>{$('play').disabled=mode==='ready'&&!roundModels.ready;}});

function clearInput(){keys.clear();touch.clear();mx=my=0;trigger=fireQueued=dragging=false;lastTouch=null;aimPointer=null;}
function releaseMouse(){if(document.pointerLockElement===canvas)document.exitPointerLock?.();}
function resize(){
  const aspect=window.innerWidth/window.innerHeight;
  renderWidth=320;renderHeight=Math.round(renderWidth/aspect);
  if(renderHeight>720){renderHeight=720;renderWidth=clamp(Math.round(720*aspect),64,1280);}
  renderHeight=clamp(renderHeight,36,720);
  canvas.width=renderWidth;canvas.height=renderHeight;draw();postRound.resize();
}
function draw(){
  ctx.putImageData(new ImageData(renderFrame(sim,{width:renderWidth,height:renderHeight}),renderWidth,renderHeight),0,0);
  drawnTick=sim.tick;
}
function showMenu(){
  const ended=sim.terminated||sim.truncated;
  if(mode!=='ready'&&ended&&sim.replay?.summary){
    $('menu').hidden=true;document.body.classList.add('menu-open');$('respawn').hidden=true;
    if(!postRound.isOpen)postRound.show(sim.replay);
    return;
  }
  postRound.hide();
  $('menu').hidden=false;document.body.classList.add('menu-open');
  $('new-match').hidden=mode==='ready'||ended;
  $('round-config').hidden=mode!=='ready';
  $('play').textContent=mode==='ready'?'Play':'Resume';
  $('play').disabled=mode==='ready'&&!roundModels.ready;
  $('menu-title').textContent='ZERO FRONT';
  $('menu-subtitle').textContent=mode==='ready'?"You're blue. First team to 25 wins.":'Round paused';
  $('respawn').hidden=true;
  if(mode==='ready')roundModels.refresh();
}
function updateHUD(){
  const p=sim.player;
  $('blue-score').textContent=sim.scores[0];$('red-score').textContent=sim.scores[1];
  $('match-target').textContent=`FIRST TO ${sim.scoreLimit}`;
  const remaining=Math.max(0,Math.ceil(observe(sim).timeLimit-sim.time));
  $('round-timer').textContent=`${Math.floor(remaining/60)}:${String(remaining%60).padStart(2,'0')}`;
  $('round-timer').classList.toggle('urgent',remaining<=30);
  $('health').textContent=Math.ceil(p.health);document.querySelector('.health').classList.toggle('low',p.health<30);
  const reloading=p.reloadTimer>0;
  $('ammo').textContent=reloading?'RELOADING':p.ammo;
  document.querySelector('.ammo').classList.toggle('reloading',reloading);
  $('ammo-total').hidden=reloading;
  $('reload-hint').hidden=p.ammo>0||reloading||!p.active||mode!=='human';
  $('damage').classList.toggle('active',p.damageFlash>0&&mode==='human');
  $('respawn').hidden=p.active||mode!=='human';
  $('respawn-count').textContent=`Respawning in ${Math.max(1,Math.ceil(p.respawnTimer||0))}`;
  $('controls-hint').hidden=mode!=='human'||(document.pointerLockElement===canvas&&performance.now()-startedAt>8000);
}
function setMode(next){
  if(!['human','paused','stepped'].includes(next))throw new TypeError('Mode must be human, paused, or stepped.');
  mode=next;resumeOnFocus=desktop&&next==='human';accumulator=0;clearInput();
  if(next!=='human')releaseMouse();
  if(next==='human'||next==='stepped'){postRound.hide();$('menu').hidden=true;document.body.classList.remove('menu-open');}
  else {releaseMouse();showMenu();}
  updateHUD();return {mode};
}
function capture(){
  if(mode!=='human'||!document.hasFocus())return;
  canvas.focus({preventScroll:true});
  if(document.pointerLockElement===canvas||capturePending)return;
  const fallback=()=>{$('controls-hint').textContent='WASD move · Right-drag / arrows aim · LMB / Space fire · R reload · Esc pause';};
  try{
    if(!canvas.requestPointerLock){fallback();return;}
    capturePending=true;
    Promise.resolve(canvas.requestPointerLock()).catch(fallback).finally(()=>{capturePending=false;});
  }catch{capturePending=false;fallback();}
}
function start(){
  if(mode==='ready'||sim.terminated||sim.truncated){
    if(!roundModels.ready)return;
    newMatch(false);
  }
  setMode('human');startedAt=performance.now();capture();
  try{audio??=new (window.AudioContext||window.webkitAudioContext)();audio.resume().catch(()=>{});}catch{}
}
function configureRound(){
  mode='ready';resumeOnFocus=false;accumulator=0;clearInput();releaseMouse();postRound.hide();
  showMenu();updateHUD();$('play').focus({preventScroll:true});
}
function newMatch(play=true){
  const controllers=roundModels.controllers();
  postRound.hide();
  const values=new Uint32Array(1);
  let seed=window.crypto?.getRandomValues?window.crypto.getRandomValues(values)[0]:Math.floor(Math.random()*4294967296);
  if(seed===sim.seed)seed=(seed+1)>>>0;
  // Human matches alternate overall structures; explicit agent seeds stay untouched.
  if(ARENA_LAYOUTS.length>1)while(seed%ARENA_LAYOUTS.length===sim.seed%ARENA_LAYOUTS.length)seed=(seed+1)>>>0;
  reset(sim,{seed,teamSize:Number($('team-size').value),scoreLimit:25,recordReplay:true});
  for(const actor of sim.actors)if(actor!==sim.player&&controllers[actor.team])setController(sim,actor.id,controllers[actor.team]);
  clearInput();accumulator=0;draw();updateHUD();
  if(play){setMode('human');startedAt=performance.now();capture();}
}
function gunSound(hit){
  if(!audio||audio.state!=='running')return;
  const t=audio.currentTime,osc=audio.createOscillator(),gain=audio.createGain();
  osc.type='sawtooth';osc.frequency.setValueAtTime(hit?210:145,t);osc.frequency.exponentialRampToValueAtTime(42,t+.07);
  gain.gain.setValueAtTime(.055,t);gain.gain.exponentialRampToValueAtTime(.001,t+.08);
  osc.connect(gain);gain.connect(audio.destination);osc.start(t);osc.stop(t+.085);
}
function action(){
  const a={
    forward:Number(keys.has('KeyW')||touch.has('forward'))-Number(keys.has('KeyS')||touch.has('back')),
    strafe:Number(keys.has('KeyD')||touch.has('right'))-Number(keys.has('KeyA')||touch.has('left')),
    turn:clamp((Number(keys.has('ArrowRight')||touch.has('turnRight'))-Number(keys.has('ArrowLeft')||touch.has('turnLeft')))*2.4+mx*.0023*60,-12,12),
    pitch:clamp((Number(keys.has('ArrowUp'))-Number(keys.has('ArrowDown')))*1.3-my*.0023*60,-6,6),
    fire:trigger||fireQueued||keys.has('Space')||touch.has('fire'),reload:keys.has('KeyR')||touch.has('reload')
  };mx=my=0;fireQueued=false;return a;
}
function advance(a,ticks=1){
  const beforeShots=sim.player.shots,beforeHits=sim.player.hits;
  const result=step(sim,a,ticks);
  if(mode==='human'&&sim.player.shots>beforeShots)gunSound(sim.player.hits>beforeHits);
  if(sim.terminated||sim.truncated){mode='paused';resumeOnFocus=false;clearInput();releaseMouse();showMenu();}
  return result;
}
function frame(now){
  const dt=previous?Math.min((now-previous)/1000,.15):0;previous=now;
  if(mode==='human'){
    accumulator+=dt;
    while(accumulator>=1/60&&mode==='human'){advance(action());accumulator-=1/60;}
  }
  if(drawnTick!==sim.tick&&now-lastDraw>=1000/60){draw();lastDraw=now;}
  if(now-lastHUD>80){updateHUD();lastHUD=now;}
  postRound.update(dt);
  requestAnimationFrame(frame);
}
$('play').addEventListener('click',start);
$('new-match').addEventListener('click',configureRound);
canvas.addEventListener('pointerdown',e=>{
  if(mode!=='human')return;
  dragging=true;aimPointer=e.pointerId;canvas.focus({preventScroll:true});
  if(e.pointerType==='touch'){lastTouch={x:e.clientX,y:e.clientY};canvas.setPointerCapture(e.pointerId);}
  else if(document.pointerLockElement!==canvas)capture();
});
// Mouse buttons have independent down/up events, including LMB while RMB is held.
// Keep a short click until the next simulation tick instead of losing it on release.
canvas.addEventListener('mousedown',e=>{
  if(mode==='human'&&e.button===0){e.preventDefault();trigger=true;fireQueued=true;}
});
window.addEventListener('mouseup',e=>{if(e.button===0)trigger=false;});
function releasePointer(e){if(e.pointerId!==aimPointer)return;dragging=false;lastTouch=null;aimPointer=null;}
window.addEventListener('pointerup',releasePointer);
window.addEventListener('pointercancel',e=>{releasePointer(e);if(e.pointerType==='mouse')trigger=fireQueued=false;});
window.addEventListener('mousemove',e=>{if(mode==='human'&&(document.pointerLockElement===canvas||dragging)){mx+=e.movementX;my+=e.movementY;}});
canvas.addEventListener('pointermove',e=>{if(e.pointerType==='touch'&&dragging){if(lastTouch){mx+=e.clientX-lastTouch.x;my+=e.clientY-lastTouch.y;}lastTouch={x:e.clientX,y:e.clientY};}});
canvas.addEventListener('contextmenu',e=>e.preventDefault());
function pauseForFocusLoss(){
  const resume=desktop&&(mode==='human'||resumeOnFocus);
  if(mode==='human')setMode('paused');
  clearInput();resumeOnFocus=resume;
}
function pauseManually(){
  resumeOnFocus=false;
  if(postRound.isOpen){postRound.handleEscape();return;}
  if(mode==='human')setMode('paused');
}
document.addEventListener('pointerlockchange',()=>{
  if(desktop)console.info(`[Zero Front] Mouse ${document.pointerLockElement===canvas?'captured':'released'}`);
  if(document.pointerLockElement===canvas){if(mode!=='human')releaseMouse();}
  else if(mode==='human')pauseForFocusLoss();
});
document.addEventListener('keydown',e=>{
  if(e.code==='Escape'){pauseManually();return;}
  if(postRound.isOpen){postRound.handleKey(e);return;}
  if(mode!=='human'||/INPUT|SELECT|BUTTON/.test(e.target.tagName))return;
  if(['KeyW','KeyA','KeyS','KeyD','ArrowLeft','ArrowRight','ArrowUp','ArrowDown','Space','KeyR'].includes(e.code)){e.preventDefault();keys.add(e.code);}
});
document.addEventListener('keyup',e=>keys.delete(e.code));
window.addEventListener('blur',pauseForFocusLoss);
document.addEventListener('visibilitychange',()=>{if(document.hidden)pauseForFocusLoss();});
if(desktop){
  window.addEventListener('arena-desktop-blur',pauseForFocusLoss);
  window.addEventListener('arena-desktop-pause',pauseManually);
  window.addEventListener('arena-desktop-focus',()=>{
    if(resumeOnFocus&&mode==='paused'&&!sim.terminated&&!sim.truncated)start();
    else if(mode==='human')capture();
  });
}
document.querySelectorAll('[data-control]').forEach(b=>{
  b.addEventListener('pointerdown',e=>{e.preventDefault();touch.add(b.dataset.control);b.setPointerCapture(e.pointerId);});
  for(const name of ['pointerup','pointercancel','lostpointercapture'])b.addEventListener(name,()=>touch.delete(b.dataset.control));
});
window.addEventListener('resize',resize);

// Agent access stays in code; it adds no controls to the game screen.
window.FPSLab=Object.freeze({
  version:'0.2.0',setMode,
  reset(options={}){reset(sim,options);setMode('stepped');$('team-size').value=sim.teamSize;draw();updateHUD();return observe(sim);},
  step(a={},ticks=1){const result=step(sim,a,ticks);setMode(sim.terminated||sim.truncated?'paused':'stepped');draw();updateHUD();return {observation:observe(sim),...result};},
  observe({pixels=false,semantic=false}={}){
    const observation=observe(sim);if(!pixels)return {observation};
    const width=320,height=180,rgba=renderFrame(sim,{width,height,semantic}),rgb=new Uint8Array(width*height*3);
    for(let i=0,j=0;i<rgba.length;i+=4){rgb[j++]=rgba[i];rgb[j++]=rgba[i+1];rgb[j++]=rgba[i+2];}
    return {observation,pixels:{width,height,format:'rgb8',semantic:Boolean(semantic),data:rgb}};
  }
});
resize();updateHUD();showMenu();requestAnimationFrame(frame);
