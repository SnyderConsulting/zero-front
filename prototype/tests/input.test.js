import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
import {createSimulation,reset,step,observe,setController} from '../dist/simulation.js';
import {renderFrame} from '../dist/renderer.js';
import {ARENA_LAYOUTS} from '../dist/mapgen.js';

// Execute the actual input controller against a small DOM/event shell. Simulation
// and rendering remain real; only browser timing, focus, and input are supplied.
function game({desktop=false,pointerLock=true}={}){
  class Target{
    listeners=new Map();
    addEventListener(name,listener){
      if(!this.listeners.has(name))this.listeners.set(name,[]);
      this.listeners.get(name).push(listener);
    }
    dispatch(name,properties={}){
      const event={target:this,preventDefault(){},...properties};
      for(const listener of this.listeners.get(name)||[])listener(event);
    }
  }
  const classes=()=>({add(){},remove(){},toggle(){}});
  const document=new Target(),window=new Target(),elements=new Map();
  let postRound;
  let roundModels;
  const createRoundModels=()=>roundModels={
    ready:true,selected:[null,null],refreshCount:0,
    refresh(){this.refreshCount++;},controllers(){return this.selected;},
  };
  const createPostRound=({onNewMatch})=>postRound={
    isOpen:false,isReplaying:false,recording:null,keyCount:0,
    show(recording){this.recording=recording;this.isOpen=true;this.isReplaying=false;},
    hide(){this.isOpen=false;this.isReplaying=false;},
    update(){},resize(){},handleKey(){this.keyCount++;},
    handleEscape(){this.isReplaying=false;},
    newMatch:onNewMatch,
  };
  let focused=true,now=1000,nextFrame,nextSeed=100;
  function element(id){
    if(!elements.has(id))elements.set(id,Object.assign(new Target(),{
      id,tagName:id==='viewport'?'CANVAS':'DIV',hidden:false,textContent:'',
      value:id==='team-size'?'1':'',classList:classes(),firstChild:{textContent:''},
      focus(){document.activeElement=this;},setPointerCapture(){},
    }));
    return elements.get(id);
  }
  const canvas=element('viewport');
  canvas.getContext=()=>({putImageData(){}});
  canvas.requestPointerLock=()=>{
    if(!pointerLock)throw new Error('Mouse capture unavailable');
    document.pointerLockElement=canvas;
    document.dispatch('pointerlockchange');
    return Promise.resolve();
  };
  Object.assign(document,{
    body:{classList:classes()},hidden:false,pointerLockElement:null,
    hasFocus:()=>focused,getElementById:element,querySelector:element,
    querySelectorAll:()=>[],
    exitPointerLock(){document.pointerLockElement=null;document.dispatch('pointerlockchange');},
  });
  Object.assign(window,{innerWidth:1280,innerHeight:720,crypto:{getRandomValues(values){values[0]=nextSeed;nextSeed+=ARENA_LAYOUTS.length;return values;}},
    location:{search:desktop?'?desktop=1':''}});
  const context=vm.createContext({window,document,URLSearchParams,
    createSimulation,reset,step,observe,setController,renderFrame,ARENA_LAYOUTS,createPostRound,createRoundModels,
    ImageData:class{constructor(data,width,height){Object.assign(this,{data,width,height});}},
    performance:{now:()=>now},requestAnimationFrame:callback=>{nextFrame=callback;},
  });
  const source=readFileSync(new URL('../dist/app.js',import.meta.url),'utf8')
    .replace(/^import .+;\r?\n/gm,'');
  vm.runInContext(source,context,{filename:'app.js'});
  nextFrame(now); // Establish the animation clock before injecting input.
  return{
    canvas,document,window,postRound,roundModels,element,
    start(){element('play').dispatch('click');},
    newMatch(){element('new-match').dispatch('click');},
    observation:()=>window.FPSLab.observe().observation,
    frames(count){for(let i=0;i<count;i++){now+=1000/60+.001;nextFrame(now);}},
    mouseDown(button=0,{first=true}={}){
      if(first)canvas.dispatch('pointerdown',{pointerType:'mouse',pointerId:1,button});
      canvas.dispatch('mousedown',{button});
    },
    mouseUp(button=0,{last=true}={}){
      window.dispatch('mouseup',{button});
      if(last)window.dispatch('pointerup',{pointerType:'mouse',pointerId:1,button});
    },
    blur(){focused=false;window.dispatch('blur');},
    nativeFocus(){focused=true;window.dispatch('arena-desktop-focus');},
  };
}

test('quick LMB click between ticks fires once even without mouse capture',()=>{
  const g=game({pointerLock:false});g.start();
  g.mouseDown();g.mouseUp();
  assert.equal(g.observation().ammo,20);
  g.frames(1);assert.equal(g.observation().ammo,19);
  g.frames(15);assert.equal(g.observation().ammo,19);
  assert.equal(g.observation().shots,1);
});

test('holding LMB repeats fire and releasing it stops fire',()=>{
  const g=game();g.start();g.mouseDown();g.frames(24);
  const shots=g.observation().shots;
  assert.ok(shots>=2);assert.equal(g.observation().ammo,20-shots);
  g.mouseUp();g.frames(12);
  assert.equal(g.observation().shots,shots);
});

test('LMB fires independently while RMB is held and after RMB is released',()=>{
  const g=game({pointerLock:false});g.start();
  g.mouseDown(2);g.mouseDown(0,{first:false});g.frames(1);
  assert.equal(g.observation().shots,1);
  g.mouseUp(2,{last:false});g.frames(12);
  assert.equal(g.observation().shots,2);
  g.mouseUp(0);g.frames(12);
  assert.equal(g.observation().shots,2);
});

test('Space remains a firing control',()=>{
  const g=game();g.start();
  g.document.dispatch('keydown',{code:'Space',target:g.canvas});g.frames(1);
  assert.equal(g.observation().ammo,19);
  g.document.dispatch('keyup',{code:'Space'});g.frames(12);
  assert.equal(g.observation().ammo,19);
});

test('desktop blur releases capture, pauses, and resumes without stuck fire on native focus',async()=>{
  const g=game({desktop:true});g.start();
  await new Promise(resolve=>setImmediate(resolve));
  assert.equal(g.document.pointerLockElement,g.canvas);
  g.mouseDown();g.frames(1);g.blur();
  const paused=g.observation();
  assert.equal(g.document.pointerLockElement,null);
  g.window.dispatch('arena-desktop-blur'); // Duplicate loss notifications preserve resume intent.
  g.frames(12);assert.equal(g.observation().tick,paused.tick);
  g.nativeFocus();g.frames(12);
  assert.equal(g.document.pointerLockElement,g.canvas);
  assert.ok(g.observation().tick>paused.tick);
  assert.equal(g.observation().shots,paused.shots);
});

test('explicit native pause stays paused when desktop focus returns',async()=>{
  const g=game({desktop:true});g.start();
  await new Promise(resolve=>setImmediate(resolve));
  g.frames(1);g.window.dispatch('arena-desktop-pause');
  const tick=g.observation().tick;
  g.blur();g.nativeFocus();g.frames(12);
  assert.equal(g.observation().tick,tick);
  assert.equal(g.document.pointerLockElement,null);
});

test('new matches change layout family even when random seeds collide, while resume keeps the map',()=>{
  const g=game();g.start();
  const firstSeed=g.observation().seed;
  g.document.dispatch('keydown',{code:'Escape',target:g.canvas});g.start();
  assert.equal(g.observation().seed,firstSeed);
  g.newMatch();assert.equal(g.observation().seed,firstSeed);
  assert.equal(g.element('round-config').hidden,false);
  g.start();assert.notEqual(g.observation().seed,firstSeed);
  assert.notEqual(g.observation().seed%ARENA_LAYOUTS.length,firstSeed%ARENA_LAYOUTS.length);
  assert.equal(g.observation().tick,0);
});

test('completed rounds open results with complete replay and do not recapture on desktop focus',async()=>{
  const g=game({desktop:true});g.start();
  await new Promise(resolve=>setImmediate(resolve));
  g.window.FPSLab.reset({seed:42,teamSize:2,scoreLimit:1});
  for(let i=0;i<18&&!g.observation().terminated&&!g.observation().truncated;i++)g.window.FPSLab.step({},600);
  assert.ok(g.observation().terminated||g.observation().truncated);
  assert.equal(g.postRound.isOpen,true);
  assert.equal(g.postRound.recording.frameCount,g.observation().tick+1);
  assert.equal(g.postRound.recording.summary.players.length,4);
  assert.equal(g.document.pointerLockElement,null);
  const finalTick=g.observation().tick;
  g.blur();g.nativeFocus();g.frames(2);
  assert.equal(g.observation().tick,finalTick);
  assert.equal(g.postRound.isOpen,true);assert.equal(g.document.pointerLockElement,null);
  g.postRound.isReplaying=true;
  g.document.dispatch('keydown',{code:'Space',target:g.canvas});
  assert.equal(g.postRound.keyCount,1);
  g.window.dispatch('arena-desktop-pause');
  assert.equal(g.postRound.isOpen,true);assert.equal(g.postRound.isReplaying,false);
  const lastReplay=g.postRound.recording;
  g.postRound.newMatch();g.frames(1);
  assert.equal(g.postRound.isOpen,false);assert.equal(g.observation().tick,finalTick);
  assert.equal(g.element('round-config').hidden,false);assert.equal(g.document.pointerLockElement,null);
  g.start();g.frames(1);
  assert.ok(g.observation().tick<finalTick);
  assert.equal(lastReplay.frameCount,finalTick+1);
  assert.equal(g.document.pointerLockElement,g.canvas);
});

test('native focus leaves round setup idle and selected models must load before Play',()=>{
  const g=game({desktop:true});g.nativeFocus();g.frames(2);
  assert.equal(g.observation().tick,0);assert.equal(g.document.pointerLockElement,null);
  assert.equal(g.element('round-config').hidden,false);
  g.roundModels.ready=false;g.start();g.frames(2);
  assert.equal(g.observation().tick,0);assert.equal(g.document.pointerLockElement,null);
  g.roundModels.ready=true;g.start();g.frames(2);
  assert.ok(g.observation().tick>0);assert.equal(g.document.pointerLockElement,g.canvas);
});

test('round model selections control their bot teams, preserve the human, and change only on a new round',()=>{
  const g=game(),blueCalls=[],redCalls=[],nextCalls=[];
  g.element('team-size').value='2';
  g.roundModels.selected=[(sim,actor)=>{blueCalls.push(actor.id);return{};},(sim,actor)=>{redCalls.push(actor.id);return{};}];
  g.start();g.frames(2);
  assert.deepEqual([...new Set(blueCalls)],['blue-2']);
  assert.deepEqual([...new Set(redCalls)],['red-1','red-2']);
  assert.equal(g.observation().shots,0);
  g.document.dispatch('keydown',{code:'Escape',target:g.canvas});
  g.roundModels.selected=[null,(sim,actor)=>{nextCalls.push(actor.id);return{};}];
  const prior=redCalls.length;g.start();g.frames(1);
  assert.ok(redCalls.length>prior);assert.equal(nextCalls.length,0);
  g.newMatch();g.start();g.frames(1);
  assert.deepEqual([...new Set(nextCalls)],['red-1','red-2']);
});
