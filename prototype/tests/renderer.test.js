import test from 'node:test';
import assert from 'node:assert/strict';
import { renderFrame, SEMANTIC_PALETTE } from '../dist/renderer.js';

function fixture() {
  return {
    map: ['#######','#.....#','#.....#','#..#..#','#.....#','#.....#','#######'],
    width:7,height:7,player:{x:1.5,y:3.5,yaw:0,pitch:0,team:0,health:100,ammo:12,cooldown:0,active:true},entities:[],tick:0,time:0,
  };
}

test('frames have the requested RGBA shape with opaque pixels', () => {
  const sim=fixture();
  for(const [width,height] of [[160,90],[320,180],[640,360]]) {
    const frame=renderFrame(sim,{width,height});
    assert.equal(frame.length,width*height*4);
    for(let i=3;i<frame.length;i+=4) assert.equal(frame[i],255);
  }
});

test('unchanged simulation produces exactly identical observations', () => {
  const sim=fixture();
  assert.deepEqual(renderFrame(sim),renderFrame(sim));
});

test('a combatant fully occluded by a wall does not leak into pixels or labels', () => {
  const sim=fixture();
  const plain=renderFrame(sim), labels=renderFrame(sim,{semantic:true});
  sim.entities.push({id:'hidden',type:'enemy',team:1,x:4.5,y:3.5,health:100,active:true,radius:.34,height:1.6});
  assert.deepEqual(renderFrame(sim),plain);
  assert.deepEqual(renderFrame(sim,{semantic:true}),labels);
});

test('visible combatant adds foreground pixels and dead combatant disappears', () => {
  const sim=fixture();
  const plain=renderFrame(sim,{semantic:true});
  sim.entities.push({id:'visible',type:'enemy',team:1,x:2.5,y:3.5,health:100,active:true,radius:.34,height:1.6});
  const withTarget=renderFrame(sim,{semantic:true});
  assert.notDeepEqual(withTarget,plain);
  let count=0;
  const color=SEMANTIC_PALETTE.enemy;
  for(let i=0;i<withTarget.length;i+=4) if(withTarget[i]===color[0]&&withTarget[i+1]===color[1]&&withTarget[i+2]===color[2]) count++;
  assert.ok(count>20);
  sim.entities[0].active=false;
  assert.deepEqual(renderFrame(sim,{semantic:true}),plain);
});

test('both RGB and semantic frames distinguish allies from enemies', () => {
  const sim=fixture();
  const actor={id:'fighter',type:'enemy',team:0,x:2.5,y:3.5,health:100,active:true,radius:.34,height:1.6,yaw:Math.PI};
  sim.entities.push(actor);
  const allyRGB=renderFrame(sim), allyLabels=renderFrame(sim,{semantic:true});
  actor.team=1;
  const enemyRGB=renderFrame(sim), enemyLabels=renderFrame(sim,{semantic:true});
  assert.notDeepEqual(allyRGB,enemyRGB);
  assert.notDeepEqual(allyLabels,enemyLabels);
  const count=(frame,color)=>{let n=0;for(let i=0;i<frame.length;i+=4)if(frame[i]===color[0]&&frame[i+1]===color[1]&&frame[i+2]===color[2])n++;return n;};
  assert.ok(count(allyLabels,SEMANTIC_PALETTE.ally)>20);
  assert.equal(count(allyLabels,SEMANTIC_PALETTE.enemy),0);
  assert.ok(count(enemyLabels,SEMANTIC_PALETTE.enemy)>20);
});

test('fire and hit feedback change visible frames without contaminating labels', () => {
  const sim=fixture(), original=renderFrame(sim), labels=renderFrame(sim,{semantic:true});
  sim.player.muzzleFlash=.08;
  assert.notDeepEqual(renderFrame(sim),original);
  assert.deepEqual(renderFrame(sim,{semantic:true}),labels);
  sim.player.muzzleFlash=0;
  sim.player.hitMarker=.12;
  assert.notDeepEqual(renderFrame(sim),original);
  assert.deepEqual(renderFrame(sim,{semantic:true}),labels);
});

test('invalid resolution cannot allocate unbounded observation buffers', () => {
  for(const width of [0,NaN,320.5,999999]) assert.throws(()=>renderFrame(fixture(),{width}),RangeError);
});
