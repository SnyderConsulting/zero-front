import test from 'node:test';
import assert from 'node:assert/strict';
import {fitReplayCamera,replayWorldAt,zoomReplayCamera} from '../dist/replay-view.js';

const near=(a,b)=>assert.ok(Math.abs(a-b)<1e-9,`${a} != ${b}`);
test('fit camera includes the whole map and centers it',()=>{
  const camera=fitReplayCamera(56,32,900,500);
  near(camera.x,28);near(camera.y,16);
  assert.ok(camera.scale*56<=852);assert.ok(camera.scale*32<=452);
});
test('zoom remains anchored at the pointer and stays within its scale bounds',()=>{
  const camera=fitReplayCamera(56,32,900,500),before=replayWorldAt(camera,600,300,900,500);
  const zoomed=zoomReplayCamera(camera,2,600,300,900,500,56,32);
  const after=replayWorldAt(zoomed,600,300,900,500);
  near(before.x,after.x);near(before.y,after.y);
  assert.equal(zoomReplayCamera(camera,1e6,450,250,900,500,56,32).scale,camera.maxScale);
  assert.equal(zoomReplayCamera(camera,1e-6,450,250,900,500,56,32).scale,camera.minScale);
});
