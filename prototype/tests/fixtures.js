import {createSimulation} from '../dist/simulation.js';

// Combat rules should not depend on a particular procedural seed's geometry.
// A clear firing lane at y=11.5 and a known wall at x=7,y=4..6 let tests
// exercise actual movement/hitscan without mocking collision or damage.
export function combatFixture(options={}){
  const sim=createSimulation(options),width=32,height=24;
  const grid=Array.from({length:height},(_,y)=>Array.from({length:width},(_,x)=>
    x===0||y===0||x===width-1||y===height-1?'#':'.'));
  for(let y=4;y<7;y++)for(let x=7;x<9;x++)grid[y][x]='1';
  const blue=[[3.5,11.5],[4.5,8.5],[4.5,15.5],[3.5,18.5],[3.5,5.5],[5.5,10.5],[5.5,13.5],[4.5,20.5]]
    .map(([x,y])=>({x,y}));
  const spawns=[blue,blue.map(({x,y})=>({x:width-x,y}))];
  Object.assign(sim,{map:grid.map(row=>row.join('')),width,height,spawns,rooms:[]});
  for(const actor of sim.actors){
    Object.assign(actor,spawns[actor.team][actor.slot],{yaw:actor.team===0?0:Math.PI,pitch:0});
  }
  return sim;
}
