import test from 'node:test';
import assert from 'node:assert/strict';
import {generateArena} from '../dist/mapgen.js';

const floor=(map,x,y)=>map[y]?.[x]==='.';
const idFor=(map,point)=>Math.floor(point.y)*map[0].length+Math.floor(point.x);
function flood(map,start){
  const width=map[0].length,first=idFor(map,start),queue=[first],previous=new Map([[first,-1]]);
  for(let i=0;i<queue.length;i++){
    const x=queue[i]%width,y=Math.floor(queue[i]/width);
    for(const [nx,ny] of [[x-1,y],[x+1,y],[x,y-1],[x,y+1]]){
      const next=ny*width+nx;
      if(floor(map,nx,ny)&&!previous.has(next)){previous.set(next,queue[i]);queue.push(next);}
    }
  }
  return previous;
}
function clearSight(map,a,b){
  const length=Math.hypot(a.x-b.x,a.y-b.y),steps=Math.ceil(length*20);
  for(let i=0;i<=steps;i++){
    const t=i/steps;
    if(!floor(map,Math.floor(a.x+(b.x-a.x)*t),Math.floor(a.y+(b.y-a.y)*t)))return false;
  }
  return true;
}
function reflected(arena,x,y,continuous=false){
  const width=arena.map[0].length,height=arena.map.length,offset=continuous?0:1;
  assert.ok(['mirror-x','mirror-y','half-turn'].includes(arena.symmetry),`Unknown symmetry: ${arena.symmetry}`);
  return [arena.symmetry==='mirror-y'?x:width-offset-x,arena.symmetry==='mirror-x'?y:height-offset-y];
}
function silhouette(map){
  // Compare structure at room/corridor scale, independent of map dimensions.
  const columns=12,rows=10,result=[];
  for(let by=0;by<rows;by++)for(let bx=0;bx<columns;bx++){
    let count=0,total=0;
    for(let y=Math.floor(by*map.length/rows);y<Math.floor((by+1)*map.length/rows);y++){
      for(let x=Math.floor(bx*map[0].length/columns);x<Math.floor((bx+1)*map[0].length/columns);x++){
        total++;if(floor(map,x,y))count++;
      }
    }
    result.push(count/total);
  }
  return result;
}
function profile(arena){
  const {map}=arena,width=map[0].length,height=map.length;
  let floors=0,broad=0;
  for(let y=0;y<height;y++)for(let x=0;x<width;x++){
    if(!floor(map,x,y))continue;
    floors++;
    let open=true;
    for(let dy=-2;dy<=2&&open;dy++)for(let dx=-2;dx<=2;dx++)if(!floor(map,x+dx,y+dy)){open=false;break;}
    if(open)broad++;
  }
  return {area:floors/(width*height),broad:broad/floors,rooms:arena.rooms.length};
}

test('seeds reproduce their maps and select structurally different map families',()=>{
  assert.deepEqual(generateArena(42),generateArena(42));
  const arenas=Array.from({length:50},(_,seed)=>generateArena(seed));
  assert.equal(new Set(arenas.map(a=>a.map.join('\n'))).size,arenas.length);
  assert.ok(new Set(arenas.map(a=>a.layout)).size>=5,'Expected at least five distinct layout families');
  assert.ok(new Set(arenas.map(a=>`${a.map[0].length}x${a.map.length}`)).size>=3,'Map footprints should vary');
  const profiles=arenas.map(profile);
  assert.ok(new Set(profiles.map(p=>p.rooms)).size>=3,'Room count must vary beyond the old fixed room graph');
  assert.ok(Math.max(...profiles.map(p=>p.area))-Math.min(...profiles.map(p=>p.area))>.15,'Open and enclosed layouts should have different floor coverage');
  assert.ok(Math.max(...profiles.map(p=>p.broad))-Math.min(...profiles.map(p=>p.broad))>.12,'Broad open spaces and corridor-heavy maps should differ geometrically');
  for(const layout of new Set(arenas.map(a=>a.layout))){
    const shapes=arenas.filter(a=>a.layout===layout).map(a=>silhouette(a.map));
    let largestDifference=0;
    for(let i=0;i<shapes.length;i++)for(let j=i+1;j<shapes.length;j++){
      const difference=shapes[i].reduce((sum,value,k)=>sum+Math.abs(value-shapes[j][k]),0)/shapes[i].length;
      largestDifference=Math.max(largestDifference,difference);
    }
    assert.ok(largestDifference>.06,`${layout} must vary its geometry within the family, not just choose a fixed template`);
  }
  const examples=[42,1,813].map(seed=>generateArena(seed));
  assert.equal(new Set(examples.map(a=>a.layout)).size,3,'The previously similar example seeds should use different structures');
  const shapes=examples.map(a=>silhouette(a.map));
  for(let i=0;i<shapes.length;i++)for(let j=i+1;j<shapes.length;j++){
    const difference=shapes[i].reduce((sum,value,k)=>sum+Math.abs(value-shapes[j][k]),0)/shapes[i].length;
    assert.ok(difference>.20,`Example map silhouettes remain too similar (${difference.toFixed(3)})`);
  }
  for(const seed of [-1,1.5,4294967296,NaN])assert.throws(()=>generateArena(seed),TypeError);
});

test('generated floors stay connected, enclosed, wide enough, and symmetric across seeds',()=>{
  for(const seed of [...Array.from({length:120},(_,i)=>i),813,0xffffffff]){
    const arena=generateArena(seed),{map,spawns}=arena,width=map[0].length,height=map.length;
    const reached=flood(map,spawns[0][0]);let total=0;
    assert.ok(width<=64&&height<=64&&width>=24&&height>=24);
    for(let y=0;y<height;y++)for(let x=0;x<width;x++){
      const walkable=floor(map,x,y),[otherX,otherY]=reflected(arena,x,y);
      assert.equal(walkable,floor(map,otherX,otherY),`Asymmetric geometry seed ${seed} at ${x},${y}`);
      if(x===0||y===0||x===width-1||y===height-1)assert.equal(walkable,false);
      if(!walkable)continue;
      total++;
      const wideEnough=[[-1,-1],[-1,0],[0,-1],[0,0]].some(([dx,dy])=>
        floor(map,x+dx,y+dy)&&floor(map,x+dx+1,y+dy)&&floor(map,x+dx,y+dy+1)&&floor(map,x+dx+1,y+dy+1));
      assert.ok(wideEnough,`Narrow floor at seed ${seed} ${x},${y}`);
    }
    assert.equal(reached.size,total,`Disconnected floor at seed ${seed}`);
    assert.ok(total>width*height*.20&&total<width*height*.85,`Room/wall balance at seed ${seed}`);
  }
});

test('team spawns have clearance, equivalent placement, and no direct enemy spawn sightline',()=>{
  for(let seed=0;seed<80;seed++){
    const arena=generateArena(seed),{map,spawns}=arena;
    for(const team of spawns){
      assert.equal(team.length,8);
      assert.equal(new Set(team.map(s=>`${s.x},${s.y}`)).size,8);
      for(const s of team){
        assert.ok(Number.isFinite(s.yaw));
        for(const dx of [-.44,.44])for(const dy of [-.44,.44])assert.ok(floor(map,Math.floor(s.x+dx),Math.floor(s.y+dy)));
      }
      for(let i=0;i<team.length;i++)for(let j=i+1;j<team.length;j++)assert.ok(Math.hypot(team[i].x-team[j].x,team[i].y-team[j].y)>=.88);
    }
    for(let i=0;i<8;i++){
      const [x,y]=reflected(arena,spawns[0][i].x,spawns[0][i].y,true);
      assert.equal(spawns[1][i].x,x);assert.equal(spawns[1][i].y,y);
      const xSign=arena.symmetry==='mirror-y'?1:-1,ySign=arena.symmetry==='mirror-x'?1:-1;
      assert.ok(Math.abs(Math.cos(spawns[0][i].yaw)*xSign-Math.cos(spawns[1][i].yaw))<1e-9);
      assert.ok(Math.abs(Math.sin(spawns[0][i].yaw)*ySign-Math.sin(spawns[1][i].yaw))<1e-9);
    }
    for(const blue of spawns[0])for(const red of spawns[1])assert.equal(clearSight(map,blue,red),false,`Spawn sightline seed ${seed}`);
  }
});

test('a blocked patch along the main combat route leaves a flanking route',()=>{
  for(let seed=0;seed<30;seed++){
    const {map,spawns}=generateArena(seed),width=map[0].length;
    const previous=flood(map,spawns[0][0]),end=idFor(map,spawns[1][0]);
    assert.ok(previous.has(end),`Disconnected spawns at seed ${seed}`);
    const path=[];for(let id=end;id!==-1;id=previous.get(id))path.push(id);
    for(const fraction of [.35,.5,.65]){
      const position=path[Math.floor(path.length*fraction)],cx=position%width,cy=Math.floor(position/width);
      const blocked=map.map((row,y)=>[...row].map((cell,x)=>Math.abs(x-cx)<=1&&Math.abs(y-cy)<=1?'#':cell).join(''));
      assert.ok(flood(blocked,spawns[0][0]).has(end),`No alternate route at seed ${seed}, path fraction ${fraction}`);
    }
  }
});
