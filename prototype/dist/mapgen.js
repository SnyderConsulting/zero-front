/** Different seeded arena grammars; random generation never consumes combat RNG. */
export const ARENA_LAYOUTS=Object.freeze(['room-web','ring','warehouse','courtyards','switchback']);

function seeded(seed){
  let state=(seed^0xa511e9b3)>>>0;
  return ()=>{
    state=(state+0x6d2b79f5)>>>0;
    let value=state;
    value=Math.imul(value^(value>>>15),value|1);
    value^=value+Math.imul(value^(value>>>7),value|61);
    return ((value^(value>>>14))>>>0)/4294967296;
  };
}

function validFloor(grid){
  const height=grid.length,width=grid[0].length;
  const floor=(x,y)=>grid[y]?.[x]==='.';
  let start=-1,total=0;
  for(let y=1;y<height-1;y++)for(let x=1;x<width-1;x++)if(floor(x,y)){
    total++;if(start<0)start=y*width+x;
    if(![[-1,-1],[-1,0],[0,-1],[0,0]].some(([dx,dy])=>
      floor(x+dx,y+dy)&&floor(x+dx+1,y+dy)&&floor(x+dx,y+dy+1)&&floor(x+dx+1,y+dy+1)))return false;
  }
  if(start<0)return false;
  const seen=new Set([start]),queue=[start];
  for(let i=0;i<queue.length;i++){
    const x=queue[i]%width,y=Math.floor(queue[i]/width);
    for(const [nx,ny] of [[x+1,y],[x-1,y],[x,y+1],[x,y-1]]){
      const next=ny*width+nx;
      if(floor(nx,ny)&&!seen.has(next)){seen.add(next);queue.push(next);}
    }
  }
  return seen.size===total;
}

/** Every grammar is symmetric under its declared team-exchange transformation. */
export function generateArena(seed=42){
  if(!Number.isInteger(seed)||seed<0||seed>4294967295)throw new TypeError('Map seed must be an unsigned 32-bit integer.');
  const random=seeded(seed),pick=(low,high)=>low+Math.floor(random()*(high-low+1));
  const family=seed%ARENA_LAYOUTS.length,layout=ARENA_LAYOUTS[family];
  const dimensions=[[[48,36],[52,40]],[[40,40],[44,44]],[[52,28],[56,32]],[[48,32],[52,36]],[[36,40],[40,44]]];
  const [width,height]=dimensions[family][pick(0,1)];
  const symmetry=family===0||family===3||family===4?'half-turn':'mirror-x';
  const grid=Array.from({length:height},()=>Array(width).fill('#'));
  const rooms=[],corridors=[];
  const transform=r=>({...r,x:width-r.x-r.width,y:symmetry==='half-turn'?height-r.y-r.height:r.y});
  const pair=r=>{
    const other=transform(r);
    return other.x===r.x&&other.y===r.y?[r]:[r,other];
  };
  function carve(r){
    for(let y=Math.max(1,r.y);y<Math.min(height-1,r.y+r.height);y++)
      for(let x=Math.max(1,r.x);x<Math.min(width-1,r.x+r.width);x++)grid[y][x]='.';
  }
  function room(x,y,w,h,kind='room'){
    const r={x,y,width:w,height:h,kind};
    for(const value of pair(r)){carve(value);rooms.push(value);}
    return r;
  }
  const center=r=>({x:Math.floor(r.x+r.width/2),y:Math.floor(r.y+r.height/2)});
  function segment(a,b,w=2){
    const offset=Math.floor((w-1)/2);
    const r={x:Math.min(a.x,b.x)-offset,y:Math.min(a.y,b.y)-offset,
      width:Math.abs(a.x-b.x)+w,height:Math.abs(a.y-b.y)+w};
    for(const value of pair(r)){carve(value);corridors.push(value);}
  }
  function connect(a,b,w=2,horizontalFirst=random()<.5){
    const elbow=horizontalFirst?{x:b.x,y:a.y}:{x:a.x,y:b.y};
    segment(a,elbow,w);segment(elbow,b,w);
  }
  function cover(x,y,w,h){
    const rectangles=pair({x,y,width:w,height:h});
    for(const r of rectangles)for(let yy=r.y-2;yy<r.y+r.height+2;yy++)
      for(let xx=r.x-2;xx<r.x+r.width+2;xx++)if(grid[yy]?.[xx]!=='.')return false;
    const changed=[];
    for(const r of rectangles)for(let yy=r.y;yy<r.y+r.height;yy++)for(let xx=r.x;xx<r.x+r.width;xx++){
      if(grid[yy][xx]==='.'){changed.push([xx,yy]);grid[yy][xx]='#';}
    }
    if(validFloor(grid))return true;
    for(const [xx,yy] of changed)grid[yy][xx]='.';
    return false;
  }

  // Two separate exits leave the inward-facing spawn wall intact.
  // Plaza maps approach on a diagonal, unlike the opposing side entrances used
  // by the warehouse/ring. This changes the attack axis as well as room shapes.
  const spawnY=family===3?pick(3,5):Math.floor(height/2)-5;
  room(2,spawnY,8,10,'spawn');
  const north={x:6,y:spawnY-2},south={x:6,y:spawnY+11};
  segment({x:6,y:spawnY+2},north,3);
  segment({x:6,y:spawnY+7},south,3);
  function attachBase(top,bottom){
    connect(north,{x:top.x,y:north.y},3,true);connect({x:top.x,y:north.y},top,3,false);
    connect(south,{x:bottom.x,y:south.y},3,true);connect({x:bottom.x,y:south.y},bottom,3,false);
  }

  if(family===0){
    // Small chambers on a randomized ladder graph, with optional diagonal links.
    const rows=pick(3,4),nodes=[],xColumns=[13,pick(20,22)];
    for(let row=0;row<rows;row++){
      const y=3+Math.round(row*(height-11)/(rows-1)),line=[];
      for(let col=0;col<2;col++)line.push(room(xColumns[col]-1,y+pick(0,1),pick(4,6),pick(4,6)));
      nodes.push(line);connect(center(line[0]),center(line[1]),pick(2,3));
    }
    for(let row=1;row<rows;row++){
      for(let col=0;col<2;col++)connect(center(nodes[row-1][col]),center(nodes[row][col]),2);
      if(random()<.55)connect(center(nodes[row-1][0]),center(nodes[row][1]),2,random()<.5);
    }
    for(const row of new Set([0,rows-1,pick(0,rows-1)])){
      const p=center(nodes[row][1]);connect(p,{x:width-p.x-1,y:height-p.y-1},pick(2,3));
    }
    attachBase(center(nodes[0][0]),center(nodes.at(-1)[0]));
    for(const line of nodes)for(const r of line)if(random()<.35)cover(r.x+2,r.y+2,1,1);
  }else if(family===1){
    // Circuit around a large impassable core, with corner chambers and side bays.
    const left=12,right=width-13,top=pick(3,5),bottom=height-pick(5,7),lane=pick(3,5);
    segment({x:left,y:top},{x:right,y:top},lane);
    segment({x:left,y:bottom},{x:right,y:bottom},lane);
    segment({x:left,y:top},{x:left,y:bottom},lane);
    for(const y of [top,bottom-4])room(left-1,y,7,pick(6,8));
    if(random()<.6)room(left-1,pick(12,height-18),pick(5,7),pick(5,8));
    if(random()<.5)room(Math.floor(width/2)-4,top-1,8,6,'hall');
    attachBase({x:left,y:top+2},{x:left,y:bottom-1});
    for(const r of rooms.filter(r=>r.kind==='room'&&r.x<width/2))cover(r.x+3,r.y+3,1,2);
  }else if(family===2){
    // Broad warehouse divided into long aisles by substantial storage islands.
    room(12,3,width-24,height-6,'warehouse');
    attachBase({x:14,y:5},{x:14,y:height-6});
    if(random()<.5){
      for(let x=16;x<width/2;x+=pick(5,7)){
        const length=pick(6,height-12),y=pick(6,height-length-6);
        cover(x,y,pick(2,3),length);
      }
    }else{
      for(let y=7;y<height-7;y+=pick(5,7))cover(16,y,pick(5,Math.max(5,width/2-17)),pick(2,3));
    }
    for(let i=0;i<5;i++)cover(pick(15,Math.floor(width/2)-2),pick(6,height-7),pick(1,2),pick(1,2));
  }else if(family===3){
    // Offset plazas separated by thick building masses, joined by broad streets.
    const upper=room(12,pick(2,4),pick(8,11),pick(8,11),'courtyard');
    const lower=room(12,height-pick(12,14),pick(8,11),pick(8,10),'courtyard');
    const a=center(upper),b=center(lower),across=p=>({x:width-p.x-1,y:height-p.y-1});
    connect(a,b,pick(3,4),true);
    connect(a,across(b),pick(3,4),random()<.5);connect(b,across(a),pick(3,4),random()<.5);
    if(random()<.65){
      const link=room(Math.floor(width/2)-3,Math.floor(height/2)-3,6,6,'room');
      connect(a,center(link),2);connect(b,center(link),2);
    }
    attachBase({x:13,y:upper.y+2},{x:13,y:lower.y+lower.height-3});
    for(const r of [upper,lower]){
      cover(r.x+3,r.y+3,pick(2,3),pick(2,3));
      if(random()<.5)cover(r.x+r.width-3,r.y+r.height-3,1,1);
    }
  }else{
    // Tall industrial galleries joined at alternating ends, with widened bends.
    const left=12,right=width-14,levels=[4,pick(12,14),height-pick(14,16),height-6],lane=pick(2,3);
    for(const y of levels)segment({x:left,y},{x:right,y},lane);
    for(let i=1;i<levels.length;i++){
      const x=i%2?left:right;segment({x,y:levels[i-1]},{x,y:levels[i]},lane);
    }
    const side=random()<.5?left:right;
    segment({x:side,y:levels[0]},{x:side,y:levels.at(-1)},2);
    for(const y of [levels[0],levels.at(-1)-3])room(left,y,5,5,'room');
    room(right-2,levels[1],5,5,'room');
    if(random()<.6)room(left,levels[2],6,5,'hall');
    attachBase({x:left,y:levels[0]+1},{x:left,y:levels.at(-1)});
  }

  if(!validFloor(grid))throw new Error(`Invalid generated ${layout} floor for seed ${seed}`);
  const blue=[];
  for(let row=0;row<4;row++)for(let col=0;col<2;col++)
    blue.push({x:5.5+col*2,y:spawnY+1.5+row*2,yaw:row<2?-Math.PI/2:Math.PI/2});
  const red=blue.map(s=>({x:width-s.x,y:symmetry==='half-turn'?height-s.y:s.y,
    yaw:symmetry==='half-turn'?Math.atan2(-Math.sin(s.yaw),-Math.cos(s.yaw)):Math.atan2(Math.sin(s.yaw),-Math.cos(s.yaw))}));
  const map=grid.map((row,y)=>row.map((cell,x)=>{
    if(cell==='.')return cell;
    if(x===0||y===0||x===width-1||y===height-1)return '#';
    return x<width/2-3?'1':x>=width/2+3?'2':'3';
  }).join(''));
  return {map,spawns:[blue,red],rooms,corridors,layout,symmetry};
}
