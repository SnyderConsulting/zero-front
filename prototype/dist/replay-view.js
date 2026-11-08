import {sampleReplay} from './replay.js';

const clamp=(value,min,max)=>Math.max(min,Math.min(max,value));
const TEAM_COLORS=['#53c5ec','#ff826d'];
const timeLabel=value=>{
  const tenths=Math.round(Math.max(0,Number.isFinite(value)?value:0)*10);
  return `${Math.floor(tenths/600)}:${((tenths%600)/10).toFixed(1).padStart(4,'0')}`;
};

/** Camera coordinates are world units; scale is CSS pixels per world unit. */
export function fitReplayCamera(mapWidth,mapHeight,width,height){
  const scale=Math.max(.1,Math.min(Math.max(1,width-48)/mapWidth,Math.max(1,height-48)/mapHeight));
  return {x:mapWidth/2,y:mapHeight/2,scale,minScale:scale*.75,maxScale:Math.max(scale*12,40)};
}
export function replayWorldAt(camera,screenX,screenY,width,height){
  return {x:camera.x+(screenX-width/2)/camera.scale,y:camera.y+(screenY-height/2)/camera.scale};
}
export function zoomReplayCamera(camera,factor,screenX,screenY,width,height,mapWidth,mapHeight){
  const anchor=replayWorldAt(camera,screenX,screenY,width,height);
  const scale=clamp(camera.scale*factor,camera.minScale,camera.maxScale);
  return {...camera,scale,
    x:clamp(anchor.x-(screenX-width/2)/scale,0,mapWidth),
    y:clamp(anchor.y-(screenY-height/2)/scale,0,mapHeight)};
}

/** Read-only, local round viewer. Its clock never advances the live simulation. */
export function createReplayViewer(host){
  const doc=host.ownerDocument||document;
  host.classList.add('arena-replay');
  host.innerHTML=`
    <div class="replay-stage">
      <canvas class="replay-map" tabindex="0" aria-label="Round replay map. Drag to pan, scroll to zoom. Click a player to inspect."></canvas>
      <div class="replay-map-score" aria-label="Score at replay time"><span class="replay-blue">BLUE <b data-score="0">0</b></span><span class="replay-red"><b data-score="1">0</b> RED</span></div>
      <div class="replay-zoom" aria-label="Map zoom controls"><button type="button" data-action="zoom-in" aria-label="Zoom in">+</button><button type="button" data-action="zoom-out" aria-label="Zoom out">−</button><button type="button" data-action="fit">Fit map</button></div>
      <p class="replay-map-hint">Drag to pan · Scroll to zoom · Arrows show facing</p>
    </div>
    <div class="replay-transport">
      <button type="button" class="replay-play" data-action="play" aria-label="Play replay">Play</button>
      <input class="replay-timeline" type="range" min="0" max="1" value="0" step="0.0166667" aria-label="Replay time">
      <output class="replay-clock">0:00.0 / 0:00.0</output>
      <label class="replay-speed-label">Speed <select class="replay-speed" aria-label="Playback speed"><option value="0.25">¼×</option><option value="0.5">½×</option><option value="1" selected>1×</option><option value="2">2×</option><option value="4">4×</option></select></label>
    </div>
    <div class="replay-inspector">
      <label class="replay-player-label">Player <select class="replay-player" aria-label="Inspect player"></select></label>
      <button type="button" data-action="follow" aria-pressed="false">Follow</button>
      <p class="replay-player-status" aria-live="off">Select a player on the map</p>
    </div>`;
  const canvas=host.querySelector('.replay-map'),ctx=canvas.getContext('2d');
  const stage=host.querySelector('.replay-stage');
  const timeline=host.querySelector('.replay-timeline'),clock=host.querySelector('.replay-clock');
  const playButton=host.querySelector('[data-action="play"]');
  const speedSelect=host.querySelector('.replay-speed'),playerSelect=host.querySelector('.replay-player');
  const status=host.querySelector('.replay-player-status'),followButton=host.querySelector('[data-action="follow"]');
  const scoreLabels=[host.querySelector('[data-score="0"]'),host.querySelector('[data-score="1"]')];
  const mapCanvas=doc.createElement('canvas'),mapContext=mapCanvas.getContext('2d');
  let recording=null,frame=null,visible=false,playing=false,following=false,time=0,speed=1;
  let width=1,height=1,pixelRatio=1,camera=null,needsFit=true,selectedId=null,drag=null,disposed=false;

  function buildMap(){
    const tile=12;mapCanvas.width=recording.width*tile;mapCanvas.height=recording.height*tile;
    mapContext.fillStyle='#142b26';mapContext.fillRect(0,0,mapCanvas.width,mapCanvas.height);
    for(let y=0;y<recording.height;y++)for(let x=0;x<recording.width;x++)if(recording.map[y][x]==='.'){
      mapContext.fillStyle='#718479';mapContext.fillRect(x*tile,y*tile,tile,tile);
      mapContext.strokeStyle='#13261e1a';mapContext.lineWidth=.5;mapContext.strokeRect(x*tile,y*tile,tile,tile);
    }
  }
  function setPlaying(value){
    playing=Boolean(value)&&Boolean(recording)&&recording.duration>0;
    playButton.textContent=playing?'Pause':'Play';
    playButton.setAttribute('aria-label',playing?'Pause replay':'Play replay');
  }
  function setFollowing(value){following=Boolean(value);followButton.setAttribute('aria-pressed',String(following));}
  function fit(){
    if(!recording)return;
    camera=fitReplayCamera(recording.width,recording.height,width,height);
    needsFit=false;setFollowing(false);render();
  }
  function resize(){
    if(!visible||disposed)return;
    const bounds=stage.getBoundingClientRect();
    if(bounds.width<1||bounds.height<1)return;
    width=bounds.width;height=bounds.height;
    pixelRatio=Math.min(2,doc.defaultView?.devicePixelRatio||1);
    canvas.width=Math.max(1,Math.round(width*pixelRatio));canvas.height=Math.max(1,Math.round(height*pixelRatio));
    if(recording){
      const fitted=fitReplayCamera(recording.width,recording.height,width,height);
      if(needsFit||!camera)camera=fitted;
      else camera={...camera,minScale:fitted.minScale,maxScale:fitted.maxScale,scale:clamp(camera.scale,fitted.minScale,fitted.maxScale)};
      needsFit=false;
    }
    render();
  }
  function screen(point){return {x:width/2+(point.x-camera.x)*camera.scale,y:height/2+(point.y-camera.y)*camera.scale};}
  function renderActor(actor){
    const p=screen(actor),selected=actor.id===selectedId;
    if(p.x<-60||p.y<-50||p.x>width+60||p.y>height+50)return;
    const color=TEAM_COLORS[actor.team]||'#e6eee4';
    const radius=clamp(camera.scale*.34,4.5,12);
    const name=actor.id==='player'?'YOU':`${actor.team===0?'B':'R'}${(actor.slot??0)+1}`;
    ctx.save();
    if(!actor.active){
      ctx.globalAlpha=.45;ctx.strokeStyle=color;ctx.lineWidth=2;
      ctx.beginPath();ctx.moveTo(p.x-radius,p.y-radius);ctx.lineTo(p.x+radius,p.y+radius);ctx.moveTo(p.x+radius,p.y-radius);ctx.lineTo(p.x-radius,p.y+radius);ctx.stroke();
    }else{
      const cosine=Math.cos(actor.yaw),sine=Math.sin(actor.yaw),tip=radius+9;
      if(selected){
        ctx.fillStyle=actor.team===0?'#53c5ec20':'#ff826d20';ctx.beginPath();ctx.moveTo(p.x,p.y);
        ctx.arc(p.x,p.y,Math.max(26,camera.scale*2),actor.yaw-.42,actor.yaw+.42);ctx.closePath();ctx.fill();
      }
      ctx.fillStyle=color;ctx.strokeStyle='#071a15';ctx.lineWidth=1.5;
      ctx.beginPath();ctx.moveTo(p.x+cosine*tip,p.y+sine*tip);
      ctx.lineTo(p.x+cosine*(radius-1)-sine*4.5,p.y+sine*(radius-1)+cosine*4.5);
      ctx.lineTo(p.x+cosine*(radius-1)+sine*4.5,p.y+sine*(radius-1)-cosine*4.5);ctx.closePath();ctx.fill();ctx.stroke();
      ctx.beginPath();ctx.arc(p.x,p.y,radius,0,Math.PI*2);ctx.fill();ctx.stroke();
      if(actor.spawnProtection>0){ctx.strokeStyle='#f6edc6';ctx.setLineDash([2,3]);ctx.beginPath();ctx.arc(p.x,p.y,radius+3,0,Math.PI*2);ctx.stroke();ctx.setLineDash([]);}
      if(selected||actor.health<100){
        const barWidth=Math.max(19,radius*2.5),barY=p.y-radius-10;
        ctx.fillStyle='#071a15';ctx.fillRect(p.x-barWidth/2,barY,barWidth,3);
        ctx.fillStyle=actor.health<30?'#ff826d':'#c8efb8';ctx.fillRect(p.x-barWidth/2,barY,barWidth*clamp(actor.health/100,0,1),3);
      }
    }
    if(selected){ctx.globalAlpha=1;ctx.strokeStyle='#f4f6e8';ctx.lineWidth=1.5;ctx.beginPath();ctx.arc(p.x,p.y,radius+5,0,Math.PI*2);ctx.stroke();}
    ctx.font=`${selected?700:600} ${selected?11:10}px "Segoe UI",sans-serif`;
    ctx.textAlign='center';ctx.textBaseline='top';ctx.lineJoin='round';ctx.lineWidth=3;
    ctx.strokeStyle='#10241e';ctx.strokeText(name,p.x,p.y+radius+7);ctx.fillStyle='#f3f6ec';ctx.fillText(name,p.x,p.y+radius+7);
    ctx.restore();
  }
  function render(){
    if(!visible||!recording||!camera||!ctx)return;
    frame=sampleReplay(recording,time);
    const selected=frame.actors.find(actor=>actor.id===selectedId);
    if(following&&selected){camera.x=selected.x;camera.y=selected.y;}
    ctx.setTransform(pixelRatio,0,0,pixelRatio,0,0);
    ctx.clearRect(0,0,width,height);ctx.fillStyle='#0e211c';ctx.fillRect(0,0,width,height);
    const origin=screen({x:0,y:0});
    ctx.imageSmoothingEnabled=false;
    ctx.drawImage(mapCanvas,origin.x,origin.y,recording.width*camera.scale,recording.height*camera.scale);
    ctx.strokeStyle='#abc3ad40';ctx.lineWidth=1;ctx.strokeRect(origin.x,origin.y,recording.width*camera.scale,recording.height*camera.scale);
    for(const event of frame.events||[]){
      const age=time-event.time;
      if(event.type==='shot'&&age>=0&&age<.18&&Number.isFinite(event.fromX)&&Number.isFinite(event.fromY)){
        const a=screen({x:event.fromX,y:event.fromY}),b=screen(event);
        ctx.globalAlpha=(1-age/.18)*.9;ctx.strokeStyle=TEAM_COLORS[event.team]||'#e6db9e';ctx.lineWidth=1.5;
        ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.stroke();ctx.globalAlpha=1;
      }
      if(event.type==='elimination'&&age>=0&&age<.35){
        const victim=frame.actors.find(actor=>actor.id===event.victim);
        if(victim){const p=screen(victim);ctx.strokeStyle='#f7e5b8';ctx.globalAlpha=1-age/.35;ctx.lineWidth=2;ctx.beginPath();ctx.arc(p.x,p.y,9+age*22,0,Math.PI*2);ctx.stroke();ctx.globalAlpha=1;}
      }
    }
    for(const actor of frame.actors)if(actor.id!==selectedId)renderActor(actor);
    if(selected)renderActor(selected);
    timeline.value=String(time);clock.textContent=`${timeLabel(time)} / ${timeLabel(recording.duration)}`;
    timeline.setAttribute('aria-valuetext',`${timeLabel(time)} of ${timeLabel(recording.duration)}`);
    scoreLabels.forEach((label,index)=>{label.textContent=frame.scores[index];});
    if(selected){
      const firing=frame.events?.some(event=>event.type==='shot'&&event.actor===selected.id&&time-event.time<.18);
      const activity=!selected.active?`Respawning ${Math.max(0,selected.respawnTimer||0).toFixed(1)}s`:selected.reloading?'Reloading':firing?'Firing':'Alive';
      status.textContent=`${activity}  ·  ${Math.ceil(selected.health)} HP  ·  ${selected.ammo} ammo  ·  ${selected.kills} K / ${selected.deaths} D`;
    }
  }
  function seek(value){
    if(!recording)return;
    time=clamp(Number(value)||0,0,recording.duration);
    if(time>=recording.duration)setPlaying(false);
    render();
  }
  function togglePlayback(){
    if(!recording)return;
    if(!playing&&time>=recording.duration)time=0;
    setPlaying(!playing);render();
  }
  function zoom(factor,x=width/2,y=height/2){
    if(!recording||!camera)return;
    camera=zoomReplayCamera(camera,factor,x,y,width,height,recording.width,recording.height);render();
  }
  function handleKey(event){
    if(!visible||!recording||event.defaultPrevented)return false;
    if(/INPUT|SELECT|BUTTON|TEXTAREA/.test(event.target?.tagName||''))return false;
    let handled=true;
    if(event.code==='Space'){if(!event.repeat)togglePlayback();}
    else if(event.code==='ArrowLeft'){setPlaying(false);seek(time-(event.shiftKey?5:1));}
    else if(event.code==='ArrowRight'){setPlaying(false);seek(time+(event.shiftKey?5:1));}
    else if(event.code==='Home'){setPlaying(false);seek(0);}
    else if(event.code==='End'){setPlaying(false);seek(recording.duration);}
    else if(event.code==='Equal'||event.code==='NumpadAdd')zoom(1.25);
    else if(event.code==='Minus'||event.code==='NumpadSubtract')zoom(.8);
    else if(event.code==='KeyF')fit();
    else handled=false;
    if(handled)event.preventDefault();
    return handled;
  }
  playButton.addEventListener('click',togglePlayback);
  host.querySelector('[data-action="zoom-in"]').addEventListener('click',()=>zoom(1.3));
  host.querySelector('[data-action="zoom-out"]').addEventListener('click',()=>zoom(1/1.3));
  host.querySelector('[data-action="fit"]').addEventListener('click',fit);
  followButton.addEventListener('click',()=>{setFollowing(!following);render();});
  timeline.addEventListener('input',()=>{setPlaying(false);seek(timeline.value);});
  speedSelect.addEventListener('change',()=>{speed=clamp(Number(speedSelect.value)||1,.25,4);});
  playerSelect.addEventListener('change',()=>{selectedId=playerSelect.value;render();});
  host.addEventListener('keydown',handleKey);
  canvas.addEventListener('pointerdown',event=>{
    if(!recording||!camera||event.button!==0)return;
    event.preventDefault();canvas.focus({preventScroll:true});
    drag={id:event.pointerId,x:event.clientX,y:event.clientY,startX:event.clientX,startY:event.clientY,moved:false};
    canvas.setPointerCapture(event.pointerId);canvas.classList.add('dragging');
  });
  canvas.addEventListener('pointermove',event=>{
    if(!drag||event.pointerId!==drag.id)return;
    const dx=event.clientX-drag.x,dy=event.clientY-drag.y;
    drag.moved ||= Math.hypot(event.clientX-drag.startX,event.clientY-drag.startY)>4;
    if(drag.moved){setFollowing(false);camera.x=clamp(camera.x-dx/camera.scale,0,recording.width);camera.y=clamp(camera.y-dy/camera.scale,0,recording.height);render();}
    drag.x=event.clientX;drag.y=event.clientY;
  });
  function clearDrag(){
    const pointerId=drag?.id;
    drag=null;canvas.classList.remove('dragging');
    if(pointerId!==undefined&&canvas.hasPointerCapture?.(pointerId))canvas.releasePointerCapture(pointerId);
  }
  function endDrag(event){
    if(!drag||event.pointerId!==drag.id)return;
    const click=!drag.moved&&event.type==='pointerup';clearDrag();
    if(click&&frame){
      const rect=canvas.getBoundingClientRect(),x=event.clientX-rect.left,y=event.clientY-rect.top;
      const candidates=frame.actors.map(actor=>({actor,p:screen(actor)})).map(value=>({...value,distance:Math.hypot(value.p.x-x,value.p.y-y)})).sort((a,b)=>a.distance-b.distance);
      if(candidates[0]?.distance<Math.max(15,camera.scale*.6)){selectedId=candidates[0].actor.id;playerSelect.value=selectedId;render();}
    }
  }
  canvas.addEventListener('pointerup',endDrag);canvas.addEventListener('pointercancel',endDrag);
  canvas.addEventListener('lostpointercapture',event=>{if(drag?.id===event.pointerId)clearDrag();});
  canvas.addEventListener('wheel',event=>{
    if(!recording)return;event.preventDefault();
    const rect=canvas.getBoundingClientRect(),delta=event.deltaY*(event.deltaMode===1?16:event.deltaMode===2?height:1);
    zoom(Math.exp(clamp(-delta*.0015,-.7,.7)),event.clientX-rect.left,event.clientY-rect.top);
  },{passive:false});
  canvas.addEventListener('dblclick',fit);
  const observer=typeof ResizeObserver!=='undefined'?new ResizeObserver(resize):null;
  observer?.observe(stage);
  doc.defaultView?.addEventListener('resize',resize);

  return {
    load(value){
      if(disposed)throw new Error('This replay viewer has been destroyed.');
      if(!value||!Array.isArray(value.map)||!value.roster?.length)throw new TypeError('A recorded round is required.');
      recording=value;time=0;speed=1;speedSelect.value='1';setPlaying(false);setFollowing(false);
      selectedId=value.roster.some(actor=>actor.id==='player')?'player':value.roster[0].id;
      playerSelect.replaceChildren(...value.roster.map(actor=>{
        const option=doc.createElement('option');option.value=actor.id;option.textContent=`${actor.name} · ${actor.team===0?'Blue':'Red'}`;return option;
      }));
      playerSelect.value=selectedId;timeline.max=String(value.duration);timeline.step=String(1/(value.tickRate||60));
      timeline.disabled=value.duration<=0;needsFit=true;buildMap();if(visible)resize();
    },
    show(){if(disposed)return;visible=true;host.hidden=false;resize();canvas.focus({preventScroll:true});},
    hide(){visible=false;clearDrag();host.hidden=true;setPlaying(false);},
    update(dt){
      if(!visible||!recording||!playing)return;
      seek(time+Math.max(0,Math.min(Number(dt)||0,.25))*speed);
    },
    resize,handleKey,seek,fit,
    destroy(){
      if(disposed)return;
      disposed=true;visible=false;clearDrag();setPlaying(false);
      host.removeEventListener('keydown',handleKey);
      observer?.disconnect();doc.defaultView?.removeEventListener('resize',resize);
      recording=null;frame=null;camera=null;selectedId=null;
      mapCanvas.width=0;mapCanvas.height=0;host.replaceChildren();
    },
  };
}
