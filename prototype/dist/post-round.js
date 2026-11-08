import {createReplayViewer} from './replay-view.js';

const TEAM_NAMES=['Blue','Red'];
const layoutName=value=>String(value||'Arena').replace(/-/g,' ').replace(/\b\w/g,letter=>letter.toUpperCase());
const durationLabel=value=>{
  const seconds=Math.max(0,Math.floor(Number(value)||0));
  return `${Math.floor(seconds/60)}:${String(seconds%60).padStart(2,'0')}`;
};
const accuracyLabel=value=>`${Math.round(Number(value)||0)}%`;

/** Post-round screens are created only when a completed round is available. */
export function createPostRound({onNewMatch=()=>{}}={}){
  let open=false,replaying=false,recording=null,summary=null,viewer=null;
  let overlay,resultsPane,replayPane,replayHost,title,meta,blueScore,redScore,teamGrid,watchButton;

  function element(tag,className,text){
    const node=document.createElement(tag);
    if(className)node.className=className;
    if(text!==undefined)node.textContent=String(text);
    return node;
  }
  function button(text,className,handler){
    const node=element('button',className,text);
    node.type='button';node.addEventListener('click',handler);return node;
  }
  function startNextRound(){hide();onNewMatch();}
  function build(){
    if(overlay)return;
    overlay=element('section','post-round');overlay.hidden=true;
    overlay.setAttribute('role','dialog');overlay.setAttribute('aria-modal','true');
    overlay.setAttribute('aria-labelledby','round-result-title');
    overlay.id='post-round';
    resultsPane=element('div','round-results');
    const header=element('header','round-results-header');
    header.append(element('p','round-eyebrow','Round complete'));
    title=element('h1','round-result-title');title.id='round-result-title';
    title.tabIndex=-1;header.append(title);
    const scoreline=element('div','round-final-score');
    const blue=element('div','round-team-score round-blue');
    blueScore=element('strong','');blue.append(element('span','','Blue'),blueScore);
    const red=element('div','round-team-score round-red');
    redScore=element('strong','');red.append(redScore,element('span','','Red'));
    scoreline.append(blue,element('span','round-score-divider','—'),red);header.append(scoreline);
    meta=element('p','round-meta');header.append(meta);
    teamGrid=element('div','round-teams');
    const footer=element('footer','round-actions');
    watchButton=button('Watch replay','round-button round-primary',openReplay);
    watchButton.id='watch-replay';
    footer.append(watchButton,button('New match','round-button round-secondary',startNextRound));
    resultsPane.append(header,teamGrid,footer);

    replayPane=element('div','round-replay-pane');replayPane.hidden=true;
    const replayHeader=element('header','round-replay-header');
    const back=button('← Results','round-button round-secondary',showResults);
    back.setAttribute('aria-label','Back to round results');
    replayHeader.append(back,element('h2','','Round replay'),button('New match','round-button round-secondary',startNextRound));
    replayHost=element('div','round-replay-host');
    replayPane.append(replayHeader,replayHost);
    overlay.append(resultsPane,replayPane);
    (document.getElementById('game')||document.body).append(overlay);
    overlay.addEventListener('keydown',event=>{
      if(!open||event.key!=='Tab')return;
      const controls=[...overlay.querySelectorAll('button,select,input,[tabindex="0"]')]
        .filter(node=>!node.disabled&&node.getClientRects().length);
      if(!controls.length){event.preventDefault();return;}
      const first=controls[0],last=controls.at(-1),focused=document.activeElement;
      if(event.shiftKey&&(focused===first||!overlay.contains(focused))){event.preventDefault();last.focus();}
      else if(!event.shiftKey&&(focused===last||!overlay.contains(focused))){event.preventDefault();first.focus();}
    });
  }

  function teamTable(team){
    const panel=element('section',`round-team-panel ${team===0?'round-blue':'round-red'}`);
    const heading=element('h2','round-team-heading',`${TEAM_NAMES[team]} team`);
    panel.append(heading);
    const scroll=element('div','round-roster-scroll');
    scroll.tabIndex=0;scroll.setAttribute('aria-label',`${TEAM_NAMES[team]} team scoreboard`);
    const table=element('table','round-roster');
    const caption=element('caption','round-sr-only',`${TEAM_NAMES[team]} team final statistics`);table.append(caption);
    const head=element('thead',''),headRow=element('tr','');
    for(const [label,full]of[['Player','Player'],['K','Kills'],['D','Deaths'],['Accuracy','Accuracy']]){
      const th=element('th','',label);th.scope='col';th.setAttribute('aria-label',full);th.title=full;headRow.append(th);
    }
    head.append(headRow);table.append(head);
    const body=element('tbody','');
    const players=[...(summary.players||[])].filter(player=>player.team===team)
      .sort((a,b)=>b.kills-a.kills||a.deaths-b.deaths||(a.slot??0)-(b.slot??0));
    for(const player of players){
      const isYou=player.id===(summary.playerId||'player');
      const row=element('tr',isYou?'round-player-you':'');
      const name=element('th','round-player-name',isYou?'YOU':player.name||player.id);
      name.scope='row';
      if(isYou)name.setAttribute('aria-label','You');
      row.append(name,element('td','round-number',player.kills||0),element('td','round-number',player.deaths||0));
      const accuracy=element('td','round-accuracy');
      const hits=Number(player.hits)||0,shots=Number(player.shots)||0;
      accuracy.append(element('span','',accuracyLabel(player.accuracy??(shots?100*hits/shots:0))),
        element('small','',`${hits} / ${shots} hits`));
      accuracy.setAttribute('aria-label',`${accuracyLabel(player.accuracy??(shots?100*hits/shots:0))} accuracy, ${hits} hits from ${shots} shots`);
      row.append(accuracy);body.append(row);
    }
    table.append(body);scroll.append(table);panel.append(scroll);
    return panel;
  }

  function show(nextRecording,nextSummary=nextRecording?.summary){
    if(!nextSummary)throw new TypeError('A completed round summary is required.');
    build();viewer?.hide();
    recording=nextRecording;summary=nextSummary;open=true;replaying=false;
    title.textContent=summary.winner==='draw'?'DRAW':summary.winner===0?'BLUE WINS':summary.winner===1?'RED WINS':'ROUND COMPLETE';
    title.className=`round-result-title ${summary.winner===0?'round-blue':summary.winner===1?'round-red':''}`;
    blueScore.textContent=String(summary.scores?.[0]??0);redScore.textContent=String(summary.scores?.[1]??0);
    const reason=summary.reason==='time'?'Time limit':summary.reason==='score'?'Score limit reached':'Round finished';
    meta.textContent=`${durationLabel(summary.duration)} played · ${reason} · ${layoutName(summary.layout)}`;
    teamGrid.replaceChildren(teamTable(0),teamTable(1));
    watchButton.disabled=!recording;watchButton.title=recording?'Review the entire round from above':'No replay recording is available for this round';
    replayPane.hidden=true;resultsPane.hidden=false;overlay.hidden=false;overlay.removeAttribute('aria-label');
    overlay.setAttribute('aria-labelledby','round-result-title');
    watchButton.disabled?title.focus():watchButton.focus();
  }

  function openReplay(){
    if(!open||!recording)return;
    replaying=true;resultsPane.hidden=true;replayPane.hidden=false;
    overlay.setAttribute('aria-label','Round replay');overlay.removeAttribute('aria-labelledby');
    if(!viewer)viewer=createReplayViewer(replayHost);
    viewer.load(recording);viewer.show();viewer.resize();
  }
  function showResults(){
    if(!open)return;
    viewer?.hide();replaying=false;replayPane.hidden=true;resultsPane.hidden=false;
    overlay.removeAttribute('aria-label');overlay.setAttribute('aria-labelledby','round-result-title');
    watchButton.focus();
  }
  function hide(){
    viewer?.hide();viewer?.destroy();viewer=null;recording=null;summary=null;
    open=false;replaying=false;
    if(overlay)overlay.hidden=true;
  }
  return {
    show,hide,
    update(dt){if(open&&replaying)viewer?.update(dt);},
    resize(){if(open&&replaying)viewer?.resize();},
    handleKey(event){if(open&&replaying&&!event.defaultPrevented)viewer?.handleKey(event);},
    handleEscape(){if(!open)return false;if(replaying)showResults();return true;},
    get isOpen(){return open;},get isReplaying(){return replaying;},
  };
}
