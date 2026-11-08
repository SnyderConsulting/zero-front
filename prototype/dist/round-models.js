import {createPolicyController,validateModel} from './policy.js';

/** Preload selected data-only models so Play remains a synchronous user gesture. */
export function createRoundModels({onStateChange=()=>{}}={}){
  const selects=[document.getElementById('blue-model'),document.getElementById('red-model')];
  const importButton=document.getElementById('import-model'),refreshButton=document.getElementById('refresh-models');
  const fileInput=document.getElementById('model-file'),status=document.getElementById('model-status');
  const bridge=window.arenaModels;
  const localModels=new Map();
  let catalog=[],loaded=new Map(),busy=false,epoch=0,localId=0,error='',initialCatalogLoaded=false;
  const selectedIds=()=>[...new Set(selects.map(select=>select.value).filter(id=>id!=='scripted'))];
  function update(){
    for(const select of selects)select.disabled=busy;
    importButton.disabled=busy;refreshButton.disabled=busy;
    // Keep the training-fingerprinted validator intact; brand its legacy error at the UI boundary.
    status.textContent=error.replace('an Arena pixel policy','a Zero Front pixel policy');status.hidden=!error;
    onStateChange();
  }
  function fillOptions(defaultId=null){
    const available=new Set(catalog.map(entry=>entry.id));
    for(const select of selects){
      const chosen=defaultId||select.value||'scripted';
      const previousLabel=[...select.children].find(option=>option.value===chosen)?.textContent||chosen;
      const baseline=document.createElement('option');baseline.value='scripted';baseline.textContent='Scripted';
      select.replaceChildren(baseline,...catalog.map(entry=>{
        const option=document.createElement('option');option.value=entry.id;option.textContent=entry.name;return option;
      }));
      if(chosen!=='scripted'&&!available.has(chosen)){
        const missing=document.createElement('option');missing.value=chosen;missing.disabled=true;
        missing.textContent=`${previousLabel.replace(/ \(unavailable\)$/,'')} (unavailable)`;
        select.append(missing);
        error='A selected model is no longer available. Choose another model or Scripted.';
      }
      select.value=chosen;
    }
  }
  async function loadSelection({refresh=false}={}){
    const request=++epoch;busy=true;error='';update();
    try{
      if(refresh){
        const result=bridge?await bridge.list():{models:[...localModels].map(([id,model])=>({id,name:model.name})),errors:[]};
        if(request!==epoch)return;
        catalog=result.models||[];
        // Default only the bundled native policy, once. Later refreshes preserve the player's choices.
        // A corrupt bundled file stays selected and blocks Play instead of silently substituting Scripted.
        const bundledId='pixel-selfplay.json';
        const defaultId=bridge&&!initialCatalogLoaded&&
          (catalog.some(entry=>entry.id===bundledId)||result.errors?.some(entry=>entry.file===bundledId))?bundledId:null;
        initialCatalogLoaded=true;loaded.clear();fillOptions(defaultId);
        if(result.errors?.length)error=`${result.errors.length} model file${result.errors.length===1?' was':'s were'} skipped: ${result.errors[0].message}`;
      }
      const ids=selectedIds();
      if(ids.some(id=>!catalog.some(entry=>entry.id===id)))throw new Error('A selected model is no longer available. Choose another model or Scripted.');
      const next=new Map();
      await Promise.all(ids.map(async id=>{
        const model=loaded.get(id)||validateModel(bridge?await bridge.read(id):localModels.get(id));
        next.set(id,model);
      }));
      if(request!==epoch)return;
      loaded=next;
    }catch(cause){
      if(request!==epoch)return;
      error=cause?.message||'Could not load this model.';
      // Never play a silently substituted policy after a selected model fails.
      if(refresh)loaded.clear();
    }finally{if(request===epoch){busy=false;update();}}
  }
  async function importModel(){
    if(!bridge){fileInput.value='';fileInput.click();return;}
    const request=++epoch;busy=true;error='';update();
    try{
      const entry=await bridge.import();
      if(request!==epoch)return;
      if(entry)await loadSelection({refresh:true});
    }catch(cause){if(request===epoch)error=cause?.message||'Could not import this model.';}
    finally{if(request===epoch){busy=false;update();}}
  }
  for(const select of selects)select.addEventListener('change',()=>loadSelection());
  importButton.addEventListener('click',importModel);
  refreshButton.addEventListener('click',()=>loadSelection({refresh:true}));
  fileInput.addEventListener('change',async()=>{
    const file=fileInput.files?.[0];if(!file)return;
    const request=++epoch;busy=true;error='';update();
    try{
      if(file.size>8*1024*1024)throw new Error('Model files must be smaller than 8 MiB.');
      const model=validateModel(JSON.parse(await file.text()));
      if(request!==epoch)return;
      localModels.set(`local-${++localId}`,model);await loadSelection({refresh:true});
    }catch(cause){if(request===epoch)error=cause?.message||'Could not import this model.';}
    finally{if(request===epoch){busy=false;update();}}
  });
  return {
    refresh:()=>loadSelection({refresh:true}),
    get ready(){return !busy&&selectedIds().every(id=>loaded.has(id));},
    controllers(){
      if(!this.ready)throw new Error('Wait for the selected models to load.');
      return selects.map(select=>select.value==='scripted'?null:createPolicyController(loaded.get(select.value)));
    },
  };
}
