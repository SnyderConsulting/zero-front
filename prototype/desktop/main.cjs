const {app,BrowserWindow,Menu,ipcMain,dialog}=require('electron');
const path=require('node:path');
const {pathToFileURL}=require('node:url');

app.setName('Zero Front');
const entry=path.join(__dirname,'..','dist','index.html');
const entryURL=pathToFileURL(entry).href;
let arena;
let modelCatalog;

async function getModelCatalog(){
  if(!modelCatalog)modelCatalog=Promise.all([import('./model-catalog.mjs'),import('../dist/policy.js')])
    .then(([{createModelCatalog},{validateModel}])=>createModelCatalog({directory:path.join(__dirname,'..','models'),validate:validateModel}));
  return modelCatalog;
}

function checkModelSender(event){
  if(!arena||arena.isDestroyed()||event.sender!==arena.webContents||event.senderFrame!==arena.webContents.mainFrame||
      event.senderFrame.url.split('?')[0]!==entryURL)throw new Error('Model access is limited to the Zero Front window.');
}

ipcMain.handle('arena-models:list',async event=>{
  checkModelSender(event);return (await getModelCatalog()).list();
});
ipcMain.handle('arena-models:read',async(event,id)=>{
  checkModelSender(event);return (await getModelCatalog()).read(id);
});
ipcMain.handle('arena-models:import',async event=>{
  checkModelSender(event);
  const selection=await dialog.showOpenDialog(arena,{title:'Import Zero Front model',properties:['openFile'],filters:[{name:'Zero Front model (JSON)',extensions:['json']}]});
  if(selection.canceled||!selection.filePaths.length)return null;
  checkModelSender(event);
  return (await getModelCatalog()).importFile(selection.filePaths[0]);
});

function sendWindowEvent(name,userGesture=false){
  if(!arena||arena.isDestroyed()||arena.webContents.isLoadingMainFrame())return;
  arena.webContents.executeJavaScript(`window.dispatchEvent(new Event('${name}'));`,userGesture)
    .catch(error=>{if(arena&&!arena.isDestroyed())console.error('Zero Front window event failed:',error.message);});
}

function createArena(){
  arena=new BrowserWindow({
    title:'Zero Front — Team Deathmatch',width:1280,height:800,minWidth:640,minHeight:480,
    show:false,backgroundColor:'#172322',autoHideMenuBar:true,
    webPreferences:{sandbox:true,contextIsolation:true,nodeIntegration:false,webSecurity:true,partition:'arena',preload:path.join(__dirname,'preload.cjs')}
  });
  Menu.setApplicationMenu(null);
  arena.setMenu(null);
  const contents=arena.webContents;
  contents.on('console-message',details=>{
    if(details.message.startsWith('[Zero Front]'))console.log(details.message);
  });
  const isArena=webContents=>webContents===contents&&contents.getURL().split('?')[0]===entryURL;
  contents.session.setPermissionCheckHandler((webContents,permission)=>isArena(webContents)&&permission==='pointerLock');
  contents.session.setPermissionRequestHandler((webContents,permission,callback)=>callback(isArena(webContents)&&permission==='pointerLock'));
  contents.setWindowOpenHandler(()=>({action:'deny'}));
  contents.on('will-navigate',event=>event.preventDefault());
  contents.on('will-attach-webview',event=>event.preventDefault());
  contents.on('page-title-updated',event=>event.preventDefault());
  contents.on('before-input-event',(event,input)=>{
    if(input.type!=='keyDown')return;
    if(input.key==='Escape'){
      event.preventDefault();
      sendWindowEvent('arena-desktop-pause');
    }else if(input.key==='F11'){
      event.preventDefault();
      if(!input.isAutoRepeat)arena.setFullScreen(!arena.isFullScreen());
    }
  });
  contents.on('did-finish-load',()=>{if(arena.isFocused())sendWindowEvent('arena-desktop-focus',true);});
  arena.once('ready-to-show',()=>{arena.show();arena.focus();sendWindowEvent('arena-desktop-focus',true);});
  arena.on('focus',()=>sendWindowEvent('arena-desktop-focus',true));
  arena.on('blur',()=>sendWindowEvent('arena-desktop-blur'));
  arena.on('closed',()=>{arena=null;});
  arena.loadFile(entry,{query:{desktop:'1'}}).catch(error=>{console.error('Zero Front could not start:',error.message);app.quit();});
}

if(!app.requestSingleInstanceLock()){
  app.quit();
}else{
  app.on('second-instance',()=>{if(arena){if(arena.isMinimized())arena.restore();arena.show();arena.focus();}});
  app.whenReady().then(createArena);
  app.on('window-all-closed',()=>app.quit());
}
