const {contextBridge,ipcRenderer}=require('electron');

// Expose named operations only; no paths, raw IPC, filesystem, or executable code.
contextBridge.exposeInMainWorld('arenaModels',Object.freeze({
  list:()=>ipcRenderer.invoke('arena-models:list'),
  read:id=>ipcRenderer.invoke('arena-models:read',id),
  import:()=>ipcRenderer.invoke('arena-models:import')
}));
