import {createHash} from 'node:crypto';
import {mkdir,readdir,lstat,readFile,writeFile,unlink,link} from 'node:fs/promises';
import path from 'node:path';

export const MAX_MODEL_BYTES=8*1024*1024;
const VALID_ID=/^[a-zA-Z0-9][a-zA-Z0-9._-]{0,123}\.json$/i;

/** Plain JSON only. Validation is shared with the in-game policy runtime. */
export function createModelCatalog({directory,validate}){
  if(typeof directory!=='string'||typeof validate!=='function')throw new TypeError('A model directory and validator are required.');
  const root=path.resolve(directory);
  const entry=(id,model)=>({id,name:model.name,createdAt:model.createdAt,training:model.training});
  function resolveId(id){
    if(typeof id!=='string'||!VALID_ID.test(id))throw new TypeError('Invalid model identifier.');
    return path.join(root,id);
  }
  async function loadFile(filename){
    const stat=await lstat(filename);
    if(!stat.isFile())throw new TypeError('Model must be a regular JSON file.');
    if(stat.size>MAX_MODEL_BYTES)throw new RangeError('Model exceeds the 8 MiB size limit.');
    const data=await readFile(filename);
    if(data.byteLength>MAX_MODEL_BYTES)throw new RangeError('Model exceeds the 8 MiB size limit.');
    let input;
    try{input=JSON.parse(data.toString('utf8'));}
    catch{throw new TypeError('Model contains invalid JSON.');}
    return validate(input);
  }
  async function ensureDirectory(){await mkdir(root,{recursive:true});}
  return Object.freeze({
    async list(){
      await ensureDirectory();
      const models=[],errors=[];
      const files=await readdir(root,{withFileTypes:true});
      for(const file of files){
        if(!file.name.toLowerCase().endsWith('.json'))continue;
        try{
          const filename=resolveId(file.name);
          models.push(entry(file.name,await loadFile(filename)));
        }catch(error){errors.push({file:file.name,message:error.message});}
      }
      models.sort((a,b)=>a.name.localeCompare(b.name)||a.id.localeCompare(b.id));
      errors.sort((a,b)=>a.file.localeCompare(b.file));
      return {models,errors};
    },
    async read(id){return loadFile(resolveId(id));},
    /** Only the native file chooser may supply this path; it is never an IPC argument. */
    async importFile(filename){
      const model=await loadFile(filename);
      const json=JSON.stringify(model)+'\n';
      if(Buffer.byteLength(json)>MAX_MODEL_BYTES)throw new RangeError('Model exceeds the 8 MiB size limit.');
      const digest=createHash('sha256').update(json).digest('hex');
      const id=`model-${digest.slice(0,24)}.json`,destination=resolveId(id);
      await ensureDirectory();
      const temporary=path.join(root,`.import-${digest}-${process.pid}-${Math.random().toString(16).slice(2)}.tmp`);
      try{
        await writeFile(temporary,json,{flag:'wx'});
        try{await link(temporary,destination);}
        catch(error){
          if(error.code!=='EEXIST')throw error;
          // Existing identical imports are harmless; never overwrite another model.
          const existing=await loadFile(destination);
          if(JSON.stringify(existing)+'\n'!==json)throw new Error('A different model already uses this identifier.');
        }
      }finally{await unlink(temporary).catch(error=>{if(error.code!=='ENOENT')throw error;});}
      return entry(id,model);
    }
  });
}
