import {renderFrame} from './renderer.js';

export const MODEL_FORMAT='arena-pixel-policy';
export const MODEL_VERSION=1;
export const MODEL_MAX_BYTES=8*1024*1024;
export const OBSERVATION_SPEC=Object.freeze({kind:'semantic-rgb',width:64,height:36,
  scalars:Object.freeze(['health','ammo','reloading','spawnProtection','pitch','alive'])});
export const ACTION_HEADS=Object.freeze([
  {name:'forward',values:[-1,0,1]}, {name:'strafe',values:[-1,0,1]},
  {name:'turn',values:[-3.6,-1.2,0,1.2,3.6]}, {name:'pitch',values:[-1.2,0,1.2]},
  {name:'fire',values:[false,true]}, {name:'reload',values:[false,true]},
].map(head=>Object.freeze({...head,values:Object.freeze(head.values)})));
export const TENSOR_SHAPES=Object.freeze({conv1:[8,3,5,5],conv1Bias:[8],conv2:[16,8,3,3],conv2Bias:[16],
  fc:[64,1574],fcBias:[64],policy:[18,64],policyBias:[18]});
// Finer aiming keeps a six-tick action from stepping over a distant target.
export const ACTION_HEADS_V2=Object.freeze(ACTION_HEADS.map(head=>head.name==='turn'?
  Object.freeze({name:'turn',values:Object.freeze([-3.6,-1.8,-.8,-.3,0,.3,.8,1.8,3.6])}):head));
export const TENSOR_SHAPES_V2=Object.freeze({...TENSOR_SHAPES,policy:[22,64],policyBias:[22]});
export function policyActionHeads(model){return model.architecture==='cnn-v2'?ACTION_HEADS_V2:ACTION_HEADS;}
const clamp=(value,min,max)=>Math.max(min,Math.min(max,value));
const object=value=>value&&typeof value==='object'&&!Array.isArray(value);
const compiled=new WeakMap();

function metadata(value,depth=0){
  if(depth>5)throw new TypeError('Training metadata is too deeply nested.');
  if(value===null||typeof value==='boolean')return value;
  if(typeof value==='number'&&Number.isFinite(value))return value;
  if(typeof value==='string'&&value.length<=2000)return value;
  if(Array.isArray(value)&&value.length<=128)return value.map(item=>metadata(item,depth+1));
  if(object(value)&&Object.keys(value).length<=64){
    const result={};
    for(const [key,item] of Object.entries(value)){
      if(key.length>100||['__proto__','constructor','prototype'].includes(key))throw new TypeError('Invalid metadata key.');
      result[key]=metadata(item,depth+1);
    }
    return result;
  }
  throw new TypeError('Invalid training metadata.');
}

/** Strict, data-only format. Unsupported networks cannot silently run as another model. */
export function validateModel(value){
  if(!object(value)||value.format!==MODEL_FORMAT||value.version!==MODEL_VERSION||!['cnn-v1','cnn-v2'].includes(value.architecture))
    throw new TypeError('Expected an Arena pixel policy v1 (cnn-v1 or cnn-v2).');
  const v2=value.architecture==='cnn-v2',shapes=v2?TENSOR_SHAPES_V2:TENSOR_SHAPES;
  if(typeof value.name!=='string'||!value.name.trim()||value.name.length>80||/[\u0000-\u001f]/.test(value.name))
    throw new TypeError('Model name must contain 1–80 printable characters.');
  if(typeof value.createdAt!=='string'||value.createdAt.length>40||!Number.isFinite(Date.parse(value.createdAt)))
    throw new TypeError('Model createdAt must be an ISO date.');
  const observation=value.observation;
  if(!object(observation)||observation.kind!==OBSERVATION_SPEC.kind||observation.width!==64||observation.height!==36||
      JSON.stringify(observation.scalars)!==JSON.stringify(OBSERVATION_SPEC.scalars))
    throw new TypeError('Model observations must use 64×36 semantic RGB and the supported own-status fields.');
  if(value.actionSpace!==(v2?'discrete-v2':'discrete-v1')||value.actionRepeat!==6)throw new TypeError('Model action space or decision interval is unsupported.');
  if(!object(value.weights)||Object.keys(value.weights).length!==Object.keys(shapes).length)
    throw new TypeError('Model must contain all eight network tensors.');
  const weights={};
  for(const [name,shape]of Object.entries(shapes)){
    const tensor=value.weights[name],size=shape.reduce((a,b)=>a*b,1);
    if(!object(tensor)||JSON.stringify(tensor.shape)!==JSON.stringify(shape)||!Array.isArray(tensor.data)||tensor.data.length!==size)
      throw new TypeError(`Invalid ${name} tensor shape or length.`);
    if(!tensor.data.every(number=>typeof number==='number'&&Number.isFinite(number)&&Math.abs(number)<=100))
      throw new TypeError(`Invalid ${name} weights: finite values within ±100 are required.`);
    weights[name]={shape:[...shape],data:[...tensor.data]};
  }
  const training=value.training===undefined?{}:metadata(value.training);
  if(!object(training)||JSON.stringify(training).length>32768)throw new TypeError('Training metadata exceeds its limit.');
  return {format:MODEL_FORMAT,version:MODEL_VERSION,name:value.name.trim(),createdAt:value.createdAt,
    architecture:value.architecture,observation:{...OBSERVATION_SPEC,scalars:[...OBSERVATION_SPEC.scalars]},
    actionSpace:value.actionSpace,actionRepeat:6,weights,training};
}

/** Only rendered visible pixels and the actor's own status enter the network. */
export function policyObservation(sim,actorId){
  const actor=sim.actors.find(value=>value.id===actorId);
  if(!actor)throw new TypeError(`Unknown actor: ${actorId}`);
  const view={...sim,player:actor,entities:sim.actors.filter(value=>value!==actor)};
  const rgba=renderFrame(view,{width:64,height:36,semantic:true}),pixels=new Uint8Array(64*36*3);
  for(let i=0,j=0;i<rgba.length;i+=4){pixels[j++]=rgba[i];pixels[j++]=rgba[i+1];pixels[j++]=rgba[i+2];}
  const scalars=new Float32Array([actor.health/100,actor.ammo/20,Number(actor.reloadTimer>0),
    clamp(actor.spawnProtection/1.2,0,1),actor.pitch/.45,Number(actor.active)]);
  return {pixels,scalars};
}

function conv(input,channels,height,width,weights,bias,outputs,kernel,result){
  const outH=Math.floor((height-kernel)/2)+1,outW=Math.floor((width-kernel)/2)+1;
  let destination=0;
  for(let oc=0;oc<outputs;oc++)for(let y=0;y<outH;y++)for(let x=0;x<outW;x++){
    let sum=bias[oc],w=oc*channels*kernel*kernel;
    for(let ic=0;ic<channels;ic++)for(let ky=0;ky<kernel;ky++){
      let source=(ic*height+y*2+ky)*width+x*2;
      for(let kx=0;kx<kernel;kx++)sum+=input[source++]*weights[w++];
    }
    result[destination++]=Math.max(0,sum);
  }
}
function dense(input,weights,bias,result,relu){
  for(let output=0;output<result.length;output++){
    let sum=bias[output],offset=output*input.length;
    for(let i=0;i<input.length;i++)sum+=weights[offset+i]*input[i];
    result[output]=relu?Math.max(0,sum):sum;
  }
}
function compile(model){
  if(compiled.has(model))return compiled.get(model);
  const validated=validateModel(model),weights={};
  for(const [key,tensor]of Object.entries(validated.weights))weights[key]=Float32Array.from(tensor.data);
  const heads=policyActionHeads(validated);
  const runtime={weights,heads,input:new Float32Array(3*36*64),conv1:new Float32Array(8*16*30),
    combined:new Float32Array(1574),hidden:new Float32Array(64),logits:new Float32Array(heads.reduce((sum,head)=>sum+head.values.length,0))};
  compiled.set(model,runtime);return runtime;
}
/** PyTorch-compatible CHW convolution, with deterministic argmax per action head. */
export function inferPolicy(model,observation){
  const runtime=compile(model),{pixels,scalars}=observation||{};
  if(!pixels||pixels.length!==64*36*3||!scalars||scalars.length!==6)throw new TypeError('Invalid pixel policy observation shape.');
  for(let pixel=0;pixel<36*64;pixel++)for(let channel=0;channel<3;channel++){
    const value=pixels[pixel*3+channel];
    if(!Number.isFinite(value)||value<0||value>255)throw new TypeError('Pixels must be finite RGB bytes.');
    runtime.input[channel*36*64+pixel]=value/255;
  }
  for(let i=0;i<6;i++){
    if(!Number.isFinite(scalars[i])||scalars[i]<(i===4?-1:0)||scalars[i]>1)throw new TypeError('Own-status values must use the supported normalized ranges.');
    runtime.combined[1568+i]=scalars[i];
  }
  const w=runtime.weights;
  conv(runtime.input,3,36,64,w.conv1,w.conv1Bias,8,5,runtime.conv1);
  conv(runtime.conv1,8,16,30,w.conv2,w.conv2Bias,16,3,runtime.combined);
  dense(runtime.combined,w.fc,w.fcBias,runtime.hidden,true);
  dense(runtime.hidden,w.policy,w.policyBias,runtime.logits,false);
  const action={};let offset=0;
  for(const head of runtime.heads){
    let best=0;
    for(let i=1;i<head.values.length;i++)if(runtime.logits[offset+i]>runtime.logits[offset+best])best=i;
    action[head.name]=head.values[best];offset+=head.values.length;
  }
  return {action,logits:new Float32Array(runtime.logits)};
}

/** A controller can be shared across actors; each gets an independent six-tick clock. */
export function createPolicyController(model){
  compile(model);
  const states=new WeakMap();
  return (sim,actor)=>{
    const state=states.get(actor);
    if(state&&state.deaths===actor.deaths&&sim.tick>=state.tick&&sim.tick-state.tick<6)return state.action;
    const {action}=inferPolicy(model,policyObservation(sim,actor.id));
    states.set(actor,{tick:sim.tick,deaths:actor.deaths,action});return action;
  };
}
