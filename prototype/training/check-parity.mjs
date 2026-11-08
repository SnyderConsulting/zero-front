/** Verify a Python/Torch reference against the exact in-game JavaScript runtime. */
import {readFile} from 'node:fs/promises';
import {inferPolicy,validateModel} from '../dist/policy.js';

const args=process.argv.slice(2);
const value=name=>{const index=args.indexOf(name);return index<0?undefined:args[index+1];};
const modelPath=value('--model'),fixturePath=value('--fixture');
if(!modelPath||!fixturePath)throw new Error('Usage: node training/check-parity.mjs --model model.json --fixture reference.json');
const raw=JSON.parse(await readFile(modelPath,'utf8'));
const model=validateModel(raw);
const fixture=JSON.parse(await readFile(fixturePath,'utf8'));
const input={pixels:new Uint8Array(Buffer.from(fixture.pixels,'base64')),scalars:new Float32Array(fixture.scalars)};
const actual=inferPolicy(model,input);
if(actual.logits.length!==fixture.logits.length)throw new Error('Logit count differs from PyTorch.');
const error=Math.max(...actual.logits.map((value,index)=>Math.abs(value-fixture.logits[index])));
if(!Number.isFinite(error)||error>1e-4)throw new Error(`JavaScript/PyTorch logit mismatch: max error ${error}.`);
console.log(JSON.stringify({parity:'passed',maximumAbsoluteLogitError:error,tolerance:1e-4}));
