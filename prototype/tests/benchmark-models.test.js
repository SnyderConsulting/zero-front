import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,writeFile,mkdir,rm} from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {createSimulation} from '../dist/simulation.js';
import {MODEL_FORMAT,OBSERVATION_SPEC,TENSOR_SHAPES} from '../dist/policy.js';
import {configureBenchmarkTeams,loadBenchmarkInputs,runBenchmarkMatch,runBenchmark,summarizeBenchmarkMatches} from '../training/benchmark-models.mjs';

function model(name='Test model',bias=0){
  const weights=Object.fromEntries(Object.entries(TENSOR_SHAPES).map(([key,shape])=>
    [key,{shape:[...shape],data:Array(shape.reduce((a,b)=>a*b,1)).fill(0)}]));
  weights.policyBias.data[0]=bias;
  return {format:MODEL_FORMAT,version:1,name,createdAt:'2026-11-07T00:00:00Z',architecture:'cnn-v1',
    observation:{...OBSERVATION_SPEC,scalars:[...OBSERVATION_SPEC.scalars]},actionSpace:'discrete-v1',actionRepeat:6,weights,
    training:{trainingSeedRanges:[[1000000,1010000]]}};
}

test('benchmark assigns both complete teams including human slot with equal bot spread',()=>{
  const candidate=()=>({}),opponent=()=>({});
  for(const team of [0,1])for(const other of [opponent,'scripted']){
    const sim=createSimulation({teamSize:4});configureBenchmarkTeams(sim,team,candidate,other);
    assert.equal(sim.controllers.size,8);
    for(const actor of sim.actors)assert.equal(sim.controllers.get(actor.id),actor.team===team?candidate:other);
    assert.equal(sim.controllers.get('player'),team===0?candidate:other);
  }
  assert.throws(()=>configureBenchmarkTeams(createSimulation(),0,candidate,null),/explicit/);
});

test('benchmark excludes frozen and changing live candidate paths and identical policies but reports bad files',async t=>{
  const root=await mkdtemp(path.join(os.tmpdir(),'arena-benchmark-'));
  t.after(()=>rm(root,{recursive:true,force:true}));
  const library=path.join(root,'models');await mkdir(library);
  const candidate=path.join(root,'frozen.json');await writeFile(candidate,JSON.stringify(model()));
  const live=path.join(library,'live.json');await writeFile(live,JSON.stringify(model('Live changed',2)));
  await writeFile(path.join(library,'copy.json'),JSON.stringify(model('Renamed candidate')));
  await writeFile(path.join(library,'opponent.json'),JSON.stringify(model('Opponent',1)));
  await writeFile(path.join(library,'bad.json'),'{invalid');
  const loaded=await loadBenchmarkInputs(candidate,{modelsDirectory:library,seeds:[600000],excludeModels:[live]});
  assert.deepEqual(loaded.opponents.map(entry=>entry.name),['Scripted','Opponent']);
  assert.deepEqual(loaded.skipped.map(entry=>entry.file),['copy.json','live.json']);
  assert.deepEqual(loaded.errors.map(entry=>entry.file),['bad.json']);
  // Loaded weights are a snapshot, unaffected by a subsequent export to that path.
  await writeFile(path.join(library,'opponent.json'),JSON.stringify(model('Replaced',3)));
  assert.equal(loaded.opponents[1].model.weights.policyBias.data[0],1);
  await assert.rejects(loadBenchmarkInputs(candidate,{modelsDirectory:library,seeds:[1000000]}),/used for training/);
  const options={modelPath:candidate,modelsDirectory:library,seeds:[600000],excludeModels:[live],teamSize:1,maxSeconds:1};
  const serial=await runBenchmark({...options,workers:1}),parallel=await runBenchmark({...options,workers:2});
  assert.deepEqual(parallel.opponents.map(entry=>entry.matches),serial.opponents.map(entry=>entry.matches));
  assert.equal(parallel.trainingFeedback,false);assert.equal(parallel.checkpointSelection,false);
  for(const opponent of parallel.opponents){assert.equal(opponent.overall.matches,2);assert.equal(opponent.overall.winRate,null);}
});

test('short benchmark matches and unfinished rounds never become win-rate claims',()=>{
  const match=runBenchmarkMatch(model(),model('Other',1),{seed:600000,team:1,teamSize:1,maxSeconds:1});
  assert.equal(match.complete,false);assert.equal(match.partial,true);assert.equal(match.win,null);assert.equal(match.winner,null);
  const partial=summarizeBenchmarkMatches([match]);assert.equal(partial.completed,0);assert.equal(partial.winRate,null);
  const summary=summarizeBenchmarkMatches([{complete:true,win:true,draw:false},{complete:true,win:false,draw:true},match]);
  assert.equal(summary.wins,1);assert.equal(summary.draws,1);assert.equal(summary.losses,0);assert.equal(summary.winRate,.5);
  assert.equal(summary.partial,1);
});
