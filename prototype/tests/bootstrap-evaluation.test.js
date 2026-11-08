import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,writeFile,rm} from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import {MODEL_FORMAT,OBSERVATION_SPEC,TENSOR_SHAPES} from '../dist/policy.js';
import {runBootstrapMatch,summarizeBootstrapMatches,bootstrapJobs,evaluateBootstrap,assertBootstrapSeeds} from '../training/evaluate-bootstrap.mjs';

function policy({move=false}={}){
  const weights=Object.fromEntries(Object.entries(TENSOR_SHAPES).map(([key,shape])=>
    [key,{shape:[...shape],data:Array(shape.reduce((a,b)=>a*b,1)).fill(0)}]));
  for(const index of [move?2:1,4,8,12,14,16])weights.policyBias.data[index]=5;
  return {format:MODEL_FORMAT,version:1,name:'Test pixel policy',createdAt:'2026-11-07T00:00:00Z',architecture:'cnn-v1',
    observation:{...OBSERVATION_SPEC,scalars:[...OBSERVATION_SPEC.scalars]},actionSpace:'discrete-v1',actionRepeat:6,weights,
    training:{trainSeeds:[1000],validationSeeds:[2000]}};
}

test('bootstrap diagnostics count one self match per seed and both scripted sides',()=>{
  const jobs=bootstrapJobs([610000,610001]);
  assert.equal(jobs.length,6);
  assert.deepEqual(jobs.filter(row=>row.opponent==='self').map(row=>row.seed),[610000,610001]);
  for(const seed of [610000,610001])assert.deepEqual(jobs.filter(row=>row.seed===seed&&row.opponent==='scripted').map(row=>row.team),[0,1]);
});

test('self diagnostics cover both whole teams and distinguish actual movement from idle actions',()=>{
  const idle=runBootstrapMatch(policy(),{seed:610000,opponent:'self',teamSize:2,maxSeconds:1});
  assert.equal(idle.players.length,4);assert.ok(idle.players.some(actor=>actor.id==='player'));
  assert.deepEqual(new Set(idle.players.map(actor=>actor.team)),new Set([0,1]));
  assert.equal(idle.distance,0);assert.equal(idle.hits,0);assert.equal(idle.firstHitSeconds,null);
  const summary=summarizeBootstrapMatches([idle],{self:true});
  assert.equal(summary.stationaryFraction,1);assert.equal(summary.movingFraction,0);
  assert.equal(summary.zeroContactMatches,1);assert.equal(summary.winRate,null);assert.equal(summary.wins,null);
  assert.equal(idle.complete,false);assert.equal(idle.win,null);assert.equal(idle.draw,null);
  const moving=runBootstrapMatch(policy({move:true}),{seed:610000,opponent:'self',teamSize:2,maxSeconds:1});
  assert.ok(moving.distance>0);assert.ok(summarizeBootstrapMatches([moving],{self:true}).movingFraction>0);
});

test('bootstrap reports mark partials, reject training maps, and preserve parallel determinism',async t=>{
  const directory=await mkdtemp(path.join(os.tmpdir(),'arena-bootstrap-'));
  t.after(()=>rm(directory,{recursive:true,force:true}));
  const modelPath=path.join(directory,'model.json');await writeFile(modelPath,JSON.stringify(policy()));
  await assert.rejects(evaluateBootstrap({modelPath,seeds:[1000],teamSize:1,maxSeconds:1,workers:1}),/used for training/);
  const options={modelPath,seeds:[610000],teamSize:1,maxSeconds:1};
  const serial=await evaluateBootstrap({...options,workers:1}),parallel=await evaluateBootstrap({...options,workers:2});
  assert.deepEqual(serial.scripted,parallel.scripted);assert.deepEqual(serial.self,parallel.self);
  assert.equal(serial.purpose,'integration-smoke');assert.equal(serial.scripted.overall.matches,2);
  assert.equal(serial.self.overall.matches,1);assert.equal(serial.scripted.overall.winRate,null);
  assert.equal(serial.self.overall.winRate,null);assert.equal(serial.checkpointSelectionAllowed,true);
});

test('first-contact averages include only games with measured contact and self wins stay undefined',()=>{
  const rows=[{complete:true,win:false,draw:true,hits:0,kills:0,firstHitSeconds:null,firstKillSeconds:null},
    {complete:true,win:false,draw:false,hits:4,kills:1,firstHitSeconds:8,firstKillSeconds:15}];
  const result=summarizeBootstrapMatches(rows,{self:true});
  assert.equal(result.contactMatches,1);assert.equal(result.zeroContactMatches,1);assert.equal(result.killMatches,1);
  assert.equal(result.meanFirstHitSecondsAmongContactMatches,8);assert.equal(result.meanFirstKillSecondsAmongKillMatches,15);
  assert.equal(result.wins,null);assert.equal(result.losses,null);assert.equal(result.draws,1);
});

test('final bootstrap maps exclude training, validation and selection throughout recorded ancestry',async()=>{
  const provenance={trainSeeds:[1000],priorTrainingSeedRanges:[[1100,1103]],validationSeeds:[2000],
    priorValidationSeeds:[2100],priorValidationSeedRanges:[[2200,2202]],selectionSeeds:[610000],
    priorSelectionSeedRanges:[[610010,610015]],initialTraining:{trainingSeedRanges:[[1200,1205]],
      validationSeedRanges:[[2300,2305]],selectionSeedRanges:[[610020,610025]]}};
  for(const seed of [1000,1101,1201,2000,2100,2201,2301,610000,610011,610021])
    assert.throws(()=>assertBootstrapSeeds([seed],provenance,'final'),/used for training or model selection/);
  assert.doesNotThrow(()=>assertBootstrapSeeds([620000,620019],provenance,'final'));
  assert.doesNotThrow(()=>assertBootstrapSeeds([2000,2100,2301,610000,610021],provenance,'tuning'));
  assert.throws(()=>assertBootstrapSeeds([1201],provenance,'tuning'),/used for training/);
  assert.throws(()=>assertBootstrapSeeds([620000],provenance,'unknown'),/Purpose/);
  await assert.rejects(evaluateBootstrap({modelPath:'unused.json',purpose:'final',maxSeconds:1}),/complete 180/);
});
