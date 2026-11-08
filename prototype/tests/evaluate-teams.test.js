import test from 'node:test';
import assert from 'node:assert/strict';
import {createSimulation} from '../dist/simulation.js';
import {configureTeams,parseSeeds,summarizeMatches,evaluationReport,assertEvaluationSeeds} from '../training/evaluate-teams.mjs';

test('full-team evaluation assigns every role and swaps the human slot without leaving scripted teammates',()=>{
  const controller=()=>({});
  for(const team of [0,1]){
    const sim=createSimulation({teamSize:4});configureTeams(sim,team,controller);
    assert.equal(sim.controllers.size,8);
    for(const actor of sim.actors)assert.equal(sim.controllers.get(actor.id),actor.team===team?controller:'scripted');
    assert.equal(sim.controllers.get('player'),team===0?controller:'scripted');
  }
});

test('round summaries count draws and unfinished rounds correctly and require paired complete trials',()=>{
  const matches=Array.from({length:40},(_,index)=>({seed:4000+Math.floor(index/2),team:index%2,layout:'room-web',
    complete:true,win:index<32,draw:index===32,modelScore:index<32?25:10,scriptedScore:index<32?10:25}));
  const summary=summarizeMatches(matches);assert.equal(summary.winRate,.8);assert.equal(summary.losses,7);assert.equal(summary.draws,1);
  assert.ok(summary.winRateWilson95[0]<.8&&summary.winRateWilson95[1]>.8);
  const options={seeds:Array.from({length:20},(_,i)=>4000+i),teamSize:4};
  assert.equal(evaluationReport({name:'model',training:{}},'hash',matches,options).criterion.targetMet,true);
  const partial=matches.map((row,index)=>index===39?{...row,complete:false}:row);
  assert.equal(evaluationReport({name:'model',training:{}},'hash',partial,options).criterion.targetMet,false);
  const unpaired=matches.map((row,index)=>index===39?{...row,team:0}:row);
  assert.equal(evaluationReport({name:'model',training:{}},'hash',unpaired,options).criterion.targetMet,false);
});

test('tuning maps can be reused for checkpoint selection but final maps reject all seen provenance',()=>{
  const metadata={trainSeeds:[1],priorTrainingSeeds:[2],validationSeeds:[3]};
  assert.throws(()=>assertEvaluationSeeds([1],metadata,'tuning'));
  assert.doesNotThrow(()=>assertEvaluationSeeds([3],metadata,'tuning'));
  assert.throws(()=>assertEvaluationSeeds([3],metadata,'final'));
  assert.doesNotThrow(()=>assertEvaluationSeeds([50000],metadata,'final'));
  assert.deepEqual(parseSeeds('1:4,2,8'),[1,2,3,8]);
  for(const raw of ['',':2','2:1','0:101','-1','4294967296'])assert.throws(()=>parseSeeds(raw));
});

test('seed range provenance excludes optimization and checkpoint selection ranges without expanding them',()=>{
  const training={trainingSeedRanges:[[100,200]],priorTrainingSeedRanges:[[10,20]],selectionSeedRanges:[[300,305]]};
  for(const seed of [10,19,100,199])assert.throws(()=>assertEvaluationSeeds([seed],training,'tuning'));
  assert.doesNotThrow(()=>assertEvaluationSeeds([300],training,'tuning'));
  for(const seed of [10,100,300,304])assert.throws(()=>assertEvaluationSeeds([seed],training,'final'));
  assert.doesNotThrow(()=>assertEvaluationSeeds([20,99,200,305],training,'final'));
  for(const range of [[0,0],[-1,10],[0,4294967297],[0,1.5],[0],[0,1,2]])
    assert.throws(()=>assertEvaluationSeeds([50000],{trainingSeedRanges:[range]},'final'),TypeError);
});
