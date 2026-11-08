"""Train cnn-v2 against strong expert labels, optionally on learner-visited states."""
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import numpy as np
import torch
from torch import nn
import train as base

base.HEADS=[('forward',3),('strafe',3),('turn',9),('pitch',3),('fire',2),('reload',2)]
base.TENSORS['policy']=('policy.weight',[22,64])
base.TENSORS['policyBias']=('policy.bias',[22])
OldPolicy=base.PixelPolicy
class ExpertPolicy(OldPolicy):
    def __init__(self):
        super().__init__()
        self.policy=nn.Linear(64,22)
base.PixelPolicy=ExpertPolicy
old_parser=base.parser
def parser():
    result=old_parser()
    result.set_defaults(name='Pixel expert v2',output='training/runs/expert-v2.json',epochs=30,
        max_samples=120000,max_validation_samples=12000,seconds=90,
        train_seeds=base.seed_list('1000:1040'),validation_seeds=base.seed_list('2000:2005'),
        parity_output='training/runs/expert-v2-parity.json')
    result.add_argument('--learner',help='Collect DAgger labels on states visited by this model instead of expert demonstrations.')
    result.add_argument('--replay-data',action='append',help='Mix previous demonstrations from this NPZ cache. Repeat to aggregate several prior rounds; total replay matches the new sample count.')
    result.add_argument('--cache-dir',default='training/runs',help='Directory for demonstration caches with embedded provenance.')
    result.add_argument('--collection-workers',type=int,choices=range(1,5),default=1,
        help='Independent Node collectors, 1–4. Seed order and per-seed quotas stay deterministic.')
    result.add_argument('--scripted-teacher',action='store_true',help='Use the built-in scripted bots as the demonstration teacher.')
    result.add_argument('--reactive-nav',action='store_true',help='Use deterministic pixel-only wall avoidance labels instead of privileged navigation.')
    result.add_argument('--camp',action='store_true',help='Use observable scan-and-engage behavior, without navigation labels.')
    result.add_argument('--scan-rate',type=float,default=.3,help='No-target scan speed when undamaged, in radians per second (0–3.6).')
    return result
base.parser=parser

def collection_groups(seeds,max_samples,workers):
    """Preserve the serial collector's per-seed floor quota, including unused remainder."""
    if not seeds or not 1<=workers<=4 or max_samples<len(seeds):
        raise ValueError('Invalid collection partition settings.')
    count=min(workers,len(seeds));size,remainder=divmod(len(seeds),count)
    quota=max_samples//len(seeds);start=0;groups=[]
    for index in range(count):
        end=start+size+int(index<remainder)
        groups.append((seeds[start:end],quota*(end-start)))
        start=end
    return groups


def stop_process(process):
    if process.poll() is not None:return
    try:process.terminate()
    except OSError:
        if process.poll() is None:raise
    try:process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:process.kill()
        except OSError:
            if process.poll() is None:raise
        process.wait(timeout=5)


class CollectionProcesses:
    """Cancel collection as a group and reap every Node child after an error."""
    def __init__(self):
        self.lock=threading.Lock();self.cancelled=False;self.processes=set()

    def start(self,*arguments,**options):
        with self.lock:
            if self.cancelled:raise RuntimeError('Collection cancelled after another worker failed.')
            process=subprocess.Popen(*arguments,**options)
            self.processes.add(process)
            return process

    def finished(self,process):
        with self.lock:self.processes.discard(process)

    def abort(self):
        with self.lock:
            self.cancelled=True;processes=list(self.processes)
        errors=[]
        for process in processes:
            try:stop_process(process)
            except OSError as error:errors.append(error)
        if errors:raise errors[0]


def collect_worker(seeds,args,split,max_samples,processes):
    command={'command':'collect','seeds':seeds,'seconds':args.seconds,'reactiveNav':args.reactive_nav,'camp':args.camp,'scanRate':args.scan_rate,
        'scriptedTeacher':args.scripted_teacher,'teamSize':args.team_size,'actorsPerFrame':args.actors_per_frame,
        'maxSamples':max_samples}
    if args.learner:command['model']=str(base.resolve(args.learner))
    errors=tempfile.TemporaryFile(mode='w+',encoding='utf-8')
    try:
        process=processes.start([args.node,str(base.ROOT/'training/expert.mjs')],cwd=base.ROOT,
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=errors,text=True,encoding='utf-8',
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    except BaseException:
        errors.close();raise
    def diagnostics():
        errors.seek(0);return errors.read()
    pixels,scalars,labels=[],[],[]
    try:
        process.stdin.write(json.dumps(command)+'\n');process.stdin.flush()
        while True:
            line=process.stdout.readline()
            if not line:raise RuntimeError('Expert bridge stopped: '+diagnostics())
            packet=json.loads(line)
            if packet['type']=='error':raise RuntimeError(packet['error'])
            if packet['type']=='samples':
                count=packet['count']
                pixels.append(np.frombuffer(base64.b64decode(packet['pixels'],validate=True),dtype=np.uint8).reshape(count,36,64,3))
                scalars.append(np.asarray(packet['scalars'],dtype=np.float32).reshape(count,6))
                labels.append(np.asarray(packet['labels'],dtype=np.int64).reshape(count,6))
            if packet['type']=='progress':print(f'Expert {split}: seed {packet["seed"]}, total {packet["total"]}',flush=True)
            if packet['type']=='complete':break
        process.stdin.write('{"command":"close"}\n');process.stdin.flush();process.stdin.close();process.wait(timeout=10)
        if process.returncode:raise RuntimeError(diagnostics())
        if not pixels:raise RuntimeError('No expert demonstration samples collected.')
        return np.concatenate(pixels),np.concatenate(scalars),np.concatenate(labels)
    finally:
        try:stop_process(process)
        finally:
            processes.finished(process)
            for stream in (process.stdin,process.stdout,errors):
                if not stream.closed:stream.close()


def collect(seeds,args,split):
    maximum=args.max_samples if split=='train' else args.max_validation_samples
    groups=collection_groups(seeds,maximum,args.collection_workers)
    processes=CollectionProcesses()
    if len(groups)==1:
        return collect_worker(seeds,args,split,maximum,processes)
    results=[None]*len(groups)
    with ThreadPoolExecutor(max_workers=len(groups),thread_name_prefix='arena-collect') as executor:
        futures={}
        try:
            for index,(group,limit) in enumerate(groups):
                futures[executor.submit(collect_worker,group,args,f'{split} [{index+1}/{len(groups)}]',limit,processes)]=index
            for future in as_completed(futures):results[futures[future]]=future.result()
        except BaseException:
            processes.abort()
            for future in futures:future.cancel()
            raise
    return tuple(np.concatenate([result[index] for result in results]) for index in range(3))

def dataset_source_hash():
    source=hashlib.sha256()
    for filename in ('dist/simulation.js','dist/renderer.js','dist/mapgen.js','dist/policy.js','training/expert.mjs','training/train-expert.py'):
        source.update(filename.encode());source.update((base.ROOT/filename).read_bytes())
    return source.hexdigest()


def load_replay(path,validation_seeds):
    with np.load(path,allow_pickle=False) as prior:
        if 'provenance' not in prior:
            raise ValueError('Replay dataset has no seed/source provenance; recollect it with the current trainer.')
        provenance=json.loads(str(prior['provenance'].item()))
        seeds=base.known_training_seeds(provenance)
        if not seeds or seeds.intersection(validation_seeds):
            raise ValueError('Replay training seeds are missing or overlap the current validation set.')
        previous=tuple(prior['train_'+key] for key in ('pixels','scalars','labels'))
    base.validate_dataset(previous)
    return previous,provenance,seeds


def load_dataset(args):
    if not np.isfinite(args.scan_rate) or not 0<=args.scan_rate<=3.6:
        raise ValueError('--scan-rate must be between 0 and 3.6 radians per second.')
    if sum(map(bool,(args.scripted_teacher,args.reactive_nav,args.camp)))>1:
        raise ValueError('Choose only one teacher behavior.')
    if args.actors_per_frame>args.team_size:
        raise ValueError('Expert collection samples one team; actors-per-frame cannot exceed team-size.')
    if args.scripted_teacher and args.learner:
        raise ValueError('Built-in scripted teacher supports demonstration collection only; use the stateful expert for learner corrections.')
    prior_seeds=set()
    learner_hash=None
    if args.learner:
        learner_path=base.resolve(args.learner)
        learner_hash=hashlib.sha256(learner_path.read_bytes()).hexdigest()
        prior_seeds.update(base.known_training_seeds(json.loads(learner_path.read_text(encoding='utf-8')).get('training',{})))
        if prior_seeds.intersection(args.validation_seeds):
            raise ValueError('Learner training seeds overlap validation.')
    config={'trainSeeds':args.train_seeds,'validationSeeds':args.validation_seeds,'seconds':args.seconds,
        'maxSamples':args.max_samples,'maxValidationSamples':args.max_validation_samples,'reactiveNav':args.reactive_nav,'camp':args.camp,'scanRate':args.scan_rate,
        'scriptedTeacher':args.scripted_teacher,'teamSize':args.team_size,'actorsPerFrame':args.actors_per_frame,
        'sourceSha256':dataset_source_hash(),'learner':learner_hash,
        'priorTrainingSeeds':sorted(prior_seeds),
        'validationDistribution':'learner-visited held-out maps' if args.learner else 'teacher-visited held-out maps'}
    config['sourceDigest']=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    cache=base.resolve(args.cache_dir)/('expert-data-'+config['sourceDigest'][:16]+'.npz')
    if cache.exists() and not args.no_cache:
        with np.load(cache,allow_pickle=False) as data:
            if 'provenance' not in data or json.loads(str(data['provenance'].item()))!=config:
                raise ValueError('Cached demonstration provenance does not match requested configuration.')
            train=tuple(data['train_'+key] for key in ['pixels','scalars','labels'])
            valid=tuple(data['valid_'+key] for key in ['pixels','scalars','labels'])
        print(f'Loaded {cache}',flush=True)
    else:
        train=collect(args.train_seeds,args,'train');valid=collect(args.validation_seeds,args,'validation')
        if not args.no_cache:
            cache.parent.mkdir(parents=True,exist_ok=True)
            temporary=cache.with_suffix('.partial.npz')
            np.savez_compressed(temporary,provenance=np.array(json.dumps(config,sort_keys=True)),
                **{f'{split}_{key}':value for split,data in [('train',train),('valid',valid)]
                   for key,value in zip(['pixels','scalars','labels'],data)})
            os.replace(temporary,cache)
    for data in (train,valid):base.validate_dataset(data)
    config['datasetCache']=str(cache) if not args.no_cache else None
    # Worker count affects throughput only; it is deliberately outside the raw cache identity.
    config['collectionWorkers']=args.collection_workers
    config['collectedSamples']=len(train[0])
    if args.replay_data:
        replay_paths=list(dict.fromkeys(str(base.resolve(value)) for value in args.replay_data))
        if len(replay_paths)>8:raise ValueError('At most eight replay caches may be mixed.')
        rng=np.random.default_rng(args.seed)
        quota=max(1,len(train[0])//len(replay_paths));replays=[]
        for replay_path in replay_paths:
            previous,provenance,seeds=load_replay(replay_path,args.validation_seeds)
            prior_seeds.update(seeds)
            count=min(quota,len(previous[0]))
            selection=rng.choice(len(previous[0]),count,replace=False)
            train=tuple(np.concatenate([current,old[selection]]) for current,old in zip(train,previous))
            replays.append({'sourceDigest':provenance['sourceDigest'],'samples':count})
            print(f'Mixed {count} prior demonstration states from {replay_path}.',flush=True)
        config['replayDatasets']=replays
    config['priorTrainingSeeds']=sorted(prior_seeds)
    config['datasetDigest']=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    args.dataset_provenance=config
    return train,valid,config
base.load_dataset=load_dataset

def import_weights(model,path,validation_seeds):
    raw=json.loads(path.read_text(encoding='utf-8'))
    if raw.get('format')!='arena-pixel-policy' or raw.get('version')!=1 or raw.get('architecture')!='cnn-v2':
        raise ValueError('Expert resume requires an Arena cnn-v2 JSON policy.')
    if raw.get('observation')!={'kind':'semantic-rgb','width':64,'height':36,'scalars':base.SCALARS} or raw.get('actionSpace')!='discrete-v2' or raw.get('actionRepeat')!=6:
        raise ValueError('Expert resume observation/action contract differs.')
    prior=base.known_training_seeds(raw.get('training',{}))
    if prior.intersection(validation_seeds):raise ValueError('Validation seeds overlap earlier training.')
    state={}
    for name,(key,shape) in base.TENSORS.items():
        tensor=raw['weights'][name];data=np.asarray(tensor['data'],dtype=np.float32)
        if tensor['shape']!=shape or data.size!=np.prod(shape) or not np.isfinite(data).all() or np.abs(data).max()>100:raise ValueError('Invalid model tensor '+name)
        state[key]=torch.from_numpy(data.reshape(shape))
    model.load_state_dict(state);return raw
base.import_weights=import_weights
old_export=base.export_model
def export_model(model,args,metadata):
    result=old_export(model,args,metadata);result['architecture']='cnn-v2';result['actionSpace']='discrete-v2'
    result['training']['teacher']='built-in-scripted-bot' if args.scripted_teacher else 'scan-and-engage-visible-pixel-expert' if args.camp else 'reactive-visible-pixel-expert' if args.reactive_nav else 'precise-visible-pixel-aim-with-privileged-navigation'
    data=args.dataset_provenance
    result['training'].update({key:data[key] for key in ('datasetCache','datasetDigest','collectedSamples','validationDistribution','collectionWorkers')})
    if data.get('replayDatasets'):result['training']['replayDatasets']=data['replayDatasets']
    if data.get('learner'):result['training']['learnerDatasetSha256']=data['learner']
    if args.camp:result['training']['scanRate']=args.scan_rate
    result['training']['algorithm']='dagger' if args.learner else 'behavioral-cloning-expert'
    if args.learner:result['training']['learnerDatasetModel']=str(args.learner)
    if args.resume:
        prior=json.loads(base.resolve(args.resume).read_text(encoding='utf-8')).get('training',{})
        result['training']['priorValidationSeeds']=sorted(set(prior.get('priorValidationSeeds',[])+prior.get('validationSeeds',[])))
        if prior.get('selectionSeeds'):result['training']['selectionSeeds']=prior['selectionSeeds']
    return result
base.export_model=export_model

if __name__=='__main__':
    base.main()
