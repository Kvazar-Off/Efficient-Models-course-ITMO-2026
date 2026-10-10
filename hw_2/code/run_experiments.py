"""Explicit experiment list; two independent GPU queues, no test-driven tuning."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from collections import Counter
from train import save_json


def experiments():
    jobs=[]
    def add(name,stage,script,args):
        jobs.append({'name':name,'stage':stage,'script':script,'args':list(map(str,args))})
    cifar=['--dataset','cifar10','--batch-size',1000,'--workers',2,'--cache-data','--weight-decay',1e-4,
           '--bn-momentum',.05,'--momentum',.9,'--shortcut','pad','--eval-every',1000,'--channels-last']
    for seed in (0,1,2):
        mnist=['--dataset','mnist','--model','lenet','--batch-size',512,'--workers',2,'--cache-data',
               '--weight-decay',.0005,'--seed',seed,'--eval-every',1170]
        add(f'mnist_inv_seed{seed}','main','train.py',mnist+['--policy','inv','--max-lr',.01,'--epochs',85])
        add(f'mnist_sc_seed{seed}','main','train.py',mnist+['--policy','onecycle','--init-lr',.01,
            '--max-lr',.1,'--final-lr',1e-6,'--epochs',12,'--pct-start',5/12,'--anneal-fraction',2/12])
        add(f'mnist_sc_cm_seed{seed}','main','train.py',mnist+['--policy','onecycle','--init-lr',.01,
            '--max-lr',.1,'--final-lr',1e-6,'--epochs',12,'--pct-start',5/12,'--anneal-fraction',2/12,
            '--mom-max',.95,'--mom-min',.8])
        add(f'r20_pc_seed{seed}','main','train.py',cifar+['--model','resnet20','--seed',seed,
            '--policy','step','--max-lr',.35,'--iters',80000,'--drop-iters',50000,70000])
        add(f'r20_clr_seed{seed}','main','train.py',cifar+['--model','resnet20','--seed',seed,
            '--policy','clr','--min-lr',.1,'--max-lr',3,'--stepsize',10000,'--iters',20000])
    add('r56_clr_seed0','main','train.py',cifar+['--model','resnet56','--seed',0,
        '--policy','clr','--min-lr',.1,'--max-lr',3,'--stepsize',5000,'--iters',10000])
    add('r56_pc_seed0','main','train.py',cifar+['--model','resnet56','--seed',0,
        '--policy','step','--max-lr',.35,'--iters',80000,'--drop-iters',50000,70000,
        '--bn-momentum',.001])
    for model,dataset,maxlr,steps,bs in [('resnet56','cifar10',3,5000,1000),('lenet','mnist',.5,1200,512)]:
        add(f'range_{model}','ranges','range_test.py',['--model',model,'--dataset',dataset,
            '--init-lr',.01 if dataset=='mnist' else .1,'--max-lr',maxlr,'--steps',steps,
            '--batch-size',bs,'--weight-decay',.0005 if dataset=='mnist' else .0001,
            '--workers',2,'--cache-data','--eval-every',1000]+(['--channels-last'] if dataset=='cifar10' else []))
    # Controlled ablations: same architecture, seed, update count, momentum, batch and BN.
    base=cifar+['--model','resnet20','--seed',0,'--policy','clr','--min-lr',.1,
                '--max-lr',3,'--stepsize',10000,'--iters',20000]
    for wd in (1e-6,5e-4):add(f'r20_wd_{wd:g}','extras','train.py',base+['--weight-decay',wd])
    for lr in (1,6):add(f'r20_peak_{lr}','extras','train.py',base+['--max-lr',lr])
    add('r20_bn_0.001','extras','train.py',base+['--bn-momentum',.001])
    # Same fixed subset across seeds and same budget across policies; local ablation, not Table 1.
    for n in (10000,25000):
        for policy in ('clr','step'):
            args=base+['--subset',n,'--policy',policy]
            if policy=='step':args+=['--max-lr',.35,'--drop-iters',12500,17500]
            add(f'r20_subset_{n}_{policy}','extras','train.py',args)
    return jobs


def live_runs(root):
    """Find only main training processes; DataLoader children share their cmdline."""
    matches={}
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():continue
        try:
            args=entry.joinpath('cmdline').read_bytes().decode().strip('\0').split('\0')
            if '--out' not in args or not any(Path(s).name in ('train.py','range_test.py') for s in args[:2]):continue
            out=Path(args[args.index('--out')+1]).resolve()
            if out.parent != root/'runs/v2':continue
            stat=entry.joinpath('stat').read_text().rsplit(')',1)[1].split()
            if stat[0]=='Z':continue
            env=entry.joinpath('environ').read_bytes().split(b'\0')
            gpu=next((s.split(b'=',1)[1].decode() for s in env if s.startswith(b'CUDA_VISIBLE_DEVICES=')),None)
            matches[int(entry.name)]={'name':out.name,'gpu':gpu,'ppid':int(stat[1])}
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
    return {pid:run for pid,run in matches.items() if run['ppid'] not in matches}


def terminal(artifact):
    if not artifact.exists():return False
    d=json.loads(artifact.read_text());reason=(d.get('error') or '').lower()
    return d.get('status') in ('completed','nonfinite') or d.get('status')=='failed' and any(t in reason for t in ('non-finite','nonfinite','nan'))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=['all','main','ranges','extras'],nargs='?',default='all')
    p.add_argument('--gpus',nargs='+',default=['0','1'])
    p.add_argument('--attach-running',action='store_true',help='monitor existing runs without restarting them')
    p.add_argument('--slots-per-gpu',type=int,default=None,help='override measured concurrency')
    a=p.parse_args()
    root=Path(__file__).resolve().parent.parent
    os.chdir(root)
    jobs=experiments()
    performance=root/'runs/v2/performance.json'
    tuning=json.loads(performance.read_text())['gpus'] if performance.exists() else {}
    capacities={gpu:a.slots_per_gpu or tuning.get(gpu,{}).get('selected',{}).get('slots',1) for gpu in a.gpus}
    if any(n<1 for n in capacities.values()):p.error('slots per GPU must be positive')
    Path('runs/v2').mkdir(parents=True,exist_ok=True)
    save_json('runs/v2/manifest.json',{'jobs':jobs,'selection':'final fixed-budget accuracy',
        'adaptations':['single-device whole-batch BN','drop_last full batches','PyTorch kernels and data ordering'],
        'seeds_main':[0,1,2],'ablation_seeds':[0],
        'execution':{'capacities':capacities,'performance':'runs/v2/performance.json' if performance.exists() else None,
                     'existing_run_args':'history.json remains authoritative for already-started runs'}})
    chosen=[j for j in jobs if a.stage=='all' or j['stage']==a.stage]
    active=live_runs(root)
    if active and not a.attach_running:p.error('training already running; use --attach-running to preserve it')
    print(f'GPU capacities: {capacities}; attached: {[r["name"] for r in active.values()]}',flush=True)
    queue=[];artifacts={}
    for job in chosen:
        out=root/'runs/v2'/job['name']
        artifact=out/('range_test.json' if job['script']=='range_test.py' else 'history.json')
        artifacts[job['name']]=artifact
        if terminal(artifact):continue
        if any(r['name']==job['name'] for r in active.values()):continue
        if artifact.exists():raise RuntimeError(f'incomplete run {out}; use a new output directory')
        queue.append(job)
    owned={}
    while queue or active:
        for pid,run in list(active.items()):
            proc=owned.get(pid)
            if proc:finished=proc.poll() is not None
            else:
                try:finished=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[0]=='Z'
                except FileNotFoundError:finished=True
            if not finished:continue
            artifact=artifacts.get(run['name'],root/'runs/v2'/run['name']/'history.json')
            if not terminal(artifact):raise RuntimeError(f'failed {run["name"]}; see stdout.log')
            print(f'finished {run["name"]}',flush=True);del active[pid]
        counts=Counter(r['gpu'] for r in active.values())
        for gpu in a.gpus:
            while queue and counts[gpu]<capacities[gpu]:
                job=queue.pop(0);out=root/'runs/v2'/job['name'];out.mkdir(parents=True,exist_ok=True)
                workers=tuning.get(gpu,{}).get('selected',{}).get('workers',2)
                cmd=[sys.executable,str(root/'code'/job['script']),*job['args'],'--workers',str(workers),'--out',str(out)]
                with (out/'stdout.log').open('w') as log:
                    proc=subprocess.Popen(cmd,env={**os.environ,'CUDA_VISIBLE_DEVICES':gpu,'OMP_NUM_THREADS':'2'},stdout=log,stderr=subprocess.STDOUT)
                owned[proc.pid]=proc;active[proc.pid]={'name':job['name'],'gpu':gpu};counts[gpu]+=1
                print(f'GPU {gpu}: {job["name"]}; workers={workers}',flush=True)
        if active:time.sleep(2)
    if a.stage in ('all','ranges') and not Path('runs/v2/hessian_lr.json').exists():
        subprocess.run([sys.executable,'code/hessian_lr_est.py'],check=True)

if __name__=='__main__':main()
