"""Fig. 3a: global secant estimate, including Caffe BN blobs or parameters only.

Author source: lnsmith54/super-convergence/L2TermComputation3.py. https://github.com/lnsmith54/super-convergence/tree/master
"""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from data import get_loaders
from train import build_model, make_optimizer, seed_all, save_json


def snapshot(model, all_blobs):
    source = model.state_dict().values() if all_blobs else model.parameters()
    return [t.detach().clone() for t in source if t.is_floating_point()]


def plot_estimates(records, path):
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),dpi=150)
    for ax,(mode,rows) in zip(axes,records.items()):
        for key,label in [('all_blobs','EMA: parameters + BN accumulators'),('parameters','EMA: trainable parameters'),('lr','used LR')]:
            ax.plot([r['iter'] for r in rows],[r[key] for r in rows],label=label)
        ax.set_yscale('log');ax.set_title(mode)
        ax.set_xlabel('first snapshot update (1-based)');ax.set_ylabel('LR / secant estimate (log scale)')
        ax.grid(alpha=.3);ax.legend(fontsize=8)
    fig.suptitle('Secant diagnostic: trainable parameters and BN accumulators, shown separately',fontsize=10)
    fig.tight_layout();fig.savefig(path);plt.close(fig)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--iters',type=int,default=300)
    p.add_argument('--batch-size',type=int,default=1000)
    p.add_argument('--alpha',type=float,default=.1)
    p.add_argument('--model',default='resnet56')
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--bn-momentum',type=float,default=.001,help='published prototxt MAF=0.999')
    p.add_argument('--out',default='runs/v2/hessian_lr.json')
    a=p.parse_args()
    if Path(a.out).exists():p.error('output exists; choose a new path')
    if not 0<a.bn_momentum<=1:p.error('invalid BN momentum')
    if a.iters<3 or not 0<a.alpha<=1:p.error('invalid iterations/EMA')
    dev='cuda' if torch.cuda.is_available() else 'cpu'
    out={}
    for mode in ('constant','rising'):
        seed_all(0)
        loader,_,channels=get_loaders('cifar10',a.batch_size,num_workers=a.workers)
        model=build_model(a.model,channels,a.bn_momentum,'pad').to(dev).to(memory_format=torch.channels_last)
        opt=make_optimizer(model,.1,.9,1e-4)
        iterator=iter(loader); snapshots={False:[],True:[]}; records=[];emas={False:.1,True:.1}
        for i in range(a.iters+2):
            lr=.1+(2.9*i/5000 if mode=='rising' else 0)
            for g in opt.param_groups:g['lr']=lr*g.get('lr_mult',1.)
            try:x,y=next(iterator)
            except StopIteration:iterator=iter(loader);x,y=next(iterator)
            x,y=x.to(dev).contiguous(memory_format=torch.channels_last),y.to(dev)
            opt.zero_grad(set_to_none=True);loss=F.cross_entropy(model(x),y)
            if not torch.isfinite(loss):raise FloatingPointError('Hessian experiment nonfinite')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),float('inf'),error_if_nonfinite=True)
            opt.step()
            rec={'iter':i-1, 'lr': .1+(2.9*(i-1)/5000 if mode=='rising' else 0)}
            for blobs in (False,True):
                previous=snapshots[blobs];previous.append(snapshot(model,blobs))
                if len(previous)<3:continue
                t0,t1,t2=previous
                num=torch.stack([(b-c).abs().sum() for c,b in zip(t0,t1)]).sum()
                den=torch.stack([(2*b-c-d).abs().sum() for c,b,d in zip(t0,t1,t2)]).sum()
                raw=rec['lr']*(num/den).item() if den>0 else None
                if raw is None:raise FloatingPointError('zero secant denominator')
                emas[blobs]=a.alpha*raw+(1-a.alpha)*emas[blobs]
                key='all_blobs' if blobs else 'parameters'
                rec[key]=emas[blobs];rec[key+'_raw']=raw
                previous.pop(0)
            if i>=2:records.append(rec)
        out[mode]=records
        print(mode,records[-1],flush=True)
    save_json(a.out,{'config':vars(a),'implementation':'v2','records':out,
        'memory_format':'channels_last', 'updates_per_regime':a.iters+2,
        'adaptation':'single-device BN; all floating Caffe-compatible state vs trainable parameters'})
    plot_estimates(out,Path(a.out).with_suffix('.png'))

if __name__=='__main__':main()
