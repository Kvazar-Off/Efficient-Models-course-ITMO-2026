"""Linear LR range test. Thresholds describe loss rise, NOT numerical divergence.

The optional 10x heuristic is a local extension, absent from arXiv v3.
"""
import argparse
import math
import time
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from data import get_loaders
from policies import RangeTest
from train import build_model, make_optimizer, seed_all, save_json, evaluate, check_gradients


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["cifar10","mnist"], default="cifar10")
    p.add_argument("--model", choices=["lenet","logreg","resnet20","resnet56","resnet110"], default="resnet20")
    p.add_argument("--init-lr", type=float, default=1e-5)
    p.add_argument("--max-lr", type=float, default=10.0)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--bn-momentum", type=float, default=0.05)
    p.add_argument("--ema", type=float, default=0.9, help="smoothing coefficient of the loss")
    p.add_argument("--subset", type=int, default=None)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shortcut", choices=["proj", "pad"], default="pad")
    p.add_argument("--out", required=True)
    p.add_argument('--normalization', choices=['caffe','standard'], default='caffe')
    p.add_argument('--optimizer', choices=['caffe','pytorch'], default='caffe')
    p.add_argument('--eval-every', type=int, default=1000)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--channels-last', action='store_true')
    p.add_argument('--cache-data', action='store_true')
    a = p.parse_args()
    if any(isinstance(v,float) and not math.isfinite(v) for v in vars(a).values()):p.error('all numeric arguments must be finite')
    if a.dataset=='cifar10' and a.model in ('lenet','logreg'):p.error('LeNet/logreg require MNIST')
    if not 0 <= a.ema < 1 or a.eval_every < 0: p.error('invalid EMA or evaluation interval')
    if a.weight_decay < 0 or not 0 <= a.momentum < 1 or not 0 < a.bn_momentum <= 1:
        p.error('invalid weight decay or momentum')
    out=Path(a.out)
    if (out/'range_test.json').exists(): p.error('run already exists')
    out.mkdir(parents=True,exist_ok=True)
    seed_all(a.seed)
    loader,test,channels=get_loaders(a.dataset,a.batch_size,num_workers=a.workers,
        train_subset=a.subset,normalization=a.normalization,cache=a.cache_data,seed=a.seed)
    model=build_model(a.model,channels,a.bn_momentum,a.shortcut).to(a.device)
    if a.channels_last:model.to(memory_format=torch.channels_last)
    opt=make_optimizer(model,a.init_lr,a.momentum,a.weight_decay,a.optimizer=='caffe')
    sched=RangeTest(opt,a.init_lr,a.max_lr,a.steps)
    lrs,losses,smooth,accs,tests=[],[],[],[],[]
    ema=None; iterator=iter(loader);t0=time.time();status='completed'; error=None
    model.train()
    for i in range(a.steps):
        try: x,y=next(iterator)
        except StopIteration:
            iterator=iter(loader);x,y=next(iterator)
        sched.step();lr=sched.get_lr()
        x,y=x.to(a.device),y.to(a.device)
        if a.channels_last:x=x.contiguous(memory_format=torch.channels_last)
        opt.zero_grad(set_to_none=True);logits=model(x);loss=F.cross_entropy(logits,y)
        loss.backward()
        try: check_gradients(model,loss)
        except FloatingPointError:
            status='nonfinite';error=f'non-finite loss or gradient at attempted update {i+1}';break
        opt.step()
        value=loss.detach();ema=value if ema is None else a.ema*ema+(1-a.ema)*value
        lrs.append(lr);losses.append(value);smooth.append(ema)
        accs.append((logits.argmax(1)==y).float().mean()*100)
        if i+1==a.steps or a.eval_every and (i+1)%a.eval_every==0:
            try:
                acc,tl=evaluate(model,test,a.device)
                tests.append({'iter':i+1,'lr':lr,'test_acc':acc,'test_loss':tl})
            except FloatingPointError:
                status='nonfinite';error=f'non-finite evaluation at update {i+1}';break
            model.train()
        if (i+1)%1000==0: print(f'update {i+1}: lr={lr:.4g}, loss={ema.item():.4g}',flush=True)
    if not smooth: raise FloatingPointError(error)
    losses,smooth,accs=[torch.stack(values).cpu().tolist() for values in (losses,smooth,accs)]
    minimum=min(range(len(smooth)),key=smooth.__getitem__)
    start_loss=sum(smooth[:min(50,len(smooth))])/min(50,len(smooth))
    def cross(fraction):
        threshold=smooth[minimum]+fraction*max(0.,start_loss-smooth[minimum])
        return next((j for j in range(minimum+1,len(smooth)) if smooth[j]>threshold),None)
    d25,d50=cross(.25),cross(.50)
    result={'config':vars(a),'implementation':'v2','status':status,'error':error,
            'steps':len(smooth),'requested_steps':a.steps,'min_iter':minimum+1,
            'min_loss':smooth[minimum],'lr_at_min':lrs[minimum], 'start_loss':start_loss,
            'div25_iter':None if d25 is None else d25+1,'div25_lr':None if d25 is None else lrs[d25],
            'div50_iter':None if d50 is None else d50+1,'div50_lr':None if d50 is None else lrs[d50],
            'threshold_definition':'Lmin + fraction*(Lstart-Lmin); heuristic loss rise, not numerical divergence',
            'rule10_iterations':10*(minimum+1),'seconds':time.time()-t0,
            'lrs':lrs,'losses':losses,'smoothed':smooth,'accs':accs,'test':tests}
    save_json(out/'range_test.json',result)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),dpi=150)
    axes[0].plot(lrs,losses,alpha=.25,label='batch train loss')
    axes[0].plot(lrs,smooth,label=f'train loss EMA ({a.ema})')
    axes[0].axvline(lrs[minimum],ls='--',color='green',label=f'minimum at update {minimum+1}')
    if d25 is not None: axes[0].axvline(lrs[d25],ls=':',color='red',label='25% loss-rise heuristic')
    axes[0].set_yscale('log'); axes[0].set_ylabel('cross-entropy loss (log scale)')
    if tests: axes[1].plot([r['lr'] for r in tests],[r['test_acc'] for r in tests],label='full test set')
    axes[1].set_ylabel('test accuracy, %');axes[1].set_ylim(0,100)
    for ax in axes: ax.set_xlabel('used learning rate');ax.grid(alpha=.3);ax.legend(fontsize=9)
    fig.suptitle(f'{a.dataset}, {a.model}: {len(smooth)}/{a.steps} updates; {status}')
    fig.tight_layout();fig.savefig(out/'range_test.png');plt.close(fig)
    print(f'range minimum at update {minimum+1}, LR={lrs[minimum]:.5g}; status={status}',flush=True)

if __name__=='__main__':main()
