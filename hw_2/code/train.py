"""Train exact optimizer-update budgets; retain final rather than test-selected models."""
import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from data import get_loaders
from lenet import LeNet, LogisticRegression
from policies import CaffeSGD, Cosine, OneCycle, Step, TriangularCLR, Inverse
from resnet import resnet110, resnet20, resnet56


def build_model(name, channels, bn_momentum=.05, shortcut='pad'):
    if name == 'logreg': return LogisticRegression()
    if name == 'lenet': return LeNet()
    return {'resnet20': resnet20, 'resnet56': resnet56, 'resnet110': resnet110}[name](
        in_channels=channels, bn_momentum=bn_momentum, shortcut=shortcut)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def save_json(path, data):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False))
    tmp.replace(path)


def make_optimizer(model, lr, momentum, weight_decay, caffe=True, nesterov=False):
    if not caffe:
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                               weight_decay=weight_decay, nesterov=nesterov)
    grouped = {}
    for name, parameter in model.named_parameters():
        bias = name.endswith('bias') and parameter.ndim == 1
        # Convolution biases decay_mult=0; FC/Scale biases keep default decay.
        module = model.get_submodule(name.rsplit('.', 1)[0])
        scaled_bias = bias and isinstance(module, (torch.nn.Conv2d, torch.nn.Linear))
        mult = 2. if scaled_bias else 1.
        decay = 0. if bias and isinstance(module, torch.nn.Conv2d) else weight_decay
        grouped.setdefault((mult, decay), []).append(parameter)
    groups = [{'params': params, 'lr_mult': mult, 'lr': lr*mult, 'weight_decay': decay}
              for (mult, decay), params in grouped.items()]
    return CaffeSGD(groups, lr=lr, momentum=momentum, nesterov=nesterov, foreach=True)


@torch.no_grad()
def evaluate(model, loader, dev, channels_last=None):
    model.eval()
    if channels_last is None:
        channels_last = next(model.parameters()).is_contiguous(memory_format=torch.channels_last)
    correct = torch.zeros((), dtype=torch.int64, device=dev)
    loss_sum = torch.zeros((), dtype=torch.float64, device=dev)
    total = 0
    for x, y in loader:
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        if channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        logits = model(x)
        loss_sum += F.cross_entropy(logits, y, reduction='sum').double()
        correct += (logits.argmax(1) == y).sum()
        total += len(y)
    correct, loss_sum = correct.item(), loss_sum.item()
    if not math.isfinite(loss_sum): raise FloatingPointError('non-finite evaluation loss')
    return 100*correct/total, loss_sum/total


def check_gradients(model, loss):
    # One host synchronization, retaining both loss and gradient checks BEFORE SGD.
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'), error_if_nonfinite=False)
    if not torch.isfinite(torch.stack((loss.detach(), norm))).all():
        raise FloatingPointError('non-finite training loss or gradient')


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["cifar10","mnist"], default="cifar10")
    p.add_argument("--model", choices=["lenet","logreg","resnet20","resnet56","resnet110"], default="resnet20")
    p.add_argument("--policy", choices=["onecycle", "cosine", "step", "const", "clr", "inv"], default="cosine")
    p.add_argument("--max-lr", type=float, default=0.1)
    p.add_argument("--final-lr", type=float, default=None, help="onecycle final lr (default max/1e5)")
    p.add_argument("--pct-start", type=float, default=0.5, help="fraction of the cycle spent rising")
    p.add_argument("--epochs", type=int, default=450)
    p.add_argument("--iters", type=int, default=None, help="override total iterations")
    p.add_argument("--from-range-test", default=None, help="range_test.json to apply the 10x rule")
    p.add_argument("--rule", type=float, default=10.0, help="multiplier of iters-to-minimum")
    p.add_argument("--lr-select", choices=["div25", "div50", "atmin"], default="div25",
                   help="which range-test lr to use as max_lr")
    p.add_argument("--stepsize", type=int, default=10000, help="clr policy: iterations per half-cycle")
    p.add_argument("--mom-max", type=float, default=None)
    p.add_argument("--mom-min", type=float, default=None, help="cyclical momentum low bound (paper: 0.85)")
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--no-nesterov", action="store_true")
    p.add_argument("--bn-momentum", type=float, default=0.05)
    p.add_argument("--shortcut", choices=["proj", "pad"], default="pad",
                   help="pad = author Caffe graph; proj = conventional projected shortcut")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--subset", type=int, default=None, help="limited labeled data experiment")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=1000, help="full test every N updates; final test always runs (0=final only)")
    p.add_argument("--out", required=True)
    p.add_argument('--init-lr', type=float, default=.01)
    p.add_argument('--min-lr', type=float, default=.1, help='CLR lower bound')
    p.add_argument('--anneal-fraction', type=float, default=0.)
    p.add_argument('--drop-iters', nargs='+', type=int, default=None)
    p.add_argument('--inv-gamma', type=float, default=.0001)
    p.add_argument('--inv-power', type=float, default=.75)
    p.add_argument('--normalization', choices=['caffe', 'standard'], default='caffe')
    p.add_argument('--optimizer', choices=['caffe', 'pytorch'], default='caffe')
    p.add_argument('--nesterov', action='store_true')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--subset-seed', type=int, default=0)
    p.add_argument('--channels-last', action='store_true', help='native PyTorch NHWC tensor storage')
    p.add_argument('--cache-data', action='store_true', help='cache Caffe-normalized tensors and augment batches')
    a = p.parse_args()
    if any(isinstance(v,float) and not math.isfinite(v) for v in vars(a).values()):p.error('all numeric arguments must be finite')
    if a.dataset=='cifar10' and a.model in ('lenet','logreg'):p.error('LeNet/logreg require MNIST')
    if a.rule<=0:p.error('rule multiplier must be positive')
    if a.max_lr <= 0 or a.weight_decay < 0 or not 0 <= a.momentum < 1 or not 0 < a.bn_momentum <= 1:
        p.error('invalid LR, weight decay, momentum or BN momentum')
    if a.epochs < 1 or a.iters is not None and a.iters < 1 or a.eval_every < 0:
        p.error('invalid update/evaluation budget')
    if a.nesterov and a.no_nesterov: p.error('conflicting Nesterov flags')
    if a.policy == 'clr' and a.final_lr is not None: p.error('CLR uses --min-lr, not --final-lr')
    if a.mom_max is not None and a.mom_min is None: p.error('--mom-max requires --mom-min; fixed momentum uses --momentum')
    out = Path(a.out)
    if (out/'history.json').exists(): p.error('output already contains a run; choose a new directory')
    out.mkdir(parents=True, exist_ok=True)
    seed_all(a.seed)
    train_loader, test_loader, channels = get_loaders(a.dataset, a.batch_size, num_workers=a.workers,
        train_subset=a.subset, normalization=a.normalization, subset_seed=a.subset_seed,
        cache=a.cache_data, seed=a.seed)
    spe = len(train_loader)
    total_iters = a.iters if a.iters is not None else a.epochs*spe
    max_lr = a.max_lr
    if a.from_range_test:
        if a.policy != 'onecycle': p.error('range-based schedule requires onecycle')
        rt = json.loads(Path(a.from_range_test).read_text())
        if not a.iters: total_iters = max(3, round(a.rule*rt['min_iter']))
        max_lr = rt[{'div25':'div25_lr', 'div50':'div50_lr', 'atmin':'lr_at_min'}[a.lr_select]]
        if max_lr is None: p.error('range threshold was not crossed; choose atmin or an explicit LR')
    model = build_model(a.model, channels, a.bn_momentum, a.shortcut).to(a.device)
    if a.channels_last: model.to(memory_format=torch.channels_last)
    opt = make_optimizer(model, max_lr, a.momentum, a.weight_decay, a.optimizer == 'caffe', a.nesterov)
    final_lr = a.final_lr if a.final_lr is not None else max_lr/1e5
    if a.policy == 'onecycle':
        sched = OneCycle(opt, max_lr, total_iters, final_lr, a.pct_start,
            (a.mom_max if a.mom_max is not None else a.momentum) if a.mom_min is not None else None,
            a.mom_min, init_lr=a.init_lr, anneal_fraction=a.anneal_fraction)
    elif a.policy == 'clr': sched = TriangularCLR(opt, a.min_lr, max_lr, a.stepsize)
    elif a.policy == 'cosine': sched = Cosine(opt, max_lr, total_iters, min_lr=final_lr)
    elif a.policy == 'inv': sched = Inverse(opt, max_lr, a.inv_gamma, a.inv_power)
    elif a.policy == 'step':
        drops = a.drop_iters or [round(total_iters*f) for f in (.5, .75)]
        if any(d < 1 or d >= total_iters for d in drops): p.error('drop iterations must fit budget')
        sched = Step(opt, max_lr, 1, drops)
    else: sched = None
    cfg = {'args': vars(a), 'implementation': 'v2', 'steps_per_epoch': spe,
           'train_size': len(train_loader.dataset), 'test_size': len(test_loader.dataset),
           'schedule': {'max_lr': max_lr, 'total_iters': total_iters,
                        'final_lr': final_lr if a.policy in ('onecycle','cosine') else None,
                        'planned_terminal_lr': final_lr if a.policy in ('onecycle','cosine') else
                            sched.value(total_iters-1) if sched else max_lr},
           'code_sha256': {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob('*.py')},
           'torch_version': str(torch.__version__), 'device': str(a.device),
           'execution': {'evaluation': 'interval and final; no implicit epoch test',
                         'cache_data': a.cache_data, 'persistent_workers': a.workers > 0,
                         'metrics': 'GPU accumulation; per-update loss/gradient guard',
                         'rng': 'separate train and test loader generators'},
           'selection': 'final fixed-budget checkpoint; test best is diagnostic only'}
    history, it_log, lr_log = [], [], []
    best = {'acc': 0., 'epoch': 0., 'iter': 0}
    global_i, epoch, t0 = 0, 0, time.time()
    status, error = 'running', None
    def persist(force_lr=False):
        save_json(out/'history.json', {'config': cfg, 'history': history, 'best': best,
            'status': status, 'error': error, 'completed_iters': global_i, 'total_sec': time.time()-t0})
        save_json(out/'iter_log.json', it_log)
        # Avoid rewriting a growing 80k-entry array after every 50-step epoch.
        if force_lr or global_i % 1000 == 0:save_json(out/'lr_log.json', lr_log)
    persist()
    print(f'{a.model} seed={a.seed}: {total_iters} updates; {spe} updates/epoch', flush=True)
    run_loss = torch.zeros((), dtype=torch.float64, device=a.device)
    run_corr = torch.zeros((), dtype=torch.int64, device=a.device)
    run_n = 0
    try:
        while global_i < total_iters:
            epoch += 1
            model.train()
            for x, y in train_loader:
                if global_i >= total_iters: break
                if sched: sched.step()
                used_lr = opt.param_groups[0]['lr']/opt.param_groups[0].get('lr_mult', 1.)
                x, y = x.to(a.device, non_blocking=True), y.to(a.device, non_blocking=True)
                if a.channels_last: x = x.contiguous(memory_format=torch.channels_last)
                opt.zero_grad(set_to_none=True)
                logits = model(x)
                loss = F.cross_entropy(logits, y)
                loss.backward()
                check_gradients(model, loss)
                opt.step()
                global_i += 1
                lr_log.append({'iter': global_i, 'lr': used_lr, 'momentum': opt.param_groups[0]['momentum']})
                run_loss += loss.detach().double()*len(y)
                run_corr += (logits.argmax(1) == y).sum()
                run_n += len(y)
                if global_i == total_iters or a.eval_every and global_i % a.eval_every == 0:
                    acc, tl = evaluate(model, test_loader, a.device)
                    rec = {'iter': global_i, 'epoch': global_i/spe, 'test_acc': acc, 'test_loss': tl,
                           'train_loss': run_loss.item()/run_n,
                           'train_acc': 100*run_corr.item()/run_n, 'lr': used_lr,
                           'sec': time.time()-t0}
                    history.append(rec)
                    if acc > best['acc']: best = {'acc': acc, 'epoch': global_i/spe, 'iter': global_i}
                    persist(); model.train()
                    print(f'update {global_i}/{total_iters} lr={used_lr:.6g} test_acc={acc:.2f} time={time.time()-t0:.0f}s', flush=True)
                    run_loss.zero_();run_corr.zero_();run_n=0
            if global_i % 1000 == 0: persist()
        if not all(torch.isfinite(t).all() for t in model.state_dict().values()):
            raise FloatingPointError('non-finite model state')
        torch.save(model.state_dict(), out/'model.pt')
        status = 'completed'
    except (Exception, KeyboardInterrupt) as exc:
        status, error = 'failed', str(exc) or type(exc).__name__
        raise
    finally:
        persist(force_lr=True)
    print(f'DONE {a.out}: FINAL accuracy {history[-1]["test_acc"]:.2f}%', flush=True)


if __name__ == '__main__': main()
