"""Per-update schedules. Call step BEFORE optimizer.step; log the used LR.

Group lr_mult is retained (Caffe biases use twice the base rate).
"""
import math
import torch


class CaffeSGD(torch.optim.SGD):
    """SGD with LR inside the momentum accumulator, as in Caffe.

    Rescale PyTorch's existing buffer so h_t = mu*h_{t-1} + lr_t*g_t.
    """
    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            lr = group['lr']
            if lr <= 0: raise ValueError('CaffeSGD requires positive LR')
            previous = group.get('_previous_lr', lr)
            buffers = [self.state[p]['momentum_buffer'] for p in group['params']
                       if 'momentum_buffer' in self.state[p]]
            if buffers: torch._foreach_mul_(buffers, previous / lr)
            if group['momentum']==0:
                # PyTorch skips buffer updates at mu=0; Caffe resets h to lr*g.
                for p in group['params']:
                    if p.grad is not None:
                        grad=p.grad.detach().clone()
                        if group['weight_decay']:grad.add_(p,alpha=group['weight_decay'])
                        self.state[p]['momentum_buffer']=grad
            group['_previous_lr'] = lr
        return super().step(closure)


class ParamGroupLR:
    def __init__(self, optimizer, base_lrs=None):
        self.opt = optimizer
        self.step_num = 0
    def _set(self, lr, momentum=None):
        for group in self.opt.param_groups:
            group['lr'] = lr * group.get('lr_mult', 1.)
            if momentum is not None: group['momentum'] = momentum
    def get_lr(self):
        g = self.opt.param_groups[0]
        return g['lr'] / g.get('lr_mult', 1.)
    def step(self):
        self._set(self.value(self.step_num))
        self.step_num += 1


class RangeTest(ParamGroupLR):
    def __init__(self, optimizer, init_lr, max_lr, steps):
        super().__init__(optimizer)
        if not 0 < init_lr <= max_lr or steps < 2: raise ValueError('invalid range')
        self.init_lr, self.max_lr, self.steps = init_lr, max_lr, steps
    def value(self, i):
        return self.init_lr + min(i / (self.steps - 1), 1) * (self.max_lr - self.init_lr)


class OneCycle(ParamGroupLR):
    """One triangular cycle plus an optional final annealing phase.

    With anneal_fraction=0 this is a two-phase adaptation, not the paper's
    three-phase recipe. init_lr is independent of the much smaller final_lr.
    """
    def __init__(self, optimizer, max_lr, steps_per_cycle, final_lr=1e-6,
                 pct_start=.5, mom_max=None, mom_min=None, init_lr=None,
                 anneal_fraction=0.):
        super().__init__(optimizer)
        if steps_per_cycle < 3 or not 0 < pct_start < 1 or not 0 <= anneal_fraction < 1-pct_start:
            raise ValueError('invalid cycle length/fractions')
        self.max_lr, self.final_lr = max_lr, final_lr
        self.init_lr = final_lr if init_lr is None else init_lr
        if not 0 < self.init_lr <= max_lr or not 0 < final_lr <= max_lr: raise ValueError('invalid LR bounds')
        if (mom_max is None) != (mom_min is None): raise ValueError('provide both momentum bounds')
        if mom_min is not None and not 0 <= mom_min <= mom_max < 1: raise ValueError('invalid momentum bounds')
        self.steps = steps_per_cycle
        self.up_steps = max(1, round((self.steps-1)*pct_start))
        self.cycle_end = round((self.steps-1)*(1-anneal_fraction))
        if not self.up_steps < self.cycle_end: raise ValueError('cycle needs a descending phase')
        self.mom_max, self.mom_min = mom_max, mom_min
    def step(self):
        i = min(self.step_num, self.steps-1)
        if i <= self.up_steps:
            frac = i / self.up_steps
            lr = self.init_lr + (self.max_lr-self.init_lr)*frac
        elif i <= self.cycle_end:
            frac = 1-(i-self.up_steps)/(self.cycle_end-self.up_steps)
            bottom = self.init_lr if self.cycle_end < self.steps-1 else self.final_lr
            lr = bottom + (self.max_lr-bottom)*frac
        else:
            frac = 0.
            t = (i-self.cycle_end)/(self.steps-1-self.cycle_end)
            lr = self.init_lr + (self.final_lr-self.init_lr)*t
        mom = None if self.mom_max is None else self.mom_max-(self.mom_max-self.mom_min)*frac
        self._set(lr, mom)
        self.step_num += 1


class TriangularCLR(ParamGroupLR):
    def __init__(self, optimizer, lr_min, lr_max, stepsize):
        super().__init__(optimizer)
        if not 0 < lr_min <= lr_max or stepsize < 1: raise ValueError('invalid CLR bounds')
        self.lr_min, self.lr_max, self.stepsize = lr_min, lr_max, stepsize
    def value(self, i):
        tri = 1-abs((i/self.stepsize % 2)-1)
        return self.lr_min + (self.lr_max-self.lr_min)*tri


class Cosine(ParamGroupLR):
    def __init__(self, optimizer, max_lr, steps, min_lr=0.):
        super().__init__(optimizer)
        self.max_lr, self.steps, self.min_lr = max_lr, steps, min_lr
    def value(self, i):
        return self.min_lr + .5*(self.max_lr-self.min_lr)*(1+math.cos(math.pi*min(i/max(self.steps-1,1),1)))


class Step(ParamGroupLR):
    def __init__(self, optimizer, max_lr, steps_per_epoch, drop_epochs, gamma=.1):
        super().__init__(optimizer)
        self.max_lr, self.drops, self.gamma = max_lr, [d*steps_per_epoch for d in drop_epochs], gamma
    def value(self, i):
        return self.max_lr*self.gamma**sum(i >= d for d in self.drops)


class Inverse(ParamGroupLR):
    def __init__(self, optimizer, max_lr, gamma=.0001, power=.75):
        super().__init__(optimizer)
        self.max_lr, self.gamma, self.power = max_lr, gamma, power
    def value(self, i): return self.max_lr*(1+self.gamma*i)**(-self.power)
