"""Small regression check for architecture, schedules and variable-LR momentum."""
import torch
from torch import nn
from lenet import LeNet
from resnet import BasicBlock, CaffeBatchNorm, resnet20, resnet56
from policies import CaffeSGD, OneCycle, RangeTest, Step, TriangularCLR

def main():
    torch.set_num_threads(2)
    for factory in (resnet20, resnet56):
        m = factory()
        assert not any(isinstance(b.conv2[-1], nn.ReLU) for b in m.modules() if isinstance(b, BasicBlock))
        assert m(torch.randn(2, 3, 32, 32)).shape == (2, 10)
        m(torch.randn(2, 3, 32, 32)).sum().backward()
        m.eval(); assert torch.isfinite(m(torch.randn(2, 3, 32, 32))).all()
    b = BasicBlock(16, 32, 2)
    y = b(torch.randn(2, 16, 32, 32))
    assert y.shape == (2, 32, 16, 16) and y[:,16:].count_nonzero() == 0
    bn = CaffeBatchNorm(3, .05)
    x = torch.randn(2,3,4,4)
    bn(x); assert torch.allclose(bn.running_mean, x.mean((0,2,3)))
    bn(x); assert torch.allclose(bn.stat_weight, torch.tensor(1.95))
    negative=BasicBlock(16,16).eval()
    with torch.no_grad():
        for conv in (negative.conv1[0],negative.conv2[0]):
            conv.weight.zero_();conv.bias.zero_()
        negative.conv2[1].bias.fill_(-1.)
    assert negative(torch.ones(2,16,4,4)).count_nonzero()==0
    assert LeNet()(torch.randn(2,1,28,28)).shape == (2,10)
    p = nn.Parameter(torch.tensor(1.))
    opt = torch.optim.SGD([{'params':[p], 'lr_mult':2}], lr=.1)
    rt = RangeTest(opt, .1, 3., 11)
    lrs=[]
    for _ in range(11): rt.step(); lrs.append(rt.get_lr())
    assert lrs[0] == .1 and lrs[-1] == 3. and opt.param_groups[0]['lr'] == 6.
    sc=OneCycle(opt, 1., 13, .00001, 5/12, mom_max=.95,mom_min=.8,init_lr=.01, anneal_fraction=2/12)
    lrs=[]
    for i in range(13):
        sc.step();lrs.append(sc.get_lr())
        if i==5:assert opt.param_groups[0]['momentum']==.8
    assert opt.param_groups[0]['momentum']==.95
    assert lrs[0] == .01 and abs(lrs[5]-1.) < 1e-9 and abs(lrs[10]-.01)<1e-9 and abs(lrs[-1]-1e-5)<1e-9
    clr=TriangularCLR(opt,.1,3.,5)
    assert clr.value(0)==.1 and clr.value(5)==3. and clr.value(10)==.1
    step=Step(opt,.35,1,[5,7]); assert abs(step.value(5)-.035)<1e-9
    q=nn.Parameter(torch.tensor(1.)); opt=CaffeSGD([q],lr=.1,momentum=.9)
    expected, velocity=1.,0.
    for lr,grad in [(.1,2.),(.3,1.),(.02,3.)]:
        opt.param_groups[0]['lr']=lr; q.grad=torch.tensor(grad);opt.step()
        velocity=.9*velocity+lr*grad;expected-=velocity
        assert abs(q.item()-expected)<1e-6
    q=nn.Parameter(torch.tensor(1.));opt=CaffeSGD([q],lr=.1,momentum=.9)
    expected,velocity=1.,0.
    for lr,grad,mom in [(.1,2.,.9),(.2,4.,0.),(.1,3.,.9)]:
        opt.param_groups[0].update(lr=lr,momentum=mom);q.grad=torch.tensor(grad);opt.step()
        velocity=mom*velocity+lr*grad;expected-=velocity
        assert abs(q.item()-expected)<1e-6
    print('architecture, BN, schedules, LR multipliers and Caffe momentum: PASS')

if __name__=='__main__': main()
