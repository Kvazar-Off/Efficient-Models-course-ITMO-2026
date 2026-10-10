"""CIFAR ResNet; pad mode follows lnsmith54/super-convergence Caffe graph.

Caffe uses Conv-BN-Scale-ReLU, Conv-BN-Scale, Sum-ReLU; downsampling
adds at the old width and appends zero channels AFTER the sum.
"""
import torch
from torch import nn
from torch.nn import functional as F


class CaffeBatchNorm(nn.BatchNorm2d):
    """Caffe accumulated statistics, followed by learnable Scale.

    momentum here is the NEW observation weight (1 - Caffe MAF).
    Single-device BN is an explicit adaptation of the author's multi-GPU BN.
    """
    def __init__(self, channels, momentum):
        super().__init__(channels, eps=1e-5, momentum=momentum)
        self.running_var.zero_()
        self.register_buffer('stat_weight', torch.tensor(0.))

    def forward(self, x):
        if self.training:
            mean, var = self.running_mean.clone(), self.running_var.clone()
            out = F.batch_norm(x, mean, var, self.weight, self.bias, True, 1., self.eps)
            with torch.no_grad():
                maf = 1 - self.momentum
                self.running_mean.mul_(maf).add_(mean)
                self.running_var.mul_(maf).add_(var)
                self.stat_weight.mul_(maf).add_(1)
                self.num_batches_tracked.add_(1)
            return out
        scale = self.stat_weight.clamp_min(1)
        return F.batch_norm(x, self.running_mean / scale, self.running_var / scale,
                            self.weight, self.bias, False, 0., self.eps)


def conv3x3(cin, cout, stride=1, bn_momentum=.05, relu=True, caffe=True):
    bn = CaffeBatchNorm(cout, bn_momentum) if caffe else nn.BatchNorm2d(cout, momentum=bn_momentum)
    layers = [nn.Conv2d(cin, cout, 3, stride, 1, bias=caffe), bn]
    return nn.Sequential(*layers, *([nn.ReLU(inplace=True)] if relu else []))


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, bn_momentum=.05, shortcut='pad'):
        super().__init__()
        self.pad = planes - in_planes if shortcut == 'pad' else 0
        width = in_planes if self.pad else planes
        self.conv1 = conv3x3(in_planes, width, stride, bn_momentum, caffe=shortcut == 'pad')
        self.conv2 = conv3x3(width, width, 1, bn_momentum, relu=False, caffe=shortcut == 'pad')
        if stride == 1 and in_planes == planes:
            self.shortcut = nn.Identity()
        elif shortcut == 'pad':
            self.shortcut = nn.AvgPool2d(3, stride, padding=0, ceil_mode=True)
        else:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, planes, 1, stride, bias=False),
                                          nn.BatchNorm2d(planes, momentum=bn_momentum))

    def forward(self, x):
        out = F.relu(self.conv2(self.conv1(x)) + self.shortcut(x), inplace=True)
        return F.pad(out, (0, 0, 0, 0, 0, self.pad)) if self.pad else out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, in_channels=3,
                 planes=(16, 32, 64), bn_momentum=.05, shortcut='pad'):
        super().__init__()
        self.conv1 = conv3x3(in_channels, planes[0], bn_momentum=bn_momentum, caffe=shortcut == 'pad')
        width = planes[0]
        for k, (target, count) in enumerate(zip(planes, num_blocks), 1):
            blocks = []
            for i in range(count):
                blocks.append(block(width, target, 2 if k > 1 and i == 0 else 1, bn_momentum, shortcut))
                width = target
            setattr(self, f'layer{k}', nn.Sequential(*blocks))
        self.linear = nn.Linear(width, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -(3 / m.in_features)**.5, (3 / m.in_features)**.5)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.layer3(self.layer2(self.layer1(self.conv1(x))))
        return self.linear(F.adaptive_avg_pool2d(x, 1).flatten(1))


def resnet20(**kw): return ResNet(BasicBlock, [3, 3, 3], **kw)
def resnet56(**kw): return ResNet(BasicBlock, [9, 9, 9], **kw)
def resnet110(**kw): return ResNet(BasicBlock, [18, 18, 18], **kw)
