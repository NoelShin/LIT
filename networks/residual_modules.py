import torch
import torch.nn as nn
import numpy as np
from .base_modules import ChannelAttentionLayer


class ResidualBlock(nn.Module):
    def __init__(self, n_ch, kernel_size=3):
        super(ResidualBlock, self).__init__()
        ps = kernel_size // 2
        block = [nn.ReflectionPad2d(ps), nn.Conv2d(n_ch, n_ch, kernel_size), nn.InstanceNorm2d(n_ch),
                 nn.ReLU(inplace=True)]
        block += [nn.ReflectionPad2d(ps), nn.Conv2d(n_ch, n_ch, kernel_size), nn.InstanceNorm2d(n_ch)]
        self.add_module('ResidualBlock', nn.Sequential(*block))
        setattr(self, 'ResidualWeight', nn.Parameter(torch.tensor(1.0)))

    def forward(self, x):
        return x + torch.mul(getattr(self, 'ResidualBlock')(x), getattr(self, 'ResidualWeight'))


class ResidualNetwork(nn.Module):
    def __init__(self, n_blocks, n_ch):
        super(ResidualNetwork, self).__init__()
        # network = [ResidualBlock(n_ch) for i in range(n_blocks)]
        # self.add_module('ResidualNetwork', nn.Sequential(*network))
        for i in range(n_blocks):
            self.add_module('ResidualBlock{}'.format(i), ResidualBlock(n_ch))

        self.n_blocks = n_blocks

    def forward(self, x):
        # return getattr(self, 'ResidualNetwork')(x)
        intermediate = [x]
        for i in range(self.n_blocks):
            intermediate += [getattr(self, 'ResidualBlock{}'.format(i))(intermediate[-1])]
        return intermediate[-1], intermediate


class ResidualChannelAttentionBlock(nn.Module):
    def __init__(self, n_ch, reduction_rate, kernel_size=3):
        super(ResidualChannelAttentionBlock, self).__init__()
        ps = kernel_size // 2
        block = []

        block += [nn.ReflectionPad2d(ps), nn.Conv2d(n_ch, n_ch, kernel_size), nn.InstanceNorm2d(n_ch),
                  nn.ReLU(inplace=True)]
        block += [nn.ReflectionPad2d(ps), nn.Conv2d(n_ch, n_ch, kernel_size), nn.InstanceNorm2d(n_ch),
                  nn.ReLU(inplace=True)]

        block += [ChannelAttentionLayer(n_ch, reduction_rate)]
        self.add_module('ResidualChannelAttentionBlock', nn.Sequential(*block))

    def forward(self, x):
        return x + getattr(self, 'ResidualChannelAttentionBlock')(x)


class BasicResidualBlock(nn.Module):
    def __init__(self, n_ch, kernel_size=3):
        super(BasicResidualBlock, self).__init__()
        ps = kernel_size // 2
        block = [nn.ReLU(inplace=True), nn.ReflectionPad2d(ps), nn.Conv2d(n_ch, n_ch, kernel_size),
                 nn.InstanceNorm2d(n_ch)]
        block += [nn.ReLU(inplace=True), nn.ReflectionPad2d(ps), nn.Conv2d(n_ch, n_ch, kernel_size),
                  nn.InstanceNorm2d(n_ch)]
        self.add_module('BasicResidualBlock', nn.Sequential(*block))

    def forward(self, x):
        return x + getattr(self, 'BasicResidualBlock')(x)


class ResidualGroup(nn.Module):
    def __init__(self, n_blocks, n_ch):
        super(ResidualGroup, self).__init__()
        group = [BasicResidualBlock(n_ch) for _ in range(n_blocks)]
        self.group = nn.Sequential(*group)

    def forward(self, x):
        return x + self.group(x)


class ResidualInResidualNetwork(nn.Module):
    def __init__(self, n_groups, n_blocks, n_ch):
        super(ResidualInResidualNetwork, self).__init__()
        network = [ResidualGroup(n_blocks, n_ch) for _ in range(n_groups)]
        self.network = nn.Sequential(*network)
        self.add_module('StyleExpansion', nn.Conv2d(n_ch, 1024, kernel_size=1, bias=False))

    def forward(self, x):
        return getattr(self,'StyleExpansion')(self.network(x))
