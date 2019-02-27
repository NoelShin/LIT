import torch.nn as nn
from .base_modules import ChannelAttentionLayer


class ResidualBlock(nn.Module):
    def __init__(self, n_ch, kernel_size=3):
        super(ResidualBlock, self).__init__()
        ps = kernel_size // 2
        block = [nn.ReflectionPad2d(ps), nn.Conv2d(n_ch, n_ch, kernel_size), nn.InstanceNorm2d(n_ch),
                 nn.ReLU(inplace=True)]
        block += [nn.ReflectionPad2d(ps), nn.Conv2d(n_ch, n_ch, kernel_size), nn.InstanceNorm2d(n_ch)]
        self.add_module('ResidualBlock', nn.Sequential(*block))

    def forward(self, x):
        return x + getattr(self, 'ResidualBlock')(x)


class ResidualNetwork(nn.Module):
    def __init__(self, n_blocks, n_ch):
        super(ResidualNetwork, self).__init__()
        network = [ResidualBlock(n_ch) for i in range(n_blocks)]
        self.add_module('ResidualNetwork', nn.Sequential(*network))

    def forward(self, x):
        return getattr(self, 'ResidualNetwork')(x)


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


class ResidualGroup(nn.Module):
    def __init__(self, n_blocks, n_ch, reduction_rate, kernel_size=3):
        super(ResidualGroup, self).__init__()
        group = [ResidualChannelAttentionBlock(n_ch, reduction_rate) for _ in range(n_blocks)]
        group += [nn.ReflectionPad2d(1), nn.Conv2d(n_ch, n_ch, kernel_size), nn.InstanceNorm2d(n_ch),
                  nn.ReLU(inplace=True)]
        self.group = nn.Sequential(*group)

    def forward(self, x):
        return x + self.group(x)
