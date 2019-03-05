import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, n_ch, kernel_size=3, block_weight=False):
        super(ResidualBlock, self).__init__()
        ps = kernel_size // 2
        block = [nn.ReflectionPad2d(ps), nn.Conv2d(n_ch, n_ch, kernel_size), nn.InstanceNorm2d(n_ch),
                 nn.ReLU(inplace=True)]
        block += [nn.ReflectionPad2d(ps), nn.Conv2d(n_ch, n_ch, kernel_size), nn.InstanceNorm2d(n_ch)]
        self.add_module('ResidualBlock', nn.Sequential(*block))
        setattr(self, 'BlockWeight', nn.Parameter(torch.tensor(1.0))) if block_weight else None
        self.block_weight = block_weight

    def forward(self, x):
        if self.block_weight:
            return x + torch.mul(getattr(self, 'ResidualBlock')(x), getattr(self, 'BlockWeight'))
        else:
            return x + getattr(self, 'ResidualBlock')(x), getattr(self, 'BlockWeight')


class ResidualNetwork(nn.Module):
    def __init__(self, n_blocks, n_ch, block_weight=False):
        super(ResidualNetwork, self).__init__()
        for i in range(n_blocks):
            self.add_module('ResidualBlock{}'.format(i), ResidualBlock(n_ch, block_weight))
        self.n_blocks = n_blocks

    def forward(self, x):
        for i in range(self.n_blocks):
            x = getattr(self, 'ResidualBlock{}'.format(i))(x)
        return x


class BasicResidualBlock(nn.Module):
    def __init__(self, n_ch, kernel_size=3, block_weight=False):
        super(BasicResidualBlock, self).__init__()
        ps = kernel_size // 2
        block = [nn.ReLU(inplace=True), nn.ReflectionPad2d(ps), nn.Conv2d(n_ch, n_ch, kernel_size),
                 nn.InstanceNorm2d(n_ch)]
        block += [nn.ReLU(inplace=True), nn.ReflectionPad2d(ps), nn.Conv2d(n_ch, n_ch, kernel_size),
                  nn.InstanceNorm2d(n_ch)]
        self.add_module('BasicResidualBlock', nn.Sequential(*block))
        setattr(self, 'BlockWeight', nn.Parameter(torch.tensor(1.0))) if block_weight else None

    def forward(self, x):
        if self.block_weight:
            return x + torch.mul(getattr(self, 'BasicResidualBlock')(x), getattr(self, 'BlockWeight'))
        else:
            return x + getattr(self, 'BasicResidualBlock')(x)


class ResidualGroup(nn.Module):
    def __init__(self, n_blocks, n_ch, rir_ch, block_weight=False):
        super(ResidualGroup, self).__init__()
        group = [nn.Conv2d(n_ch, rir_ch, kernel_size=1)]
        group += [BasicResidualBlock(rir_ch, block_weight) for _ in range(n_blocks)]
        group += [nn.Conv2d(rir_ch, n_ch, kernel_size=1)]
        self.group = nn.Sequential(*group)

    def forward(self, x):
        return x + self.group(x)


class ResidualInResidualNetwork(nn.Module):
    def __init__(self, n_blocks, n_groups, n_ch, rir_ch, block_weight=False):
        super(ResidualInResidualNetwork, self).__init__()
        network = [ResidualGroup(n_blocks, n_ch, rir_ch, block_weight) for _ in range(n_groups)]
        self.network = nn.Sequential(*network)

    def forward(self, x):
        return self.network(x)
