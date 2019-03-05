import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class DenseLayer(nn.Module):
    def __init__(self, n_ch, growth_rate, efficient=True):
        super(DenseLayer, self).__init__()

        layer = [nn.ReflectionPad2d(1), nn.Conv2d(n_ch, growth_rate, kernel_size=3), nn.InstanceNorm2d(growth_rate),
                 nn.ReLU(inplace=True)]
        # without activation at the last, it consumes more memory without reason.
        # plz advice us if you have any idea about this phenomenon.

        self.add_module('Dense_layer', nn.Sequential(*layer))
        self.efficient = efficient

    def function(self, *inputs):
        return getattr(self, 'Dense_layer')(torch.cat(inputs, dim=1))

    def forward(self, *inputs):
        if self.efficient and any(input.requires_grad for input in inputs):
            x = checkpoint(self.function, *inputs)
        else:
            x = getattr(self, 'Dense_layer')(torch.cat(inputs, dim=1))
        return x


class DenseBlock(nn.Module):
    def __init__(self, n_ch, growth_rate, n_dense_layers, efficient=True, block_weight=False):
        super(DenseBlock, self).__init__()
        self.add_module('Entry', nn.Conv2d(nn.Conv2d(n_ch, growth_rate, kernel_size=1)))
        for i in range(n_dense_layers):
            self.add_module('Dense_layer_{}'.format(i), DenseLayer(n_ch + growth_rate * i, growth_rate,
                                                                   efficient=efficient))
        self.add_module('Exit', nn.Conv2d(nn.Conv2d(growth_rate, n_ch, kernel_size=1)))
        setattr(self, 'BlockWeight', nn.Parameter(torch.tensor(1.0))) if block_weight else None
        self.block_weight = block_weight
        self.n_dense_layers = n_dense_layers

    def forward(self, x):
        features = [getattr(self, 'Entry')(x)]
        for i in range(self.n_dense_layers):
            features += [getattr(self, 'Dense_layer_{}'.format(i))(*features)]

        if self.block_weight:
            return getattr(self, 'BlockWeight') * getattr(self, 'Exit')(features[-1])
        else:
            return getattr(self, 'Exit')(features[-1])


class DenseNetwork(nn.Module):
    def __init__(self, n_blocks, n_ch, growth_rate, n_dense_layers, efficient=True, block_weight=False):
        super(DenseNetwork, self).__init__()
        for i in range(n_blocks):
            self.add_module('Dense_block_{}'.format(i), DenseBlock(n_ch, growth_rate, n_dense_layers,
                                                                   efficient=efficient, block_weight=block_weight))
        self.n_blocks = n_blocks

    def forward(self, x):
        features = [x]
        for i in range(self.n_blocks):
            features += [getattr(self, 'Dense_block_{}'.format(i))(features[-1])]
        return checkpoint(self.function, *features)
