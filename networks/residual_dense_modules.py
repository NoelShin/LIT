import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from .base_modules import ChannelAttentionLayer
from .dense_modules import DenseLayer


class ResidualDenseBlock(nn.Module):
    def __init__(self, n_ch, growth_rate, n_dense_layers, kernel_size=3, efficient=True):
        super(ResidualDenseBlock, self).__init__()
        init_ch = n_ch
        entry_layer = [ChannelAttentionLayer(n_ch), nn.ReflectionPad2d(1), nn.Conv2d(n_ch, growth_rate, kernel_size),
                       nn.InstanceNorm2d(growth_rate)]
        self.add_module('Entry_layer', nn.Sequential(*entry_layer))
        for i in range(n_dense_layers):
            self.add_module('Dense_layer_{}'.format(i), DenseLayer(growth_rate * (i + 1), growth_rate,
                                                                   efficient=efficient))

        SE = [nn.Conv2d(n_dense_layers * growth_rate, init_ch, kernel_size=1), ChannelAttentionLayer(init_ch),
              nn.LeakyReLU(1.0, True)]
        self.add_module('SE', nn.Sequential(*SE))  # style expansion
        # self.add_module('AR', AggregatedResidualBlock(init_ch, init_ch, 128, act, norm, pad))

        self.efficient = efficient
        self.n_dense_layers = n_dense_layers

    def function(self, *features):
        return getattr(self, 'SE')(torch.cat(features, dim=1))

    def forward(self, x):
        features = [getattr(self, 'Entry_layer')(x)]
        for i in range(self.n_dense_layers):
            features += [getattr(self, 'Dense_layer_{}'.format(i))(*features)]
        # if self.efficient:
        #   x = x + checkpoint(self.function, features[-1])
        # else:
        #    x = x + getattr(self, 'SE')(features)
        x = checkpoint(self.function, *features[1:]) + x
        return x


class ResidualDenseNetwork(nn.Module):
    def __init__(self, n_blocks, n_ch, growth_rate, n_dense_layers):
        super(ResidualDenseNetwork, self).__init__()
        for i in range(n_blocks):
            self.add_module('ResidualDenseBlock_{}'.format(i), ResidualDenseBlock(n_ch, growth_rate, n_dense_layers))
        self.n_blocks = n_blocks

    def forward(self, x):
        for i in range(self.n_blocks):
            x += getattr(self, 'ResidualDenseBlock_{}'.format(i))(x)
        return x
