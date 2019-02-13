import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from .base_module import BaseModule
from .residual_channel_attention_block import ChannelAttentionLayer


class DenseLayer(BaseModule):
    def __init__(self, n_ch, growth_rate, kernel_size=3, act='relu', norm='InstanceNorm2d', pad='reflection',
                 pre_activation=False, efficient=True):
        super(DenseLayer, self).__init__()
        act = self.get_act_layer(act, inplace=True) if isinstance(act, str) else act
        norm = self.get_norm_layer(norm) if isinstance(norm, str) else norm
        pad = self.get_pad_layer(pad) if isinstance(pad, str) else pad

        if pre_activation:
            layer = [norm(n_ch), act, pad(1), nn.Conv2d(n_ch, growth_rate, kernel_size=kernel_size, bias=False)]

        else:
            layer = [pad(1), nn.Conv2d(n_ch, growth_rate, kernel_size=kernel_size, bias=False), norm(growth_rate), act]
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


class ResidualDenseBlock(BaseModule):
    def __init__(self, n_ch, growth_rate, n_dense_layers, act, norm, pad, kernel_size=3, pre_activation=False,
                 efficient=True):
        super(ResidualDenseBlock, self).__init__()
        init_ch = n_ch
        for i in range(n_dense_layers):
            self.add_module('Dense_layer_{}'.format(i), DenseLayer(n_ch, growth_rate, kernel_size, act, norm, pad,
                                                                   pre_activation=pre_activation, efficient=efficient))
            n_ch += growth_rate
        self.add_module('LFF', nn.Conv2d(n_ch, init_ch, kernel_size=1, bias=False))  # local feature fusion

        self.efficient = efficient
        self.n_dense_layers = n_dense_layers

    def function(self, *inputs):
        return getattr(self, 'LFF')(torch.cat(inputs, dim=1))

    def forward(self, x):
        features = [x]
        for i in range(self.n_dense_layers):
            features += [getattr(self, 'Dense_layer_{}'.format(i))(*features)]

        if self.efficient:
            x = x + checkpoint(self.function, *features)
        else:
            x = x + getattr(self, 'LFF')((torch.cat(features, dim=1)))  # local residual learning
        return x


class ResidualChannelAttentionDenseBlock(ResidualDenseBlock):
    def __init__(self, n_ch, growth_rate, n_dense_layers, act, norm, pad, kernel_size=3, pre_activation=False):
        super(ResidualChannelAttentionDenseBlock, self).__init__(n_ch, growth_rate, n_dense_layers, act, norm, pad,
                                                                 kernel_size=kernel_size, pre_activation=pre_activation)
        self.add_module('CA', ChannelAttentionLayer(n_ch))

    def forward(self, x):
        features = [x]
        for i in range(self.n_dense_layers):
            features += [getattr(self, 'Dense_layer_{}'.format(i))(*features)]

        if self.efficient:
            x = x + checkpoint(self.function, *features)
        else:
            x = x + getattr(self, 'LFF')((torch.cat(*features, dim=1)))  # local residual learning
        x = getattr(self, 'CA')(x)
        return x
