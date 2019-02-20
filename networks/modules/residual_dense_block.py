import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from .base_module import BaseModule
from .residual_channel_attention_block import ChannelAttentionLayer
from . residual_block import ResidualBlock


class DenseLayer(BaseModule):
    def __init__(self, n_ch, growth_rate, kernel_size=3, act='relu', norm='InstanceNorm2d', pad='reflection',
                 pre_activation=False, efficient=True):
        super(DenseLayer, self).__init__()
        act = self.get_act_layer(act, inplace=True, negative_slope=0.2) if isinstance(act, str) else act
        norm = self.get_norm_layer(norm) if isinstance(norm, str) else norm
        pad = self.get_pad_layer(pad) if isinstance(pad, str) else pad

        if pre_activation:
            layer = [act, pad(1),
                     nn.Conv2d(n_ch, growth_rate, kernel_size=kernel_size),
                     norm(growth_rate), nn.LeakyReLU(1.0, inplace=True)]
            # without activation at the last, it consumes more memory without reason.
            # plz advice us if you have any idea about this phenomenon.

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
        entry_layer = [act, pad(1), nn.Conv2d(n_ch, growth_rate, kernel_size, bias=False), norm(growth_rate)]
        self.add_module('Entry_layer', nn.Sequential(*entry_layer))
        for i in range(n_dense_layers):
            self.add_module('Dense_layer_{}'.format(i), DenseLayer(growth_rate * (i + 1), growth_rate, kernel_size, act,
                                                                   norm, pad, pre_activation=pre_activation,
                                                                   efficient=efficient))

        if pre_activation:
            SE = [nn.Conv2d(n_dense_layers * growth_rate, init_ch, kernel_size=1, bias=False),
                  ChannelAttentionLayer(init_ch),
                  nn.LeakyReLU(1.0, True)]
        # else:
        #    SE = [nn.Conv2d(growth_rate, init_ch, kernel_size=1, bias=False), norm(init_ch), nn.LeakyReLU(1.0, True)]

        self.add_module('SE', nn.Sequential(*SE))  # style expansion
        # self.add_module('AR', AggregatedResidualBlock(init_ch, init_ch, 128, act, norm, pad))

        self.efficient = efficient
        self.init_ch = init_ch
        self.n_ch = n_ch
        self.n_dense_layers = n_dense_layers

    def function(self, *feature):
        return getattr(self, 'SE')(torch.cat(feature, dim=1))

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


class ResidualChannelAttentionDenseBlock(ResidualDenseBlock):
    def __init__(self, n_ch, growth_rate, n_dense_layers, act, norm, pad, kernel_size=3, pre_activation=False):
        super(ResidualChannelAttentionDenseBlock, self).__init__(n_ch, growth_rate, n_dense_layers, act, norm, pad,
                                                                 kernel_size=kernel_size, pre_activation=pre_activation)
        delattr(self, 'LFF')
        if pre_activation:
            LFF = [nn.Conv2d(self.n_ch, self.init_ch, kernel_size=1, bias=False),
                   norm(self.init_ch), nn.LeakyReLU(1.0, True)]
        else:
            LFF = [ChannelAttentionLayer(self.n_ch), nn.Conv2d(self.n_ch, self.init_ch, kernel_size=1, bias=False),
                   norm(self.init_ch), nn.LeakyReLU(1.0, True)]

        self.add_module('LFF', nn.Sequential(*LFF))  # local feature fusion

    def forward(self, x):
        features = [x]
        for i in range(self.n_dense_layers):
            features += [getattr(self, 'Dense_layer_{}'.format(i))(*features)]

        if self.efficient:
            x = x + checkpoint(self.function, *features)
        else:
            x = x + getattr(self, 'LFF')((torch.cat(*features, dim=1)))  # local residual learning
        return x


class AggregatedResidualBlock(BaseModule):
    def __init__(self, in_ch, out_ch, cardinality, act, norm, pad):
        super(AggregatedResidualBlock, self).__init__()
        block = [nn.Conv2d(in_ch, 4 * cardinality // 2, kernel_size=1, bias=False), act, norm(4 * cardinality)]
        block += [pad(1), nn.Conv2d(4 * cardinality // 2, 4 * cardinality // 2, kernel_size=3, groups=cardinality,
                                    bias=False), act, norm(4 * cardinality)]
        block += [nn.Conv2d(4 * cardinality // 2, out_ch, kernel_size=1, bias=False), act, norm(out_ch)]
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return x + self.block(x)
