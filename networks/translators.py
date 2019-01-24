import torch
import torch.nn as nn
from base_network import BaseNetwork
from modules.residual_block import ResidualBlock
from modules.residual_dense_block import ResidualDenseBlock
from modules.residual_channel_attention_block import ResidualGroup


class ResidualChannelAttentionNetwork(BaseNetwork):
    def __init__(self, n_groups, n_blocks, n_ch, reduction_rate, act, norm, pad, kernel_size=3):
        super(ResidualChannelAttentionNetwork, self).__init__()
        ps = kernel_size // 2
        network = [ResidualGroup(n_blocks, n_ch, reduction_rate, kernel_size, act, norm, pad)
                   for _ in range(n_groups)]
        network += [pad(ps), nn.Conv2d(n_ch, n_ch, kernel_size=kernel_size, padding=0, stride=1, bias=True)]

        self.network = nn.Sequential(*network)

    def forward(self, x):
        return x + self.network(x)


class ResidualDenseNetwork(BaseNetwork):
    def __init__(self, input_ch, n_blocks, n_ch, growth_rate, n_dense_layers, act, norm, pad, kernel_size=3):
        super(ResidualDenseNetwork, self).__init__()

        F0 = [nn.Conv2d(input_ch, n_ch, kernel_size=1, padding=0, stride=1, bias=True)]
        F0 += self.add_norm_act_layer(norm, n_ch=n_ch)

        for i in range(n_blocks):
            setattr(self, 'ResidualDenseBlock_{}'.format(i), ResidualDenseBlock(n_ch, growth_rate, n_dense_layers, act,
                                                                                norm, pad, kernel_size=kernel_size))

        GFF = [nn.Conv2d(n_blocks * n_ch, n_ch, kernel_size=1, padding=0, stride=1, bias=True)]
        GFF += self.add_norm_act_layer(norm, n_ch=n_ch)
        GFF += [pad(1), nn.Conv2d(n_ch, n_ch, kernel_size=3, padding=0, stride=1, bias=True)]
        GFF += self.add_norm_act_layer(norm, n_ch=n_ch)

        F_last = [nn.Conv2d(n_ch, input_ch, kernel_size=1, padding=0, stride=1, bias=True)]
        F_last += self.add_norm_act_layer(norm, n_ch=input_ch, act=act)

        self.F0 = nn.Sequential(*F0)
        self.GFF = nn.Sequential(*GFF)
        self.n_blocks = n_blocks
        self.F_last = nn.Sequential(*F_last)

    def forward(self, x):
        x = self.F0(x)
        results = [x]
        for i in range(self.n_blocks):
            results.append(getattr(self, 'ResidualDenseBlock_{}'.format(i))(results[-1]))
        cat = torch.cat(results[1:], dim=1)
        y = self.GFF(cat)
        x = x + y

        x = self.F_last(x)

        return x


class ResidualNetwork(BaseNetwork):
    def __init__(self, n_blocks, n_ch, act, norm, pad, kernel_size=3):
        super(ResidualNetwork, self).__init__()

        for i in range(n_blocks):
            setattr(self, 'ResidualBlock_{}'.format(i), ResidualBlock(n_ch, act, norm, pad, kernel_size=kernel_size))

        self.n_block = n_blocks

    def forward(self, x):
        for i in range(self.n_block):
            x = getattr(self, 'ResidualBlock_{}'.format(i))(x)

        return x
