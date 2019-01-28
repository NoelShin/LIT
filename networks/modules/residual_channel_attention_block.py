import torch.nn as nn
from .base_module import BaseModule


class ChannelAttentionLayer(BaseModule):
    def __init__(self, n_ch, reduction_rate=16):
        super(ChannelAttentionLayer, self).__init__()
        layer = [nn.AdaptiveAvgPool2d(1)]
        layer += [nn.Conv2d(n_ch, n_ch // reduction_rate, kernel_size=1, padding=0, stride=1, bias=True),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(n_ch // reduction_rate, n_ch, kernel_size=1, padding=0, stride=1, bias=True),
                  nn.Sigmoid()]

        self.layer = nn.Sequential(*layer)

    def forward(self, x):
        return x * self.layer(x)


class ResidualChannelAttentionBlock(BaseModule):
    def __init__(self, n_ch, reduction_rate, kernel_size, act, norm, pad):
        super(ResidualChannelAttentionBlock, self).__init__()
        ps = kernel_size // 2
        block = [pad(ps), nn.Conv2d(n_ch, n_ch, kernel_size=kernel_size, padding=0, stride=1, bias=True)]
        block += self.add_norm_act_layer(norm, n_ch=n_ch, act=act)
        block += [pad(ps), nn.Conv2d(n_ch, n_ch, kernel_size=kernel_size, padding=0, stride=1, bias=True)]
        block += self.add_norm_act_layer(norm, n_ch=n_ch)

        block += [ChannelAttentionLayer(n_ch, reduction_rate)]

        self.block = nn.Sequential(*block)

    def forward(self, x):
        return x + self.block(x)


class ResidualGroup(BaseModule):
    def __init__(self, n_blocks, n_ch, reduction_rate, kernel_size, act, norm, pad):
        super(ResidualGroup, self).__init__()
        ps = kernel_size //2
        group = [ResidualChannelAttentionBlock(n_ch, reduction_rate, kernel_size, act, norm, pad)
                 for _ in range(n_blocks)]
        group += [pad(ps), nn.Conv2d(n_ch, n_ch, kernel_size=kernel_size)]
        group += self.add_norm_act_layer(norm, n_ch=n_ch)

        self.group = nn.Sequential(*group)

    def forward(self, x):
        return x + self.group(x)
