import torch
import torch.nn as nn
from .base_module import BaseModule
from .residual_channel_attention_block import ChannelAttentionLayer


class DenseLayer(BaseModule):
    def __init__(self, n_ch, growth_rate, kernel_size=3, act='relu', norm='InstanceNorm2d', pad='reflection'):
        super(DenseLayer, self).__init__()
        act = self.get_act_layer(act, inplace=True) if isinstance(act, str) else act
        norm = self.get_norm_layer(norm) if isinstance(norm, str) else norm
        pad = self.get_pad_layer(pad) if isinstance(pad, str) else pad

        layer = [pad(1), nn.Conv2d(n_ch, growth_rate, kernel_size=kernel_size, padding=0, stride=1, bias=False)]
        layer += self.add_norm_act_layer(norm, n_ch=growth_rate, act=act)

        self.layer = nn.Sequential(*layer)

    def forward(self, x):
        y = self.layer(x)
        x = torch.cat([x, y], dim=1)

        return x


class ResidualDenseBlock(BaseModule):
    def __init__(self, n_ch, growth_rate, n_dense_layers, act, norm, pad, kernel_size=3):
        super(ResidualDenseBlock, self).__init__()
        init_n_ch = n_ch

        block = []
        for _ in range(n_dense_layers):
            block += [DenseLayer(n_ch, growth_rate, kernel_size=kernel_size, act=act, norm=norm, pad=pad)]
            n_ch += growth_rate
        block += [nn.Conv2d(n_ch, init_n_ch, kernel_size=1, padding=0, stride=1, bias=False)]  # local feature fusion
        self.CA = ChannelAttentionLayer(init_n_ch)
        self.block = nn.Sequential(*block)

    def forward(self, x):
        x = self.CA(x) + self.block(x)  # local residual learning

        return x
