import torch.nn as nn
from .base_module import BaseModule


class ResidualBlock(BaseModule):
    def __init__(self, n_ch, act, norm, pad, kernel_size=3):
        super(ResidualBlock, self).__init__()
        block = [pad(1), nn.Conv2d(n_ch, n_ch, kernel_size=kernel_size, padding=0, stride=1)]
        block += self.add_norm_act_layer(norm, n_ch=n_ch, act=act)
        block += [pad(1), nn.Conv2d(n_ch, n_ch, kernel_size=kernel_size, padding=0, stride=1)]
        block += self.add_norm_act_layer(norm, n_ch=n_ch)

        self.block = nn.Sequential(*block)

    def forward(self, x):
        x = x + self.block(x)

        return x