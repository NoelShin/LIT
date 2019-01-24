import torch.nn as nn
from base_network import BaseNetwork


class Encoder(BaseNetwork):
    def __init__(self, input_ch, n_gf, n_downsample, act='relu', kernel_size=3, norm='InstanceNorm2d', pad=None):
        super(Encoder, self).__init__()
        act = self.get_act_layer(act, inplace=True) if isinstance(act, str) else act
        norm = self.get_norm_layer(norm) if isinstance(norm, str) else norm
        pad = self.get_pad_layer(pad) if isinstance(pad, str) else pad

        encoder = [pad(1), nn.Conv2d(input_ch, n_gf, kernel_size=kernel_size, padding=0, stride=1)]
        encoder += self.add_norm_act_layer(norm, n_ch=n_gf, act=act)

        for _ in range(n_downsample):
            encoder += [pad(1), nn.Conv2d(n_gf, 2 * n_gf, kernel_size=kernel_size, padding=0, stride=2)]
            encoder += self.add_norm_act_layer(norm, n_ch=2 * n_gf, act=act)
            n_gf *= 2

        self.model = nn.Sequential(*encoder)

    def forward(self, x):
        return self.model(x)
