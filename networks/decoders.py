import torch.nn as nn
from .base_network import BaseNetwork


class Decoder(BaseNetwork):
    def __init__(self, input_ch, output_ch, n_upsample, kernel_size=3, act=None, norm=None, pad=None, tanh=True):
        super(Decoder, self).__init__()
        ch = input_ch
        ps = kernel_size // 2  # padding size

        up_layers = []
        for i in range(n_upsample):
            up_layers += [nn.ConvTranspose2d(ch, ch//2, kernel_size=kernel_size, padding=1, stride=2, output_padding=1,
                                             bias=True)]
            up_layers += self.add_norm_act_layer(norm, n_ch=ch//2, act=act)
            ch //= 2

        up_layers += [pad(3), nn.Conv2d(ch, output_ch, kernel_size=7, padding=0, stride=1, bias=True)]
        up_layers += [nn.Tanh()] if tanh else None

        self.model = nn.Sequential(*up_layers)

    def forward(self, x):
        return self.model(x)


class ProgressiveDecoder(BaseNetwork):
    def __init__(self, input_ch, output_ch, n_upsample, act, kernel_size=3, norm=None, pad=None, tanh=True):
        super(ProgressiveDecoder, self).__init__()
        ps = kernel_size // 2  # padding size
        rgb_layers = []
        up_layers = []

        ch = input_ch
        if tanh:
            rgb_layers += [[pad(ps), nn.Conv2d(ch, output_ch, kernel_size=kernel_size, padding=0, stride=1, bias=True),
                            nn.Tanh()]]

            for _ in range(n_upsample):
                ch //= 2
                rgb_layers += [[pad(ps), nn.Conv2d(ch, output_ch, kernel_size=kernel_size, padding=0, stride=1,
                                                   bias=True), nn.Tanh()]]

        else:
            rgb_layers += [[pad(ps), nn.Conv2d(ch, output_ch, kernel_size=kernel_size, padding=0, stride=1, bias=True)]]

            for _ in range(n_upsample):
                ch //= 2
                rgb_layers += [[pad(ps), nn.Conv2d(ch, output_ch, kernel_size=kernel_size, padding=0, stride=1,
                                                   bias=True)]]

        ch = input_ch
        for _ in range(n_upsample):
            up_layers += [[nn.ConvTranspose2d(ch, ch // 2, kernel_size=kernel_size, padding=1, stride=2,
                                              output_padding=1, bias=True),
                           *self.add_norm_act_layer(norm, n_ch=ch // 2, act=act)]]
            ch //= 2

        for i, layer in enumerate(rgb_layers):
            setattr(self, 'RGB_level_{}'.format(i), nn.Sequential(*layer))

        for i, layer in enumerate(up_layers):
            setattr(self, 'Fractional_strided_conv_{}'.format(i), nn.Sequential(*layer))

    def forward(self, x, level):
        for i in range(level):
            x = getattr(self, 'Fractional_strided_conv_{}'.format(i))(x)

        x = getattr(self, 'RGB_level_{}'.format(level))(x)

        return x
