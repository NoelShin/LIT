from functools import partial
import torch
import torch.nn as nn


class BaseModule(nn.Module):
    def __init__(self):
        super(BaseModule, self).__init__()

    @staticmethod
    def add_norm_act_layer(norm=None, act=None, n_ch=None):
        layer = []

        if isinstance(norm, partial):
            layer += [norm(n_ch)]

        elif not norm:
            pass

        elif norm.__name__ == 'PixelNorm':
            layer += [norm]

        else:
            raise NotImplementedError

        if act:
            layer += [act]

        return layer

    @staticmethod
    def get_act_layer(type, inplace=True, negative_slope=None):
        if type == 'leaky_relu':
            assert negative_slope != 0.0
            layer = nn.LeakyReLU(negative_slope=0.2, inplace=inplace)

        elif type == 'prelu':
            layer = nn.PReLU

        elif type == 'relu':
            layer = nn.ReLU(inplace=inplace)

        else:
            raise NotImplementedError("Invalid activation {}. Please check your activation type.".format(type))

        return layer

    @staticmethod
    def get_norm_layer(type):
        if type == 'BatchNorm2d':
            layer = partial(nn.BatchNorm2d, affine=True)

        elif type == 'InstanceNorm2d':
            layer = partial(nn.InstanceNorm2d, affine=False)

        elif type == 'PixelNorm':
            layer = PixelNorm()

        else:
            layer = None
            print("Normalization layer is not selected.")

        return layer

    @staticmethod
    def get_pad_layer(type):
        if type == 'reflection':
            layer = nn.ReflectionPad2d

        elif type == 'replication':
            layer = nn.ReplicationPad2d

        elif type == 'zero':
            layer = nn.ZeroPad2d

        else:
            raise NotImplementedError(
                "Padding type {} is not valid. Please choose among ['reflection', 'replication', 'zero']".format(type))

        return layer

    def forward(self, x):
        pass


class ChannelAttentionLayer(nn.Module):
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


class SeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, bias=True):
        super(SeparableConv, self).__init__()
        self.add_module('Spatial_conv', nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding,
                                                  groups=in_channels, bias=bias))
        self.add_module('Cross_channel_conv', nn.Conv2d(in_channels, out_channels, 1, bias=bias))
        self.model = nn.Sequential(getattr(self, 'Spatial_conv'), getattr(self, 'Cross_channel_conv'))

    def forward(self, x):
        return self.model(x)


class PixelNorm(nn.Module):
    def __init__(self, eps=1e-8):
        super(PixelNorm, self).__init__()
        self.__name__ = 'PixelNorm'
        self.eps = eps

    def forward(self, x):
        x = x * torch.rsqrt(torch.mean(x.pow(2), dim=1, keepdim=True) + self.eps)

        return x
