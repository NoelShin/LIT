import torch
import torch.nn as nn
from .modules.base_module import BaseModule
from .translators import ResidualDenseNetwork, ResidualNetwork
from .modules.residual_channel_attention_block import ResidualChannelAttentionBlock


class BaseNetwork(BaseModule):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    @staticmethod
    def init_weights(module, type='kaiming_normal', mode='fan_in', negative_slope=0.2, nonlinearity='leaky_relu'):
        classname = module.__class__.__name__
        if classname.find('Conv') != -1:
            if type == 'kaiming_normal':
                nn.init.kaiming_normal_(module.weight.detach(), a=negative_slope, mode=mode, nonlinearity=nonlinearity)

            elif type == 'normal':
                nn.init.normal_(module.weight.detach(), 0.0, 0.02)

            else:
                raise NotImplementedError("Weight init type {} is not valid.".format(type))

        else:
            pass

    def set_attribute(self, list, name, sequential=True, progressive=False):
        if progressive:
            if sequential:
                for i in range(len(list)):
                    setattr(self, name + '_level_{}'.format(i), nn.Sequential(*list[i]))
            else:
                for i in range(len(list)):
                    setattr(self, name + '_level_{}'.format(i), list[i])

        else:
            if sequential:
                for i in range(len(list)):
                    setattr(self, name + '_{}'.format(i), nn.Sequential(*list[i]))
            else:
                for i in range(len(list)):
                    setattr(self, name + '_{}'.format(i), list[i])

    def to_CUDA(self, gpu_id):
        gpu_id = gpu_id[0] if isinstance(gpu_id, list) else gpu_id
        if gpu_id != -1:
            self.to(torch.device('cuda', 0))
        else:
            pass

    def forward(self, x):
        pass


class BaseGenerator(BaseNetwork):
    def __init__(self):
        super(BaseGenerator, self).__init__()

    @staticmethod
    def get_decoder(opt, act, norm, pad, kernel_size=3):
        if opt.progression:
            from networks.decoders import ProgressiveDecoder
            decoder = ProgressiveDecoder(opt.n_gf * 2 ** opt.n_downsample, opt.output_ch, opt.n_downsample,
                                         kernel_size=kernel_size, act=act, norm=norm, pad=pad, tanh=True)
        else:
            from networks.decoders import Decoder
            decoder = Decoder(opt.n_gf * 2 ** opt.n_downsample, opt.output_ch, opt.n_downsample,
                              kernel_size=kernel_size, act=act, norm=norm, pad=pad, tanh=True)
        return decoder

    @staticmethod
    def get_encoder(opt, act, norm, pad, kernel_size=3):
        from networks.encoders import Encoder
        encoder = Encoder(opt.input_ch, opt.n_gf, opt.n_downsample, kernel_size=kernel_size,
                          act=act, norm=norm, pad=pad)
        return encoder

    @staticmethod
    def get_enhance_layer(opt, n_ch, act, norm, pad):
        n_enhance = opt.n_enhance_blocks
        trans_network = opt.trans_network

        if trans_network == 'RCAN':
            enhance_layer = [ResidualChannelAttentionBlock(n_ch, opt.reduction_rate, 3, act, norm, pad) for _ in
                             range(n_enhance)]

        elif trans_network == 'RDN':
            enhance_layer = [ResidualDenseNetwork(n_enhance, n_ch, n_ch // 8, opt.n_dense_layers, act, norm, pad)]

        elif trans_network == 'RN':
            enhance_layer = [ResidualNetwork(n_enhance, n_ch, act, norm, pad)]

        else:
            raise NotImplementedError

        return enhance_layer

    @staticmethod
    def get_trans_network(opt, act, norm, pad, kernel_size=3, input_ch=None):
        if opt.trans_network == 'RCAN':
            from networks.translators import ResidualChannelAttentionNetwork
            net = ResidualChannelAttentionNetwork(opt.n_RG, opt.n_RCAB, opt.RCA_ch,
                                                  opt.reduction_rate, kernel_size=kernel_size, act=act, norm=norm,
                                                  pad=pad, input_ch=input_ch)

        elif opt.trans_network == 'RN':
            from networks.translators import ResidualNetwork
            net = ResidualNetwork(opt.n_RB, opt.n_gf * 2 ** opt.n_downsample, kernel_size=kernel_size, act=act,
                                  norm=norm, pad=pad)

        elif opt.trans_network == 'RDN':
            from networks.translators import ResidualDenseNetwork
            net = ResidualDenseNetwork(opt.n_RDB, opt.n_gf * 2 ** opt.n_downsample, opt.growth_rate,
                                       opt.n_dense_layers, kernel_size=kernel_size, act=act, norm=norm, pad=pad)

        else:
            raise NotImplementedError("Invalid translation unit {}. Please check trans_type option.".
                                      format(opt.trans_unit))
        return net
    
