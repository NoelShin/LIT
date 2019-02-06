from functools import partial
import torch
import torch.nn as nn
import torchvision
from networks.base_network import BaseNetwork
from networks.translators import ResidualDenseNetwork, ResidualNetwork
from networks.modules.residual_channel_attention_block import ChannelAttentionLayer, ResidualChannelAttentionBlock


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


class Critic(BaseNetwork):
    def __init__(self, opt):
        super(Critic, self).__init__()
        self.n_C = opt.n_C
        for i in range(opt.n_C):
            setattr(self, 'Scale_{}'.format(i), PatchCritic(opt))

        self.apply(partial(self.init_weights, type=opt.init_type, mode=opt.fan_mode, negative_slope=opt.negative_slope,
                           nonlinearity=opt.C_act))

        self.to_CUDA(opt.gpu_ids)

        print(self)
        print("the number of C parameters: ", sum(p.numel() for p in self.parameters() if p.requires_grad))

    def forward(self, x):
        result = []
        for i in range(self.n_C):
            result.append(getattr(self, 'Scale_{}'.format(i))(x))
            x = nn.AvgPool2d(kernel_size=3, padding=1, stride=2, count_include_pad=False)(x)

        return result


class GatedUNetGenerator(BaseGenerator):
    def __init__(self, opt):
        super(GatedUNetGenerator, self).__init__()
        act = self.get_act_layer(opt.G_act, inplace=True)
        norm = self.get_norm_layer(opt.norm_type)
        pad = self.get_pad_layer(opt.pad_type)

        n_ch = opt.n_gf
        n_downsample = opt.n_downsample
        layer_first = [pad(3), nn.Conv2d(opt.input_ch, n_ch, kernel_size=7, padding=0)]
        layer_first += self.add_norm_act_layer(norm, n_ch=n_ch, act=act)
        self.layer_first = nn.Sequential(*layer_first)

        layer_last = [pad(3), nn.Conv2d(n_ch, opt.output_ch, kernel_size=7, padding=0)]
        layer_last += [nn.Tanh()] if opt.tanh else None
        self.layer_last = nn.Sequential(*layer_last)

        gate_layer = [[ChannelAttentionLayer(n_ch)]]
        for i in range(n_downsample):
            encoder_layer = [pad(1), nn.Conv2d(n_ch, 2 * n_ch, kernel_size=3, padding=0, stride=2)]
            encoder_layer += self.add_norm_act_layer(norm, n_ch=2 * n_ch, act=act)

            gate_layer += [[ChannelAttentionLayer(2 * n_ch)]] if i != n_downsample - 1 else []

            decoder_layer = [nn.ConvTranspose2d(2 * n_ch, n_ch, kernel_size=3, padding=1, stride=2,
                                                output_padding=1)]
            decoder_layer += self.add_norm_act_layer(norm, n_ch=n_ch, act=act)
            n_ch *= 2

            setattr(self, 'Down_layer_{}'.format(i), nn.Sequential(*encoder_layer))
            setattr(self, 'Up_layer_{}'.format(i), nn.Sequential(*decoder_layer))

        for i in range(len(gate_layer)):
            setattr(self, 'Gate_layer_{}'.format(i), nn.Sequential(*gate_layer[i]))

        self.translator = self.get_trans_network(opt, act, norm, pad, input_ch=n_ch)
        self.n_downsample = n_downsample

        self.apply(partial(self.init_weights, type=opt.init_type, mode=opt.fan_mode, negative_slope=opt.negative_slope,
                           nonlinearity=opt.G_act))

        self.to_CUDA(opt.gpu_ids)

        print(self)
        print("the number of G parameters: ", sum(p.numel() for p in self.parameters() if p.requires_grad))

    def forward(self, x):
        results = [self.layer_first(x)]
        for i in range(self.n_downsample):
            results += [getattr(self, 'Down_layer_{}'.format(i))(results[-1])]
        # results 0 1 2 3 4

        x = self.translator(results[-1])
        for i in range(self.n_downsample - 1, -1, -1):  # 3 2 1 0
            x = getattr(self, 'Up_layer_{}'.format(i))(x) + getattr(self, 'Gate_layer_{}'.format(i))(results[i])

        x = self.layer_last(x)
        return x


class Generator(BaseGenerator):
    def __init__(self, opt):
        super(Generator, self).__init__()
        act = self.get_act_layer(opt.G_act, inplace=True)
        norm = self.get_norm_layer(opt.norm_type)
        pad = self.get_pad_layer(opt.pad_type)

        self.encoder = self.get_encoder(opt, act, norm, pad)
        self.translator = self.get_trans_network(opt, act, norm, pad, input_ch=1024)
        self.decoder = self.get_decoder(opt, act, norm, pad)

        self.apply(partial(self.init_weights, type=opt.init_type, mode=opt.fan_mode, negative_slope=opt.negative_slope,
                           nonlinearity=opt.G_act))

        self.to_CUDA(opt.gpu_ids)

        print(self)
        print("the number of G parameters: ", sum(p.numel() for p in self.parameters() if p.requires_grad))

    def forward(self, x):
        x = self.decoder(self.translator(self.encoder(x)))

        return x


class PatchCritic(BaseNetwork):
    def __init__(self, opt):
        super(BaseNetwork, self).__init__()
        self.act = nn.LeakyReLU(0.2, inplace=False)  # to avoid inplace activation. inplace activation cause error WGAN_GP
        input_channel = opt.input_ch + opt.output_ch if opt.C_condition else opt.output_ch
        n_df = opt.n_df
        norm = self.get_norm_layer(opt.norm_type) if opt.C_norm else None
        patch_size = opt.patch_size

        blocks = [[nn.Conv2d(input_channel, n_df, kernel_size=4, padding=1, stride=2)]]
        blocks += [[nn.Conv2d(n_df, 2 * n_df, kernel_size=4, padding=1, stride=2),
                    *self.add_norm_act_layer(norm, n_ch=2 * n_df)]]
        if patch_size == 16:
            blocks += [[nn.Conv2d(2 * n_df, 4 * n_df, kernel_size=4, padding=1, stride=1),
                        *self.add_norm_act_layer(norm, n_ch=4 * n_df)]]
            blocks += [[nn.Conv2d(4 * n_df, 1, kernel_size=4, padding=1, stride=1)]]

        elif patch_size == 70:
            blocks += [[nn.Conv2d(2 * n_df, 4 * n_df, kernel_size=4, padding=1, stride=2),
                        *self.add_norm_act_layer(norm, n_ch=4 * n_df)]]
            blocks += [[nn.Conv2d(4 * n_df, 8 * n_df, kernel_size=4, padding=1, stride=1),
                        *self.add_norm_act_layer(norm, n_ch=8 * n_df)]]
            blocks += [[nn.Conv2d(8 * n_df, 1, kernel_size=4, padding=1, stride=1)]]

        self.n_blocks = len(blocks)
        for i in range(self.n_blocks):
            setattr(self, 'block_{}'.format(i), nn.Sequential(*blocks[i]))

    def forward(self, x):
        result = [x]
        for i in range(self.n_blocks - 1):
            result.append(self.act(getattr(self, 'block_{}'.format(i))(result[-1])))
        result.append(getattr(self, 'block_{}'.format(self.n_blocks - 1))(result[-1]))

        return result[1:]  # except for the input


class ProgressiveGenerator(BaseGenerator):
    def __init__(self, opt):
        super(ProgressiveGenerator, self).__init__()
        act = self.get_act_layer(opt.G_act, inplace=True)
        norm = self.get_norm_layer(opt.norm_type)
        pad = self.get_pad_layer(opt.pad_type)

        n_ch = opt.n_gf
        n_downsample = opt.n_downsample
        layer_first = [pad(1), nn.Conv2d(opt.input_ch, n_ch, kernel_size=3, padding=0, stride=1)]
        self.layer_first = nn.Sequential(*layer_first)

        RGB_layer = [[pad(1), nn.Conv2d(n_ch, opt.output_ch, kernel_size=3, padding=0, stride=1)]]
        RGB_layer[-1].append(nn.Tanh()) if opt.tanh else None

        for i in range(n_downsample):
            encoder_layer = self.add_norm_act_layer(norm, n_ch=n_ch, act=act)
            encoder_layer += [pad(1), nn.Conv2d(n_ch, 2 * n_ch, kernel_size=3, padding=0, stride=2)]
            encoder_layer += self.add_norm_act_layer(norm, n_ch=2 * n_ch, act=act) if i == n_downsample - 1 else []

            RGB_layer += [[pad(1), nn.Conv2d(2 * n_ch, opt.output_ch, kernel_size=3, padding=0, stride=1)]]
            RGB_layer[-1].append(nn.Tanh()) if opt.tanh else None

            decoder_layer = [nn.ConvTranspose2d(2 * n_ch, n_ch, kernel_size=3, padding=1, stride=2, output_padding=1)]
            decoder_layer += self.add_norm_act_layer(norm, n_ch=n_ch, act=act)

            n_ch *= 2

            setattr(self, 'Down_layer_{}'.format(i), nn.Sequential(*encoder_layer))
            setattr(self, 'Up_layer_{}'.format(i), nn.Sequential(*decoder_layer))

        n_RGB = len(RGB_layer)
        for i in range(n_RGB):  # 0 1 2 3 4
            setattr(self, 'RGB_layer_{}'.format(n_RGB - 1 - i), nn.Sequential(*RGB_layer[i]))

        self.translator = self.get_trans_network(opt, act, norm, pad, input_ch=n_ch)
        self.n_downsample = opt.n_downsample
        self.apply(partial(self.init_weights, type=opt.init_type, mode=opt.fan_mode, negative_slope=opt.negative_slope,
                           nonlinearity=opt.G_act))
        self.to_CUDA(opt.gpu_ids)

        print(self)
        print("the number of G parameters: ", sum(p.numel() for p in self.parameters() if p.requires_grad))

    def forward(self, x, level):
        results = [self.layer_first(x)]
        for i in range(self.n_downsample):
            results.append(getattr(self, 'Down_layer_{}'.format(i))(results[-1]))

        x = self.translator(results[-1])
        for i in range(level):  # 0 1 2 3
            x = getattr(self, 'Up_layer_{}'.format(self.n_downsample - 1 - i))(x)

        x = getattr(self, 'RGB_layer_{}'.format(level))(x)
        return x


class ProgressiveGatedUNetGenerator(BaseGenerator):
    def __init__(self, opt):
        super(ProgressiveGatedUNetGenerator, self).__init__()
        act = self.get_act_layer(opt.G_act, inplace=True)
        norm = self.get_norm_layer(opt.norm_type)
        pad = self.get_pad_layer(opt.pad_type)

        n_ch = opt.n_gf
        n_downsample = opt.n_downsample
        layer_first = [pad(3), nn.Conv2d(opt.input_ch, n_ch, kernel_size=7, padding=0, stride=1)]
        layer_first += self.add_norm_act_layer(norm, n_ch=n_ch, act=act)
        self.layer_first = nn.Sequential(*layer_first)

        RGB_layer = []
        gate_layer = [[ChannelAttentionLayer(n_ch)]]
        for i in range(n_downsample):
            encoder_layer = [pad(1), nn.Conv2d(n_ch, 2 * n_ch, kernel_size=3, padding=0, stride=2)]
            encoder_layer += self.add_norm_act_layer(norm, n_ch=2 * n_ch, act=act)
            gate_layer += [[ChannelAttentionLayer(2 * n_ch)]] if i != n_downsample - 1 else []

            decoder_layer = [nn.ConvTranspose2d(2 * n_ch, n_ch, kernel_size=3, padding=1, stride=2, output_padding=1)]
            decoder_layer += self.add_norm_act_layer(norm, n_ch=n_ch, act=act)

            enhance_layer = [ResidualNetwork(opt.n_enhance_blocks, n_ch, act, norm, pad)]

            RGB_layer += [[pad(1), nn.Conv2d(n_ch, opt.output_ch, kernel_size=3, padding=0, stride=1)]]
            RGB_layer[-1].append(nn.Tanh()) if opt.tanh else None

            n_ch *= 2

            setattr(self, 'Down_layer_{}'.format(i), nn.Sequential(*encoder_layer))
            setattr(self, 'Up_layer_{}'.format(i), nn.Sequential(*decoder_layer))
            setattr(self, 'Enhance_layer_{}'.format(i), nn.Sequential(*enhance_layer))

        RGB_layer += [[pad(1), nn.Conv2d(n_ch, opt.output_ch, kernel_size=3, padding=0, stride=1)]]
        RGB_layer[-1].append(nn.Tanh()) if opt.tanh else None

        n_RGB = len(RGB_layer)
        for i in range(n_RGB): # 0 1 2 3 4
            setattr(self, 'RGB_layer_{}'.format(i), nn.Sequential(*RGB_layer[i]))

        n_gate = len(gate_layer)
        for i in range(n_gate):
            setattr(self, 'Gate_layer_{}'.format(i), nn.Sequential(*gate_layer[i]))

        self.translator = self.get_trans_network(opt, act, norm, pad, input_ch=n_ch)
        self.n_downsample = n_downsample
        self.n_gate = n_gate
        self.n_RGB = n_RGB
        self.apply(partial(self.init_weights, type=opt.init_type, mode=opt.fan_mode, negative_slope=opt.negative_slope,
                           nonlinearity=opt.G_act))
        self.to_CUDA(opt.gpu_ids)

        print(self)
        print("the number of G parameters: ", sum(p.numel() for p in self.parameters() if p.requires_grad))

    def forward(self, x, level, level_in):
        results = [self.layer_first(x)]
        for i in range(self.n_downsample):
            results.append(getattr(self, 'Down_layer_{}'.format(i))(results[-1]))

        x = self.translator(results[-1])
        out = getattr(self, 'RGB_layer_{}'.format(self.n_RGB - 1))(x)

        for i in range(level):
            x = getattr(self, 'Gate_layer_{}'.format(self.n_gate - 1 - i))(results[self.n_downsample - 1 - i])\
                + getattr(self, 'Up_layer_{}'.format(self.n_downsample - 1 - i))(x)
            x = getattr(self, 'Enhance_layer_{}'.format(self.n_downsample - 1 - i))(x)
            y = getattr(self, 'RGB_layer_{}'.format(self.n_RGB - 2 - i))(x)
            out = nn.functional.interpolate(out, scale_factor=2, mode='nearest')
            out = torch.lerp(out, y, level_in - level)

        return out


class UNetGenerator(BaseGenerator):
    def __init__(self, opt):
        super(UNetGenerator, self).__init__()
        act = self.get_act_layer(opt.G_act, inplace=True)
        norm = self.get_norm_layer(opt.norm_type)
        pad = self.get_pad_layer(opt.pad_type)

        n_ch = opt.n_gf
        n_downsample = opt.n_downsample
        layer_first = [pad(3), nn.Conv2d(opt.input_ch, n_ch, kernel_size=7, padding=0)]
        layer_first += self.add_norm_act_layer(norm, n_ch=n_ch, act=act)
        self.layer_first = nn.Sequential(*layer_first)

        layer_last = [pad(3), nn.Conv2d(n_ch, opt.output_ch, kernel_size=7, padding=0)]
        layer_last += [nn.Tanh()] if opt.tanh else None
        self.layer_last = nn.Sequential(*layer_last)

        for i in range(n_downsample):
                encoder_layer = [pad(1), nn.Conv2d(n_ch, 2 * n_ch, kernel_size=3, padding=0, stride=2)]
                encoder_layer += self.add_norm_act_layer(norm, n_ch=2 * n_ch, act=act)

                decoder_layer = [nn.ConvTranspose2d(2 * n_ch, n_ch, kernel_size=3, padding=1, stride=2,
                                                    output_padding=1)]
                decoder_layer += self.add_norm_act_layer(norm, n_ch=n_ch, act=act)
                n_ch *= 2

                setattr(self, 'Down_layer_{}'.format(i), nn.Sequential(*encoder_layer))
                setattr(self, 'Up_layer_{}'.format(i), nn.Sequential(*decoder_layer))

        self.translator = self.get_trans_network(opt, act, norm, pad, input_ch=n_ch)
        self.n_downsample = n_downsample

        self.apply(partial(self.init_weights, type=opt.init_type, mode=opt.fan_mode, negative_slope=opt.negative_slope,
                           nonlinearity=opt.G_act))

        self.to_CUDA(opt.gpu_ids)

        print(self)
        print("the number of G parameters: ", sum(p.numel() for p in self.parameters() if p.requires_grad))

    def forward(self, x):
        results = [self.layer_first(x)]
        for i in range(self.n_downsample):
            results += [getattr(self, 'Down_layer_{}'.format(i))(results[-1])]
        # results 0 1 2 3 4

        x = self.translator(results[-1])
        for i in range(self.n_downsample - 1, -1, -1):  # 3 2 1 0
            x = results[i] + getattr(self, 'Up_layer_{}'.format(i))(x)
        x = self.layer_last(x)

        return x


class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        vgg_pretrained_layers = torchvision.models.vgg19(pretrained=True).features
        self.block_1 = nn.Sequential()
        self.block_2 = nn.Sequential()
        self.block_3 = nn.Sequential()
        self.block_4 = nn.Sequential()
        self.block_5 = nn.Sequential()

        for i in range(2):
            self.block_1.add_module(str(i), vgg_pretrained_layers[i])

        for i in range(2, 7):
            self.block_2.add_module(str(i), vgg_pretrained_layers[i])

        for i in range(7, 12):
            self.block_3.add_module(str(i), vgg_pretrained_layers[i])

        for i in range(12, 21):
            self.block_4.add_module(str(i), vgg_pretrained_layers[i])

        for i in range(21, 30):
            self.block_5.add_module(str(i), vgg_pretrained_layers[i])

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        out_1 = self.block_1(x)
        out_2 = self.block_2(out_1)
        out_3 = self.block_3(out_2)
        out_4 = self.block_4(out_3)
        out_5 = self.block_5(out_4)
        out = [out_1, out_2, out_3, out_4, out_5]

        return out
