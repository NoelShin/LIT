from functools import partial
import torch
import torch.nn as nn
import torchvision
from networks.base_network import BaseNetwork
from networks.translators import ResidualDenseNetwork, ResidualNetwork
from networks.modules.residual_block import ResidualBlock
from networks.modules.residual_channel_attention_block import ChannelAttentionLayer, ResidualChannelAttentionBlock
from networks.modules.residual_dense_block import ResidualDenseBlock


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
            n_dense_layers = opt.n_dense_layers
            n_downsample = opt.n_downsample
            n_gf = opt.n_gf
            n_RDB = opt.n_RDB
            net = ResidualDenseNetwork(n_RDB, n_gf * 2 ** n_downsample, n_gf * 2 ** n_downsample//n_dense_layers,
                                       n_dense_layers, kernel_size=kernel_size, act=act, norm=norm, pad=pad)

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


class Generator(BaseGenerator):
    def __init__(self, opt):
        super(Generator, self).__init__()
        act = self.get_act_layer(opt.G_act, inplace=True)
        norm = self.get_norm_layer(opt.norm_type)
        pad = self.get_pad_layer(opt.pad_type)

        input_ch = opt.input_ch
        max_ch = opt.max_ch
        n_ch = opt.n_gf
        n_down = opt.n_downsample
        n_RB = opt.n_RB
        output_ch = opt.output_ch
        pre_activation = opt.pre_activation

        down_blocks = []
        res_blocks = []
        up_blocks = []
        if pre_activation:
            down_blocks += self.add_norm_act_layer(norm, n_ch=n_ch, act=act)
            down_blocks += [pad(3), nn.Conv2d(input_ch, n_ch, kernel_size=7, padding=0, stride=1)]
            for _ in range(n_down):
                down_blocks += self.add_norm_act_layer(norm, n_ch=min(n_ch, max_ch), act=act)
                down_blocks += [pad(1), nn.Conv2d(min(n_ch, max_ch), min(2 * n_ch, max_ch), kernel_size=3, padding=0,
                                                  stride=2)]
                n_ch *= 2

            for _ in range(n_RB):
                res_blocks += [ResidualBlock(min(n_ch, max_ch), act, norm, pad, pre_activation=True)]

            for _ in range(n_down):
                up_blocks += self.add_norm_act_layer(norm, n_ch=min(n_ch, max_ch), act=act)
                up_blocks += [nn.ConvTranspose2d(min(n_ch, max_ch), min(n_ch//2, max_ch), kernel_size=3, padding=1,
                                                 stride=2, output_padding=1)]
                n_ch //= 2

            up_blocks += self.add_norm_act_layer(norm, n_ch=n_ch, act=act)
            up_blocks += [pad(3), nn.Conv2d(n_ch, output_ch, kernel_size=7, padding=0, stride=1)]

        else:
            down_blocks += [pad(3), nn.Conv2d(input_ch, n_ch, kernel_size=7, padding=0, stride=1)]
            down_blocks += self.add_norm_act_layer(norm, n_ch=n_ch, act=act)
            for _ in range(n_down):
                down_blocks += [pad(1), nn.Conv2d(min(n_ch, max_ch), min(2 * n_ch, max_ch), kernel_size=3, padding=0,
                                                  stride=2)]
                down_blocks += self.add_norm_act_layer(norm, n_ch=min(n_ch, max_ch), act=act)
                n_ch *= 2

            for _ in range(n_RB):
                res_blocks += [ResidualBlock(min(n_ch, max_ch), act, norm, pad, pre_activation=False)]

            for _ in range(n_down):
                up_blocks += [nn.ConvTranspose2d(min(n_ch, max_ch), min(n_ch // 2, max_ch), kernel_size=3, padding=1,
                                              stride=2, output_padding=1)]
                up_blocks += self.add_norm_act_layer(norm, n_ch=min(n_ch, max_ch), act=act)
                n_ch //= 2

            up_blocks += [pad(3), nn.Conv2d(n_ch, output_ch, kernel_size=7, padding=0, stride=1)]
        up_blocks += [nn.Tanh()] if opt.tanh else None

        self.down_blocks = nn.Sequential(*down_blocks)
        self.res_blocks = nn.Sequential(*res_blocks)
        self.up_blocks = nn.Sequential(*up_blocks)

        self.apply(partial(self.init_weights, type=opt.init_type, mode=opt.fan_mode, negative_slope=opt.negative_slope,
                           nonlinearity=opt.G_act))

        self.to_CUDA(opt.gpu_ids)

        print(self)
        print("the number of G parameters: ", sum(p.numel() for p in self.parameters() if p.requires_grad))

    def forward(self, x):
        return self.up_blocks(self.res_blocks(self.down_blocks(x)))


class ProgressiveGenerator(Generator):
    def __init__(self, opt):
        super(ProgressiveGenerator, self).__init__(opt)
        act = self.get_act_layer(opt.G_act, inplace=True)
        norm = self.get_norm_layer(opt.norm_type)
        max_ch = opt.max_ch
        n_down = opt.n_downsample
        output_ch = opt.output_ch
        pre_activation = opt.pre_activation

        n_ch = opt.n_gf * 2 ** n_down
        del self.up_blocks
        rgb_blocks = [[nn.Conv2d(min(n_ch, max_ch), output_ch, kernel_size=3, padding=0, stride=1)]]
        rgb_blocks[-1].append(nn.Tanh()) if opt.tanh else None
        if pre_activation:
            for i in range(n_down):
                up_block = self.add_norm_act_layer(norm, n_ch=min(n_ch, max_ch), act=act)
                up_block += [nn.ConvTranspose2d(min(n_ch, max_ch), min(n_ch // 2, max_ch), kernel_size=3, padding=1,
                                                stride=2, output_padding=1)]
                rgb_blocks += [[nn.Conv2d(min(n_ch // 2, max_ch), output_ch, kernel_size=3, padding=0, stride=1)]]
                rgb_blocks[-1].append(nn.Tanh()) if opt.tanh else None
                setattr(self, 'Up_block_{}'.format(i), nn.Sequential(*up_block))
                n_ch //= 2
        else:
            for i in range(n_down):
                up_block = [nn.ConvTranspose2d(min(n_ch, max_ch), min(n_ch // 2, max_ch), kernel_size=3, padding=1,
                                               stride=2, output_padding=1)]
                up_block += self.add_norm_act_layer(norm, n_ch=min(n_ch // 2, max_ch), act=act)

                rgb_blocks += [[nn.Conv2d(min(n_ch // 2, max_ch), output_ch, kernel_size=3, padding=0, stride=1)]]
                rgb_blocks[-1].append(nn.Tanh()) if opt.tanh else None
                setattr(self, 'Up_block_{}'.format(i), nn.Sequential(*up_block))
                n_ch //= 2

        for i in range(len(rgb_blocks)):
            setattr(self, 'RGB_block_{}'.format(i), nn.Sequential(*rgb_blocks[i]))

    def forward(self, x, level, level_in):
        x = self.res_blocks(self.down_blocks(x))
        out = getattr(self, 'RGB_block_0')(x)
        for i in range(level):
            x = getattr(self, 'Up_block_{}'.format(i))(x)
            y = getattr(self, 'RGB_block_{}'.format(i + 1))(x)
            out = nn.functional.interpolate(out, scale_factor=2, mode='nearest')
            out = torch.lerp(out, y, level_in - level)

        return out


class PatchCritic(BaseNetwork):
    def __init__(self, opt):
        super(PatchCritic, self).__init__()
        self.act = nn.LeakyReLU(0.2, inplace=False)  # to avoid inplace activation. inplace activation cause error GP
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


class ResidualPatchCritic(BaseNetwork):
    def __init__(self, opt):
        super(ResidualPatchCritic, self).__init__()
        # self.act = nn.LeakyReLU(0.2, inplace=False)  # to avoid inplace activation. inplace activation cause error GP
        act = self.get_act_layer(opt.C_act, inplace=True, negative_slope=opt.C_act_negative_slope)
        norm = self.get_norm_layer(opt.norm_type) if opt.C_norm else None
        pad = self.get_pad_layer(opt.pad_type_C)

        input_ch = opt.input_ch + opt.output_ch if opt.C_condition else opt.output_ch
        max_ch = opt.max_ch
        n_ch = opt.n_df
        n_down = opt.n_downsample_C
        n_RB_C = opt.n_RB_C
        pre_activation = opt.pre_activation

        down_blocks = []
        res_blocks = []
        if pre_activation:
            in_conv = [act, nn.Conv2d(input_ch, n_ch, kernel_size=3, padding=1, stride=1)]
            for _ in range(n_down):
                down_blocks += [[norm(min(n_ch, max_ch)), act, nn.Conv2d(min(n_ch, max_ch), min(2 * n_ch, max_ch),
                                                                    kernel_size=3, padding=1, stride=2)]]
                n_ch *= 2

            for _ in range(n_RB_C):
                res_blocks += [[ResidualBlock(min(n_ch, max_ch), act, norm, pad, pre_activation=True)]]

        else:
            in_conv = [nn.Conv2d(input_ch, n_ch, kernel_size=3, padding=1, stride=1), act]
            for _ in range(n_down):
                down_blocks += [[nn.Conv2d(min(n_ch, max_ch), min(2 * n_ch, max_ch), kernel_size=3, padding=1, stride=2),
                            norm(min(2 * n_ch, max_ch)), act]]
                n_ch *= 2

            for _ in range(n_RB_C):
                res_blocks += [[ResidualBlock(min(n_ch, max_ch), act, norm, pad, pre_activation=False)]]

        self.in_conv = nn.Sequential(*in_conv)
        self.out_conv = nn.Conv2d(min(n_ch, max_ch), kernel_size=3, padding=1, stride=1)

        self.n_down_blocks = len(down_blocks)
        self.n_res_blocks = len(res_blocks)

        for i in range(self.n_down_blocks):
            setattr(self, 'Down_block_{}'.format(i), nn.Sequential(*down_blocks[i]))

        for i in range(self.n_res_blocks):
            setattr(self, 'Residual_block_{}'.format(i), nn.Sequential(*down_blocks[i]))

        self.apply(partial(self.init_weights, type=opt.init_type, mode=opt.fan_mode, negative_slope=opt.negative_slope,
                           nonlinearity=opt.C_act))

        self.to_CUDA(opt.gpu_ids)

        print(self)
        print("the number of C parameters: ", sum(p.numel() for p in self.parameters() if p.requires_grad))

    def forward(self, x):
        results = [self.in_conv(x)]
        for i in range(self.n_down_blocks):
            results += [getattr(self, 'Down_block_{}'.format(i))(results[-1])]

        for i in range(self.n_res_blocks):
            results += [getattr(self, 'Residual_block{}'.format(i))(results[-1])]

        results += [self.out_conv(results[-1])]

        return results


class ProgressiveResidualPatchCritic(ResidualPatchCritic):
    def __init__(self, opt):
        super(ProgressiveResidualPatchCritic, self).__init__(opt)
        act = self.get_act_layer(opt.C_act, inplace=True, negative_slope=opt.C_act_negative_slope)

        input_ch = opt.input_ch + opt.output_ch if opt.C_condition else opt.output_ch
        n_ch = opt.n_df
        n_down = opt.n_downsample_C
        pre_activation = opt.pre_activation

        del self.in_conv
        n_in_conv = n_down + 1
        if pre_activation:
            for i in range(n_in_conv):
                in_conv = [act, nn.Conv2d(input_ch, n_ch, kernel_size=3, padding=1, stride=1)]
                setattr(self, 'in_conv_{}'.format(i), nn.Sequential(*in_conv))
                n_ch *= 2

        else:
            for i in range(n_in_conv):
                in_conv = [nn.Conv2d(input_ch, n_ch, kernel_size=3, padding=1, stride=1), act]
                setattr(self, 'in_conv_{}'.format(i), nn.Sequential(*in_conv))

        self.n_in_conv = n_in_conv

    def forward(self, tensor, level, level_in):
        results = []
        x = getattr(self, 'in_conv_{}'.format(self.n_in_conv - 1 - level))(tensor)
        for i in range(level): # 0 1 2 3 4
            tensor = nn.AvgPool2d(kernel_size=2, stride=2)(tensor)
            y = getattr(self, 'in_conv_{}'.format(self.n_in_conv - 1 - i))(tensor)
            x = getattr(self, 'Down_block_{}'.format(self.n_down_blocks - 1 - i))(x)
            x = torch.lerp(y, x, level_in - level)
            results += [x]

        for i in range(self.n_res_blocks):
            results += [getattr(self, 'Residual_block{}'.format(i))(results[-1])]

        results += [self.out_conv(results[-1])]

        return results


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
