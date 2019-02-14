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
    def get_trans_module(opt, act, norm, pad, kernel_size=3):
        trans_module = opt.trans_module
        pre_activation = opt.pre_activation
        if trans_module == 'RB':
            from networks.modules.residual_block import ResidualBlock
            module = partial(ResidualBlock, act=act, norm=norm, pad=pad, kernel_size=kernel_size,
                             pre_activation=pre_activation)

        elif trans_module == 'RDB':
            from networks.modules.residual_dense_block import ResidualDenseBlock
            module = partial(ResidualDenseBlock, growth_rate=opt.growth_rate, n_dense_layers=opt.n_dense_layers,
                             act=act, norm=norm, pad=pad, kernel_size=kernel_size, pre_activation=pre_activation,
                             efficient=opt.efficient)

        elif opt.trans_module == 'RCAN':
            from networks.module.residual_channel_attention_block import ResidualChannelAttentionBlock
            module = partial(ResidualChannelAttentionBlock, reduction_rate=opt.reduction_rate, act=act, norm=norm,
                             pad=pad, kernel_size=kernel_size, pre_activation=pre_activation)

        elif opt.trans_module == 'RCADB':
            from networks.module.residual_dense_block import ResidualChannelAttentionDenseBlock
            module = partial(ResidualChannelAttentionDenseBlock, growth_rate=opt.growth_rate,
                             n_dense_layer=opt.n_dense_layers, act=act, norm=norm, pad=pad, kernel_size=kernel_size,
                             pre_activation=pre_activation)

        return module

    @staticmethod
    def get_trans_network(opt, act, norm, pad, kernel_size=3, pre_activation=False):
        if opt.trans_network == 'RCAN':
            from networks.translators import ResidualChannelAttentionNetwork
            net = ResidualChannelAttentionNetwork(opt.n_RG, opt.n_RCAB, opt.RCA_ch,
                                                  opt.reduction_rate, kernel_size=kernel_size, act=act, norm=norm,
                                                  pad=pad, pre_activation=pre_activation)

        elif opt.trans_network == 'RN':
            from networks.translators import ResidualNetwork
            net = ResidualNetwork(opt.n_RB, opt.n_gf * 2 ** opt.n_downsample, kernel_size=kernel_size, act=act,
                                  norm=norm, pad=pad, pre_activation=pre_activation)

        elif opt.trans_network == 'RDN':
            from networks.translators import ResidualDenseNetwork
            growth_rate = opt.growth_rate
            n_dense_layers = opt.n_dense_layers
            n_downsample = opt.n_downsample
            n_gf = opt.n_gf
            n_RDB = opt.n_RDB
            net = ResidualDenseNetwork(n_RDB, n_gf * 2 ** n_downsample, growth_rate,
                                       n_dense_layers, kernel_size=kernel_size, act=act, norm=norm, pad=pad,
                                       pre_activation=pre_activation)

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
        max_ch = opt.max_ch_G
        n_ch = opt.n_gf
        n_down = opt.n_downsample
        n_RB = opt.n_RB
        output_ch = opt.output_ch
        pixel_shuffle = opt.pixel_shuffle
        pre_activation = opt.pre_activation
        trans_module = self.get_trans_module(opt, act, norm, pad)

        down_blocks = []
        trans_blocks = []
        up_blocks = []
        if pre_activation:
            down_blocks += [pad(3), nn.Conv2d(input_ch, n_ch, kernel_size=7)]
            for i in range(n_down):
                down_blocks += [norm(min(n_ch, max_ch)), act, pad(1),
                                nn.Conv2d(min(n_ch, max_ch), min(2 * n_ch, max_ch), kernel_size=3, stride=2)]
                n_ch *= 2
            down_blocks += [norm(min(n_ch, max_ch))]

            trans_blocks += [trans_module(n_ch=min(n_ch, max_ch)) for _ in range(n_RB)]

            for _ in range(n_down):
                up_blocks += [nn.ConvTranspose2d(min(n_ch, max_ch), min(n_ch//2, max_ch), kernel_size=3, padding=1,
                                                 stride=2, output_padding=1), norm(min(n_ch//2, max_ch)), act]
                n_ch //= 2

            if pixel_shuffle:
                up_blocks = [pad(1), nn.Conv2d(n_ch, n_down * n_ch, kernel_size=3), nn.PixelShuffle(2 ** n_down)]
                n_ch = opt.n_gf

            else:
                up_blocks += [pad(3), nn.Conv2d(n_ch, output_ch, kernel_size=7)]

        else:
            down_blocks += [pad(3), nn.Conv2d(input_ch, n_ch, kernel_size=7), norm(n_ch), act]
            for _ in range(n_down):
                down_blocks += [pad(1), nn.Conv2d(min(n_ch, max_ch), min(2 * n_ch, max_ch), kernel_size=3, stride=2),
                                norm(min(2 * n_ch, max_ch)), act]
                n_ch *= 2

            trans_blocks += [trans_module(n_ch=min(n_ch, max_ch)) for _ in range(n_RB)]

            if pixel_shuffle:
                up_blocks = [pad(1), nn.Conv2d(n_ch, n_down * n_ch, kernel_size=3), nn.PixelShuffle(2 ** n_down)]
                n_ch = opt.n_gf

            else:
                for _ in range(n_down):
                    up_blocks += [nn.ConvTranspose2d(min(n_ch, max_ch), min(n_ch // 2, max_ch), kernel_size=3, padding=1,
                                                     stride=2, output_padding=1), norm(min(n_ch // 2, max_ch)), act]
                    n_ch //= 2

            up_blocks += [pad(3), nn.Conv2d(n_ch, output_ch, kernel_size=7)]
        up_blocks += [nn.Tanh()] if opt.tanh else []

        self.down_blocks = nn.Sequential(*down_blocks)
        self.translator = nn.Sequential(*trans_blocks)
        self.up_blocks = nn.Sequential(*up_blocks)

        self.apply(partial(self.init_weights, type=opt.init_type, mode=opt.fan_mode, negative_slope=opt.negative_slope,
                           nonlinearity=opt.G_act))

        self.to_CUDA(opt.gpu_ids)

        print(self)
        print("the number of G parameters: ", sum(p.numel() for p in self.parameters() if p.requires_grad))

    def forward(self, x):
        return self.up_blocks(self.translator(self.down_blocks(x)))


class ProgressiveGenerator(Generator):
    def __init__(self, opt):
        super(ProgressiveGenerator, self).__init__(opt)
        act = self.get_act_layer(opt.G_act, inplace=True)
        norm = self.get_norm_layer(opt.norm_type)
        pad = self.get_pad_layer(opt.pad_type)

        max_ch = opt.max_ch_G
        n_down = opt.n_downsample
        output_ch = opt.output_ch
        pre_activation = opt.pre_activation

        n_ch = opt.n_gf * 2 ** n_down
        rgb_blocks = [[norm(min(n_ch, max_ch)), act, pad(1),
                       nn.Conv2d(min(n_ch, max_ch), output_ch, kernel_size=3)]]
        rgb_blocks[-1].append(nn.Tanh()) if opt.tanh else None
        delattr(self, 'up_blocks')
        if pre_activation:
            for i in range(n_down):
                up_block = self.add_norm_act_layer(norm, n_ch=min(n_ch, max_ch), act=act)
                up_block += [nn.ConvTranspose2d(min(n_ch, max_ch), min(n_ch // 2, max_ch), kernel_size=3, padding=1,
                                                stride=2, output_padding=1)]
                rgb_blocks += [[norm(min(n_ch // 2, max_ch)), act, pad(1),
                                nn.Conv2d(min(n_ch // 2, max_ch), output_ch, kernel_size=3)]]
                rgb_blocks[-1].append(nn.Tanh()) if opt.tanh else None
                setattr(self, 'Up_block_{}'.format(i), nn.Sequential(*up_block))
                n_ch //= 2
        else:
            for i in range(n_down):
                up_block = [nn.ConvTranspose2d(min(n_ch, max_ch), min(n_ch // 2, max_ch), kernel_size=3, padding=1,
                                               stride=2, output_padding=1)]
                up_block += self.add_norm_act_layer(norm, n_ch=min(n_ch // 2, max_ch), act=act)

                rgb_blocks += [[nn.Conv2d(min(n_ch // 2, max_ch), output_ch, kernel_size=3)]]
                rgb_blocks[-1].append(nn.Tanh()) if opt.tanh else None
                setattr(self, 'Up_block_{}'.format(i), nn.Sequential(*up_block))
                n_ch //= 2

        for i in range(len(rgb_blocks)):
            setattr(self, 'RGB_block_{}'.format(i), nn.Sequential(*rgb_blocks[i]))

        self.apply(partial(self.init_weights, type=opt.init_type, mode=opt.fan_mode, negative_slope=opt.negative_slope,
                           nonlinearity=opt.G_act))

        self.to_CUDA(opt.gpu_ids)

        print(self)
        print("the number of G parameters: ", sum(p.numel() for p in self.parameters() if p.requires_grad))

    def forward(self, x, level, level_in):
        x = self.translator(self.down_blocks(x))
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
        # self.act = nn.LeakyReLU(0.2, inplace=False)  # to avoid inplace activation. inplace activation cause error GP
        act = nn.LeakyReLU(0.2, inplace=True)
        input_channel = opt.input_ch + opt.output_ch if opt.C_condition else opt.output_ch
        n_df = opt.n_df
        norm = self.get_norm_layer(opt.norm_type) if opt.C_norm else None
        patch_size = opt.patch_size
        pre_activaton = opt.pre_activation

        if pre_activaton:
            blocks = [[nn.Conv2d(input_channel, n_df, kernel_size=4, padding=1, stride=2)]]
            blocks += [[act, nn.Conv2d(n_df, 2 * n_df, kernel_size=4, padding=1, stride=2)]]

            if patch_size == 16:
                blocks += [[*self.add_norm_act_layer(norm, n_ch=2 * n_df, act=act),
                            nn.Conv2d(2 * n_df, 4 * n_df, kernel_size=4, padding=1)]]
                blocks += [[nn.Conv2d(4 * n_df, 1, kernel_size=4, padding=1)]]

            elif patch_size == 70:
                blocks += [[*self.add_norm_act_layer(norm, n_ch=2 * n_df, act=act),
                            nn.Conv2d(2 * n_df, 4 * n_df, kernel_size=4, padding=1, stride=2)]]

                blocks += [[*self.add_norm_act_layer(norm, n_ch=4 * n_df, act=act),
                            nn.Conv2d(4 * n_df, 8 * n_df, kernel_size=4, padding=1)]]
                blocks += [[*self.add_norm_act_layer(norm, n_ch=8 * n_df, act=act),
                            nn.Conv2d(8 * n_df, 1, kernel_size=4, padding=1)]]

        else:
            blocks = [[nn.Conv2d(input_channel, n_df, kernel_size=4, padding=1, stride=2), act]]
            blocks += [[nn.Conv2d(n_df, 2 * n_df, kernel_size=4, padding=1, stride=2),
                        *self.add_norm_act_layer(norm, n_ch=2 * n_df, act=act)]]
            if patch_size == 16:
                blocks += [[nn.Conv2d(2 * n_df, 4 * n_df, kernel_size=4, padding=1),
                            *self.add_norm_act_layer(norm, n_ch=4 * n_df, act=act)]]
                blocks += [[nn.Conv2d(4 * n_df, 1, kernel_size=4, padding=1)]]

            elif patch_size == 70:
                blocks += [[nn.Conv2d(2 * n_df, 4 * n_df, kernel_size=4, padding=1, stride=2),
                            *self.add_norm_act_layer(norm, n_ch=4 * n_df, act=act)]]
                blocks += [[nn.Conv2d(4 * n_df, 8 * n_df, kernel_size=4, padding=1),
                            *self.add_norm_act_layer(norm, n_ch=8 * n_df, act=act)]]
                blocks += [[nn.Conv2d(8 * n_df, 1, kernel_size=4, padding=1)]]

        self.n_blocks = len(blocks)
        for i in range(self.n_blocks):
            setattr(self, 'block_{}'.format(i), nn.Sequential(*blocks[i]))

    def forward(self, x):
        result = [x]
        for i in range(self.n_blocks):
            result += [getattr(self, 'block_{}'.format(i))(result[-1])]

        return result[1:]  # except for the input


class ResidualPatchCritic(BaseNetwork):
    def __init__(self, opt):
        super(ResidualPatchCritic, self).__init__()
        # self.act = nn.LeakyReLU(0.2, inplace=False)  # to avoid inplace activation. inplace activation cause error GP
        act = self.get_act_layer(opt.C_act, inplace=True, negative_slope=opt.C_act_negative_slope)
        norm = self.get_norm_layer(opt.norm_type) if opt.C_norm else None
        pad = self.get_pad_layer(opt.pad_type_C)

        input_ch = opt.input_ch + opt.output_ch if opt.C_condition else opt.output_ch
        max_ch = opt.max_ch_C
        n_ch = opt.n_df
        n_down = opt.n_downsample if opt.progression else opt.n_downsample_C
        n_RB_C = opt.n_RB_C
        pre_activation = opt.pre_activation

        down_blocks = []
        res_blocks = []
        if pre_activation:
            in_conv = [[nn.Conv2d(input_ch, n_ch, kernel_size=3, padding=1, stride=2)]]
            in_conv += [[act, nn.Conv2d(n_ch, 2 * n_ch, kernel_size=3, padding=1, stride=2)]]
            n_ch *= 2
            for _ in range(n_down - len(in_conv)):
                down_blocks += [[norm(min(n_ch, max_ch)), act, nn.Conv2d(min(n_ch, max_ch), min(2 * n_ch, max_ch),
                                                                         kernel_size=3, padding=1, stride=2)]]
                n_ch *= 2

            for _ in range(n_RB_C):
                res_blocks += [[ResidualBlock(min(n_ch, max_ch), act, norm, pad, pre_activation=True)]]

            out_conv = [norm(min(n_ch, max_ch)), act, nn.Conv2d(min(n_ch, max_ch), 1, kernel_size=3, padding=1)]
            self.out_conv = nn.Sequential(*out_conv)

        else:
            in_conv = [[nn.Conv2d(input_ch, n_ch, kernel_size=3, padding=1, stride=2), act]]
            for _ in range(n_down - len(in_conv)):
                down_blocks += [[nn.Conv2d(min(n_ch, max_ch), min(2 * n_ch, max_ch), kernel_size=3, padding=1, stride=2),
                                 norm(min(2 * n_ch, max_ch)), act]]
                n_ch *= 2

            for _ in range(n_RB_C):
                res_blocks += [[ResidualBlock(min(n_ch, max_ch), act, norm, pad, pre_activation=False)]]
            self.out_conv = nn.Conv2d(min(n_ch, max_ch), 1, kernel_size=3, padding=1)

        self.n_in_conv = len(in_conv)
        self.n_down_blocks = len(down_blocks)
        self.n_res_blocks = len(res_blocks)

        for i in range(self.n_in_conv):
            setattr(self, 'In_conv_{}'.format(i), nn.Sequential(*in_conv[i]))

        for i in range(self.n_down_blocks):
            setattr(self, 'Down_block_{}'.format(i), nn.Sequential(*down_blocks[i]))

        for i in range(self.n_res_blocks):
            setattr(self, 'Residual_block_{}'.format(i), nn.Sequential(*res_blocks[i]))

        self.apply(partial(self.init_weights, type=opt.init_type, mode=opt.fan_mode, negative_slope=opt.negative_slope,
                           nonlinearity=opt.C_act))

        self.to_CUDA(opt.gpu_ids)

        print(self)
        print("the number of C parameters: ", sum(p.numel() for p in self.parameters() if p.requires_grad))

    def forward(self, x):
        results = [x]
        for i in range(self.n_in_conv):
            results += [getattr(self, 'In_conv_{}'.format(i))(results[-1])]
        for i in range(self.n_down_blocks):
            results += [getattr(self, 'Down_block_{}'.format(i))(results[-1])]
        for i in range(self.n_res_blocks):
            results += [getattr(self, 'Residual_block_{}'.format(i))(results[-1])]

        results += [self.out_conv(results[-1])]

        return [results[1:]]


class ProgressiveResidualPatchCritic(BaseNetwork):
    def __init__(self, opt):
        super(ProgressiveResidualPatchCritic, self).__init__()
        act = self.get_act_layer(opt.C_act, inplace=True, negative_slope=opt.C_act_negative_slope)
        norm = self.get_norm_layer(opt.norm_type)
        pad = self.get_pad_layer(opt.pad_type_C)

        input_ch = opt.input_ch + opt.output_ch if opt.C_condition else opt.output_ch
        max_ch = opt.max_ch_C
        n_ch = opt.n_df
        n_down = opt.n_downsample
        n_RB_C = opt.n_RB_C
        pre_activation = opt.pre_activation

        n_in_conv = n_down + 1
        if pre_activation:
            for i in range(n_in_conv):  # 0, 1, 2, 3, 4
                in_conv = [nn.Conv2d(input_ch, n_ch, kernel_size=3, padding=1, stride=1)]
                setattr(self, 'In_conv_{}'.format(i), nn.Sequential(*in_conv))
                n_ch *= 2

            n_ch = opt.n_df
            for i in range(n_down):
                down_block = [act, nn.Conv2d(n_ch, 2 * n_ch, kernel_size=3, padding=1, stride=2)]
                setattr(self, 'Down_block_{}'.format(i), nn.Sequential(*down_block))
                n_ch *= 2

            for i in range(n_RB_C):
                setattr(self, 'Residual_block_{}'.format(i), ResidualBlock(n_ch, act, norm, pad, pre_activation=True))

            out_conv = [norm(min(n_ch, max_ch)), act, nn.Conv2d(min(n_ch, max_ch), 1, kernel_size=3, padding=1)]
            self.out_conv = nn.Sequential(*out_conv)

        else:
            for i in range(n_in_conv):
                in_conv = [nn.Conv2d(input_ch, n_ch, kernel_size=3, padding=1), act]
                setattr(self, 'In_conv_{}'.format(i), nn.Sequential(*in_conv))
                n_ch *= 2

            n_ch = opt.n_df
            for i in range(n_down):
                down_block = [nn.Conv2d(n_ch, 2 * n_ch, kernel_size=3, stride=2, padding=1), norm(2 * n_ch), act]
                setattr(self, 'Down_block_{}'.format(i), nn.Sequential(*down_block))
                n_ch *= 2

            for i in range(n_RB_C):
                setattr(self, 'Residual_block_{}'.format(i), ResidualBlock(n_ch, act, norm, pad, pre_activation=False))
            self.out_conv = nn.Conv2d(n_ch, 1, kernel_size=3, padding=1)

        self.n_df = opt.n_df
        self.n_down = n_down
        self.n_in_conv = n_in_conv
        self.n_res_blocks = n_RB_C
        self.norm = norm

        self.apply(partial(self.init_weights, type=opt.init_type, mode=opt.fan_mode, negative_slope=opt.negative_slope,
                           nonlinearity=opt.C_act))

        self.to_CUDA(opt.gpu_ids)

        print(self)
        print("the number of C parameters: ", sum(p.numel() for p in self.parameters() if p.requires_grad))

    def forward(self, tensor, level, level_in):
        results = []
        x = getattr(self, 'In_conv_{}'.format(self.n_in_conv - 1 - level))(tensor)
        for i in range(level, 0, -1):  # 4 3 2 1
            tensor = nn.AvgPool2d(kernel_size=2, stride=2)(tensor)
            y = getattr(self, 'In_conv_{}'.format(self.n_in_conv - i))(tensor)
            x = getattr(self, 'Down_block_{}'.format(self.n_down - i))(x)
            x = torch.lerp(y, x, level_in - level)
            results += [x]
            x = self.norm(self.n_df * 2 ** (self.n_down + 1 - i))(x) if i != 1 else x

        for i in range(self.n_res_blocks):
            results += [getattr(self, 'Residual_block_{}'.format(i))(x)]

        results += [self.out_conv(results[-1])]

        return [results]


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
