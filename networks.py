from functools import partial
import torch
import torch.nn as nn
from utils import PixelNorm


class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    @staticmethod
    def add_norm_act_layer(norm=None, act=None, n_ch=None, double_par=False):
        layer = []
        if not double_par:
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

        else:
            if isinstance(norm, partial):
                layer += [[norm(n_ch)]]

            elif not norm:
                pass

            elif norm.__name__ == 'PixelNorm':
                layer += [[norm]]

            else:
                raise NotImplementedError

            if act:
                layer[-1].append(act)

        return layer

    @staticmethod
    def get_act_layer(type, inplace=True, negative_slope=None):
        if type == 'leaky_relu':
            assert negative_slope != 0.0
            layer = nn.LeakyReLU(negative_slope=negative_slope, inplace=inplace)

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
            self.to(torch.device('cuda', gpu_id))
        else:
            pass

    def forward(self, x):
        pass


class PatchCritic(BaseNetwork):
    def __init__(self, opt):
        super(BaseNetwork, self).__init__()
        # act = self.get_act_layer(opt.C_act, negative_slope=opt.C_act_negative_slope)
        self.act = nn.LeakyReLU(0.2, inplace=False)  # to avoid inplace activation. inplace activation cause error WGAN_GP
        fan_mode = opt.fan_mode
        gpu_id = opt.gpu_ids
        init_type = opt.init_type
        input_channel = opt.input_ch + opt.output_ch if opt.C_condition else opt.output_ch
        n_df = opt.n_df
        negative_slope = opt.C_act_negative_slope
        norm = self.get_norm_layer(opt.norm_type) if opt.C_norm else None

        blocks = [[nn.Conv2d(input_channel, n_df, kernel_size=4, padding=1, stride=2)]]
        blocks += [[nn.Conv2d(n_df, 2 * n_df, kernel_size=4, padding=1, stride=2),
                    *self.add_norm_act_layer(norm, n_ch=2 * n_df)]]
        blocks += [[nn.Conv2d(2 * n_df, 4 * n_df, kernel_size=4, padding=1, stride=2),
                    *self.add_norm_act_layer(norm, n_ch=4 * n_df)]]
        blocks += [[nn.Conv2d(4 * n_df, 8 * n_df, kernel_size=4, padding=1, stride=1),
                    *self.add_norm_act_layer(norm, n_ch=8 * n_df)]]
        blocks += [[nn.Conv2d(8 * n_df, 1, kernel_size=4, padding=1, stride=1)]]

        self.n_blocks = len(blocks)
        for i in range(self.n_blocks):
            setattr(self, 'block_{}'.format(i), nn.Sequential(*blocks[i]))

        self.apply(partial(self.init_weights, type=init_type, mode=fan_mode, negative_slope=negative_slope,
                           nonlinearity=opt.C_act))

        self.to_CUDA(gpu_id)

        print(self)
        print("the number of D parameters: ", sum(p.numel() for p in self.parameters() if p.requires_grad))

    def forward(self, x):
        result = [x]
        for i in range(self.n_blocks - 1):
            result.append(self.act(getattr(self, 'block_{}'.format(i))(result[-1])))
        result.append(getattr(self, 'block_{}'.format(self.n_blocks - 1))(result[-1]))

        return result[1:]  # except for the input


class Generator(BaseNetwork):
    def __init__(self, opt):
        super(Generator, self).__init__()
        act = self.get_act_layer(opt.G_act, inplace=True)
        fan_mode = opt.fan_mode
        gpu_id = opt.gpu_ids
        init_type = opt.init_type
        input_ch = opt.input_ch
        n_downsample = opt.n_downsample
        n_gf = opt.n_gf
        n_residual = opt.n_residual
        negative_slope = opt.G_act_negative_slope
        norm = self.get_norm_layer(opt.norm_type)
        output_ch = opt.output_ch
        pad = self.get_pad_layer(opt.pad_type)
        trans_unit = self.get_trans_unit(opt.trans_unit)

        self.encoder = Encoder(input_ch, n_gf, n_downsample, act=act, norm=norm, pad=pad)

        if trans_unit.__name__ == 'ResidualBlock':
            self.translator = ResidualNetwork(n_residual, n_gf, act, norm, pad, kernel_size=3)

        elif trans_unit.__name__ == 'ResidualDenseBlock':
            growth_rate = opt.growth_rate
            RDB_ch = opt.RDB_ch
            n_dense_layer = opt.n_dense_layer

            self.translator = ResidualDenseNetwork(n_gf * 2 ** n_downsample, n_residual, RDB_ch, growth_rate,
                                                   n_dense_layer, act, norm, pad, kernel_size=3)

        self.decoder = ProgressiveDecoder(n_gf * 2 ** n_downsample, output_ch, n_downsample, act=act, norm=norm,
                                          pad=pad, tanh=True)

        self.apply(partial(self.init_weights, type=init_type, mode=fan_mode, negative_slope=negative_slope,
                           nonlinearity=opt.G_act))

        self.to_CUDA(gpu_id)

        print(self)
        print("the number of G parameters: ", sum(p.numel() for p in self.parameters() if p.requires_grad))

    def get_trans_unit(self, trans_type):
        if trans_type == 'RB':
            unit = ResidualBlock
        elif trans_type == 'RDB':
            unit = ResidualDenseBlock
        else:
            raise NotImplementedError("Invalid transfer unit {}. Please check transfer_type option.".format(trans_type))

        return unit

    def forward(self, x, level):
        x = self.decoder(self.translator(self.encoder(x)), level)

        return x


class DenseLayer(BaseNetwork):
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


class Decoder(BaseNetwork):
    def __init__(self, input_ch, output_ch, n_upsample, act, kernel_size=3, pad=None, tanh=True):
        super(Decoder, self).__init__()
        ch = input_ch
        ps = kernel_size // 2  # padding size

        up_layers = []
        for i in range(n_upsample):
            up_layers += [nn.ConvTranspose2d(ch, ch//2, kernel_size=kernel_size, padding=1, stride=2, output_padding=1,
                                             bias=True), act]
            ch //= 2
        up_layers += [pad(ps), nn.Conv2d(ch, output_ch, kernel_size=kernel_size, padding=0, stride=1, bias=True)]
        up_layers[-1].append(nn.Tanh()) if tanh else None

        self.model = nn.Sequential(*up_layers)

    def forward(self, x):
        return self.model(x)


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


class ResidualBlock(BaseNetwork):
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


class ResidualNetwork(BaseNetwork):
    def __init__(self, n_block, n_ch, act, norm, pad, kernel_size=3):
        super(ResidualNetwork, self).__init__()

        for i in range(n_block):
            setattr(self, 'ResidualBlock_{}'.format(i), ResidualBlock(n_ch, act, norm, pad, kernel_size=kernel_size))

        self.n_block = n_block

    def forward(self, x):
        for i in range(self.n_block):
            x = getattr(self, 'ResidualBlock_{}'.format(i))(x)

        return x


class ResidualDenseBlock(BaseNetwork):
    def __init__(self, n_ch, growth_rate, n_dense_layer, act, norm, pad, kernel_size=3):
        super(ResidualDenseBlock, self).__init__()
        init_n_ch = n_ch

        block = []
        for _ in range(n_dense_layer):
            block += [DenseLayer(n_ch, growth_rate, kernel_size=kernel_size, act=act, norm=norm, pad=pad)]
            n_ch += growth_rate
        block += [nn.Conv2d(n_ch, init_n_ch, kernel_size=1, padding=0, stride=1, bias=False)]  # local feature fusion
        self.block = nn.Sequential(*block)

    def forward(self, x):
        y = self.block(x)
        x = x + y  # local residual learning

        return x


class ResidualDenseNetwork(BaseNetwork):
    def __init__(self, input_ch, n_block, n_ch, growth_rate, n_dense_layer, act, norm, pad, kernel_size=3):
        super(ResidualDenseNetwork, self).__init__()

        F0 = [nn.Conv2d(input_ch, n_ch, kernel_size=1, padding=0, stride=1, bias=True)]
        F0 += self.add_norm_act_layer(norm, n_ch=n_ch)

        for i in range(n_block):
            setattr(self, 'ResidualDenseBlock_{}'.format(i), ResidualDenseBlock(n_ch, growth_rate, n_dense_layer, act,
                                                                                norm, pad, kernel_size=kernel_size))

        GFF = [nn.Conv2d(n_block * n_ch, n_ch, kernel_size=1, padding=0, stride=1, bias=True)]
        GFF += self.add_norm_act_layer(norm, n_ch=n_ch)
        GFF += [pad(1), nn.Conv2d(n_ch, n_ch, kernel_size=3, padding=0, stride=1, bias=True)]
        GFF += self.add_norm_act_layer(norm, n_ch=n_ch)

        F_last = [nn.Conv2d(n_ch, input_ch, kernel_size=1, padding=0, stride=1, bias=True)]
        F_last += self.add_norm_act_layer(norm, n_ch=input_ch, act=act)

        self.F0 = nn.Sequential(*F0)
        self.GFF = nn.Sequential(*GFF)
        self.n_block = n_block
        self.F_last = nn.Sequential(*F_last)

    def forward(self, x):
        x = self.F0(x)
        results = [x]
        for i in range(self.n_block):
            results.append(getattr(self, 'ResidualDenseBlock_{}'.format(i))(results[-1]))
        cat = torch.cat(results[1:], dim=1)
        y = self.GFF(cat)
        x = x + y

        x = self.F_last(x)

        return x
