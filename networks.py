from functools import partial, partialmethod
import torch
import torch.nn as nn


class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    @staticmethod
    def get_act_layer(type, inplace=True, negative_slope=None):
        if type == 'leaky_relu':
            assert not negative_slope and negative_slope != 0.0
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
        act = self.get_act_layer(opt.C_act, negative_slope=opt.C_act_negative_slope)
        fan_mode = opt.fan_mode
        gpu_id = opt.gpu_ids
        init_type = opt.init_type
        input_channel = opt.input_ch + opt.ouput_ch if opt.C_condition else opt.output_ch
        n_df = opt.n_df
        negative_slope = opt.C_act_negative_slope
        norm = self.get_norm_layer(opt.norm_type) if opt.C_norm else None

        blocks = []
        if opt.C_norm:
            blocks += [[nn.Conv2d(input_channel, n_df, kernel_size=4, padding=1, stride=2), act]]
            blocks += [[nn.Conv2d(n_df, 2 * n_df, kernel_size=4, padding=1, stride=2), norm(2 * n_df), act]]
            blocks += [[nn.Conv2d(2 * n_df, 4 * n_df, kernel_size=4, padding=1, stride=2), norm(4 * n_df), act]]
            blocks += [[nn.Conv2d(4 * n_df, 8 * n_df, kernel_size=4, padding=1, stride=1), norm(8 * n_df), act]]
            blocks += [[nn.Conv2d(8 * n_df, 1, kernel_size=4, padding=1, stride=1)]]

        elif not opt.C_norm:
            blocks += [[nn.Conv2d(input_channel, n_df, kernel_size=4, padding=1, stride=2), act]]
            blocks += [[nn.Conv2d(n_df, 2 * n_df, kernel_size=4, padding=1, stride=2), act]]
            blocks += [[nn.Conv2d(2 * n_df, 4 * n_df, kernel_size=4, padding=1, stride=2), act]]
            blocks += [[nn.Conv2d(4 * n_df, 8 * n_df, kernel_size=4, padding=1, stride=1), act]]
            blocks += [[nn.Conv2d(8 * n_df, 1, kernel_size=4, padding=1, stride=1)]]

        self.n_blocks = len(blocks)
        for i in range(self.n_blocks):
            setattr(self, 'block_{}'.format(i), nn.Sequential(*blocks[i]))

        self.apply(partial(self.init_weights, type=init_type, mode=fan_mode, negative_slope=negative_slope,
                           nonlinearity=act))

        self.to_CUDA(gpu_id)

        print(self)
        print("the number of D parameters: ", sum(p.numel() for p in self.parameters() if p.requires_grad))

    def forward(self, x):
        result = [x]
        for i in range(self.n_blocks):
            block = getattr(self, 'block_{}'.format(i))
            result.append(block(result[-1]))

        return result[1:]  # except for the input


class Generator(BaseNetwork):
    def __init__(self, opt):
        super(Generator, self).__init__()
        act = self.get_act_layer(opt.G_act, inplace=True)
        fan_mode = opt.fan_mode
        gpu_id = opt.gpu_id
        init_type = opt.init_type
        input_ch = opt.input_ch
        n_downsample = opt.n_downsample
        n_gf = opt.n_gf
        n_residual = opt.n_residual
        negative_slope = opt.G_act_negative_slope
        norm = self.get_norm_layer(opt.norm_type)
        output_ch = opt.output_ch
        pad = self.get_pad_layer(opt.pad_type)
        trans_unit = self.get_trans_unit(opt.trans_type)
        self.opt = opt

        encoder = []
        encoder += [pad(3), nn.Conv2d(input_ch, n_gf, kernel_size=7, padding=0)]
        encoder += [norm(n_gf), act] if norm else [act]
        for _ in range(n_downsample):
            encoder += [pad(1), nn.Conv2d(n_gf, 2 * n_gf, kernel_size=3, padding=0, stride=2)]
            encoder += [norm(n_gf), act] if norm else [act]
            n_gf *= 2

        translator = []
        for _ in range(n_residual):
            translator += [trans_unit(n_gf, pad=pad, norm=norm, act=act)]

        decoder_layer = []
        for _ in range(n_downsample):
            if norm:
                decoder_layer += [[nn.ConvTranspose2d(n_gf, n_gf // 2, kernel_size=3, padding=1, stride=2,
                                                      output_padding=1), norm(n_gf), act]]
            else:
                decoder_layer += [[nn.ConvTranspose2d(n_gf, n_gf // 2, kernel_size=3, padding=1, stride=2,
                                                      output_padding=1), act]]
            n_gf //= 2

        decoder_layer += [[pad(3), nn.Conv2d(n_gf, output_ch, kernel_size=7, padding=0), nn.Tanh()]]

        self.encoder = nn.Sequential(*encoder)
        self.translator = nn.Sequential(*translator)

        for i, layer in enumerate(decoder_layer):
            setattr(self, 'decoder_level_{}'.format(i), nn.Sequential(*layer))

        self.apply(partial(self.init_weights, type=init_type, mode=fan_mode, negative_slope=negative_slope,
                           nonlinearity=act))

        self.to_CUDA(gpu_id)

        print(self)
        print("the number of G parameters: ", sum(p.numel() for p in self.parameters() if p.requires_grad))

    def get_trans_unit(self, trans_type):
        if trans_type == 'RB':
            unit = ResidualBlock
        elif trans_type == 'RDB':
            unit = partialmethod(ResidualDenseBlock.__init__, growth_rate=self.opt.growth_rate,
                                 n_dense_layer=self.opt.n_dense_layer)
        else:
            raise NotImplementedError("Invalid transfer unit {}. Please check transfer_type option.".format(trans_type))

        return unit

    def forward(self, x, lod):
        x = self.translator(self.encoder(x))

        for i in range(lod):
            x = getattr(self, 'decoder_level_{}'.format(i))(x)

        return x


class DenseLayer(BaseNetwork):
    def __init__(self, n_ch, growth_rate, kernel_size=3, act='relu', norm='InstanceNorm2d', pad='reflection'):
        super(DenseLayer, self).__init__()
        act = self.get_act_layer(act, inplace=True) if isinstance(act, str) else act
        norm = self.get_norm_layer(norm) if isinstance(norm, str) else norm
        pad = self.get_pad_layer(pad) if isinstance(pad, str) else pad

        layer = []
        layer += [pad(1), nn.Conv2d(n_ch, growth_rate, kernel_size=kernel_size, padding=0, stride=1, bias=False)]
        layer += [norm(growth_rate), act] if norm else [act]

        self.layer = nn.Sequential(*layer)

    def forward(self, x):
        y = self.layer(x)
        x = torch.cat([x, y], dim=1)

        return x


class ResidualBlock(nn.Module):
    def __init__(self, n_ch, act, norm, pad, kernel_size=3):
        super(ResidualBlock, self).__init__()
        block = []
        block += [pad(1), nn.Conv2d(n_ch, n_ch, kernel_size=kernel_size, padding=0, stride=1)]
        block += [norm(n_ch), act] if norm else [act]
        block += [pad(1), nn.Conv2d(n_ch, n_ch, kernel_size=kernel_size, padding=0, stride=1)]
        block += [norm(n_ch)] if norm else []

        self.block = nn.Sequential(*block)

    def forward(self, x):
        x = x + self.block(x)

        return x


class ResidualDenseBlock(BaseNetwork):
    def __init__(self, n_ch, growth_rate, n_dense_layer, act, norm, pad, kernel_size=3):
        super(ResidualDenseBlock, self).__init__()
        init_n_ch = n_ch

        block = []
        for i in range(n_dense_layer):
            block.append(DenseLayer(n_ch, growth_rate, kernel_size=kernel_size, act=act, norm=norm, pad=pad))
            n_ch += growth_rate
        block.append(nn.Conv2d(n_ch, init_n_ch, kernel_size=1, padding=0, stride=1, bias=False))
        self.block = nn.Sequential(*block)

    def forward(self, x):
        y = self.block(x)
        x = x + y

        return x
