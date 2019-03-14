import torch
import torch.nn as nn
import torchvision


class Critic(nn.Module):
    def __init__(self, opt):
        super(Critic, self).__init__()
        self.n_C = opt.n_C
        for i in range(opt.n_C):
            setattr(self, 'Scale_{}'.format(i), PatchCritic(opt))

    def forward(self, x):
        result = []
        for i in range(self.n_C):
            result.append(getattr(self, 'Scale_{}'.format(i))(x))
            x = nn.AvgPool2d(kernel_size=3, padding=1, stride=2, count_include_pad=False)(x)
        return result


class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        act = nn.ReLU(inplace=True)
        norm = nn.InstanceNorm2d
        pad = nn.ReflectionPad2d

        input_ch = opt.input_ch
        max_ch = opt.max_ch_G
        n_ch = opt.n_gf
        n_down = opt.n_downsample
        output_ch = opt.output_ch
        trans_module = opt.trans_module

        down_blocks = []
        up_blocks = []
        down_blocks += [pad(3), nn.Conv2d(input_ch, n_ch, kernel_size=7), norm(n_ch), act]
        for _ in range(n_down - 1):
            down_blocks += [nn.Conv2d(min(n_ch, max_ch), min(2 * n_ch, max_ch), kernel_size=3, padding=1, stride=2),
                            norm(min(2 * n_ch, max_ch)), act]
            n_ch *= 2

        if trans_module == 'DB':
            down_blocks += [nn.Conv2d(min(n_ch, max_ch), min(2 * n_ch, max_ch), kernel_size=3, padding=1, stride=2),
                            norm(min(2 * n_ch, max_ch))]
        elif trans_module == 'RIR':
            down_blocks += [nn.Conv2d(min(n_ch, max_ch), min(2 * n_ch, max_ch), kernel_size=3, padding=1, stride=2),
                            norm(min(2 * n_ch, max_ch))]
            # down_blocks += [nn.Conv2d(min(n_ch, max_ch), opt.rir_ch, kernel_size=1), norm(opt.rir_ch)]
        else:
            down_blocks += [nn.Conv2d(min(n_ch, max_ch), min(2 * n_ch, max_ch), kernel_size=3, padding=1, stride=2),
                            norm(min(2 * n_ch, max_ch)), act]

        for _ in range(n_down):
            up_blocks += [nn.ConvTranspose2d(min(2 * n_ch, max_ch), min(n_ch, max_ch), kernel_size=3, padding=1,
                                             stride=2, output_padding=1), norm(min(n_ch, max_ch)), act]
            n_ch //= 2

        up_blocks += [pad(3), nn.Conv2d(2 * n_ch, output_ch, kernel_size=7), nn.Tanh()]

        self.down_blocks = nn.Sequential(*down_blocks)
        self.translator = self.get_trans_network(opt, 1024)
        self.up_blocks = nn.Sequential(*up_blocks)

    def forward(self, x):
        result = self.translator(self.down_blocks(x))
        return self.up_blocks(result)

    @staticmethod
    def get_trans_network(opt, n_ch):
        trans_module = opt.trans_module
        if trans_module == 'DB':
            from networks.dense_modules import DenseNetwork
            network = DenseNetwork(opt.n_blocks, n_ch, opt.growth_rate, opt.n_dense_layers, efficient=opt.efficient,
                                   block_weight=opt.block_weight)
        elif trans_module == 'RB':
            from networks.residual_modules import ResidualNetwork
            network = ResidualNetwork(opt.n_blocks, n_ch, block_weight=opt.block_weight)
        elif trans_module == 'RDB':
            from networks.residual_dense_modules import ResidualDenseNetwork
            network = ResidualDenseNetwork(opt.n_blocks, n_ch, opt.growth_rate, opt.n_dense_layers)
        elif trans_module == 'RIR':
            from networks.residual_modules import ResidualInResidualNetwork
            network = ResidualInResidualNetwork(opt.n_groups, opt.n_blocks, n_ch, opt.rir_ch,
                                                block_weight=opt.block_weight)
        else:
            raise NotImplementedError
        return network


class ProgressiveGenerator(Generator):
    def __init__(self, opt):
        super(ProgressiveGenerator, self).__init__(opt)
        act = nn.ReLU(inplace=True)
        norm = nn.InstanceNorm2d
        pad = nn.ReflectionPad2d

        max_ch = opt.max_ch_G
        n_down = opt.n_downsample
        output_ch = opt.output_ch

        n_ch = opt.n_gf * 2 ** n_down
        rgb_blocks = [[norm(n_ch), act, pad(1), nn.Conv2d(min(n_ch, max_ch), output_ch, kernel_size=3)]]
        rgb_blocks[-1].append(nn.Tanh()) if opt.tanh else None
        delattr(self, 'up_blocks')
        for i in range(n_down):
            up_block = [nn.ConvTranspose2d(min(n_ch, max_ch), min(n_ch // 2, max_ch), kernel_size=3, padding=1,
                                           stride=2, output_padding=1), norm(min(n_ch // 2, max_ch)), act,
                        pad(1), nn.Conv2d(min(n_ch // 2, max_ch), min(n_ch // 2, max_ch), kernel_size=3),
                        norm(min(n_ch // 2, max_ch)), act]
            rgb_blocks += [[pad(1), nn.Conv2d(min(n_ch // 2, max_ch), output_ch, kernel_size=3)]]
            rgb_blocks[-1].append(nn.Tanh()) if opt.tanh else None
            setattr(self, 'Up_block_{}'.format(i), nn.Sequential(*up_block))
            n_ch //= 2

        for i in range(len(rgb_blocks)):
            setattr(self, 'RGB_block_{}'.format(i), nn.Sequential(*rgb_blocks[i]))

    def forward(self, x, level, level_in):
        x = self.translator(self.down_blocks(x))
        out = getattr(self, 'RGB_block_0')(x)
        for i in range(level):
            x = getattr(self, 'Up_block_{}'.format(i))(x)
            y = getattr(self, 'RGB_block_{}'.format(i + 1))(x)
            out = nn.functional.interpolate(out, scale_factor=2, mode='nearest')
            out = torch.lerp(out, y, level_in - level)
        return out


class PatchCritic(nn.Module):
    def __init__(self, opt):
        super(PatchCritic, self).__init__()
        # act = nn.LeakyReLU(0.2, inplace=True)
        C_norm = opt.C_norm
        input_channel = opt.input_ch + opt.output_ch if opt.C_condition else opt.output_ch
        n_df = opt.n_df
        norm = nn.InstanceNorm2d
        patch_size = opt.patch_size

        blocks = [[nn.Conv2d(input_channel, n_df, kernel_size=4, padding=1, stride=2)]]
        blocks += [[nn.Conv2d(n_df, 2 * n_df, kernel_size=4, padding=1, stride=2)]]
        blocks[-1].append(norm(2 * n_df)) if opt.C_norm else None
        if patch_size == 16:
            blocks += [[nn.Conv2d(2 * n_df, 4 * n_df, kernel_size=4, padding=1)]]
            blocks[-1].append(norm(4 * n_df)) if opt.C_norm else None
            blocks += [[nn.Conv2d(4 * n_df, 1, kernel_size=4, padding=1)]]

        elif patch_size == 70:
            blocks += [[nn.Conv2d(2 * n_df, 4 * n_df, kernel_size=4, padding=1, stride=2)]]
            blocks[-1].append(norm(4 * n_df)) if opt.C_norm else None
            blocks += [[nn.Conv2d(4 * n_df, 8 * n_df, kernel_size=4, padding=1)]]
            blocks[-1].append(norm(8 * n_df)) if opt.C_norm else None
            blocks += [[nn.Conv2d(8 * n_df, 1, kernel_size=4, padding=1)]]

        self.CT = opt.CT
        self.dropout = nn.Dropout2d(p=0.4, inplace=False) if opt.CT else None
        self.GAN_type = opt.GAN_type
        self.n_blocks = len(blocks)
        for i in range(self.n_blocks):
            setattr(self, 'block_{}'.format(i), nn.Sequential(*blocks[i]))

        self.act = nn.LeakyReLU(0.2, inplace=False)  # to avoid inplace activation. inplace activation cause error GP

    def forward(self, x):
        result = [x]
        for i in range(self.n_blocks):
            if i == self.n_blocks - 1:
                result += [getattr(self, 'block_{}'.format(i))(result[-1])]
            else:
                if self.CT and self.GAN_type == 'WGAN':
                    result += [self.dropout(self.act(getattr(self, 'block_{}'.format(i))(result[-1])))]
                else:
                    result += [self.act(getattr(self, 'block_{}'.format(i))(result[-1]))]
        return result[1:]  # except for the input


class ProgressivePatchCritic(nn.Module):
    def __init__(self, opt):
        super(ProgressivePatchCritic, self).__init__()
        act = nn.LeakyReLU(0.2, inplace=True)
        norm = nn.InstanceNorm2d

        input_ch = opt.input_ch + opt.output_ch if opt.C_condition else opt.output_ch
        max_ch = opt.max_ch_C
        n_ch = opt.n_df
        n_down = opt.n_downsample
        n_RB_C = opt.n_RB_C

        n_in_conv = n_down + 1
        for i in range(n_in_conv):  # 0, 1, 2, 3, 4, 5
            in_conv = [nn.Conv2d(input_ch, min(n_ch, max_ch), kernel_size=3, padding=1, stride=1), act]
            setattr(self, 'In_conv_{}'.format(i), nn.Sequential(*in_conv))
            n_ch *= 2

        n_ch = opt.n_df
        for i in range(n_down):  # 0, 1, 2, 3, 4
            down_block = [nn.Conv2d(n_ch, min(2 * n_ch, max_ch), kernel_size=3, padding=1, stride=2),
                          norm(min(2 * n_ch, max_ch)), act]
            setattr(self, 'Down_block_{}'.format(i), nn.Sequential(*down_block))
            n_ch *= 2

        self.add_module('Out_conv', nn.Conv2d(min(n_ch, max_ch), 1, kernel_size=3, padding=1))

        self.n_df = opt.n_df
        self.n_down = n_down
        self.n_in_conv = n_in_conv
        self.n_res_blocks = n_RB_C
        self.norm = norm

    def forward(self, tensor, level, level_in):
        x = getattr(self, 'In_conv_{}'.format(self.n_in_conv - 1 - level))(tensor)
        results = [x]
        for i in range(level, 0, -1):  # 4 3 2 1
            x = getattr(self, 'Down_block_{}'.format(self.n_down - i))(x)
            results += [x]

        results += [getattr(self, 'Out_conv')(results[-1])]

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
