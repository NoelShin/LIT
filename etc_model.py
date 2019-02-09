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


class ProgressiveGatedUNetGenerator(BaseGenerator):
    def __init__(self, opt):
        super(ProgressiveGatedUNetGenerator, self).__init__()
        act = self.get_act_layer(opt.G_act, inplace=True)
        norm = self.get_norm_layer(opt.norm_type)
        pad = self.get_pad_layer(opt.pad_type)

        n_dense_layers = opt.n_dense_layers

        n_ch = opt.n_gf
        n_downsample = opt.n_downsample
        n_enhance = opt.n_enhance_blocks
        layer_first = [pad(3), nn.Conv2d(opt.input_ch, n_ch, kernel_size=7, padding=0, stride=1)]
        layer_first += self.add_norm_act_layer(norm, n_ch=n_ch, act=act)
        self.layer_first = nn.Sequential(*layer_first)

        RGB_layer = []
        gate_layer = [[ChannelAttentionLayer(n_ch)]]
        fusion_layer = [[nn.Conv2d(2 * n_ch, n_ch, kernel_size=1, padding=0, stride=1, bias=True)]]
        for i in range(n_downsample):
            encoder_layer = [pad(1), nn.Conv2d(n_ch, 2 * n_ch, kernel_size=3, padding=0, stride=2)]
            encoder_layer += self.add_norm_act_layer(norm, n_ch=2 * n_ch, act=act)
            gate_layer += [[ChannelAttentionLayer(2 * n_ch)]] if i != n_downsample - 1 else []
            fusion_layer += [[nn.Conv2d(2 * n_ch, n_ch, kernel_size=1, padding=0, stride=1, bias=True)]]

            decoder_layer = [nn.ConvTranspose2d(2 * n_ch, n_ch, kernel_size=3, padding=1, stride=2, output_padding=1)]
            decoder_layer += self.add_norm_act_layer(norm, n_ch=n_ch, act=act)

            enhance_layer = [ResidualDenseBlock(n_ch, n_ch//n_dense_layers, n_dense_layers, act, norm, pad)
                             for _ in range(n_enhance)]

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
            setattr(self, 'Fusion_layer_{}'.format(i), nn.Sequential(*fusion_layer[i]))

        self.translator = self.get_trans_network(opt, act, norm, pad, input_ch=n_ch)
        self.n_downsample = n_downsample
        self.n_fusion = len(fusion_layer)
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
            x = torch.cat(getattr(self, 'Gate_layer_{}'.format(self.n_gate - 1 - i))(results[self.n_downsample - 1 - i]),
                          getattr(self, 'Up_layer_{}'.format(self.n_downsample - 1 - i))(x), dim=1)
            x = getattr(self, 'Fusion_layer_{}'.format(self.n_fusion - 1 - i))(x)
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