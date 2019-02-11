import torch
import torch.nn as nn
from torch.autograd import grad
import numpy as np


class Loss(object):
    def __init__(self, opt):
        if opt.FM:
            self.equal_FM_weights = opt.equal_FM_weights
            self.FM = True
            self.FM_criterion = self.get_loss(opt.FM_criterion)
            self.FM_lambda = opt.FM_lambda

            if opt.Res_C:
                self.n_down = opt.n_downsample_C if not opt.progression else opt.n_downsample
                self.n_RB_C = opt.n_RB_C

        else:
            self.FM = False

        if opt.GP:
            self.GP = True
            self.GP_lambda = opt.GP_lambda

        else:
            self.GP = False

        if opt.VGG:
            from models import VGG19
            self.VGG = True
            self.VGGNet = VGG19()
            self.VGG_lambda = opt.VGG_lambda

            if opt.gpu_ids != -1:
                self.VGGNet = self.VGGNet.cuda(0)
                self.VGG_weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]

        else:
            self.VGG = False

        self.gpu_id = opt.gpu_ids
        self.n_C = opt.n_C

    def calc_FM(self, fake_features, real_features, weights=None):
        FM = 0
        if weights:
            for i in range(len(real_features)):
                FM += weights[i] * self.FM_criterion(fake_features[i], real_features[i].detach())

        elif not weights:
            for i in range(len(real_features)):
                FM += self.FM_criterion(fake_features[i], real_features[i].detach())

        return FM

    def calc_GP(self, C, output, target):
        GP = 0
        alpha = torch.FloatTensor(np.random.random((target.shape[0], 1, 1, 1)))
        alpha = alpha.cuda(self.gpu_id) if self.gpu_id != -1 else alpha

        differences = target - output
        interp = (alpha * differences + output).requires_grad_(True)

        for i in range(self.n_C):
            interp_score = getattr(C, 'Scale_{}'.format(i))(interp)[-1]
            output_grid = torch.ones(interp_score.shape)
            output_grid = output_grid.cuda(self.gpu_id) if self.gpu_id != -1 else output_grid
            gradient = grad(outputs=interp_score, inputs=interp, grad_outputs=output_grid,
                            create_graph=True, retain_graph=True, only_inputs=True)[0]

            GP = (((gradient ** 2 + 1e-8).sqrt() - 1.) ** 2).mean()

            interp = nn.AvgPool2d(kernel_size=3, padding=1, stride=2, count_include_pad=False)(interp)

        return GP

    @staticmethod
    def get_loss(type):
        if type == 'L1':
            criterion = nn.L1Loss()

        elif type == 'MSE':
            criterion = nn.MSELoss()

        else:
            raise NotImplementedError("Invalid loss type {}. Please choose among ['L1', 'MSE']".format(type))

        return criterion


class LSGANLoss(Loss):
    def __init__(self, opt):
        super(LSGANLoss, self).__init__(opt)
        self.criterion = nn.MSELoss()
        self.condition = opt.C_condition
        self.gpu_id = opt.gpu_ids
        self.n_C = opt.n_C
        self.progression = opt.progression

    def get_grid(self, tensor, is_real=True):
        if is_real:
            grid = torch.FloatTensor(tensor.shape).fill_(1.0)

        elif not is_real:
            grid = torch.FloatTensor(tensor.shape).fill_(0.0)

        if self.gpu_id != -1:
            grid = grid.cuda(0)

        return grid

    def __call__(self, C, G, data_dict, level=None, level_in=None):
        loss_C = 0
        loss_G = 0
        package = {}

        input = data_dict['input_tensor']
        fake = G(x=input, level=level, level_in=level_in) if self.progression else G(input)
        target = data_dict['target_tensor']

        if self.condition:
            input = data_dict['C_input_tensor']
            input_fake = torch.cat([input, fake.detach()], dim=1)
            input_real = torch.cat([input, target], dim=1)

        else:
            input_fake = fake.detach()
            input_real = target

        fake_features = C(input_fake, level, level_in) if self.progression else C(input_fake)
        real_features = C(input_real, level, level_in) if self.progression else C(input_real)

        for i in range(self.n_C):
            fake_grid = self.get_grid(fake_features[i][-1], is_real=False)
            real_grid = self.get_grid(real_features[i][-1], is_real=True)

            loss_C += (self.criterion(real_features[i][-1], real_grid) +
                       self.criterion(fake_features[i][-1], fake_grid)) * 0.5

        if self.condition:
            input_fake = torch.cat([input, fake], dim=1)

        else:
            input_fake = fake

        fake_features = C(input_fake, level, level_in) if self.progression else C(input_fake)
        for i in range(self.n_C):
            real_grid = self.get_grid(fake_features[i][-1], is_real=True)
            loss_G += self.criterion(fake_features[i][-1], real_grid)

            if self.FM:
                loss_FM = 0
                n_layers = len(fake_features[0])
                if self.equal_FM_weights:
                    weights = [1.0 for i in range(n_layers)]
                else:
                    weights = [1.0 for i in range(self.n_down)] + [2 ** -(i + 1) for i in range(self.n_RB_C)] + [1.0]

                for j in range(n_layers):
                    loss_FM += weights[j] * self.FM_criterion(fake_features[i][j], real_features[i][j].detach())
                loss_G += loss_FM * self.FM_lambda * 1/self.n_C
                package.update({'FM': loss_FM.detach().item()})

        if self.VGG:
            loss_VGG = 0
            fake_features_VGG, real_features_VGG = self.VGGNet(fake), self.VGGNet(target)
            loss_VGG += self.calc_FM(fake_features_VGG, real_features_VGG, self.VGG_weights)

            loss_G += self.FM_lambda * loss_VGG

        package.update({'A_loss': loss_C, 'G_loss': loss_G, 'generated_tensor': fake.detach(),
                        'A_state_dict': C.state_dict(), 'G_state_dict': G.state_dict(), 'target_tensor': target})

        return package


class WGANGPLoss(Loss):
    def __init__(self, opt):
        super(WGANGPLoss, self).__init__(opt)
        self.condition = opt.C_condition
        self.drift_lambda = opt.drift_lambda
        self.GP_lambda = opt.GP_lambda
        self.gpu_id = opt.gpu_ids
        self.progression = opt.progression

    def __call__(self, C, G, data_dict, lod=None):
        C_loss = 0
        G_loss = 0
        package = {}

        fake = G(data_dict['input_tensor'], lod) if self.progression else G(data_dict['input_tensor'])
        target = data_dict['target_tensor']

        if self.condition:
            input = data_dict['C_input_tensor']
            input_fake = torch.cat([input, fake.detach()], dim=1)
            input_real = torch.cat([input, target], dim=1)
        else:
            input_fake = fake.detach()
            input_real = target

        fake_features = C(input_fake)
        real_features = C(input_real)

        for i in range(self.n_C):
            C_loss += (fake_features[i][-1] - real_features[i][-1]).mean()

        GP = self.calc_GP(C, output=input_fake.detach(), target=input_real)
        C_loss += self.GP_lambda * GP

        if self.condition:
            input_fake = torch.cat([input, fake], dim=1)
        else:
            input_fake = fake

        fake_features = C(input_fake)

        for i in range(self.n_C):
            G_loss += -fake_features[i][-1].mean()

        if self.FM:
            FM = self.calc_FM(fake_features=fake_features, real_features=real_features)
            package.update({'FM': FM.detach().item()})
            G_loss += self.FM_lambda * FM

        if self.VGG:
            VGG = 0
            fake_features_VGG, real_features_VGG = self.VGGNet(fake), self.VGGNet(target)
            VGG += self.calc_FM(fake_features=fake_features_VGG, real_features=real_features_VGG,
                                weights=self.VGG_weights)

            G_loss += self.VGG_lambda * VGG

        package.update({'A_loss': C_loss, 'G_loss': G_loss, 'GP': GP.detach().item(), 'generated_tensor': fake.detach(),
                        'A_state_dict': C.state_dict(), 'G_state_dict': G.state_dict(), 'target_tensor': target})

        return package
