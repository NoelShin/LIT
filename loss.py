import torch
import torch.nn as nn
from torch.autograd import grad
import numpy as np


class Loss(object):
    def __init__(self, opt):
        if opt.FM:
            self.FM = True
            self.FM_criterion = self.get_loss(opt.FM_criterion)
            self.FM_lambda = opt.FM_lambda

        elif not opt.FM:
            self.FM = False

        else:
            raise NotImplementedError("Invalid FM (Feature Matching loss) option. Please check FM keyword in option.")

    def calc_FM(self, fake_features, real_features):
        FM = 0
        for i in range(len(real_features)):
            FM += self.FM_criterion(fake_features[i], real_features[i].detach())

        return FM

    @staticmethod
    def get_grid(tensor, is_real=True):
        if is_real:
            grid = torch.FloatTensor(tensor.shape).fill_(1.0)

        elif not is_real:
            grid = torch.FloatTensor(tensor.shape).fill_(0.0)

        return grid

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
        self.opt = opt
        self.criterion = nn.MSELoss()
        self.D_condition = opt.C_condition

    def __call__(self, D, G, lod, data_dict):
        loss_D = 0
        loss_G = 0
        loss_G_FM = 0

        input = data_dict['input_tensor']
        fake = G(input, lod)
        target = data_dict['target_tensor']

        if self.D_condition:
            D_input = data_dict['D_input_tensor']
            D_input_real = torch.cat([D_input, target], dim=1)
            D_input_fake = torch.cat([D_input, fake.detach()], dim=1)

        else:
            D_input_real = target
            D_input_fake = fake.detach()

        real_features = D(D_input_real)
        fake_features = D(D_input_fake)

        real_grid = self.get_grid(real_features[-1], is_real=True)
        fake_grid = self.get_grid(fake_features[-1], is_real=False)

        if self.opt.gpu_ids != -1:
            real_grid = real_grid.cuda(self.opt.gpu_ids)
            fake_grid = fake_grid.cuda(self.opt.gpu_ids)

        loss_D += (self.criterion(real_features[-1], real_grid) +
                   self.criterion(fake_features[-1], fake_grid)) * 0.5

        if self.D_condition:
            D_input_fake = torch.cat([D_input, fake], dim=1)

        else:
            D_input_fake = fake

        fake_features = D(D_input_fake)

        for i in range(len(fake_features[0])):
            loss_G_FM += self.FM_criterion(fake_features[i], real_features[i].detach())

        real_grid = self.get_grid(fake_features[-1], is_real=True)
        if self.opt.gpu_ids != -1:
            real_grid = real_grid.cuda(self.opt.gpu_ids)
        loss_G += self.criterion(fake_features[-1], real_grid)

        loss_G += loss_G_FM * self.opt.FM_lambda

        return loss_D, loss_G, fake


class WGANGPLoss(Loss):
    def __init__(self, opt):
        super(WGANGPLoss, self).__init__(opt)
        self.C_condition = opt.C_condition
        self.GP_lambda = opt.GP_lambda
        self.gpu_id = opt.gpu_ids

    def calc_GP(self, C, output, target):
        alpha = torch.FloatTensor(np.random.random((target.shape[0], 1, 1, 1)))
        alpha = alpha.cuda(self.gpu_id) if self.gpu_id != -1 else alpha

        differences = target - output
        interp = (alpha * differences + output).requires_grad_(True)

        interp_score = C(interp)[-1]

        output_grid = torch.ones(interp_score.shape)
        output_grid = output_grid.cuda(self.gpu_id) if self.gpu_id != -1 else output_grid

        gradient = grad(outputs=interp_score, inputs=interp, grad_outputs=output_grid,
                        create_graph=True, retain_graph=True, only_inputs=True)[0]

        GP = (((gradient ** 2).sqrt() - 1.) ** 2).mean()

        return GP

    def __call__(self, C, G, lod, data_dict):
        C_loss = 0
        G_loss = 0

        fake = G(data_dict['input_tensor'], lod)
        target = data_dict['target_tensor']

        if self.C_condition:
            C_input = data_dict['D_input_tensor']
            C_input_fake = torch.cat([C_input, fake.detach()], dim=1)
            C_input_real = torch.cat([C_input, target], dim=1)

        else:
            C_input_fake = fake.detach()
            C_input_real = target

        fake_features = C(C_input_fake)
        real_features = C(C_input_real)
        C_loss += (fake_features[-1] - real_features[-1]).mean()

        C_loss += self.GP_lambda * self.calc_GP(C, output=C_input_fake.detach(), target=C_input_real.detach())

        if self.C_condition:
            C_input_fake = torch.cat([C_input, fake], dim=1)

        else:
            C_input_fake = fake

        fake_features = C(C_input_fake)
        G_loss += -fake_features[-1].mean()

        if self.FM:
            G_loss += self.FM_lambda * self.calc_FM(fake_features=fake_features, real_features=real_features)

        return C_loss, G_loss, fake
