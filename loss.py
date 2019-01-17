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
            FM += self.FM_criterion(fake_features[i], real_features[i])

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

    def __call__(self, D, G, data_dict):
        loss_D = 0
        loss_G = 0
        loss_G_FM = 0

        input = data_dict['input_tensor']
        fake = G(input)
        target = data_dict['target_tensor']

        real_features = D(torch.cat([input, target], dim=1))
        fake_features = D(torch.cat([input, fake.detach()], dim=1))

        for i in range(self.opt.n_D):
            real_grid = self.get_grid(real_features[i][-1], is_real=True)
            fake_grid = self.get_grid(fake_features[i][-1], is_real=False)

            if self.opt.gpu_ids != -1:
                real_grid = real_grid.cuda(self.opt.gpu_ids)
                fake_grid = fake_grid.cuda(self.opt.gpu_ids)

            loss_D += (self.criterion(real_features[i][-1], real_grid) +
                       self.criterion(fake_features[i][-1], fake_grid)) * 0.5

        fake_features = D(torch.cat([input, fake], dim=1))

        for i in range(self.opt.n_D):
            for j in range(len(fake_features[0])):
                loss_G_FM += self.FM_criterion(fake_features[i][j], real_features[i][j].detach())

            real_grid = self.get_grid(fake_features[i][-1], is_real=True)
            if self.opt.gpu_ids != -1:
                real_grid = real_grid.cuda(self.opt.gpu_ids)
            loss_G += self.criterion(fake_features[i][-1], real_grid)

        loss_G += loss_G_FM * (1.0/self.opt.n_D) * self.opt.lambda_FM

        return loss_D, loss_G, target, fake


class WGANGPLoss(Loss):
    def __init__(self, opt):
        super(WGANGPLoss, self).__init__(opt)
        self.C_condition = opt.C_condition
        self.GP_lambda = opt.GP_lambda
        self.gpu_id = opt.gpu_id

    def calc_GP(self, output, target, C):
        alpha = torch.FloatTensor(np.random.random((target.shape[0], 1, 1, 1)))
        alpha = alpha.cuda(self.gpu_id) if self.gpu_id != -1 else alpha

        differences = target - output
        interp = (alpha * differences + output).requires_grad_(True)

        if self.C_condition:
            interp_score = C(torch.cat([input, interp], dim=1))[-1]

        else:
            interp_score = C(interp)[-1]

        output_grid = torch.ones(interp_score.shape)
        output_grid = output_grid.cuda(self.gpu_id) if self.gpu_id != -1 else alpha

        gradient = grad(outputs=interp_score, inputs=interp, grad_outputs=output_grid,
                        create_graph=True, retain_graph=True, only_inputs=True)[0]

        print(grad(outputs=interp_score, inputs=interp, grad_outputs=output_grid,
                   create_graph=True, retain_graph=True, only_inputs=True))

        GP = (((gradient ** 2).sqrt() - 1.) ** 2).mean()

        return GP

    def __call__(self, C, G, lod, data_dict):
        C_loss = 0
        G_loss = 0

        fake = G(data_dict['input_tensor'], lod)
        target = data_dict['target_tensor']

        fake_features = C(fake.detach())
        real_features = C(target)

        C_loss += fake_features[-1] - real_features[-1]
        C_loss += self.GP_lambda * self.calc_GP(output=fake.detach(), target=target, C=C)

        fake_features = C(fake)
        G_loss += -fake_features[-1]

        if self.FM:
            G_loss += self.FM_lambda * self.calc_FM(fake_features, real_features)

        return C_loss, G_loss, fake
