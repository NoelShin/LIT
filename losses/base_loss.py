import torch
import torch.nn as nn
from torch.autograd import grad
import numpy as np


class Loss(object):
    def __init__(self, opt):
        self.condition = opt.C_condition
        self.device = torch.device('cuda' if opt.USE_CUDA else 'cpu', 0)
        self.n_C = opt.n_C
        self.n_critics = opt.n_critics
        self.progression = opt.progression
        self.USE_CUDA = opt.USE_CUDA

        if opt.CT:
            self.CT = True
            self.CT_lambda = opt.CT_lambda
            self.CT_factor = opt.CT_factor
        else:
            self.CT = False

        if opt.FM:
            self.FM = True
            self.FM_criterion = self.get_criterion(opt.FM_criterion)
            self.FM_lambda = opt.FM_lambda

        else:
            self.FM = False

        if opt.GAN_type == 'BWGAN':
            self.drift_lambda = opt.drift_lambda
            self.drift_loss = opt.drift_loss
            self.exponent = opt.exponent
            self.dual_exponent = 1 / (1 - 1 / self.exponent) if self.exponent != 1 else np.inf
            self.sobolev_c = opt.sobolev_c
            self.sobolev_s = opt.sobolev_s

        elif opt.GAN_type == 'WGANDiv':
            self.k = opt.k
            self.p = opt.p

        elif opt.GAN_type == 'WGANGP':
            self.GP_lambda = opt.GP_lambda

        else:
            raise NotImplementedError

        if opt.VGG:
            from models import VGG19
            self.VGG = True
            self.VGGNet = VGG19().to(self.device)
            self.VGG_lambda = opt.VGG_lambda
            self.VGG_weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

        else:
            self.VGG = False

    def calc_CT(self, real_features_1, real_features_2):
        CT = 0
        for i in range(self.n_C):
            CT += ((real_features_1[i][-1] - real_features_2[i][-1]) ** 2).mean()
            CT += 0.1 * ((real_features_1[i][-2] - real_features_2[i][-2]) ** 2).mean()
            CT = torch.max(torch.tensor(0.0).to(self.device), CT - self.CT_factor)
        return CT

    def calc_FM(self, fake_features, real_features, weights=None):
        weights = [1.0 for _ in range(len(real_features))] if not weights else weights
        FM = 0
        for i in range(len(real_features)):
            FM += weights[i] * self.FM_criterion(fake_features[i], real_features[i].detach())
        return FM

    def calc_GP(self, C, output, target):
        GP = 0
        alpha = torch.FloatTensor(torch.rand((target.shape[0], 1, 1, 1))).expand(target.shape).to(self.device)

        interp = (target + alpha * (output - target)).requires_grad_(True)

        for i in range(self.n_C):
            interp_score = getattr(C, 'Scale_{}'.format(i))(interp)[-1]
            weight_grid = torch.ones_like(interp_score).to(self.device)
            gradient = grad(outputs=interp_score, inputs=interp, grad_outputs=weight_grid,
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
            gradient = gradient.view(gradient.shape[0], -1)
            GP += ((gradient.norm(2, dim=1) - 1) ** 2).mean()
            interp = nn.AvgPool2d(kernel_size=3, padding=1, stride=2, count_include_pad=False)(interp)
        return GP

    @staticmethod
    def get_criterion(type):
        if type == 'L1':
            criterion = nn.L1Loss()

        elif type == 'MSE':
            criterion = nn.MSELoss()

        else:
            raise NotImplementedError("Invalid loss type {}. Please choose among ['L1', 'MSE']".format(type))

        return criterion
