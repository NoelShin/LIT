import torch
import torch.nn as nn
from torch.autograd import grad
import numpy as np


class Loss(object):
    def __init__(self, opt):
        self.condition = opt.C_condition
        self.n_C = opt.n_C
        self.n_critics = opt.n_critics
        self.progression = opt.progression
        self.USE_CUDA = True if opt.gpu_ids != -1 else False

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

        if opt.GP_mode == 'Banach':
            self.drift_lambda = opt.drift_lambda
            self.drift_loss = opt.drift_loss
            self.exponent = opt.exponent
            self.dual_exponent = 1 / (1 - 1 / self.exponent) if self.exponent != 1 else np.inf
            self.sobolev_s = opt.sobolev_s
            self.sobolev_c = opt.sobolev_c

        elif opt.GP_mode == 'div':
            self.k = opt.k
            self.p = opt.p

        elif opt.GP_mode == 'GP':
            self.GP_lambda = opt.GP_lambda

        else:
            raise NotImplementedError

        if opt.VGG:
            from models import VGG19
            self.VGG = True
            self.VGGNet = VGG19().cuda(0) if opt.gpu_ids != -1 else VGG19()
            self.VGG_lambda = opt.VGG_lambda
            self.VGG_weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

        else:
            self.VGG = False

    def calc_CT(self, real_features_1, real_features_2):
        CT = 0
        for i in range(self.n_C):
            CT += ((real_features_1[i][-1] - real_features_2[i][-1]) ** 2).mean()
            CT += 0.1 * ((real_features_1[i][-2] - real_features_2[i][-2]) ** 2).mean()
            CT = torch.max(torch.tensor(0.0).cuda(0) if self.USE_CUDA else torch.tensor(0.0), CT - self.CT_factor)
        return CT

    def calc_FM(self, fake_features, real_features, weights=None):
        weights = [1.0 for _ in range(len(real_features))] if not weights else weights
        FM = 0
        for i in range(len(real_features)):
            FM += weights[i] * self.FM_criterion(fake_features[i], real_features[i].detach())
        return FM

    def calc_GP(self, C, output, target):
        GP = 0
        alpha = torch.FloatTensor(torch.rand((target.shape[0], 1, 1, 1))).expand(target.shape)
        alpha = alpha.cuda(0) if self.USE_CUDA else alpha

        interp = (target + alpha * (output - target)).requires_grad_(True)

        for i in range(self.n_C):
            interp_score = getattr(C, 'Scale_{}'.format(i))(interp)[-1]
            weight_grid = torch.ones_like(interp_score)
            weight_grid = weight_grid.cuda(0) if self.USE_CUDA else weight_grid
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


class BWGANLoss(Loss):
    def __init__(self, opt):
        super(BWGANLoss, self).__init__(opt)

    def lp_norm(self, x, p=None, epsilon=1e-5):
        x = x.view(x.shape[0], -1).type(torch.cuda.FloatTensor if self.USE_CUDA else torch.FloatTensor)
        alpha, _ = torch.max(x.abs() + epsilon, dim=-1)
        return alpha * torch.norm(x / alpha[:, None], p=p, dim=1)

    def sobolev_filter(self, x, dual=False):
        real_x = x  # real part
        imaginary_x = torch.zeros_like(x)  # imaginary part
        x_fft = torch.fft(torch.stack([real_x, imaginary_x], dim=-1), signal_ndim=2)  # fourier transform

        dx = x_fft.shape[3]
        dy = x_fft.shape[2]

        x = torch.arange(dx)
        x = torch.min(x, dx - x)
        x = x / (dx // 2)

        y = torch.arange(dy)
        y = torch.min(y, dy - y)
        y = y / (dy // 2)

        # constructing the \xi domain
        X, Y = torch.meshgrid([y, x])
        X = X[None, None].float()
        Y = Y[None, None].float()

        # computing the scale (1 + |\xi|^2)^{s/2}
        scale = (1 + self.sobolev_c * (X ** 2 + Y ** 2)) ** (self.sobolev_s / 2 if not dual else -self.sobolev_s / 2)
        # scale is a real number which scales both real and imaginary parts by multiplying
        scale = torch.stack([scale, scale], dim=-1).float()
        scale = scale.cuda(0) if self.USE_CUDA else scale
        x_fft *= scale
        return torch.ifft(x_fft, signal_ndim=2)[..., 0]  # only real part

    def get_constants(self, real):
        transformed_real = self.sobolev_filter(real).view([real.shape[0], -1])
        lamb = self.lp_norm(transformed_real, p=self.exponent).mean()

        dual_transformed_real = self.sobolev_filter(real, dual=True).view([real.shape[0], -1])
        gamma = self.lp_norm(dual_transformed_real, p=self.dual_exponent).mean()
        lamb, gamma = (lamb.cuda(0), gamma.cuda(0)) if self.USE_CUDA else (lamb, gamma)
        return lamb, gamma

    def __call__(self, C, G, data_dict, current_step, level=None, level_in=None):
        loss_C = 0
        loss_G = 0
        package = dict()

        input = data_dict['input_tensor']
        fake = G(input, level=level, level_in=level_in) if self.progression else G(input)
        target = data_dict['target_tensor']

        if self.condition and self.progression:
            input = data_dict['C_input_tensor']
            input_fake, input_real = torch.cat([input, fake.detach()], dim=1), torch.cat([input, target], dim=1)
        elif self.condition and not self.progression:
            input_fake, input_real = torch.cat([input, fake.detach()], dim=1), torch.cat([input, target], dim=1)
        else:
            input_fake, input_real = fake.detach(), target

        lamb, gamma = self.get_constants(target)
        fake_features, real_features = C(input_fake), C(input_real)

        score_C = 0
        for i in range(self.n_C):
            score_C += (real_features[i][-1] - fake_features[i][-1]).mean() / gamma
            score_C += self.drift_lambda * (real_features[i][-1].mean() ** 2)
        loss_C += score_C

        GP = 0
        alpha = torch.FloatTensor(torch.rand((input_real.shape[0], 1, 1, 1))).expand(input_real.shape)
        alpha = alpha.cuda(0) if self.USE_CUDA else alpha
        interp = (input_real + alpha * (input_fake.detach() - input_real)).requires_grad_(True)

        for i in range(self.n_C):
            interp_score = getattr(C, 'Scale_{}'.format(i))(interp)[-1]
            weight_grid = torch.ones_like(interp_score).cuda(0) if self.USE_CUDA else torch.ones_like(interp_score)
            gradient = grad(outputs=interp_score, inputs=interp, grad_outputs=weight_grid,
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
            GP += ((self.lp_norm(self.sobolev_filter(gradient, dual=True), self.dual_exponent) / gamma - 1) ** 2).mean()

            interp = nn.AvgPool2d(kernel_size=3, padding=1, stride=2, count_include_pad=False)(interp)

        loss_C += lamb * GP
        package.update({'A_score': score_C.detach().item(), 'GP': GP.detach().item(), 'total_A_loss': loss_C,
                        'A_state_dict': C.state_dict()})

        if current_step % self.n_critics == 0:
            input_fake = torch.cat([input, fake], dim=1) if self.condition else fake

            fake_features = C(input_fake)

            G_score = 0
            for i in range(self.n_C):
                G_score += fake_features[i][-1].mean()
            loss_G += G_score / gamma
            package.update({'G_score': G_score.detach().item()})

            if self.FM:
                FM = 0
                n_layers = len(fake_features[0])
                for j in range(n_layers):
                    FM += self.FM_criterion(fake_features[i][j], real_features[i][j].detach())
                loss_G += FM * self.FM_lambda / self.n_C
                package.update({'FM': FM.detach().item() / self.n_C})
            else:
                package.update({'FM': 0.0})

            if self.VGG:
                VGG = self.calc_FM(self.VGGNet(fake), self.VGGNet(target), weights=self.VGG_weights)
                loss_G += self.VGG_lambda * VGG
                package.update({'VGG': VGG.detach().item()})
            else:
                package.update({'VGG': 0.0})

            package.update({'total_G_loss': loss_G, 'generated_tensor': fake.detach(),
                            'G_state_dict': G.state_dict(), 'target_tensor': target, 'CT': 0.0})
        return package


class LSGANLoss(Loss):
    def __init__(self, opt):
        super(LSGANLoss, self).__init__(opt)
        self.criterion = nn.MSELoss()

    def get_grid(self, tensor, is_real=False):
        grid = torch.FloatTensor(tensor.shape).fill_(1.0 if is_real else 0.0)
        grid = grid.cuda(0) if self.USE_CUDA else grid
        return grid

    def __call__(self, C, G, data_dict, level=None, level_in=None):
        loss_C = 0
        loss_G = 0
        package = dict()

        input, target = data_dict['input_tensor'], data_dict['target_tensor']
        fake = G(x=input, level=level, level_in=level_in) if self.progression else G(input)

        if self.condition and self.progression:
            input = data_dict['C_input_tensor']
            input_fake, input_real = torch.cat([input, fake.detach()], dim=1), torch.cat([input, target], dim=1)

        elif self.condition and not self.progression:
            input_fake, input_real = torch.cat([input, fake.detach()], dim=1), torch.cat([input, target], dim=1)
        else:
            input_fake, input_real = fake.detach(), target

        fake_features = C(input_fake, level, level_in) if self.progression else C(input_fake)
        real_features = C(input_real, level, level_in) if self.progression else C(input_real)

        C_score = 0
        for i in range(self.n_C):
            fake_grid, real_grid = self.get_grid(fake_features[i][-1]), self.get_grid(real_features[i][-1], True)

            C_score += self.criterion(real_features[i][-1], real_grid) + self.criterion(fake_features[i][-1], fake_grid)
            C_score *= 0.5

        loss_C += C_score
        package.update({'A_score': C_score.detach().item()})

        if self.CT:
            real_features_2 = C(input_real)
            CT = self.CT_lambda * self.calc_CT(real_features, real_features_2)
            loss_C += CT
            package.update({'CT': CT.detach().item()})
        else:
            package.update({'CT': 0.0})

        if self.GP:
            GP = self.calc_GP(C, output=input_fake.detach(), target=input_real.detach())
            loss_C += self.GP_lambda * GP
            package.update({'GP': GP.detach().item()})
        else:
            package.update({'GP': 0.0})

        input_fake = torch.cat([input, fake], dim=1) if self.condition else fake
        fake_features = C(input_fake, level, level_in) if self.progression else C(input_fake)

        G_score = 0
        for i in range(self.n_C):
            real_grid = self.get_grid(fake_features[i][-1], True)
            G_score += self.criterion(fake_features[i][-1], real_grid)
        loss_G += G_score
        package.update({'G_score': G_score.detach().item()})

        if self.FM:
            FM = 0
            n_layers = len(fake_features[0])
            for j in range(n_layers):
                FM += self.FM_criterion(fake_features[i][j], real_features[i][j].detach())
            loss_G += FM * self.FM_lambda / self.n_C
            package.update({'FM': FM.detach().item() / self.n_C})
        else:
            package.update({'FM': 0.0})

        if self.VGG:
            VGG = 0
            VGG += self.calc_FM(self.VGGNet(fake), self.VGGNet(target), weights=self.VGG_weights)
            loss_G += self.VGG_lambda * VGG
            package.update({'VGG': VGG.detach().item()})
        else:
            package.update({'VGG': 0.0})

        package.update({'total_A_loss': loss_C, 'total_G_loss': loss_G, 'generated_tensor': fake.detach(),
                        'A_state_dict': C.state_dict(), 'G_state_dict': G.state_dict(), 'target_tensor': target})
        return package


class WGANLoss(Loss):
    def __init__(self, opt):
        super(WGANLoss, self).__init__(opt)
        self.GP_mode = opt.GP_mode

    def calc_GP(self, C, output, target):
        GP = 0
        alpha = torch.FloatTensor(torch.rand((target.shape[0], 1, 1, 1))).expand(target.shape)
        alpha = alpha.cuda(0) if self.USE_CUDA else alpha

        interp = (target + alpha * (output - target)).requires_grad_(True)

        for i in range(self.n_C):
            interp_score = getattr(C, 'Scale_{}'.format(i))(interp)[-1]
            weight_grid = torch.ones_like(interp_score).cuda(0) if self.USE_CUDA else torch.ones_like(interp_score)
            gradient = grad(outputs=interp_score, inputs=interp, grad_outputs=weight_grid,
                            create_graph=True, retain_graph=True, only_inputs=True)[0]

            gradient = gradient.view(gradient.shape[0], -1)
            GP += ((gradient.norm(2, dim=-1) - 1) ** 2).mean()
            interp = nn.AvgPool2d(kernel_size=3, padding=1, stride=2, count_include_pad=False)(interp)
        return GP

    def __call__(self, C, G, data_dict, current_step, level=None, level_in=None):
        loss_C = 0
        loss_G = 0
        package = {}

        input, target = data_dict['input_tensor'], data_dict['target_tensor']
        fake = G(input, level=level, level_in=level_in) if self.progression else G(input)

        if self.condition and self.progression:
            input = data_dict['C_input_tensor']
            input_fake, input_real = torch.cat([input, fake.detach()], dim=1), torch.cat([input, target], dim=1)
        elif self.condition and not self.progression:
            input_fake, input_real = torch.cat([input, fake.detach()], dim=1), torch.cat([input, target], dim=1)
        else:
            input_fake, input_real = fake.detach(), target

        fake_features, real_features = C(input_fake), C(input_real)

        C_score = 0
        for i in range(self.n_C):
            C_score += (fake_features[i][-1] - real_features[i][-1]).mean()
        loss_C += C_score

        GP = self.calc_GP(C, output=input_fake.detach(), target=input_real.detach())
        loss_C += self.GP_lambda * GP

        if self.CT:
            CT = self.calc_CT(real_features, C(input_real))
            loss_C += self.CT_lambda * CT
            package.update({'CT': CT.detach().item()})
        else:
            package.update({'CT': 0.0})

        package.update({"A_score": C_score.detach().item(), 'GP': GP.detach().item(), 'total_A_loss': loss_C,
                        'A_state_dict': C.state_dict()})

        if current_step % self.n_critics == 0:

            input_fake = torch.cat([input, fake], dim=1) if self.condition else fake
            fake_features = C(input_fake)

            G_score = 0
            for i in range(self.n_C):
                G_score += -fake_features[i][-1].mean()
            loss_G += G_score

            if self.FM:
                FM = 0
                n_layers = len(fake_features[0])

                for j in range(n_layers):
                    FM += self.FM_criterion(fake_features[i][j], real_features[i][j].detach())
                loss_G += self.FM_lambda * FM / self.n_C
                package.update({'FM': FM.detach().item() / self.n_C})
            else:
                package.update({'FM': 0.0})

            if self.VGG:
                VGG = 0
                VGG += self.calc_FM(self.VGGNet(fake), self.VGGNet(target), weights=self.VGG_weights)
                loss_G += self.VGG_lambda * VGG
                package.update({'VGG': VGG.detach().item()})
            else:
                package.update({'VGG': 0.0})

            package.update({'G_score': G_score.detach().item(), 'total_G_loss': loss_G,
                            'generated_tensor': fake.detach(), 'G_state_dict': G.state_dict(), 'target_tensor': target})
        return package


class WGANDivLoss(Loss):
    def __init__(self, opt):
        super(WGANDivLoss, self).__init__(opt)

    def calc_GP(self, C, output, target):
        GP = 0
        alpha = torch.FloatTensor(torch.rand((target.shape[0], 1, 1, 1))).expand(target.shape)
        alpha = alpha.cuda(0) if self.USE_CUDA else alpha

        interp = (target + alpha * (output - target)).requires_grad_(True)

        for i in range(self.n_C):
            interp_score = getattr(C, 'Scale_{}'.format(i))(interp)[-1]
            weight_grid = torch.ones_like(interp_score)
            weight_grid = weight_grid.cuda(0) if self.USE_CUDA else weight_grid
            gradient = grad(outputs=interp_score, inputs=interp, grad_outputs=weight_grid,
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
            gradient = gradient.view(gradient.shape[0], -1)
            GP += gradient.norm(self.p, dim=1).mean()
            interp = nn.AvgPool2d(kernel_size=3, padding=1, stride=2, count_include_pad=False)(interp)
        return GP

    def __call__(self, C, G, data_dict, current_step, level=None, level_in=None):
        loss_C = 0
        loss_G = 0
        package = {}

        input, target = data_dict['input_tensor'], data_dict['target_tensor']
        fake = G(input, level=level, level_in=level_in) if self.progression else G(input)

        if self.condition and self.progression:
            input = data_dict['C_input_tensor']
            input_fake, input_real = torch.cat([input, fake.detach()], dim=1), torch.cat([input, target], dim=1)
        elif self.condition and not self.progression:
            input_fake, input_real = torch.cat([input, fake.detach()], dim=1), torch.cat([input, target], dim=1)
        else:
            input_fake, input_real = fake.detach(), target

        fake_features, real_features = C(input_fake), C(input_real)

        C_score = 0
        for i in range(self.n_C):
            C_score += (-fake_features[i][-1] + real_features[i][-1]).mean()
        loss_C += C_score

        GP = self.calc_GP(C, output=input_fake.detach(), target=input_real.detach())
        loss_C += self.k * GP

        package.update({"A_score": C_score.detach().item(), 'GP': GP.detach().item(), 'total_A_loss': loss_C,
                        'A_state_dict': C.state_dict()})

        if current_step % self.n_critics == 0:

            input_fake = torch.cat([input, fake], dim=1) if self.condition else fake

            fake_features = C(input_fake)

            G_score = 0
            for i in range(self.n_C):
                G_score += fake_features[i][-1].mean()
            loss_G += G_score
            package.update({'G_score': G_score.detach().item()})

            if self.FM:
                FM = 0
                n_layers = len(fake_features[0])
                for j in range(n_layers):
                    FM += self.FM_criterion(fake_features[i][j], real_features[i][j].detach())
                loss_G += FM * self.FM_lambda / self.n_C
                package.update({'FM': FM.detach().item() / self.n_C})
            else:
                package.update({'FM': 0.0})

            if self.VGG:
                VGG = self.calc_FM(self.VGGNet(fake), self.VGGNet(target), weights=self.VGG_weights)
                loss_G += self.VGG_lambda * VGG
                package.update({'VGG': VGG.detach().item()})
            else:
                package.update({'VGG': 0.0})

            package.update({'total_G_loss': loss_G, 'generated_tensor': fake.detach(),
                            'G_state_dict': G.state_dict(), 'target_tensor': target})
        return package
