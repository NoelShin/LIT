import torch
import torch.nn as nn
from torch.autograd import grad
from base_loss import Loss


class BWGANLoss(Loss):
    def __init__(self, opt):
        super(BWGANLoss, self).__init__(opt)

    def lp_norm(self, x, p=None, epsilon=1e-5):
        x = x.view(x.shape[0], -1)
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

        input, target = data_dict['input_tensor'], data_dict['target_tensor']
        fake = G(input, level=level, level_in=level_in) if self.progression else G(input)

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

        if current_step % self.n_critics == 0:
            input_fake = torch.cat([input, fake], dim=1) if self.condition else fake

            fake_features = C(input_fake)

            G_score = 0
            for i in range(self.n_C):
                G_score += fake_features[i][-1].mean()
            loss_G += G_score / gamma

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

            package.update({'A_score': score_C.detach().item(), 'GP': GP.detach().item(), 'total_A_loss': loss_C,
                            'A_state_dict': C.state_dict(), 'G_score': G_score.detach().item(), 'total_G_loss': loss_G,
                            'generated_tensor': fake.detach(), 'G_state_dict': G.state_dict(), 'target_tensor': target,
                            'CT': 0.0})
        return package
    
