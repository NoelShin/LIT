import torch
import torch.nn as nn
from torch.autograd import grad
from base_loss import Loss


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
        package.update({'A_score': C_score.detach().item(), 'GP': GP.detach().item(), 'total_A_loss': loss_C,
                        'A_state_dict': C.state_dict()})
        if current_step % self.n_critics == 0:

            input_fake = torch.cat([input, fake], dim=1) if self.condition else fake

            fake_features = C(input_fake)

            G_score = 0
            for i in range(self.n_C):
                G_score += fake_features[i][-1].mean()
            loss_G += G_score

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

            package.update({'G_score': G_score.detach().item(), 'total_G_loss': loss_G,
                            'generated_tensor': fake.detach(), 'G_state_dict': G.state_dict(), 'target_tensor': target})
        return package
