import torch
import torch.nn as nn
from .base_loss import Loss


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

        package.update({'A_score': C_score.detach().item(), 'G_score': G_score.detach().item(), 'total_A_loss': loss_C,
                        'total_G_loss': loss_G, 'generated_tensor': fake.detach(), 'A_state_dict': C.state_dict(),
                        'G_state_dict': G.state_dict(), 'target_tensor': target})
        return package
