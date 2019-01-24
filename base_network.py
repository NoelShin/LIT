import torch
import torch.nn as nn
from modules.base_module import BaseModule


class BaseNetwork(BaseModule):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    @staticmethod
    def init_weights(module, type='kaiming_normal', mode='fan_in', negative_slope=0.2, nonlinearity='leaky_relu'):
        classname = module.__class__.__name__
        if classname.find('Conv') != -1:
            if type == 'kaiming_normal':
                nn.init.kaiming_normal_(module.weight.detach(), a=negative_slope, mode=mode, nonlinearity=nonlinearity)

            elif type == 'normal':
                nn.init.normal_(module.weight.detach(), 0.0, 0.02)

            else:
                raise NotImplementedError("Weight init type {} is not valid.".format(type))

        else:
            pass

    def set_attribute(self, list, name, sequential=True, progressive=False):
        if progressive:
            if sequential:
                for i in range(len(list)):
                    setattr(self, name + '_level_{}'.format(i), nn.Sequential(*list[i]))
            else:
                for i in range(len(list)):
                    setattr(self, name + '_level_{}'.format(i), list[i])

        else:
            if sequential:
                for i in range(len(list)):
                    setattr(self, name + '_{}'.format(i), nn.Sequential(*list[i]))
            else:
                for i in range(len(list)):
                    setattr(self, name + '_{}'.format(i), list[i])

    def to_CUDA(self, gpu_id):
        gpu_id = gpu_id[0] if isinstance(gpu_id, list) else gpu_id
        if gpu_id != -1:
            self.to(torch.device('cuda', gpu_id))
        else:
            pass

    def forward(self, x):
        pass

