import argparse
from utils import configure


class BaseOption(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument('--debug', action='store_true', default=False, help='for checking code')

        self.parser.add_argument('--batch_size', type=int, default=1, help='the number of batch_size')
        self.parser.add_argument('--dataset_name', type=str, default='Cityscapes', help='[Cityscapes, Custom]')
        self.parser.add_argument('--growth_rate', type=int, default=32)
        self.parser.add_argument('--gpu_ids', type=int, default=2, help='gpu number. If -1, use cpu')
        self.parser.add_argument('--image_height', type=int, default=512, help='[512, 1024]')
        self.parser.add_argument('--image_mode', type=str, default='png', help='extension for saving image')
        self.parser.add_argument('--leaky_inc', type=float, default=0.2, help='inclination for negative part')
        self.parser.add_argument('--max_ch', type=int, default=2 ** 10, help='max_nb of channels in model')
        self.parser.add_argument('--n_dense_layer', type=int, default=6, help='how many dense layers in a RDB')
        self.parser.add_argument('--n_downsample', type=int, default=4,
                                 help='how many times you want downsample the original data')
        self.parser.add_argument('--n_residual', type=int, default=16, help='the number of residual (dense) blocks')
        self.parser.add_argument('--n_workers', type=int, default=2, help='how many threads you want to use')
        self.parser.add_argument('--norm_type', type=str, default='InstanceNorm2d',
                                 help='[BatchNorm2d, InstanceNorm2d]')
        self.parser.add_argument('--pad_type', type=str, default='reflection',
                                 help='[reflection, replication, zero]')
        self.parser.add_argument('trans_unit', type=str, default='RDB',
                                 help='Unit you want to use for image translation. "RB" for residual block, "RDB"'
                                      ' for Residual dense block')
        self.parser.add_argument('--use_boundary_map', action='store_true', default=True,
                                 help='if you want to use boundary map')

    def parse(self):
        self.opt = self.parser.parse_args()
        configure(self.opt)

        return self.opt


class TrainOption(BaseOption):
    def __init__(self):
        super(TrainOption, self).__init__()

        self.parser.add_argument('--is_train', action='store_true', default=True, help='train flag')

        self.parser.add_argument('--beta1', type=float, default=0.0)
        self.parser.add_argument('--beta2', type=float, default=0.9)
        self.parser.add_argument('--display_freq', type=int, default=500)
        self.parser.add_argument('--C_act', type=str, default='leaky_relu', help='which activation to use in Critic')
        self.parser.add_argument('--C_act_negative_slope', type=float, default=0.2, help='negative slope of C_act')
        self.parser.add_argument('--C_condition', action='store_true', default=False, help='whether to condition Critic')
        self.parser.add_argument('--C_norm', action='store_true', default=True, help='whether to use norm layer in Critic')
        self.parser.add_argument('--epoch_decay', type=int, default=100, help='when to start decay the lr')
        self.parser.add_argument('--eps', type=float, default=1e-8)
        self.parser.add_argument('--G_act', type=str, default='relu', help='which activation to use in Generator')
        self.parser.add_argument('--G_act_negative_slope', type=float, default=0.0, help='negative slope of G_act')
        self.parser.add_argument('--GP_lambda', type=int, default=10, help='weight for gradient penalty')
        self.parser.add_argument('--FM', action='store_true', default=True, help='switch for feature matching loss')
        self.parser.add_argument('--FM_criterion', type=str, default='L1', help='[L1, MSE]')
        self.parser.add_argument('--FM_lambda', type=int, default=10, help='weight for feature matching loss')
        self.parser.add_argument('--flip', action='store_true', default=True, help='switch for flip input data')
        self.parser.add_argument('--GAN_type', type=str, default='WGAN_GP', help='[LSGAN, WGAN_GP]')
        self.parser.add_argument('--lr', type=float, default=0.0002)
        self.parser.add_argument('--n_epochs', type=int, default=100)
        self.parser.add_argument('--report_freq', type=int, default=5)
        self.parser.add_argument('--save_freq', type=int, default=100000)
        self.parser.add_argument('--shuffle', action='store_true', default=True,
                                 help='if you want to shuffle the order')


class TestOption(BaseOption):
    def __init__(self):
        super(TestOption, self).__init__()

        self.parser.add_argument('--is_train', action='store_true', default=False, help='test flag')

        self.parser.add_argument('--shuffle', action='store_true', default=False,
                                 help='if you want to shuffle the order')