import argparse
from utils import configure


class BaseOption(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument('--debug', action='store_true', default=False, help='for checking code')
        self.parser.add_argument('--gpu_ids', type=int, default=3, help='gpu number. If -1, use cpu')

        self.parser.add_argument('--batch_size', type=int, default=1, help='the number of batch_size')
        self.parser.add_argument('--block_weight', action='store_true', default=False, help='learnable block weights')
        self.parser.add_argument('--dataset_name', type=str, default='Cityscapes', help='[Cityscapes, Custom]')
        self.parser.add_argument('--fan_mode', type=str, default='fan_in',
                                 help='He init keyword. Choose among [fan_in, fan_out]')
        self.parser.add_argument('--flip', action='store_true', default=True, help='switch for flip input data')
        self.parser.add_argument('--G_act', type=str, default='relu', help='which activation to use in Generator')
        self.parser.add_argument('--image_height', type=int, default=512, help='[512, 1024]')
        self.parser.add_argument('--image_mode', type=str, default='png', help='extension for saving image')
        self.parser.add_argument('--init_type', type=str, default='normal',
                                 help='Init type. Choose among [kaiming_normal, normal]')
        self.parser.add_argument('--negative_slope', type=float, default=0.2, help='inclination for negative part')
        self.parser.add_argument('--max_ch_G', type=int, default=2 ** 10, help='max nb of channels in generator')
        self.parser.add_argument('--norm_type', type=str, default='InstanceNorm2d',
                                 help='[BatchNorm2d, InstanceNorm2d, PixelNorm]')
        self.parser.add_argument('--n_workers', type=int, default=0, help='how many threads you want to use')

        # about RIR
        self.parser.add_argument('--n_groups', type=int, default=4, help='the number of residual groups')
        self.parser.add_argument('--rir_ch', type=int, default=512)

        # about DB
        self.parser.add_argument('--efficient', action='store_true', default=True)
        self.parser.add_argument('--growth_rate', type=int, default=256)
        self.parser.add_argument('--n_dense_layers', type=int, default=4, help='how many dense layers in a RDB')

        # about G
        self.parser.add_argument('--trans_module', type=str, default='RB', help='[RB, DB, RDB, RIR]')
        self.parser.add_argument('--n_blocks', type=int, default=9, help='the number of residual blocks')
        self.parser.add_argument('--FM_lambda', type=float, default=10, help='weight for feature matching loss')
        self.parser.add_argument('--progression', action='store_true', default=False,
                                 help='if you want progressive training')
        self.parser.add_argument('--n_enhance_blocks', type=int, default=2,
                                 help='the number of enhancement blocks per level in decoder')
        self.parser.add_argument('--pad_type', type=str, default='reflection', help='[reflection, replication, zero]')
        self.parser.add_argument('--tanh', action='store_true', default=True, help='if you want to use tanh for RGB')
        self.parser.add_argument('--use_boundary_map', action='store_true', default=True,
                                 help='if you want to use boundary map')
        self.parser.add_argument('--reduction_rate', type=int, default=16)

        # Basic GAN option
        self.parser.add_argument('--GAN_type', type=str, default='WGAN', help='[LSGAN, WGAN]')
        self.parser.add_argument('--GP_lambda', type=int, default=10, help='weight for gradient penalty')
        self.parser.add_argument('--GP_mode', type=str, default='div', help='[Banach, div, GP]')

        # about Banach Wasserstein GAN (BWGAN)
        self.parser.add_argument('--drift_lambda', type=float, default=1e-5, help='weight for drift loss')
        self.parser.add_argument('--drift_loss', action='store_true', default=True, help='drift loss switch')
        self.parser.add_argument('--exponent', type=float, default=4.0, help='Exponent of norm')
        self.parser.add_argument('--sobolev_c', type=float, default=5.0)
        self.parser.add_argument('--sobolev_s', type=float, default=0.0)

        # about Consistenty Term Wasserstein GAN (CTGAN)
        self.parser.add_argument('--CT', action='store_true', default=False, help='CTGAN switch')
        self.parser.add_argument('--CT_factor', type=float, default=0.2, help='factor for CT term')
        self.parser.add_argument('--CT_lambda', type=int, default=2, help='weight for CT term')

        # about Wasserstein-div GAN (WGAN-div)
        self.parser.add_argument('--k', type=int, default=2, help='weight for div term')
        self.parser.add_argument('--p', type=int, default=6, help='p-norm for gradient')

        self.parser.add_argument('--C_condition', action='store_true', default=True, help='whether to condition Critic')
        self.parser.add_argument('--C_norm', action='store_true', default=True,
                                 help='whether to use norm layer in Critic')

        self.parser.add_argument('--max_ch_C', type=int, default=2 ** 9, help='max nb of channels in critic')
        self.parser.add_argument('--n_critics', type=int, default=1, help='how many C updates per G update')
        self.parser.add_argument('--n_downsample_C', type=int, default=4)
        self.parser.add_argument('--pad_type_C', type=str, default='zero')

    def parse(self):
        self.opt = self.parser.parse_args()
        configure(self.opt)

        return self.opt


class TrainOption(BaseOption):
    def __init__(self):
        super(TrainOption, self).__init__()

        self.parser.add_argument('--is_train', action='store_true', default=True, help='train flag')

        self.parser.add_argument('--display_freq', type=int, default=100)
        self.parser.add_argument('--C_act', type=str, default='leaky_relu', help='which activation to use in Critic')
        self.parser.add_argument('--C_act_negative_slope', type=float, default=0.2, help='negative slope of C_act')
        self.parser.add_argument('--epoch_decay', type=int, default=100, help='when to start decay the lr')
        self.parser.add_argument('--eps', type=float, default=1e-8)
        self.parser.add_argument('--FM', action='store_true', default=True, help='switch for feature matching loss')
        self.parser.add_argument('--FM_criterion', type=str, default='L1', help='[L1, MSE]')

        self.parser.add_argument('--G_act_negative_slope', type=float, default=0.0, help='negative slope of G_act')

        self.parser.add_argument('--lr', type=float, default=0.0002)
        self.parser.add_argument('--n_epochs', type=int, default=200)
        self.parser.add_argument('--n_epochs_per_lod', type=int, default=40)

        self.parser.add_argument('--report_freq', type=int, default=5)
        self.parser.add_argument('--save_freq', type=int, default=29750)
        self.parser.add_argument('--shuffle', action='store_true', default=True,
                                 help='if you want to shuffle the order')
        self.parser.add_argument('--VGG_lambda', type=int, default=10, help='weight for VGG loss')


class TestOption(BaseOption):
    def __init__(self):
        super(TestOption, self).__init__()

        self.parser.add_argument('--is_train', action='store_true', default=False, help='test flag')

        self.parser.add_argument('--shuffle', action='store_true', default=False,
                                 help='if you want to shuffle the order')
