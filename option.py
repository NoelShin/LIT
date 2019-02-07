import argparse
from utils import configure


class BaseOption(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument('--debug', action='store_true', default=False, help='for checking code')
        self.parser.add_argument('--gpu_ids', type=int, default=2, help='gpu number. If -1, use cpu')

        self.parser.add_argument('--batch_size', type=int, default=1, help='the number of batch_size')
        self.parser.add_argument('--dataset_name', type=str, default='Cityscapes', help='[Cityscapes, Custom]')
        self.parser.add_argument('--GAN_type', type=str, default='LSGAN', help='[LSGAN, WGAN_GP]')
        self.parser.add_argument('--image_height', type=int, default=512, help='[512, 1024]')
        self.parser.add_argument('--image_mode', type=str, default='png', help='extension for saving image')
        self.parser.add_argument('--negative_slope', type=float, default=0.2, help='inclination for negative part')
        self.parser.add_argument('--max_ch', type=int, default=2 ** 10, help='max_nb of channels in model')

        # about RCAN
        self.parser.add_argument('--n_RCAB', type=int, default=1,
                                 help='the number of RCAB blocks per a residual group')
        self.parser.add_argument('--n_RG', type=int, default=6, help='the number of residual groups')
        self.parser.add_argument('--RCA_ch', type=int, default=1024, help='the number of ch RCA has')
        self.parser.add_argument('--reduction_rate', type=int, default=16)

        # about RDN
        self.parser.add_argument('--growth_rate', type=int, default=512)
        self.parser.add_argument('--n_dense_layers', type=int, default=8, help='how many dense layers in a RDB')
        self.parser.add_argument('--n_RDB', type=int, default=4, help='the number of residual dense blocks')
        self.parser.add_argument('--RDB_ch', type=int, default=1024, help='the number of ch RDN started with')

        # about RN
        self.parser.add_argument('--n_RB', type=int, default=9, help='the number of residual blocks')

        # about architecture
        self.parser.add_argument('--progression', action='store_true', default=True,
                                 help='if you want progressive training')
        self.parser.add_argument('--trans_network', type=str, default='RDN',
                                 help='Network you want to use for image translation. "RN" for residual network, "RDN" for Residual dense network, "RCAN" for residual channel attention network')
        self.parser.add_argument('--U_net', action='store_true', default=True,
                                 help='if you want to use U-net skip connection')
        self.parser.add_argument('--U_net_gate', action='store_true', default=True, help='if you want gating for U-net')
        self.parser.add_argument('--n_enhance_blocks', type=int, default=2,
                                 help='the number of enhancement blocks per level in decoder')

        self.parser.add_argument('--n_workers', type=int, default=2, help='how many threads you want to use')
        self.parser.add_argument('--pad_type', type=str, default='reflection', help='[reflection, replication, zero]')
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

        self.parser.add_argument('--display_freq', type=int, default=100)
        self.parser.add_argument('--C_act', type=str, default='leaky_relu', help='which activation to use in Critic')
        self.parser.add_argument('--C_act_negative_slope', type=float, default=0.2, help='negative slope of C_act')
        self.parser.add_argument('--C_condition', action='store_true', default=True, help='whether to condition Critic')
        self.parser.add_argument('--C_norm', action='store_true', default=True,
                                 help='whether to use norm layer in Critic')
        self.parser.add_argument('--epoch_decay', type=int, default=100, help='when to start decay the lr')
        self.parser.add_argument('--eps', type=float, default=1e-8)
        self.parser.add_argument('--FM', action='store_true', default=True, help='switch for feature matching loss')
        self.parser.add_argument('--FM_criterion', type=str, default='L1', help='[L1, MSE]')
        self.parser.add_argument('--FM_lambda', type=int, default=10, help='weight for feature matching loss')
        self.parser.add_argument('--flip', action='store_true', default=True, help='switch for flip input data')
        self.parser.add_argument('--fan_mode', type=str, default='fan_in',
                                 help='He init keyword. Choose among [fan_in, fan_out]')
        self.parser.add_argument('--G_act', type=str, default='relu', help='which activation to use in Generator')
        self.parser.add_argument('--G_act_negative_slope', type=float, default=0.0, help='negative slope of G_act')
        self.parser.add_argument('--GP', action='store_true', default=False, help='if you want to add GP for GANs')
        self.parser.add_argument('--GP_lambda', type=int, default=50, help='weight for gradient penalty')
        self.parser.add_argument('--init_type', type=str, default='normal',
                                 help='Init type. Choose among [kaiming_normal, normal]')
        self.parser.add_argument('--lr', type=float, default=0.0002)
        self.parser.add_argument('--n_epochs', type=int, default=100)
        self.parser.add_argument('--n_epochs_per_lod', type=int, default=80)
        self.parser.add_argument('--norm_type', type=str, default='InstanceNorm2d',
                                 help='[BatchNorm2d, InstanceNorm2d, PixelNorm]')
        self.parser.add_argument('--report_freq', type=int, default=5)
        self.parser.add_argument('--save_freq', type=int, default=29750)
        self.parser.add_argument('--shuffle', action='store_true', default=True,
                                 help='if you want to shuffle the order')
        self.parser.add_argument('--tanh', action='store_true', default=True, help='if you want to use tanh for RGB')
        self.parser.add_argument('--VGG', action='store_true', default=True, help='if you want to add a VGG loss')
        self.parser.add_argument('--VGG_lambda', type=int, default=10, help='weight for VGG loss')


class TestOption(BaseOption):
    def __init__(self):
        super(TestOption, self).__init__()

        self.parser.add_argument('--is_train', action='store_true', default=False, help='test flag')

        self.parser.add_argument('--shuffle', action='store_true', default=False,
                                 help='if you want to shuffle the order')
