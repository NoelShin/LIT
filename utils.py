import os
import copy
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def configure(opt):
    image_height = opt.image_height
    is_train = opt.is_train
    trans_module = opt.trans_module
    progression = opt.progression
    opt.USE_CUDA = True if opt.gpu_ids != -1 else False
    opt.format = 'png'
    opt.n_df = 64

    dataset_name = opt.dataset_name
    if dataset_name == 'Cityscapes':
        opt.n_data = 2975
        opt.input_ch = 36 if opt.use_boundary_map else 35
        opt.output_ch = 3

        if image_height == 512:
            opt.image_size = (512, 1024)
            opt.n_downsample = 4
            opt.n_df = 64
            opt.n_gf = 64

        elif image_height == 1024:
            opt.image_size = (1024, 2048)
            opt.n_downsample = 5
            opt.n_df = 16
            opt.n_gf = 16
        else:
            raise NotImplementedError("Invalid image_height: {}".format(image_height))

    elif dataset_name == 'NYU':
        opt.n_data = 1200
        opt.output_ch = 3
        opt.input_ch = 896 if opt.use_boundary_map else 895
        opt.image_size = (561, 427)
        opt.n_downsample = 4
        opt.n_df = 64
        opt.n_gf = 64

    else:
        opt.input_ch = 1
        opt.output_ch = 1

        if image_height == 512:
            opt.image_size = (512, 512)
            opt.n_downsample = 4
            opt.n_df = 64
            opt.n_gf = 64

        elif image_height == 1024:
            opt.image_size = (1024, 1024)
            opt.n_downsample = 5
            opt.n_df = 32
            opt.n_gf = 32

    if progression:
        opt.beta1, opt.beta2 = (0.0, 0.9)
        opt.n_C = 1
        opt.patch_size = 16
        opt.save_freq = 2975 * 20
        opt.VGG = False

    else:
        opt.beta1, opt.beta2 = (0.5, 0.9)
        opt.n_C = 1 if opt.Res_C else 2
        opt.patch_size = 70
        opt.VGG = True

    opt.min_image_size = (2 ** (np.log2(opt.image_size[0]) - opt.n_downsample),
                          2 ** (np.log2(opt.image_size[1]) - opt.n_downsample))

    args = list()
    args.append(trans_module)
    args.append(opt.n_blocks)
    # args.append('LR')
    # args.append('Ex')
    # args.append('Entry')
    # args.append('0.0001')
    args.append('prelu') if opt.G_act is 'prelu' else None

    kwargs = dict()
    kwargs.update({'prog': progression}) if progression else None
    kwargs.update({'G': opt.n_groups, 'C': opt.rir_ch}) if trans_module == 'RIR' else None
    kwargs.update({'L': opt.n_dense_layers, 'K': opt.growth_rate}) if trans_module in ['RDB', 'DB'] else None
    kwargs.update({'C_down': opt.n_downsample_C, 'RB_C': opt.n_RB_C}) if opt.Res_C else None

    model_name = model_namer(*args, **kwargs)
    make_dir(dataset_name, model_name, is_train=is_train)

    opt.analysis_dir = os.path.join('./checkpoints', dataset_name, model_name, 'Analysis')
    opt.image_dir = os.path.join('./checkpoints', dataset_name, model_name,  'Image', '{}'.format('Training' if is_train
                                                                                                  else 'Test'))

    opt.model_dir = os.path.join('./checkpoints', dataset_name, model_name, 'Model')
    log_path = os.path.join(opt.model_dir, 'opt.txt')

    if opt.debug:
        opt.display_freq = 100
        opt.n_epochs = 4
        opt.n_epochs_per_lod = 1
        opt.report_freq = 5
        opt.save_freq = 1000000000

    if os.path.isfile(log_path) and not opt.debug and is_train:
        permission = input(
            "{} log already exists. Do you really want to overwrite this log? Y/N. : ".format(model_name + '/opt'))
        if permission == 'Y':
            pass

        else:
            raise NotImplementedError("Please check {}".format(log_path))

    args = vars(opt)
    with open(log_path, 'wt') as log:
        log.write('-' * 50 + 'Options' + '-' * 50 + '\n')
        print('-' * 50 + 'Options' + '-' * 50)
        for k, v in sorted(args.items()):
            log.write('{}: {}\n'.format(str(k), str(v)))
            print("{}: {}".format(str(k), str(v)))
        log.write('-' * 50 + 'End' + '-' * 50)
        print('-' * 50 + 'End' + '-' * 50)
        log.close()


def init_weights(module, type='kaiming_normal', mode='fan_in', negative_slope=0.2, nonlinearity='leaky_relu'):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
        if type == 'kaiming_normal':
            nn.init.kaiming_normal_(module.weight.detach(), a=negative_slope, mode=mode, nonlinearity=nonlinearity)

        elif type == 'normal':
            nn.init.normal_(module.weight.detach(), 0.0, 0.02)

        else:
            raise NotImplementedError("Weight init type {} is not valid.".format(type))

    else:
        pass


def make_dir(dataset_name=None, model_name=None, is_train=False):
    assert dataset_name in ['Cityscapes']
    assert model_name, "model_name keyword should be specified for type='checkpoints'"
    if is_train:
        os.makedirs(os.path.join('./checkpoints', dataset_name, model_name, 'Image', 'Training'), exist_ok=True)
        os.makedirs(os.path.join('./checkpoints', dataset_name, model_name, 'Model'), exist_ok=True)
        os.makedirs(os.path.join('./checkpoints', dataset_name, model_name, 'Analysis'), exist_ok=True)
    else:
        os.makedirs(os.path.join('./checkpoints', dataset_name, model_name, 'Image', 'Test'), exist_ok=True)


def model_namer(*elements, **k_elements):
    name = ''

    for v in elements:
        name += str(v) + '_'

    for k, v in sorted(k_elements.items()):
        name += str(k) + '_' + str(v) + '_'

    return name.strip('_')


class Manager(object):
    def __init__(self, opt):
        self.analysis_dir = opt.analysis_dir
        self.GAN_type = opt.GAN_type
        self.model_dir = opt.model_dir
        self.log = os.path.join(self.model_dir, 'log.txt')
        self.n_blocks = opt.n_blocks
        self.n_ch_trans = 1024
        self.signal_log = os.path.join(self.model_dir, 'ResidualSignals.txt')
        self.weight_log = os.path.join(self.model_dir, 'ResidualWeights.txt')

        # with open(self.signal_log, 'wt') as log:
        #    log.write('Epoch, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9\n')

        # with open(self.weight_log, 'wt') as log:
        #     log.write('Epoch, 1, 2, 3, 4, 5, 6, 7, 8, 9\n')

        if self.GAN_type == 'LSGAN':
            with open(self.log, 'wt') as log:
                log.write('Epoch, Current_step, C_loss, G_loss, FM_loss, Runtime\n')

        elif self.GAN_type == 'WGAN_GP':
            with open(self.log, 'wt') as log:
                log.write('Epoch, Current_step, C_loss, G_loss, GP_loss, FM_loss, Runtime\n')

        else:
            raise NotImplementedError

        self.image_dir = opt.image_dir
        self.image_mode = opt.image_mode

        if opt.is_train:
            self.display_freq = opt.display_freq
            self.progression = opt.progression
            self.report_freq = opt.report_freq
            self.save_freq = opt.save_freq

    def report_loss(self, package):
        if self.GAN_type == 'LSGAN':
            inf = [package['Epoch'], package['Current_step'], package['A_loss'].detach().item(),
                   package['G_loss'].detach().item(), package['FM']]
            print("Epoch: {} Current_step: {} A_loss: {:.{prec}}  G_loss: {:.{prec}} FM: {:.{prec}}".format(*inf, prec=4))

            with open(self.log, 'a') as log:
                log.write('{}, {}, {:.{prec}}, {:.{prec}}, {:.{prec}}\n'.format(*inf, prec=4))

        elif self.GAN_type == 'WGAN_GP':
            inf = [package['Epoch'], package['Current_step'], package['A_loss'].detach().item(),
                   package['G_loss'].detach().item(), package['GP'], package['FM']]
            print("Epoch: {} Current_step: {} A_loss: {:.{prec}}  G_loss: {:.{prec}} GP: {:.{prec}} FM: {:.{prec}},"
                  .format(*inf, prec=4))

            with open(self.log, 'a') as log:
                log.write('{}, {}, {:.{prec}}, {:.{prec}}, {:.{prec}}, {:.{prec}}\n'.format(*inf, prec=4))

    @staticmethod
    def adjust_dynamic_range(data, drange_in, drange_out):
        if drange_in != drange_out:
            scale = (np.float32(drange_out[1]) - np.float32(drange_out[0])) / (
                        np.float32(drange_in[1]) - np.float32(drange_in[0]))
            bias = (np.float32(drange_out[0]) - np.float32(drange_in[0]) * scale)
            data = data * scale + bias
        return data

    @staticmethod
    def write_log(log_path, informations, epoch, header=None):
        with open(log_path, 'wt' if epoch == 0 else 'a') as log:
            log.write(header) if not header else None
            log.write(informations + '\n')
            log.close()

    def layer_magnitude(self, G, epoch):
        names = list()
        magnitudes = list()
        for name, m in G.named_modules:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                names.append(name)
                magnitudes.append(m.weight.detach().abs().mean().cpu().item())
        self.write_log(os.path.join(self.analysis_dir, 'Layer_magnitudes.txt'), ','.join(map(str, magnitudes)), epoch,
                       header='Epoch, ' + ','.join(names) if epoch == 0 else None)
        magnitudes = np.array(magnitudes)
        plt.figure()
        plt.axhline(y=magnitudes.mean(), linestyle='--')
        plt.xticks(range(len(magnitudes) + 1))
        plt.xlabel('Layer index')
        plt.ylabel('Average absolute magnitude per layer')
        plt.plot(len(magnitudes), magnitudes, linestyle='--', marker='^', color='g')
        plt.savefig(os.path.join(self.analysis_dir, 'layer_magnitude_{}.png'.format(epoch)))
        plt.close()

    def tensor2image(self, image_tensor):
        np_image = image_tensor[0].cpu().float().numpy()
        if len(np_image.shape) == 3:
            np_image = np.transpose(np_image, (1, 2, 0))  # HWC
        else:
            pass

        np_image = self.adjust_dynamic_range(np_image, drange_in=[-1., 1.], drange_out=[0, 255])
        np_image = np.clip(np_image, 0, 255).astype(np.uint8)
        return np_image

    def save_image(self, image_tensor, path):
        np_image = self.tensor2image(image_tensor)
        pil_image = Image.fromarray(np_image)
        pil_image.save(path, self.image_mode)

    def save(self, package, image=False, model=False):
        if image:
            if self.progression:
                path_real = os.path.join(self.image_dir, str(package['Level']) + '_' + str(package['Epoch']) + '_' + 'real.png')
                path_fake = os.path.join(self.image_dir, str(package['Level']) + '_' + str(package['Epoch']) + '_' + 'fake.png')
            else:
                path_real = os.path.join(self.image_dir,  str(package['Epoch']) + '_' + 'real.png')
                path_fake = os.path.join(self.image_dir, str(package['Epoch']) + '_' + 'fake.png')
            self.save_image(package['target_tensor'], path_real)
            self.save_image(package['generated_tensor'], path_fake)

        elif model:
            path_A = os.path.join(self.model_dir, str(package['Epoch']) + '_' + 'A.pt')
            path_G = os.path.join(self.model_dir, str(package['Epoch']) + '_' + 'G.pt')
            torch.save(package['A_state_dict'], path_A)
            torch.save(package['G_state_dict'], path_G)

    def save_weight_figure(self, weight,  epoch, init_weight=1.0):
        weight = np.array(weight)
        delta = max(abs(weight.max()), abs(weight.min()))
        # print("{}, {}\n".format(epoch, mean_arr))
        plt.figure(figsize=[9.6, 7.2])
        # plt.axhline(y=init_weight, linestyle='--', color='k')

        plt.axis([0, len(weight) + 1, 0, 2])
        for i in range(11):
            plt.axhline(y=0.2 * i, c='gray', alpha=0.3)
        plt.xlabel('Residual index')
        plt.xticks(range(len(weight) + 1))
        plt.ylabel('Weight per residual block')
        plt.yticks([0.2 * i for i in range(11)])
        plt.plot(range(1, len(weight) + 1), weight, linestyle='-', marker='^', color='r')
        plt.savefig(os.path.join(self.model_dir, 'Epoch_{}_weights.png'.format(epoch)))
        plt.close()

        size = abs(weight).sum()
        fraction_arr = weight / size
        delta = max(abs(fraction_arr.max()), abs(fraction_arr.min()))
        plt.figure(figsize=[9.6, 7.2])
        plt.xlabel('Residual index')
        plt.ylabel('Fraction over final feature')
        plt.xticks(range(len(fraction_arr) + 1))
        plt.axhline(y=0, linestyle='--', color='k')
        plt.axis([0, len(fraction_arr) + 1, -delta * 1.2, delta * 1.2])
        plt.plot(range(1, len(fraction_arr) + 1), fraction_arr, linestyle=':', marker='_', color='b')
        for i in range(len(fraction_arr)):
            plt.vlines(i + 1, min(0, fraction_arr[i]), max(0, fraction_arr[i]), linestyles='-', colors='b')
        plt.savefig(os.path.join(self.model_dir, 'Epoch_{}_fractions.png'.format(epoch)))
        plt.close()
        with open(self.weight_log, 'a') as f:
            f.write("{}, {:.{prec}}, {:.{prec}}, {:.{prec}}, {:.{prec}}, {:.{prec}}, {:.{prec}}, {:.{prec}}, {:.{prec}},"
                    " {:.{prec}}\n".format(epoch, *weight, prec=6))
            f.close()

    def __call__(self, package):
        if package['Current_step'] % self.display_freq == 0:
            self.save(package, image=True)

        if package['Current_step'] % self.report_freq == 0:
            self.report_loss(package)

        if package['Current_step'] % self.save_freq == 0:
            self.save(package, model=True)


def update_lr(old_lr, n_epoch_decay, D_optim, G_optim):
    delta_lr = old_lr/n_epoch_decay
    new_lr = old_lr - delta_lr

    for param_group in D_optim.param_groups:
        param_group['lr'] = new_lr

    for param_group in G_optim.param_groups:
        param_group['lr'] = new_lr

    print("Learning rate has been updated from {} to {}.".format(old_lr, new_lr))

    return new_lr
