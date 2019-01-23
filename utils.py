import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image


def configure(opt):
    opt.USE_CUDA = True if opt.gpu_ids != -1 else False

    opt.format = 'png'
    opt.n_df = 64
    if opt.dataset_name == 'Cityscapes':
        if opt.use_boundary_map:
            opt.input_ch = 36

        else:
            opt.input_ch = 35

        if opt.image_height == 512:
            opt.half = True
            opt.image_size = (512, 1024)
            opt.n_gf = 64
        elif opt.image_height == 1024:
            opt.half = False
            opt.image_size = (1024, 2048)
            opt.n_gf = 32
        opt.max_lod = opt.n_downsample
        opt.min_image_size = (2 ** (np.log2(opt.image_size[0]) - opt.max_lod),
                              2 ** (np.log2(opt.image_size[1]) - opt.max_lod))
        opt.n_data = 2975
        opt.output_ch = 3

    elif opt.dataset_name == 'Custom':
        opt.input_ch = 1

        if opt.image_height == 512:
            opt.half = True
            opt.image_size = (512, 512)
            opt.n_gf = 64
        elif opt.image_height == 1024:
            opt.half = False
            opt.image_size = (1024, 1024)
            opt.n_gf = 32
        opt.max_lod = opt.n_downsample
        opt.min_image_size = (2 ** (np.log2(opt.image_size[0]) - opt.max_lod),
                              2 ** (np.log2(opt.image_size[1]) - opt.max_lod))
        opt.output_ch = 1

    else:
        raise NotImplementedError("Please check dataset_name. It should be in ['Cityscapes', 'Custom'].")

    dataset_name = opt.dataset_name
    model_name = model_namer(opt.trans_unit, opt.image_height, opt.GAN_type, opt.norm_type, A_norm=opt.C_norm,
                             condition=opt.C_condition)
    make_dir(dataset_name, model_name, type='checkpoints')

    if opt.is_train:
        opt.image_dir = os.path.join('./checkpoints', dataset_name, 'Image/Training', model_name)

    elif not opt.is_train:
        opt.image_dir = os.path.join('./checkpoints', dataset_name, 'Image/Test', model_name)

    opt.model_dir = os.path.join('./checkpoints', dataset_name, 'Model', model_name)
    log_path = os.path.join('./checkpoints/', dataset_name, 'Model', model_name, 'opt.txt')

    if os.path.isfile(log_path) and not opt.debug:
        permission = input(
            "{} log already exists. Do you really want to overwrite this log? Y/N. : ".format(model_name + '/opt'))
        if permission == 'Y':
            pass

        else:
            raise NotImplementedError("Please check {}".format(log_path))

    if opt.debug:
        opt.display_freq = 1
        opt.n_epochs = 4
        opt.report_freq = 1
        opt.save_freq = 4

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


def make_dir(dataset_name=None, model_name=None, type='checkpoints'):
    assert dataset_name in ['Cityscapes']
    if type == 'checkpoints':
        assert model_name, "model_name keyword should be specified for type='checkpoints'"
        os.makedirs(os.path.join('./checkpoints', dataset_name, 'Image', 'Training', model_name), exist_ok=True)
        os.makedirs(os.path.join('./checkpoints', dataset_name, 'Image', 'Test', model_name), exist_ok=True)
        os.makedirs(os.path.join('./checkpoints', dataset_name, 'Model', model_name), exist_ok=True)

    else:
        """
        for other type of directory
        """
        pass


def model_namer(*elements, **k_elements):
    name = ''

    for k, v in sorted(k_elements.items()):
        name += str(k) + '_' + str(v) + '_'

    for v in elements:
        name += str(v) + '_'

    name = name.strip('_')

    return name


class Manager(object):
    def __init__(self, opt):
        self.GAN_type = opt.GAN_type
        self.model_dir = opt.model_dir
        self.log = os.path.join(self.model_dir, 'log.txt')

        if self.GAN_type == 'LSGAN':
            with open(self.log, 'wt') as log:
                log.write('Level, Current_step, D_loss, G_loss, FM_loss\n')

        elif self.GAN_type == 'WGAN_GP':
            with open(self.log, 'wt') as log:
                log.write('Level, Current_step, C_loss, G_loss, GP_loss, FM_loss\n')

        else:
            raise NotImplementedError

        self.display_freq = opt.display_freq
        self.image_dir = opt.image_dir
        self.image_mode = opt.image_mode
        self.report_freq = opt.report_freq
        self.save_freq = opt.save_freq

    def report_loss(self, package):
        if self.GAN_type == 'LSGAN':
            inf = [package['Level'], package['Current_step'], package['A_loss'].detach().item(),
                   package['G_loss'].detach().item(), package['FM']]

            print("Level: {} Current_step: {} A_loss: {:.{prec}}  G_loss: {:.{prec}} FM: {:.{prec}}".
                  format(*inf, prec=4))

            with open(self.log, 'a') as log:
                log.write('{}, {}, {:.{prec}}, {:.{prec}, {:.{prec}}\n'.format(*inf, prec=4))

        elif self.GAN_type == 'WGAN_GP':
            inf = [package['Level'], package['Current_step'], package['A_loss'].detach().item(),
                   package['G_loss'].detach().item(), package['GP'], package['FM']]
            print("Level: {} Current_step: {} A_loss: {:.{prec}}  G_loss: {:.{prec}} GP: {:.{prec}} FM: {:.{prec}}".
                  format(*inf, prec=4))

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

    def tensor2image(self, image_tensor):
        np_image = image_tensor[0].cpu().float().numpy()
        # assert np_image.shape[0] in [1, 3], print("The channel is ", np_image.shape)
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
            path_real = os.path.join(self.image_dir, str(package['Current_step']) + '_' + 'real.png')
            path_fake = os.path.join(self.image_dir, str(package['Current_step']) + '_' + 'fake.png')
            self.save_image(package['target_tensor'], path_real)
            self.save_image(package['generated_tensor'], path_fake)

        elif model:
            path_A = os.path.join(self.model_dir, str(package['Current_step']) + '_' + 'A.pt')
            path_G = os.path.join(self.model_dir, str(package['Current_step']) + '_' + 'G.pt')
            torch.save(package['A_state_dict'], path_A)
            torch.save(package['G_state_dict'], path_G)

    def __call__(self, package):
        if package['Current_step'] % self.display_freq == 0:
            self.save(package, image=True)

        if package['Current_step'] % self.report_freq == 0:
            self.report_loss(package)

        if package['Current_step'] % self.save_freq == 0:
            self.save(package, model=True)


class PixelNorm(nn.Module):
    def __init__(self, eps=1e-8):
        super(PixelNorm, self).__init__()
        self.__name__ = 'PixelNorm'
        self.eps = eps

    def forward(self, x):
        x = x * torch.rsqrt(torch.mean(x.pow(2), dim=1, keepdim=True) + self.eps)

        return x
    
