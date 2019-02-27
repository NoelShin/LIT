if __name__ == '__main__':
    import os
    from functools import partial
    import torch
    import numpy as np
    from option import TrainOption
    from pipeline import CustomDataset
    from utils import init_weights, Manager, update_lr
    import datetime
    opt = TrainOption().parse()
    if opt.progression:
        from models import ProgressiveGenerator as Generator
        from models import ProgressivePatchCritic as Adversarial

    else:
        from models import Generator
        from models import Critic as Adversarial

    if opt.GAN_type == 'LSGAN':
        from loss import LSGANLoss as Loss
    elif opt.GAN_type == 'WGAN_GP':
        from loss import WGANGPLoss as Loss

    USE_CUDA = opt.USE_CUDA
    torch.backends.cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu_ids)
    device = torch.device('cuda' if USE_CUDA else 'cpu', 0)

    G = Generator(opt).apply(partial(init_weights, type=opt.init_type, mode=opt.fan_mode,
                                     negative_slope=opt.negative_slope, nonlinearity=opt.G_act))
    G.to(device)
    print(G, "the number of G parameters: ", sum(p.numel() for p in G.parameters() if p.requires_grad))

    A = Adversarial(opt).apply(partial(init_weights, type=opt.init_type, mode=opt.fan_mode,
                                       negative_slope=opt.negative_slope, nonlinearity=opt.C_act))
    A.to(device)
    print(A, "the number of A parameters: ", sum(p.numel() for p in A.parameters() if p.requires_grad))

    criterion = Loss(opt)

    G_optim = torch.optim.Adam(G.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2), eps=opt.eps)
    A_optim = torch.optim.Adam(A.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2), eps=opt.eps)

    manager = Manager(opt)

    current_step = 0
    lr = opt.lr
    n_epochs_per_lod = opt.n_epochs_per_lod
    n_iter_per_lod = n_epochs_per_lod * opt.n_data
    nb_transition = n_iter_per_lod / 2
    package = {}
    start_time = datetime.datetime.now()
    if opt.progression:
        for level in range(opt.n_downsample + 1):  # 0 1 2 3 4 5
            level_in = level
            dataset = CustomDataset(opt, level=level)
            data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                      batch_size=opt.batch_size,
                                                      num_workers=opt.n_workers,
                                                      shuffle=opt.shuffle)
            for epoch in range(n_epochs_per_lod):
                package.update({'Epoch': epoch + 1})
                for _, data_dict in enumerate(data_loader):
                    current_step += 1

                    level_in += 1 / nb_transition
                    level_in = np.clip(level_in, level, level + 1.0)

                    if USE_CUDA:
                        for k, v in data_dict.items():
                            data_dict.update({k: v.to(device)})

                    package.update(criterion(A, G, data_dict, level=level, level_in=level_in))
                    A_optim.zero_grad()
                    package['A_loss'].backward()
                    A_optim.step()

                    G_optim.zero_grad()
                    package['G_loss'].backward()
                    G_optim.step()

                    package.update({'Level': level, 'Current_step': current_step})

                    manager(package)

                    if opt.debug:
                        break

                if opt.debug:
                    break

    else:
        dataset = CustomDataset(opt)
        data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                  batch_size=opt.batch_size,
                                                  num_workers=opt.n_workers,
                                                  shuffle=opt.shuffle)
        for epoch in range(opt.n_epochs):
            package.update({'Epoch': epoch + 1})
            for _, data_dict in enumerate(data_loader):
                time = datetime.datetime.now()
                current_step += 1

                if USE_CUDA:
                    for k, v in data_dict.items():
                        data_dict.update({k: v.to(device)})

                package.update(criterion(A, G, data_dict))
                A_optim.zero_grad()
                package['A_loss'].backward()
                A_optim.step()

                G_optim.zero_grad()
                package['G_loss'].backward()
                G_optim.step()

                package.update({'Current_step': current_step, 'running_time': datetime.datetime.now() - time})

                manager(package)

                if opt.debug:
                    break

            if epoch > opt.epoch_decay:
                lr = update_lr(lr, opt.n_epochs - opt.epoch_decay, A_optim, G_optim)

    print("Total time taken: ", datetime.datetime.now() - start_time)
