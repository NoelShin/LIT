if __name__ == '__main__':
    import os
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    import torch
    from models import Critic as Adversarial
    from option import TrainOption
    from pipeline import CustomDataset
    from utils import Manager, update_lr
    import datetime

    torch.backends.cudnn.benchmark = True

    opt = TrainOption().parse()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu_ids)
    if opt.progression:
        if opt.U_net:
            if opt.U_net_gate:
                from models import ProgressiveGatedUNetGenerater as Generator
            else:
                from models import ProgressiveUNetGenerator as Generator
        else:
            from models import ProgressiveGenerator as Generator
    else:
        if opt.U_net:
            if opt.U_net_gate:
                from models import GatedUNetGenerator as Generator

            else:
                from models import UNetGenerator as Generator
        else:
            from models import Generator

    if opt.GAN_type == 'LSGAN':
        from loss import LSGANLoss as Loss
    elif opt.GAN_type == 'WGAN_GP':
        from loss import WGANGPLoss as Loss

    USE_CUDA = opt.USE_CUDA

    G = Generator(opt)
    A = Adversarial(opt)

    criterion = Loss(opt)

    G_optim = torch.optim.Adam(G.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2), eps=opt.eps)
    A_optim = torch.optim.Adam(A.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2), eps=opt.eps)

    manager = Manager(opt)

    current_step = 0
    lr = opt.lr
    package = {}
    start_time = datetime.datetime.now()
    if opt.progression:
        for lod in range(opt.n_downsample + 1):
            dataset = CustomDataset(opt, lod=lod)
            data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                      batch_size=opt.batch_size,
                                                      num_workers=opt.n_workers,
                                                      shuffle=opt.shuffle)
            for epoch in range(opt.n_epochs_per_lod):
                package.update({'Epoch': epoch})
                for _, data_dict in enumerate(data_loader):
                    current_step += 1

                    if USE_CUDA:
                        device = torch.device('cuda', 0)
                        for k, v in data_dict.items():
                            data_dict.update({k: v.to(device)})

                    package.update(criterion(A, G, data_dict, lod))
                    A_optim.zero_grad()
                    package['A_loss'].backward()
                    A_optim.step()

                    G_optim.zero_grad()
                    package['G_loss'].backward()
                    G_optim.step()

                    package.update({'Level': lod, 'Current_step': current_step})

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
            package.update({'Epoch': epoch})
            for _, data_dict in enumerate(data_loader):
                time = datetime.datetime.now()
                current_step += 1
                if USE_CUDA:
                    device = torch.device('cuda', 0)
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
