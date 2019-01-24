if __name__ == '__main__':
    import os
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    import torch
    from networks import Critic as Adversarial, Generator
    from option import TrainOption
    from pipeline import CustomDataset
    from utils import Manager
    import datetime

    torch.backends.cudnn.benchmark = True

    opt = TrainOption().parse()
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
    #nb_transition = opt.n_data * opt.n_epochs / 2
    start_time = datetime.datetime.now()
    if opt.progression:
        total_step = opt.n_data * opt.n_epochs * opt.max_lod
        for lod in range(opt.max_lod + 1):
            dataset = CustomDataset(opt, lod)
            data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                      batch_size=opt.batch_size,
                                                      num_workers=opt.n_workers,
                                                      shuffle=opt.shuffle)
            for epoch in range(opt.n_epochs):
                for _, data_dict in enumerate(data_loader):

                    if USE_CUDA:
                        device = torch.device('cuda', opt.gpu_ids)
                        for k, v in data_dict.items():
                            data_dict.update({k: v.to(device)})

                    package = criterion(A, G, lod, data_dict)
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

    else:
        dataset = CustomDataset(opt)
        data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                  batch_size=opt.batch_size,
                                                  num_workers=opt.n_workers,
                                                  shuffle=opt.shuffle)
        for epoch in range(opt.n_epochs):
            for _, data_dict in enumerate(data_loader):

                if USE_CUDA:
                    device = torch.device('cuda', opt.gpu_ids)
                    for k, v in data_dict.items():
                        data_dict.update({k: v.to(device)})

                package = criterion(A, G, data_dict)
                A_optim.zero_grad()
                package['A_loss'].backward()
                A_optim.step()

                G_optim.zero_grad()
                package['G_loss'].backward()
                G_optim.step()

                package.update({'Current_step': current_step})

                manager(package)

                if opt.debug:
                    break

    print("Total time taken: ", datetime.datetime.now() - start_time)
    
