if __name__ == '__main__':
    import os
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    import torch
    from option import TrainOption
    from pipeline import CustomDataset
    from utils import Manager
    import datetime

    torch.backends.cudnn.benchmark = True

    opt = TrainOption().parse()
    if opt.GAN_type == 'LSGAN':
        from networks import PatchCritic as Adversarial, Generator
        from loss import LSGANLoss as Loss

    elif opt.GAN_type == 'WGAN_GP':
        from networks import PatchCritic as Adversarial, Generator
        from loss import WGANGPLoss as Loss

    USE_CUDA = opt.USE_CUDA

    G = Generator(opt)
    A = Adversarial(opt)

    criterion = Loss(opt)

    G_optim = torch.optim.Adam(G.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2), eps=opt.eps)
    A_optim = torch.optim.Adam(A.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2), eps=opt.eps)

    manager = Manager(opt)

    current_step = 0
    nb_transition = opt.n_data * opt.n_epochs / 2
    total_step = opt.n_data * opt.n_epochs * opt.max_lod
    start_time = datetime.datetime.now()
    for lod in range(opt.max_lod + 1):
        # lod_in = lod
        dataset = CustomDataset(opt, lod)
        data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                  batch_size=opt.batch_size,
                                                  num_workers=opt.n_workers,
                                                  shuffle=opt.shuffle)
        for epoch in range(opt.n_epochs):
            for _, data_dict in enumerate(data_loader):
                current_step += 1
                # lod_in += 1. / nb_transition
                # lod_in = np.clip(lod_in, lod, lod + 1.0)

                if USE_CUDA:
                    device = torch.device('cuda', opt.gpu_ids)
                    for k, v in data_dict.items():
                        data_dict.update({k: v.to(device)})

                A_loss, G_loss, generated_tensor = criterion(A, G, lod, data_dict)
                A_optim.zero_grad()
                A_loss.backward()
                A_optim.step()

                G_optim.zero_grad()
                G_loss.backward()
                G_optim.step()

                package = {'lod': lod,
                           'current_step': current_step,
                           'total_step': total_step,
                           'A_loss': A_loss.detach().item(),
                           'G_loss': G_loss.detach().item(),
                           'A_state_dict': A.state_dict(),
                           'G_state_dict': G.state_dict(),
                           'target_tensor': data_dict['target_tensor'],
                           'generated_tensor': generated_tensor.detach()}

                manager(package)

                if opt.debug:
                    break

    print("Total time taken: ", datetime.datetime.now() - start_time)
