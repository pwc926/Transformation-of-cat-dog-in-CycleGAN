from model import *
from dataset import *

import itertools
from statistics import mean

import torch
import torch.nn as nn

from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from torch.optim import lr_scheduler
from tqdm import tqdm


class Train:
    def __init__(self, args):
        self.mode = args.mode
        self.train_continue = args.train_continue

        self.dir_checkpoint = args.dir_checkpoint
        self.dir_log = args.dir_log

        self.dir_data = args.dir_data
        self.dir_result = args.dir_result

        self.num_epoch = args.num_epoch
        self.batch_size = args.batch_size

        self.lr_G = args.lr_G
        self.lr_D = args.lr_D

        self.wgt_c_a = args.wgt_c_a
        self.wgt_c_b = args.wgt_c_b
        self.wgt_i = args.wgt_i

        self.optimizer = args.optimizer
        self.beta1 = args.beta1

        self.ny_in = args.ny_in
        self.nx_in = args.nx_in
        self.nch_in = args.nch_in

        self.ny_load = args.ny_load
        self.nx_load = args.nx_load
        self.nch_load = args.nch_load

        self.ny_out = args.ny_out
        self.nx_out = args.nx_out
        self.nch_out = args.nch_out

        self.nch_ker = args.nch_ker

        self.data_type = args.data_type
        self.norm = args.norm

        self.num_freq_disp = args.num_freq_disp
        self.num_freq_save = args.num_freq_save

        self.direction = args.direction
        self.name_data = args.name_data

        self.n_blocks = args.n_blocks

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    def save(self, dir_chck, netG_a2b, netG_b2a, netD_a, netD_b, optimizer_G , optimizer_D, epoch):
        if not os.path.exists(dir_chck):
            os.makedirs(dir_chck)

        torch.save({'netG_a2b': netG_a2b.state_dict(), 'netG_b2a': netG_b2a.state_dict(),
                    'netD_a': netD_a.state_dict(), 'netD_b': netD_b.state_dict(),
                    'optimizer_G ': optimizer_G .state_dict(), 'optimizer_D': optimizer_D.state_dict()},
                   '%s/model_epoch%04d.pth' % (dir_chck, epoch))

    def load(self, dir_chck, netG_a2b, netG_b2a, netD_a=[], netD_b=[], optimizer_G =[], optimizer_D=[], epoch=[], mode='train'):
        if not epoch:
            ckpt = os.listdir(dir_chck)
            ckpt.sort()
            epoch = int(ckpt[-1].split('epoch')[1].split('.pth')[0])

        dict_net = torch.load('%s/model_epoch%04d.pth' % (dir_chck, epoch))

        if mode == 'train':
            netG_a2b.load_state_dict(dict_net['netG_a2b'])
            netG_b2a.load_state_dict(dict_net['netG_b2a'])
            netD_a.load_state_dict(dict_net['netD_a'])
            netD_b.load_state_dict(dict_net['netD_b'])
            optimizer_G .load_state_dict(dict_net['optimizer_G '])
            optimizer_D.load_state_dict(dict_net['optimizer_D'])

            return netG_a2b, netG_b2a, netD_a, netD_b, optimizer_G , optimizer_D, epoch

        elif mode == 'test':
            netG_a2b.load_state_dict(dict_net['netG_a2b'])
            netG_b2a.load_state_dict(dict_net['netG_b2a'])

            return netG_a2b, netG_b2a, epoch

    def preprocess(self, data):
        normalize = Normalize()
        randflip = RandomFlip()
        rescale = Rescale((self.ny_load, self.nx_load))
        randomcrop = RandomCrop((self.ny_out, self.nx_out))
        totensor = ToTensor()
        return totensor(randomcrop(rescale(randflip(normalize(data)))))

    def deprocess(self, data):
        tonumpy = ToNumpy()
        denomalize = Denomalize()
        return denomalize(tonumpy(data))

    def train(self):
        mode = self.mode

        train_continue = self.train_continue
        num_epoch = self.num_epoch

        lr_G = self.lr_G
        lr_D = self.lr_D

        wgt_c_a = self.wgt_c_a
        wgt_c_b = self.wgt_c_b
        wgt_i = self.wgt_i

        batch_size = self.batch_size
        device = self.device

        nch_in = self.nch_in
        nch_out = self.nch_out
        nch_ker = self.nch_ker

        norm = self.norm
        name_data = self.name_data

        num_freq_disp = self.num_freq_disp
        num_freq_save = self.num_freq_save

        ## setup dataset
        dir_chck = os.path.join(self.dir_checkpoint, name_data)

        dir_data_train = os.path.join(self.dir_data, name_data, 'train')

        dir_log_train = os.path.join(self.dir_log, name_data, 'train')

        transform_train = transforms.Compose([Normalize(), RandomFlip(), Rescale((self.ny_load, self.nx_load)), RandomCrop((self.ny_in, self.nx_in)), ToTensor()])
        transform_inv = transforms.Compose([ToNumpy(), Denomalize()])

        dataset_train = Dataset(dir_data_train, direction=self.direction, data_type=self.data_type, transform=transform_train)

        loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8)

        num_train = len(dataset_train)

        num_batch_train = int((num_train / batch_size) + ((num_train % batch_size) != 0))

        ## setup network
        netG_a2b = ResNet(nch_in, nch_out, nch_ker, norm, n_blocks=self.n_blocks).to(device)
        netG_b2a = ResNet(nch_in, nch_out, nch_ker, norm, n_blocks=self.n_blocks).to(device)

        netD_a = Discriminator(nch_in, nch_ker, norm).to(device)
        netD_b = Discriminator(nch_in, nch_ker, norm).to(device)
        
        init_net(netG_a2b, init_type='normal', init_gain=0.02)
        init_net(netG_b2a, init_type='normal', init_gain=0.02)

        init_net(netD_a, init_type='normal', init_gain=0.02)
        init_net(netD_b, init_type='normal', init_gain=0.02)

        # Print model structures
        print("Generator A2B structure:")
        print(netG_a2b)
        print("\nGenerator B2A structure:")
        print(netG_b2a)
        print("\nDiscriminator A structure:")
        print(netD_a)
        print("\nDiscriminator B structure:")
        print(netD_b)

        ## setup loss & optimization
        fn_Cycle = nn.L1Loss().to(device)   # L1
        fn_GAN = GANLoss().to(device)
        fn_Ident = nn.L1Loss().to(device)   # L1

        paramsG_a2b = netG_a2b.parameters()
        paramsG_b2a = netG_b2a.parameters()
        paramsD_a = netD_a.parameters()
        paramsD_b = netD_b.parameters()

        optimizer_G  = torch.optim.Adam(itertools.chain(paramsG_a2b, paramsG_b2a), lr=lr_G, betas=(self.beta1, 0.999))
        optimizer_D = torch.optim.Adam(itertools.chain(paramsD_a, paramsD_b), lr=lr_D, betas=(self.beta1, 0.999))


        # schedG = torch.optim.lr_scheduler.ExponentialLR(optimizer_G , gamma=0.9)
        # schedD = torch.optim.lr_scheduler.ExponentialLR(optimizer_D, gamma=0.9)

        ## load from checkpoints
        st_epoch = 0

        if train_continue == 'on':
            netG_a2b, netG_b2a, netD_a, netD_b, optimizer_G , optimizer_D, st_epoch = \
                self.load(dir_chck, netG_a2b, netG_b2a, netD_a, netD_b, optimizer_G , optimizer_D, mode=mode)
            # Move loaded models to device
            netG_a2b = netG_a2b.to(device)
            netG_b2a = netG_b2a.to(device)
            netD_a = netD_a.to(device)
            netD_b = netD_b.to(device)

        ## setup tensorboard
        writer_train = SummaryWriter(log_dir=dir_log_train)

        for epoch in range(st_epoch + 1, num_epoch + 1):
            ## training phase
            netG_a2b.train()
            netG_b2a.train()
            netD_a.train()
            netD_b.train()

            loss_G_a2b_train = []
            loss_G_b2a_train = []
            loss_D_a_train = []
            loss_D_b_train = []
            loss_C_a_train = []
            loss_C_b_train = []
            loss_I_a_train = []
            loss_I_b_train = []

            pbar = tqdm(enumerate(loader_train, 1), total=num_batch_train, desc=f"Epoch {epoch}/{num_epoch}", ncols=120)
            for i, data in pbar:
                def should(freq):
                    return freq > 0 and (i % freq == 0 or i == num_batch_train)

                input_a = data['dataA'].to(device)
                input_b = data['dataB'].to(device)

                # forward netG
                output_b = netG_a2b(input_a)
                output_a = netG_b2a(input_b)

                recon_b = netG_a2b(output_a)
                recon_a = netG_b2a(output_b)

                # backward netD
                set_requires_grad([netD_a, netD_b], True)
                optimizer_D.zero_grad()

                # backward netD_a
                pred_real_a = netD_a(input_a)
                pred_fake_a = netD_a(output_a.detach())

                loss_D_a_real = fn_GAN(pred_real_a, True)
                loss_D_a_fake = fn_GAN(pred_fake_a, False)
                loss_D_a = 0.5 * (loss_D_a_real + loss_D_a_fake)

                # backward netD_b
                pred_real_b = netD_b(input_b)
                pred_fake_b = netD_b(output_b.detach())

                loss_D_b_real = fn_GAN(pred_real_b, True)
                loss_D_b_fake = fn_GAN(pred_fake_b, False)
                loss_D_b = 0.5 * (loss_D_b_real + loss_D_b_fake)

                # backward netD
                loss_D = loss_D_a + loss_D_b
                loss_D.backward()
                optimizer_D.step()

                # backward netG
                set_requires_grad([netD_a, netD_b], False)
                optimizer_G .zero_grad()

                if wgt_i > 0:
                    ident_b = netG_a2b(input_b)
                    ident_a = netG_b2a(input_a)

                    loss_I_a = fn_Ident(ident_a, input_a)
                    loss_I_b = fn_Ident(ident_b, input_b)
                else:
                    loss_I_a = 0
                    loss_I_b = 0

                pred_fake_a = netD_a(output_a)
                pred_fake_b = netD_b(output_b)

                loss_G_a2b = fn_GAN(pred_fake_b, True)
                loss_G_b2a = fn_GAN(pred_fake_a, True)

                loss_C_a = fn_Cycle(input_a, recon_a)
                loss_C_b = fn_Cycle(input_b, recon_b)

                loss_G = (loss_G_a2b + loss_G_b2a) + \
                         (wgt_c_a * loss_C_a + wgt_c_b * loss_C_b) + \
                         (wgt_c_a * loss_I_a + wgt_c_b * loss_I_b) * wgt_i

                loss_G.backward()
                optimizer_G .step()

                # get losses
                loss_G_a2b_train += [loss_G_a2b.item()]
                loss_G_b2a_train += [loss_G_b2a.item()]

                loss_D_a_train += [loss_D_a.item()]
                loss_D_b_train += [loss_D_b.item()]

                loss_C_a_train += [loss_C_a.item()]
                loss_C_b_train += [loss_C_b.item()]

                if wgt_i > 0:
                    loss_I_a_train += [loss_I_a.item()]
                    loss_I_b_train += [loss_I_b.item()]

                # Update tqdm bar with current batch loss (optional, can be commented out)
                pbar.set_postfix({
                    'G_a2b': f"{loss_G_a2b.item():.4f}",
                    'G_b2a': f"{loss_G_b2a.item():.4f}",
                    'D_a': f"{loss_D_a.item():.4f}",
                    'D_b': f"{loss_D_b.item():.4f}"
                })

                if should(num_freq_disp):
                    ## show output
                    input_a = transform_inv(input_a)
                    output_a = transform_inv(output_a)
                    input_b = transform_inv(input_b)
                    output_b = transform_inv(output_b)

                    writer_train.add_images('input_a', input_a, num_batch_train * (epoch - 1) + i, dataformats='NHWC')
                    writer_train.add_images('output_a', output_a, num_batch_train * (epoch - 1) + i, dataformats='NHWC')
                    writer_train.add_images('input_b', input_b, num_batch_train * (epoch - 1) + i, dataformats='NHWC')
                    writer_train.add_images('output_b', output_b, num_batch_train * (epoch - 1) + i, dataformats='NHWC')

            # Print average losses after each epoch
            print('TRAIN: EPOCH %d: '
                  'G_a2b: %.4f G_b2a: %.4f D_a: %.4f D_b: %.4f C_a: %.4f C_b: %.4f I_a: %.4f I_b: %.4f'
                  % (epoch,
                     mean(loss_G_a2b_train), mean(loss_G_b2a_train),
                     mean(loss_D_a_train), mean(loss_D_b_train),
                     mean(loss_C_a_train), mean(loss_C_b_train),
                     mean(loss_I_a_train), mean(loss_I_b_train)))

            writer_train.add_scalar('loss_G_a2b', mean(loss_G_a2b_train), epoch)
            writer_train.add_scalar('loss_G_b2a', mean(loss_G_b2a_train), epoch)
            writer_train.add_scalar('loss_D_a', mean(loss_D_a_train), epoch)
            writer_train.add_scalar('loss_D_b', mean(loss_D_b_train), epoch)
            writer_train.add_scalar('loss_C_a', mean(loss_C_a_train), epoch)
            writer_train.add_scalar('loss_C_b', mean(loss_C_b_train), epoch)
            writer_train.add_scalar('loss_I_a', mean(loss_I_a_train), epoch)
            writer_train.add_scalar('loss_I_b', mean(loss_I_b_train), epoch)

            # # update schduler
            # # schedG.step()
            # # schedD.step()

            ## save
            if (epoch % num_freq_save) == 0:
                self.save(dir_chck, netG_a2b, netG_b2a, netD_a, netD_b, optimizer_G , optimizer_D, epoch)

        writer_train.close()

    def test(self):
        mode = self.mode

        batch_size = self.batch_size
        device = self.device

        nch_in = self.nch_in
        nch_out = self.nch_out
        nch_ker = self.nch_ker

        norm = self.norm

        name_data = self.name_data

        ## setup dataset
        dir_chck = os.path.join(self.dir_checkpoint, name_data)

        dir_result = os.path.join(self.dir_result, name_data)
        if not os.path.exists(dir_result):
            os.makedirs(dir_result)

        dir_data_test = os.path.join(self.dir_data, self.name_data, 'test')

        transform_test = transforms.Compose([Normalize(), ToTensor()])
        transform_inv = transforms.Compose([ToNumpy(), Denomalize()])

        dataset_test = Dataset(dir_data_test, data_type=self.data_type, transform=transform_test)

        loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=8)

        num_test = len(dataset_test)
        num_batch_test = int((num_test / batch_size) + ((num_test % batch_size) != 0))

        ## setup network
        netG_a2b = ResNet(nch_in, nch_out, nch_ker, norm, n_blocks=self.n_blocks).to(device)
        netG_b2a = ResNet(nch_in, nch_out, nch_ker, norm, n_blocks=self.n_blocks).to(device)

        init_net(netG_a2b, init_type='normal', init_gain=0.02)
        init_net(netG_b2a, init_type='normal', init_gain=0.02)

        ## load from checkpoints
        st_epoch = 0

        netG_a2b, netG_b2a, st_epoch = self.load(dir_chck, netG_a2b, netG_b2a, mode=mode)
        # Move loaded models to device
        netG_a2b = netG_a2b.to(device)
        netG_b2a = netG_b2a.to(device)

        ## test phase
        pdf_path = os.path.join(dir_result, "results.pdf")
        with PdfPages(pdf_path) as pdf:
            with torch.no_grad():
                netG_a2b.eval()
                netG_b2a.eval()

                img_idx = 0
                for i, data in enumerate(loader_test, 1):
                    input_a = data['dataA'].to(device)
                    input_b = data['dataB'].to(device)

                    # forward netG
                    output_b = netG_a2b(input_a)
                    output_a = netG_b2a(input_b)

                    # Only show real and fake images, align them side by side
                    # For A2B: real A, fake B
                    real_a = transform_inv(input_a)
                    fake_b = transform_inv(output_b)

                    # For B2A: real B, fake A
                    real_b = transform_inv(input_b)
                    fake_a = transform_inv(output_a)

                    # Save each pair in the PDF and as PNGs
                    for j in range(real_a.shape[0]):
                        # Save images as PNG
                        plt.imsave(os.path.join(dir_result, f"{img_idx:05d}_realA.png"), real_a[j].squeeze())
                        plt.imsave(os.path.join(dir_result, f"{img_idx:05d}_fakeB.png"), fake_b[j].squeeze())
                        plt.imsave(os.path.join(dir_result, f"{img_idx:05d}_realB.png"), real_b[j].squeeze())
                        plt.imsave(os.path.join(dir_result, f"{img_idx:05d}_fakeA.png"), fake_a[j].squeeze())

                        # Save to PDF
                        fig, axes = plt.subplots(2, 2, figsize=(6, 6))
                        axes[0, 0].imshow(real_a[j].squeeze())
                        axes[0, 0].set_title("Real A")
                        axes[0, 0].axis('off')
                        axes[0, 1].imshow(fake_b[j].squeeze())
                        axes[0, 1].set_title("Fake B")
                        axes[0, 1].axis('off')
                        axes[1, 0].imshow(real_b[j].squeeze())
                        axes[1, 0].set_title("Real B")
                        axes[1, 0].axis('off')
                        axes[1, 1].imshow(fake_a[j].squeeze())
                        axes[1, 1].set_title("Fake A")
                        axes[1, 1].axis('off')
                        plt.tight_layout()
                        pdf.savefig(fig)
                        plt.close(fig)
                        img_idx += 1


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def add_plot(output, label, writer, epoch=[], ylabel='Density', xlabel='Radius', namescope=[]):
    fig, ax = plt.subplots()

    ax.plot(output.transpose(1, 0).detach().numpy(), '-')
    ax.plot(label.transpose(1, 0).detach().numpy(), '--')

    ax.set_xlim(0, 400)

    ax.grid(True)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    writer.add_figure(namescope, fig, epoch)
