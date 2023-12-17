import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.utils as vutils

from utils import loss

from .transforms import GaussianBlur, RandomAffine


def get_method(method):
    if method == 'original':
        return OriginalTarget
    elif method == 'augment':
        return AugmentTarget
    elif method == 'generate':
        return GenerateTarget
    else:
        return None


def ova_loss_unk(out_open):
    assert len(out_open.size()) == 3
    assert out_open.size(1) == 2

    out_open = F.softmax(out_open, 1)
    label_n = torch.ones((out_open.size(0),
                          out_open.size(2))).long().cuda()
    open_loss_neg = torch.mean(torch.mean(-torch.log(out_open[:, 0, :] +
                                                1e-8) * label_n, 1))

    return open_loss_neg


class Generator(nn.Module):
    def __init__(self, nz, ngf, nc):
        super().__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.zeros_(m.bias)

    def forward(self, input):
        output = self.main(input)
        return output


class Discriminator(nn.Module):
    def __init__(self, nc, ndf):
        super().__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf * 2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 32 x 32
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 16 x 16
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 8 x 8
            nn.Conv2d(ndf * 8, ndf * 16, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(ndf * 16, 1),
            nn.Sigmoid()
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.zeros_(m.bias)

    def forward(self, input):
        output = self.main(input)
        output = self.avgpool(output)
        output = torch.flatten(output, 1)
        output = self.classifier(output).flatten()
        return output


class TargetTransfer:

    def __init__(self, G, Cs, target_loader, entropy=False, thr=0.9):
        self.unknown_target_data = self._make_dataset(
            G,
            Cs,
            target_loader,
            entropy,
            thr)

    def _make_dataset(self, G, Cs, target_loader, entropy=False, thr=0.9):
        G.eval()
        for c in Cs:
            c.eval()
        selected_idx = []
        for batch_idx, data in enumerate(target_loader):
            with torch.no_grad():
                img_t, idx_t = data[0].cuda(), data[2]
                feat_t = G(img_t)
                out_t = Cs[0](feat_t)
                pred = out_t.data.max(1)[1]
                out_t = F.softmax(out_t, 1)
                if entropy:
                    pred_unk = -torch.sum(out_t * torch.log(out_t), 1)
                    ind_unk = np.where(pred_unk.data.cpu().numpy() > thr)[0]
                else:
                    out_open = Cs[1](feat_t)
                    out_open = F.softmax(out_open.view(out_t.size(0), 2, -1),1)
                tmp_range = torch.arange(0, out_t.size(0)).long()
                pred_unk = out_open[tmp_range, 0, pred]
                ind_unk = np.where(pred_unk.data.cpu().numpy() > thr)[0]
            selected_idx.extend(idx_t[ind_unk].data.cpu().numpy())

        return torch.utils.data.Subset(target_loader.dataset, selected_idx)

    def __call__(self):
        raise NotImplementedError


class OriginalTarget(TargetTransfer):

    def __init__(self, G, Cs, target_loader, entropy=False, thr=0.9):
        super().__init__(G, Cs, target_loader, entropy, thr)
        self.loader = torch.utils.data.DataLoader(
            self.unknown_target_data,
            batch_size=target_loader.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=4)
        self.iter = iter(self.loader)

    def __call__(self):
        try:
            batch = next(self.iter)
        except:
            self.iter = iter(self.loader)
            batch = next(self.iter)
        return batch[0]


class AugmentTarget(OriginalTarget):

    def __init__(self, G, Cs, target_loader, entropy=False, thr=0.9, **kwargs):
        super().__init__(G, Cs, target_loader, entropy, thr)
        self.transform = transforms.Compose([
            RandomAffine(0.0, 0.1),
            GaussianBlur(0.1)
        ]) if 'transform' not in kwargs else kwargs.pop('transform')

    def __call__(self):
        batch = super().__call__()
        if self.transform is not None:
            batch = self.transform(batch)
        return batch


class GenerateTarget(TargetTransfer):

    def __init__(self, G, Cs, target_loader, entropy=False, thr=0.9, **kwargs):
        super().__init__(G, Cs, target_loader, entropy, thr)
        params = {}
        # Batch size during training
        params['batch_size'] = 128
        # Spatial size of training images. All images will be resized to this
        # size using a transformer.
        params['image_size'] = 64
        # Number of channels in the training images. For color images this is 3
        params['nc'] = 3
        # Size of z latent vector (i.e. size of generator input)
        params['nz'] = 100
        # Size of feature maps in generator
        params['ngf'] = 64
        # Size of feature maps in discriminator
        params['ndf'] = 64
        # Number of training epochs
        params['num_epochs'] = 25
        # Learning rate for optimizers
        params['lr'] = 0.0002
        # Beta1 hyperparameter for Adam optimizers
        params['beta1'] = 0.5

        params['gamma'] = 0.1
        params['no_adapt'] = False
        params['logname'] = 'record/logname'
        params['save_model'] = False
        params['save_path'] = "record/ova_model"
        if 'params' in kwargs:
            params.update(kwargs.pop('params'))

        G.eval()
        for c in Cs:
            c.eval()

        gan = { 'G': Generator(params['nz'], params['ngf'], params['nc']),
                'D': Discriminator(params['nc'], params['ndf']) } if 'gan' not in kwargs else kwargs.pop('gan')

        optimizer = { 'G': torch.optim.Adam(gan['G'].parameters(),
                                            lr=params['lr'],
                                            betas=(params['beta1'], 0.999)),
                      'D': torch.optim.Adam(gan['D'].parameters(),
                                            lr=params['lr'],
                                            betas=(params['beta1'], 0.999)) } if 'optimizer' not in kwargs else kwargs.pop('optimizer')

        loader = torch.utils.data.DataLoader(
            self.unknown_target_data,
            batch_size=params['batch_size'],
            shuffle=True,
            drop_last=True,
            num_workers=4)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gan['G'].to(device)
        gan['D'].to(device)

        scaler = torch.cuda.amp.GradScaler()
        gan['G'] = nn.DataParallel(gan['G'])
        gan['D'] = nn.DataParallel(gan['D'])

        # Initialize the ``BCELoss`` function
        criterion = nn.BCELoss()

        # Create batch of latent vectors that we will use to visualize
        #  the progression of the generator
        fixed_noise = torch.randn(64, params['nz'], 1, 1, device=device)

        # Establish convention for real and fake labels during training
        real_label = 1
        fake_label = 0

        resize = { 'G': transforms.Resize((params['image_size'],
                                           params['image_size'])),
                   'D': transforms.Resize((64, 64)) }

        print("Starting Training Loop...")
        # For each epoch
        for epoch in range(params['num_epochs']):
            gan['G'].train()
            gan['D'].train()

            # For each batch in the dataloader
            for i, data_t in enumerate(loader):

                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                ## Train with all-real batch
                optimizer['D'].zero_grad()
                # Format batch
                real = data_t[0]
                real = Variable(real.to(device))
                label = torch.full((params['batch_size'],),
                                   real_label,
                                   dtype=torch.float,
                                   device=device)
                # Forward pass real batch through D
                output = gan['D'](resize['D'](real))
                # Calculate loss on all-real batch
                errD_real = criterion(output, label)
                # Calculate gradients for D in backward pass
                scaler.scale(errD_real).backward()
                D_x = output.mean().item()

                # Train with all-fake batch
                # Generate batch of latent vectors
                noise = torch.randn(params['batch_size'],
                                    params['nz'], 1, 1, device=device)
                # Generate fake image batch with G
                fake = gan['G'](noise)
                label.fill_(fake_label)
                # Classify all fake batch with D
                output = gan['D'](fake.detach())
                # Calculate D's loss on the all-fake batch
                errD_fake = criterion(output, label)
                # Calculate the gradients for this batch,
                # accumulated (summed) with previous gradients
                scaler.scale(errD_fake).backward()
                D_G_z1 = output.mean().item()
                # Compute error of D as sum over the fake and the real batches
                errD = errD_real + errD_fake
                # Update D
                scaler.step(optimizer['D'])

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                optimizer['G'].zero_grad()
                label.fill_(real_label) # fake labels are real for generator cost
                # Since we just updated D, perform another forward pass of all-fake batch through D
                output = gan['D'](fake)
                # Calculate G's loss based on this output
                errG = criterion(output, label)
                if not params['no_adapt']:
                    # Align all-fake batch to some known class, but deceive
                    # the classifier by assigning them to unknown classes
                    with torch.no_grad():
                        feat = G(resize['G'](fake))
                        out_f = Cs[0](feat)
                    open_entropy = loss.entropy(out_f)
                    errG += params['gamma'] * open_entropy
                    if not entropy:
                        label_f = out_f.data.max(1)[1]
                        label_f = Variable(label_f.cuda())
                        with torch.no_grad():
                            out_open = Cs[1](feat)
                        out_open = out_open.view(out_f.size(0), 2, -1)
                        open_loss_pos, _ = loss.ova_loss(out_open, label_f)
                        errG += params['gamma'] * open_loss_pos
                # Calculate gradients for G
                scaler.scale(errG).backward()
                D_G_z2 = output.mean().item()
                # Update G
                scaler.step(optimizer['G'])
                scaler.update()

                # Output training stats
                if i % 50 == 0:
                    print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                        % (epoch, params['num_epochs'], i, len(loader),
                           errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Check how the generator is doing by saving G's output on fixed_noise
            if (epoch % 100 == 0) or (epoch == params['num_epochs']-1):
                gan['G'].eval()
                with torch.no_grad():
                    fake = gan['G'](fixed_noise)
                vutils.save_image(fake,
                    '%s-fake_samples_epoch_%03d.png' % (params['logname'],
                                                        epoch),
                    normalize=True)

            # do checkpointing
            if params['save_model']:
                torch.save(gan['G'].state_dict(),
                        '%s/ganG_epoch_%d.pth' % (params['save_path'], epoch))
                torch.save(gan['D'].state_dict(),
                        '%s/ganD_epoch_%d.pth' % (params['save_path'], epoch))

        self.params = params
        self.device = device
        self.generator = gan['G']
        self.batch_size = target_loader.batch_size
        self.resize = resize['G']
        self.generator.eval()

    def __call__(self):
        noise = torch.randn(self.batch_size,
                            self.params['nz'], 1, 1, device=self.device)
        with torch.no_grad():
            batch = self.generator(noise)
        return self.resize(batch)

