# coding: utf-8
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

import sys
import time
import numpy as np
import os.path as osp
from PIL import Image
import cv2
import easydict
import random

import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torchvision
import torch.nn as nn
from torch.autograd import grad
from torch import optim
import torchvision.utils as vutils

from networks.generator_wasserstein_gan import Generator
from networks.discriminator_wasserstein_gan import Discriminator

from data.dataset_aus import AusDataset

from tensorboardX import SummaryWriter
writer = SummaryWriter('./runs')

args = easydict.EasyDict({
    'seed': 42,
    'gpus': '[0,1,2]',
    'workers': 16,
    'dataset': '/data/tjs/GANimation/imgs',
    'save_path': 'checkpoints',

    'batch_size': 24,
    'image_size': 256,
    'cond_nc': 2,
    'do_saturate_mask': True,

    'train_G_every_n_iterations': 4,
    'lr': 1e-5,
    'adam_b1': 0.5,
    'adam_b2': 0.999,

    'lambda_D_cond': 15000,
    'lambda_gan' : 0.01,
    'lambda_cyc': 10,
    'lambda_mask': 0.002,
    'lambda_D_gp': 10,
    'lambda_mask_smooth': 1e-5,

    'print_freq': 20,
    'display_freq': 60,
    'display_num' : 5,
    'save_freq': 100,
    'val_num':5,
    'val_freq':200,
    'use_wgan':True,

    'epochs': 60,
    'decay_start_ratio': 0.66,
    'continue_train': True,
    'continue_epoch': 0,
    'load_optim':False
})

# random seed setup
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
cudnn.benchmark = True

netG = Generator(c_dim=args.cond_nc)
netG.init_weights()
netG = torch.nn.DataParallel(netG).cuda()

netD = Discriminator(image_size=args.image_size, c_dim =args.cond_nc)
netD.init_weights()
netD = torch.nn.DataParallel(netD).cuda()

G_optimizer = optim.Adam(netG.parameters(),lr=args.lr,betas=(args.adam_b1, args.adam_b2))
D_optimizer = optim.Adam(netD.parameters(),lr=args.lr,betas=(args.adam_b1, args.adam_b2))

train_loader = torch.utils.data.DataLoader(
    AusDataset(args, True),
    batch_size=args.batch_size, shuffle=True,
    num_workers=args.workers, pin_memory=True, drop_last=True)

val_loader = torch.utils.data.DataLoader(
    AusDataset(args, False),
    batch_size=1, shuffle=False,
    num_workers=args.workers, pin_memory=True)

last_count_train = 0
last_count_val = 0
if args.continue_train:
    # load checkpoint
    G_checkpoint = torch.load(osp.join(args.save_path, str(args.continue_epoch), str(
        args.continue_epoch) + '_G_net_' + 'checkpoint.pth.tar'))
    netG.load_state_dict(G_checkpoint['model_state_dict'])
    D_checkpoint = torch.load(osp.join(args.save_path, str(args.continue_epoch), str(
        args.continue_epoch) + '_D_net_' + 'checkpoint.pth.tar'))
    netD.load_state_dict(D_checkpoint['model_state_dict'])

    if args.load_optim:
        G_optim_checkpoint = torch.load(osp.join(args.save_path, str(
            args.continue_epoch), str(args.continue_epoch) + '_G_optim_' + 'checkpoint.pth.tar'))
        G_optimizer.load_state_dict(G_optim_checkpoint['optimizer_state_dict'])
        D_optim_checkpoint = torch.load(osp.join(args.save_path, str(
            args.continue_epoch), str(args.continue_epoch) + '_D_optim_' + 'checkpoint.pth.tar'))
        D_optimizer.load_state_dict(D_optim_checkpoint['optimizer_state_dict'])

    last_count_train = args.continue_epoch * len(train_loader)
    last_count_val = last_count_train * args.val_num // args.val_freq

def SmoothLoss(mat):
    return torch.sum(torch.abs(mat[:, :, :, :-1] - mat[:, :, :, 1:])) + \
           torch.sum(torch.abs(mat[:, :, :-1, :] - mat[:, :, 1:, :]))

class GANLoss(nn.Module):

    def __init__(self,use_wgan):
        super(GANLoss, self).__init__()
        self.use_wgan = use_wgan
        self.loss_func = nn.MSELoss().cuda()
  
    def forward_normal(self,logit, is_real):
        if is_real:
            target = torch.ones_like(logit)
        else:
            target = torch.zeros_like(logit)
        return self.loss_func(logit, target)

    def forward_wgan(self, logit, is_real):
        if is_real:
            target = -torch.mean(logit)
        else:
            target = torch.mean(logit)
        return target
    def forward(self,logit,is_real):
        if self.use_wgan:
            return self.forward_wgan(logit,is_real)
        else:
            return self.forward_normal(logit,is_real)

def calc_gradient_penalty(netD, real_data, fake_data, args):
    alpha = torch.rand(args.batch_size, 1, 1, 1).cuda()
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates.requires_grad = True
    disc_interpolates,_ = netD(interpolates)

    gradients = grad(outputs=disc_interpolates, inputs=interpolates,
                     grad_outputs=torch.ones(disc_interpolates.size()).cuda(), create_graph=True,
                     retain_graph=True, only_inputs=True)[0]
    # gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * args.lambda_D_gp
    gradient_penalty = gradients.view(gradients.size(0),-1)
    gradient_penalty = torch.sqrt(torch.sum(gradient_penalty ** 2, dim=1))
    gradient_penalty = torch.mean((gradient_penalty-1)**2) * args.lambda_D_gp
    return gradient_penalty

criterion_MSE = nn.MSELoss().cuda()
criterion_L1 = nn.L1Loss().cuda()
criterion_GAN = GANLoss(args.use_wgan).cuda()

def tb_view_pic(imgs):
    for i,img in enumerate(imgs):
        img = vutils.make_grid(img[:args.display_num], normalize=True, scale_each=True)
        writer.add_image(str(i), img, i)

def tensor2img(tensor):
    temp = (np.transpose(tensor.detach().cpu().numpy(),
                         (0, 2, 3, 1)) + 1) / 2.0 * 255.0
    return temp[0]

def save_pic(save_epoch, i, tensors):
    imgs = []
    for tensor in tensors:
        imgs.append(tensor2img(tensor))
    combine = np.concatenate(imgs, axis=1)
    combine = Image.fromarray(combine.astype('uint8'))
    combine.save(osp.join(save_epoch, str(i) + '.jpg'))

def train_net(args, train_loader, netG, netD, epoch, save_epoch, last_count_train, last_count_val, cur_lr):
    netG.train()
    netD.train()
    end = time.time()

    for i, (img,real_cond,desired_cond) in enumerate(train_loader):
        img, real_cond, desired_cond = img.cuda(), real_cond.cuda(), desired_cond.cuda()
        
        # update D
        for p in netD.parameters():  
            p.requires_grad = True  
        for p in netG.parameters():
            p.requires_grad = False  

        output_real, pred_cond_real = netD(img)
        lossD_real = criterion_GAN(output_real,True)
        lossD_cond = args.lambda_D_cond * criterion_MSE(real_cond,pred_cond_real) / args.batch_size

        with torch.no_grad():
            fake_img, fake_mask = netG(img,desired_cond)
        if args.do_saturate_mask:
            fake_mask = torch.clamp(0.55*torch.tanh(3*(fake_mask-0.5))+0.5, 0, 1)
        fake_img_masked = fake_mask * img + (1 - fake_mask) * fake_img

        output_fake, _ = netD(fake_img_masked)
        lossD_fake = criterion_GAN(output_fake,False)

        lossD_Gan = lossD_real + lossD_fake
        lossD = lossD_Gan + lossD_cond
        writer.add_scalar('Tdata/lossD_Gan', lossD_Gan.item(), last_count_train)
        writer.add_scalar('Tdata/lossD_cond', lossD_cond.item(), last_count_train)

        D_optimizer.zero_grad()
        lossD.backward()
        D_optimizer.step()

        if args.use_wgan:
            gradient_penalty = calc_gradient_penalty(netD, img, fake_img_masked, args)
            D_optimizer.zero_grad()
            gradient_penalty.backward()
            D_optimizer.step()

        if i%args.train_G_every_n_iterations == 0:
            # Update G
            for p in netD.parameters():  
                p.requires_grad = False  
            for p in netG.parameters():
                p.requires_grad = True 
            
            fake_img, fake_mask = netG(img,desired_cond)
            if args.do_saturate_mask:
                fake_mask = torch.clamp(0.55*torch.tanh(3*(fake_mask-0.5))+0.5, 0, 1)
            fake_img_masked = fake_mask * img + (1 - fake_mask) * fake_img

            output_fake, cond = netD(fake_img_masked)
            lossG_GAN = args.lambda_gan *criterion_GAN(output_fake,True)
            lossG_cond = args.lambda_D_cond * criterion_MSE(cond,desired_cond) / args.batch_size

            #cycle
            rec_img, rec_mask = netG(fake_img_masked,real_cond)
            if args.do_saturate_mask:
                rec_mask = torch.clamp(0.55*torch.tanh(3*(rec_mask-0.5))+0.5, 0, 1)
            rec_img_masked = rec_mask * fake_img_masked + (1 - rec_mask) * rec_img

            lossG_cyc = args.lambda_cyc * criterion_L1(img, rec_img_masked)
            
            lossG_mask_1 = torch.mean(fake_mask) * args.lambda_mask
            lossG_mask_2 = torch.mean(rec_mask) * args.lambda_mask
            lossG_mask_1_smooth = SmoothLoss(fake_mask) * args.lambda_mask_smooth
            lossG_mask_2_smooth = SmoothLoss(rec_mask) * args.lambda_mask_smooth
            lossG_mask = lossG_mask_1 + lossG_mask_2 + lossG_mask_1_smooth + lossG_mask_2_smooth

            loss_G = lossG_GAN + lossG_cond + lossG_cyc + lossG_mask
            # loss_G = lossG_GAN + lossG_cyc + lossG_cond

            G_optimizer.zero_grad()
            loss_G.backward()
            G_optimizer.step()

            last_count_train += 1
            writer.add_scalar('Tdata/lossG_GAN', lossG_GAN.item(), last_count_train)
            writer.add_scalar('Tdata/lossG_cyc', lossG_cyc.item(), last_count_train)
            writer.add_scalar('Tdata/lossG_cond', lossG_cond.item(), last_count_train)
            writer.add_scalar('Tdata/lossG_mask', lossG_mask.item(), last_count_train)

            if i% args.display_freq == 0:
                tb_view_pic([img, fake_img, fake_mask, fake_img_masked, rec_img, rec_mask, rec_img_masked])

        end = time.time()
        if i % args.print_freq == 0:
            print(
                ('Epoch: [{0}][{1}/{2}], lr: {3}'.format(epoch, i, len(train_loader), cur_lr)))

        if i % args.save_freq == 0:
            # save G
            torch.save({'model_state_dict': netG.state_dict(),
                        }, osp.join(save_epoch, str(epoch) + '_G_net_' + 'checkpoint.pth.tar'))
            # save D
            torch.save({'model_state_dict': netD.state_dict(),
                        }, osp.join(save_epoch, str(epoch) + '_D_net_' + 'checkpoint.pth.tar'))
            # save G
            # torch.save({'optimizer_state_dict': G_optimizer.state_dict(),
            #             }, osp.join(save_epoch, str(epoch) + '_G_optim_' + 'checkpoint.pth.tar'))
            # save D
            # torch.save({'optimizer_state_dict': D_optimizer.state_dict(),
            #             }, osp.join(save_epoch, str(epoch) + '_D_optim_' + 'checkpoint.pth.tar'))
        if i % args.val_freq == 0:
            last_count_val = val_net(args,val_loader, netG, netD, save_epoch, last_count_val)
            netG.train()
            netD.train()
    return last_count_train, last_count_val


def val_net(args,val_loader, netG, netD, save_epoch, last_count_val):
    netG.eval()
    netD.eval()

    for i, (img,real_cond,desired_cond) in enumerate(val_loader):
        if i >= int(args.val_num):
            break
        img, real_cond, desired_cond = img.cuda(), real_cond.cuda(), desired_cond.cuda()
        print('validating image ind {} with desired cond {}'.format(i,desired_cond))
        output_real,pred_cond_real = netD(img)
        lossD_cond = args.lambda_D_cond * criterion_MSE(pred_cond_real,real_cond) / args.batch_size

        with torch.no_grad():
            fake_img, fake_mask = netG(img,desired_cond)
            if args.do_saturate_mask:
                fake_mask = torch.clamp(0.55*torch.tanh(3*(fake_mask-0.5))+0.5, 0, 1)
            fake_img_masked = fake_mask * img + (1 - fake_mask) * fake_img

            output_fake,pred_cond_fake = netD(fake_img_masked)

            lossG_cond = args.lambda_D_cond * criterion_MSE(pred_cond_fake,desired_cond) / args.batch_size
 
            #cycle
            rec_img, rec_mask = netG(fake_img_masked,real_cond)
            if args.do_saturate_mask:
                rec_mask = torch.clamp(0.55*torch.tanh(3*(rec_mask-0.5))+0.5, 0, 1)
            rec_img_masked = rec_mask * fake_img_masked + (1 - rec_mask) * rec_img

            lossG_cyc = args.lambda_cyc * criterion_L1(img, rec_img_masked)  
            lossG_mask_1 = torch.mean(fake_mask) * args.lambda_mask
            lossG_mask_2 = torch.mean(rec_mask) * args.lambda_mask
            lossG_mask_1_smooth = SmoothLoss(fake_mask) * args.lambda_mask_smooth
            lossG_mask_2_smooth = SmoothLoss(rec_mask) * args.lambda_mask_smooth
            lossG_mask = lossG_mask_1 + lossG_mask_2 + lossG_mask_1_smooth + lossG_mask_2_smooth

        save_pic(save_epoch, i, [img, fake_img_masked,rec_img_masked])

        last_count_val += 1
        writer.add_scalar('Vdata/lossD_cond',lossD_cond.item(), last_count_val)
        writer.add_scalar('Vdata/lossG_cond', lossG_cond.item(), last_count_val)
        writer.add_scalar('Vdata/lossG_cyc',lossG_cyc.item(), last_count_val)
        writer.add_scalar('Vdata/lossG_mask', lossG_mask.item(), last_count_val)
    return last_count_val

for epoch in range(args.continue_epoch + 1, args.epochs):
    num_decay_epoch = epoch - int(args.epochs*args.decay_start_ratio)
    total_num_decay = int(args.epochs*(1-args.decay_start_ratio))
    cur_lr = args.lr
    if num_decay_epoch>0:  
        cur_lr = args.lr * (1-num_decay_epoch/total_num_decay)
        for param_group in G_optimizer.param_groups:
            param_group['lr'] = cur_lr
        for param_group in D_optimizer.param_groups:
            param_group['lr'] = cur_lr

    save_epoch = osp.join(args.save_path, str(epoch))
    if not os.path.isdir(save_epoch):
        os.makedirs(save_epoch)

    last_count_train, last_count_val = train_net(
        args, train_loader, netG, netD, epoch, save_epoch, last_count_train, last_count_val, cur_lr)

writer.close()
