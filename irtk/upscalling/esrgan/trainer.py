import argparse
import os
import numpy as np

from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torch.autograd import Variable

from model import *
from dataset import *

import torch
import torch.nn as nn
import torch.nn.functional as F

os.makedirs('images/training', exist_ok=True)
os.makedirs('saved_models', exist_ok=True)

epoch = 0
n_epochs=10
lr=0.0002
hr_height=1024
hr_width=1024
# sample_interval=100
sample_interval=2
# residual_blocks=23
residual_blocks=10
# warmup_batches=500
warmup_batches=2
lambda_adv=0.1
lambda_pixel=1
lambda_content=1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

hr_shape = (hr_height, hr_width)
channels = 1

# initialize generator and discriminator  
generator = GeneratorRRDB(channels, num_res_blocks=residual_blocks, num_upsample=3).to(device)
discriminator = Discriminator(input_shape=(channels, *hr_shape)).to(device)
feature_extractor = FeatureExtractor().to(device)

# set feature extractor to inference mode
feature_extractor.eval()

# Losses  
criterion_GAN = torch.nn.BCEWithLogitsLoss().to(device) 
criterion_content = torch.nn.L1Loss().to(device)
criterion_pixel = torch.nn.L1Loss().to(device)

if epoch != 0:
    generator.load_state_dict(torch.load('saved_models/generator_%d.pth' % epoch))
    discriminator.load_state_dict(torch.load('saved_models/discriminator_%d.pth' % epoch))

optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr)

# train_set = TrainDatasetFromFolder('ersgan_dataset/train_HR', crop_size=128, upscale_factor=8)
train_set = TrainDatasetFromFolder(R'D:\volume_registration\ersgan_dataset\train_HR\SRF_8x\target_mini', crop_size=1024, upscale_factor=8)
train_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=4, shuffle=True)

# ----------
# Training  
# ----------  

for epoch in range(epoch, n_epochs):
    for i, (data, target) in enumerate(train_loader):

        batches_done = epoch * len(train_loader) + i

        imgs_lr = Variable(data.type(torch.Tensor)).to(device)
        imgs_hr = Variable(target.type(torch.Tensor)).to(device)

        # ------------------
        #  Train Generator
        # ------------------   

        optimizer_G.zero_grad()
        gen_hr = generator(imgs_lr)
        loss_pixel = criterion_pixel(gen_hr, imgs_hr)

        if batches_done < warmup_batches:
            loss_pixel.backward()
            optimizer_G.step()
            print(
                '[Epoch %d/%d] [Batch %d/%d] [G pixel: %f]' % 
                (epoch+1, n_epochs, i+1, len(train_loader), loss_pixel.item())
            )
            continue

        pred_real = discriminator(imgs_hr).detach()
        pred_fake = discriminator(gen_hr)

        valid = torch.ones_like(pred_fake, device=device)
        fake  = torch.zeros_like(pred_fake, device=device)

        loss_GAN = (criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), valid) + 
                    criterion_GAN(pred_real - pred_fake.mean(0, keepdim=True), fake)) / 2

        gen_features = feature_extractor(gen_hr)
        real_features = feature_extractor(imgs_hr).detach()
        loss_content = criterion_content(gen_features, real_features)

        loss_G = lambda_content * loss_content + lambda_adv * loss_GAN + lambda_pixel * loss_pixel

        loss_G.backward()
        optimizer_G.step()

        # --------------------
        # Train Discriminator  
        # --------------------  

        optimizer_D.zero_grad()

        pred_real = discriminator(imgs_hr)
        pred_fake = discriminator(gen_hr.detach())  

        loss_real = criterion_GAN(pred_real - pred_fake.mean(0, keepdim=True), valid)
        loss_fake = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), fake)

        loss_D = (loss_real + loss_fake) / 2

        loss_D.backward()
        optimizer_D.step()

        # -----------------  
        # Log Progress  
        # -----------------

        print(
            '[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, content: %f, adv: %f, pixel: %f]' % 
            (
                epoch+1, 
                n_epochs, 
                i+1, 
                len(train_loader), 
                loss_D.item(), 
                loss_G.item(), 
                loss_content.item(), 
                loss_GAN.item(), 
                loss_pixel.item(), 
            )
        )
        if batches_done % sample_interval == 0:
            imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=8, mode='bicubic')
            img_grid = torch.clamp(torch.cat((imgs_lr, gen_hr, imgs_hr), -1), min=0, max=1)
            save_image(img_grid, '../../../images/training/%d.png' % batches_done, nrow=1, normalize=True)
            save_image(imgs_hr, '../../../images/training/test.png', nrow=1, normalize=True)

    torch.save(generator.state_dict(), '../../../saved_models/generator_%d.pth' % epoch)
    torch.save(discriminator.state_dict(), '../../../saved_models/discriminator_%d.pth' % epoch)