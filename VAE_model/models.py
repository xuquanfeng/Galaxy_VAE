#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 15:26:41 2022

@author: xuquanfeng
"""
import torch
import torch.nn as nn
import numpy as np
import random
from astropy.io import fits
import os
from torch.autograd import Variable


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, datatxt, transform=None, target_transform=None):
        super(MyDataset, self).__init__()
        fh = open(datatxt, 'r')  # 按照传入的路径和txt文本参数，打开这个文本，并读取内容
        imgs = []
        for line in fh:  # 按行循环txt文本中的内容
            line = line.rstrip()
            imgs.append((line))
            # 很显然，words[0]是图片path
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        fn = self.imgs[index]  # fn是图片path
        hdu = fits.open(fn)
        img = hdu[0].data
        img = np.array(img,dtype=np.float32)
        hdu.close()
        if self.transform is not None:
            img = self.transform(img)
            img = img.permute(1,0,2)
        return img
    def __len__(self):
        return len(self.imgs)

class VAE(nn.Module):
    def __init__(self,num_var, device, hidden_dims = [32, 64, 128]):
        super(VAE, self).__init__()
        modules = []
        self.device = device
        self.hidden_dims = hidden_dims
        in_channels = 3
        latent_dim = num_var
        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    # nn.BatchNorm2d(h_dim),
                    nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2),return_indices=True),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*16, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*16, latent_dim)
        
        modules = []
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 16)
        hidden_dims.reverse()
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.UpsamplingNearest2d(scale_factor=2),
                    # nn.MaxUnpool2d((2, 2), stride=(2, 2)),
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    # nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )
        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.UpsamplingNearest2d(scale_factor=2),
                            # nn.MaxUnpool2d((2, 2), stride=(2, 2)),
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            # nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= 3,
                                      kernel_size= 3, padding= 1),
                            nn.ReLU())

    def encode(self, x):
        result = x
        # idx = []
        for i in range(len(self.hidden_dims)):
            result,indices = self.encoder[i][:2](result)
            # idx.append(indices)
            result = self.encoder[i][2](result)        
        # self.idx = idx
        # result = self.encoder(x)
        result = torch.flatten(result, start_dim=1)
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return mu, log_var

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.FloatTensor(std.size()).normal_().to(self.device)
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        result = self.decoder_input(z)
        result = result.view(len(result), 128, 4, 4)
        # for i in range(len(self.hidden_dims)-1):
        #     result = self.decoder[i][0](result,self.idx[len(self.hidden_dims)-1-i])
        #     result = self.decoder[i][1:](result)
        result = self.decoder(result)
        result = self.final_layer(result)
        # result = self.final_layer[0](result,self.idx[0])
        # result = self.final_layer[1:](result)
        return result

    def forward(self, x):
        # x = x.view(len(x),-1)
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar

