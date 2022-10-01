#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 20:18:19 2021

@author: xuquanfeng
"""
import torch
from torchvision import transforms
import torch.nn as nn
from torch.autograd import Variable
from VAE_model.models import VAE, MyDataset
import numpy as np
import os
from tqdm import tqdm
import torch.nn.functional as F

if not os.path.exists('./result_ssim'):
    os.mkdir('./result_ssim')
batch_size = 512    #每次投喂数据量

def _fspecial_gauss_1d(size, sigma):
    coords = torch.arange(size).to(dtype=torch.float)
    coords -= size//2
    g = torch.exp(-(coords**2) / (2*sigma**2))
    g /= g.sum()
    return g.unsqueeze(0).unsqueeze(0)
    
def gaussian_filter(input, win):
    N, C, H, W = input.shape
    out = F.conv2d(input, win, stride=1, padding=0, groups=C)
    out = F.conv2d(out, win.transpose(2, 3), stride=1, padding=0, groups=C)
    return out

def _ssim(X, Y, win, data_range=1023, size_average=True, full=False):
    K1 = 0.01
    K2 = 0.03
    batch, channel, height, width = X.shape
    compensation = 1.0

    C1 = (K1 * data_range)**2
    C2 = (K2 * data_range)**2

    win = win.to(X.device, dtype=X.dtype)

    mu1 = gaussian_filter(X, win)
    mu2 = gaussian_filter(Y, win)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * ( gaussian_filter(X * X, win) - mu1_sq )
    sigma2_sq = compensation * ( gaussian_filter(Y * Y, win) - mu2_sq )
    sigma12   = compensation * ( gaussian_filter(X * Y, win) - mu1_mu2 )

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map

    if size_average:
        ssim_val = ssim_map.mean()
        cs = cs_map.mean()
    else:
        ssim_val = ssim_map.mean(-1).mean(-1).mean(-1)  # reduce along CHW
        cs = cs_map.mean(-1).mean(-1).mean(-1)

    if full:
        return ssim_val, cs
    else:
        return ssim_val

def ssim(X, Y, win_size=11, win_sigma=10, win=None, data_range=1, size_average=True, full=False):

    if len(X.shape) != 4:
        raise ValueError('Input images must 4-d tensor.')

    if not X.type() == Y.type():
        raise ValueError('Input images must have the same dtype.')

    if not X.shape == Y.shape:
        raise ValueError('Input images must have the same dimensions.')

    if not (win_size % 2 == 1):
        raise ValueError('Window size must be odd.')

    win_sigma = win_sigma
    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat(X.shape[1], 1, 1, 1)
    else:
        win_size = win.shape[-1]

    ssim_val, cs = _ssim(X, Y,
                         win=win,
                         data_range=data_range,
                         size_average=False,
                         full=True)
    if size_average:
        ssim_val = ssim_val.mean()
        cs = cs.mean()

    if full:
        return ssim_val, cs
    else:
        return ssim_val



train_data = MyDataset(datatxt='train.txt', transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size = batch_size, shuffle=False,num_workers=10)
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)

pt = './model/'
pp = './result_ssim/'
aaa = os.listdir(pt)
aaa1 = os.listdir(pp)
bbb = [i for i in aaa if 'best.pth' in i]
bbb1 = [i for i in aaa1 if 'result_' not in i]
ccc = [i.replace('vae_','').replace('_best.pth','') for i in bbb]
ccc1 = [i.replace('resu_','').replace('.npy','') for i in bbb1]
# print(ccc)
# print(ccc1)
ddd = []
for i in ccc:
    zai = 0
    for j in ccc1:
        if i != j:
            zai += 1
    if zai == len(ccc1):
        ddd.append(i)
print(ddd)

for kk in ddd:
    print(kk)
    cccc = kk.split('_')
    if len(cccc)==2:
        num_var = eval(cccc[0])
        k = eval(cccc[1])
    elif len(cccc)==3:
        num_var = eval(cccc[1])
        k = eval(cccc[2])
    else:
        num_var = eval(cccc[0])
        k = 1
    torch.cuda.empty_cache()
    model = torch.load('./model/vae_'+kk+'_best.pth')

    model = model.to(device)

    model.eval()
    # print(device)
    # print(torch.cuda.current_device())
    train_loss = 0
    sssi = []
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(train_loader)):
            img,fn = data
            img = Variable(img)
            img = img.to(device)
            mu, logvar = model.encode(img)
            imgs = model.decode(mu)

            yunatu = img.cpu().numpy()
            houtu = imgs.cpu().detach().numpy()
            for i in range(len(yunatu)):
                ssim_val = ssim(img[i].unsqueeze(0), imgs[i].unsqueeze(0), data_range=1, size_average=True,)
                qw = [fn[i]]
                qw.extend(mu[i].cpu().detach().numpy())
                # qw.append(lab[i].item())
                qw.append(ssim_val.cpu().detach().numpy())
                sssi.append(qw)

    dd = np.array(sssi)
    np.save('./result_ssim/resu_'+kk+'.npy',dd)
