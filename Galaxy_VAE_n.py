#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 15:26:41 2022

@author: xuquanfeng
"""
import torch
from torchvision import transforms
import torch.nn as nn
from torch.autograd import Variable
from VAE_model.models import VAE, MyDataset
import numpy as np
import random
import os
import datetime
from torch import optim

#设置随机种子
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    os.environ['PYTHONHASHSEED'] = str(seed)
setup_seed(10)
if not os.path.exists('./model'):
    os.mkdir('./model')
if not os.path.exists('./train_proces'):
    os.mkdir('./train_proces')
num_epochs = 50   #循环次数
batch_size = 512    #每次投喂数据量
learning_rate = 0.00001    #学习率
momentum = 0.8
num_var = 40

train_loss11 = open('./train_proces/train_'+str(num_var)+'.txt', 'w')

root = '/data/xqf/VAE2/'
train_data = MyDataset(datatxt=root + 'train.txt', transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size = batch_size, shuffle=True,num_workers=40)

# Device configuration  判断能否使用cuda加速
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = VAE(num_var,device=device)
net = model.to(device)
reconstruction_function = nn.MSELoss(size_average=False)
def loss_function(recon_x, x, mu, logvar):
    """
    recon_x: generating images
    x: origin images
    mu: latent mean
    logvar: latent log variance
    """
    BCE = reconstruction_function(recon_x, x)  # mse loss
    # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    # KL divergence
    return BCE + KLD

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
strattime = datetime.datetime.now()
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        img = data
        img = Variable(img)
        img = img.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(img)
        loss = loss_function(recon_batch, img, mu, logvar)
        loss.backward()
        # train_loss += loss.data[0]
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 100 == 0:
            endtime = datetime.datetime.now()
            asd = str('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f} time:{:.2f}s'.format(
                epoch,
                batch_idx * len(img),
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(img),
                (endtime-strattime).seconds))
            print(asd)
            train_loss11.write(asd+'\n')
            # torch.save(model, './model/b_vae'+str(epoch)+'_'+str(batch_idx)+'.pth')
    if epoch == 0:
        best_loss = train_loss / len(train_loader.dataset)
    if epoch > 0 and best_loss > train_loss / len(train_loader.dataset):
        best_loss = train_loss / len(train_loader.dataset)
        asds = 'Save Best Model!'
        print(asds)
        train_loss11.write(asds+'\n')
        torch.save(model, './model/vae_'+str(num_var)+'_best.pth')
    asds = str('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))
    print(asds)
    train_loss11.write(asds+'\n')
# if epoch == num_epochs-1:
#     torch.save(model, './model/vae_'+str(num_var)+'.pth')
train_loss11.close()

####训练完成 自动发送邮件提醒
# From='xxx@qq.com'
# To = 'xuquanfeng@shao.ac.cn'
# pwd='xxxx'##这是授权码
# import smtplib
# from email.mime.text import MIMEText

# # login
# smtp=smtplib.SMTP()
# smtp.connect('smtp.qq.com',25)
# smtp.login(From,pwd)
# # email
# mail = MIMEText('''The training of the model has been completed, please check.''')
# mail['Subject'] = 'Progress of training'
# mail['From'] = 'Note'
# mail['To'] = 'xxx@qq.com'
# # send
# smtp.sendmail(From, To, mail.as_string())
# print('send email success')
# smtp.quit()