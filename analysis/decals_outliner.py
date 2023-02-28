# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 15:43:18 2022

@author: xqf35
"""
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

def samplee(dat, sam = 500):
    mmax = int(dat[dat.columns[-1]].max())
    mmin = int(dat[dat.columns[-1]].min())
    cc1 = pd.DataFrame()
    for i in range(mmin,mmax+1):
        cc2 = dat[dat[dat.columns[-1]]==i]
        cc3 = cc2.sample(sam)
        cc1 = pd.concat([cc1,cc3])
    return cc1
def liqundian(dat4, nums, stds=8):
    dat5 = dat4
    for i in range(nums):
        dat5 = dat5[np.abs(dat5[dat5.columns[i]] - dat5[dat5.columns[i]].mean()) <= (stds * dat5[dat5.columns[i]].std())]
    return dat5
def find_center(dat, sam = 500):
    mmax = int(dat[dat.columns[-1]].max())
    mmin = int(dat[dat.columns[-1]].min())
    # threshold = 2  # 阈值选择3
    relative_distance = []
    cluster_centers_ = []
    for i in range(mmin,mmax+1):
        cc1 = dat[dat[dat.columns[-1]]==i]
        cc2 = cc1.mean()
        cluster_centers_.append(list(cc2)[:-1])
    dat2 = dat
    for i in range(mmin,mmax+1):
        distance = dat2.query("label == @i")[dat2.columns[:-1]] - cluster_centers_[i-mmin]  # 计算各点至簇中心点的距离
        absolute_distance = distance.apply(np.linalg.norm, axis = 1)  # 求出绝对距离
        relative_distance.append(absolute_distance / absolute_distance.median())  # 求相对距离并添加
        
    dat2['relative_distance'] = pd.concat(relative_distance)  # 合并
    
    dat3 = pd.DataFrame()
    for i in range(mmin,mmax+1):
        cc2 = dat2[dat2[dat2.columns[-2]]==i]
        cc3 = cc2.sort_values(by=['relative_distance'],ascending=True)
        cc4 = cc3.iloc[:sam,:-1]
        dat3 = pd.concat([dat3,cc4])
    
    # dat2['outlier_3'] = dat2.relative_distance.apply(lambda x: 1 if x > threshold else 0)
    return dat3
def find_center1(dat, sam = 800, sam1 = 500):
    mmax = int(dat[dat.columns[-1]].max())
    mmin = int(dat[dat.columns[-1]].min())
    # threshold = 2  # 阈值选择3
    relative_distance = []
    cluster_centers_ = []
    for i in range(mmin,mmax+1):
        cc1 = dat[dat[dat.columns[-1]]==i]
        cc2 = cc1.mean()
        cluster_centers_.append(list(cc2)[:-1])
    dat2 = dat
    for i in range(mmin,mmax+1):
        distance = dat2.query("label == @i")[dat2.columns[:-1]] - cluster_centers_[i-mmin]  # 计算各点至簇中心点的距离
        absolute_distance = distance.apply(np.linalg.norm, axis = 1)  # 求出绝对距离
        relative_distance.append(absolute_distance / absolute_distance.median())  # 求相对距离并添加
        
    dat2['relative_distance'] = pd.concat(relative_distance)  # 合并
    dat3 = pd.DataFrame()
    for i in range(mmin,mmax+1):
        cc2 = dat2[dat2[dat2.columns[-2]]==i]
        cc2 = cc2.sample(sam)
        cc3 = cc2.sort_values(by=['relative_distance'],ascending=True)
        cc4 = cc3.iloc[:sam1,:-1]
        dat3 = pd.concat([dat3,cc4])
    return dat3
def find_center2(dat, sam = 500):
    mmax = int(dat[dat.columns[-1]].max())
    mmin = int(dat[dat.columns[-1]].min())
    # threshold = 2  # 阈值选择3
    relative_distance = []
    cluster_centers_ = []
    for i in range(mmin,mmax+1):
        cc1 = dat[dat[dat.columns[-1]]==i]
        cc2 = cc1.mean()
        cluster_centers_.append(list(cc2)[:-1])
    dat2 = dat
    for i in range(mmin,mmax+1):
        distance = dat2.query("label == @i")[dat2.columns[:-1]] - cluster_centers_[i-mmin]  # 计算各点至簇中心点的距离
        absolute_distance = distance.apply(np.linalg.norm, axis = 1)  # 求出绝对距离
        relative_distance.append(absolute_distance / absolute_distance.median())  # 求相对距离并添加
        
    dat2['relative_distance'] = pd.concat(relative_distance)  # 合并
    
    dat3 = pd.DataFrame()
    for i in range(mmin,mmax+1):
        cc2 = dat2[dat2[dat2.columns[-2]]==i]
        cc3 = cc2.sort_values(by=['relative_distance'],ascending=False)
        cc4 = cc3.iloc[:sam,:-1]
        dat3 = pd.concat([dat3,cc4])
    
    # dat2['outlier_3'] = dat2.relative_distance.apply(lambda x: 1 if x > threshold else 0)
    return dat3

pt = './all/'
ppt = os.listdir(pt)
ppt.sort()
ptt = [i.replace('.fits','') for i in ppt if '.fits' in i]
dd = np.load('./result_ssim/result_100.npy')
bb = pd.DataFrame(dd[:,1:].astype("float64"))
cc = bb[bb[bb.columns[-1]]>-1]
# cc1 = bb[bb[bb.columns[-1]]==4]
# cc2 = bb[bb[bb.columns[-1]]!=4]
# cc3 = cc2.sample(len(cc1)*2)
# cc1[cc1.columns[-1]] = 0
# cc3[cc3.columns[-1]] = 1
# cc = pd.concat([cc1,cc3])
for i in range(8):
    cc1 = bb[bb[bb.columns[-1]]==i]
    print(i,cc1.shape[0])

dat2 = cc.iloc[:,:-2]
dat2['label'] = cc.iloc[:,-1]

dat3 = find_center2(dat2, sam=20)
dat4 = dd[dat3.index,0]
np.save('liqundian1.npy',dat4)
