# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 16:39:41 2022

@author: xqf35
"""
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from astropy.table import Table
from astropy.io import fits

if not os.path.exists('../all'):
    os.mkdir('../all')
# root = '/data/GZ_Decals/'
pt = '../all/'
pp = '../train.txt'
with open(pp, "r") as f:
    n_dir = f.readlines()
n_dii = [i.split('/')[-1].split("_")[:2] for i in n_dir]
n_dii = np.array(n_dii).astype("float64")

dd = []
clas = -1
ccc = os.listdir(pt)
ccc.sort()
for i in ccc:
    # print(i)
    if ".fits" in i:
        print(i)
        clas = clas + 1
        lamost = Table.read(pt+i)
        data1 = lamost.to_pandas()
        for j in tqdm(range(len(n_dii))):
            da = data1[(data1[data1.columns[0]]==n_dii[j,0]) & (data1[data1.columns[1]]==n_dii[j,1])]
            if da.size != 0 :
                re = j
                re=np.hstack((n_dir[j][:-1],re))
                re=np.hstack((re,clas))
                dd.append(re)
dd = np.array(dd)
np.save(pt+'resu.npy',dd)
