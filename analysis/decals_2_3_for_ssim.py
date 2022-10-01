# -*- coding: utf-8 -*-
"""
Created on Sat Jun 11 01:21:39 2022

@author: xqf35
"""
import os
import numpy as np
import sys
from tqdm import tqdm

pt = '../all/'
pp = '../result_ssim/'
aaa = os.listdir(pp)
bbb = [i for i in aaa if 'result_' not in i]
bbb1 = [i for i in aaa if 'result_' in i]
ccc = [i.replace('resu_','').replace('.npy','') for i in bbb]
ccc1 = [i.replace('result_','').replace('.npy','') for i in bbb1]
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
# print(ddd)
for num_var in tqdm(ddd):
    print('Dealing:'+'resu_'+num_var+'.npy')
    n_comp = np.load(pp+'resu_'+num_var+'.npy')
    n_di = np.load(pt+'resu.npy')
    data = []

    for i in range(len(n_di)):
        re = n_comp[eval(n_di[i,1])]

        pat = re[0]
        # if re[-1]>=0.9:
        if 'nomerge' in pat:
            if eval(n_di[i,2])!=4:
                re=np.hstack((re,eval(n_di[i,2])))
                data.append(re)
        else:
            re=np.hstack((re,4))
            data.append(re)

        # re=np.hstack((re,eval(n_di[i,2])))
        # data.append(re)

    dd = np.array(data)
    np.save(pp+'result_'+num_var+'.npy',dd)
