# -*- coding: utf-8 -*-
"""
Created on Sat Jun 11 01:21:39 2022

@author: xqf35
"""

import numpy as np

num_var = '50'
pt = '../all/'
pp = '../result/'
n_comp = np.load(pp+'resu_'+num_var+'.npy')
n_di = np.load(pt+'resu.npy')
data = []

for i in range(len(n_di)):
    re = n_comp[eval(n_di[i,1])]
    # re = n_comp[j,:]
    re=np.hstack((n_di[i,0],re))
    re=np.hstack((re,eval(n_di[i,2])))
    data.append(re)
dd = np.array(data)
np.save(pp+'result_'+num_var+'.npy',dd)
