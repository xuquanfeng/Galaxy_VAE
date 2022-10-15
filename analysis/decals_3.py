# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 20:22:40 2022

@author: xqf35
"""
import pandas as pd
import numpy as np

pt = '../all/'
dd = np.load('../result/result_100.npy')
bb = pd.DataFrame(dd[:,1:].astype("float64"))
cc = bb[bb[bb.columns[-1]]<3]

from qrpca.decomposition import qrpca
import torch
num = 5
dat = np.array(cc.iloc[:,:-1])
device = torch.device("cpu")
pca = qrpca(n_component_ratio=num,device=device)
dat1 = pca.fit_transform(dat)
dat2 = pd.DataFrame(dat1.astype("float64"))
dat2['label'] = cc.iloc[:,-1]

import matplotlib.pyplot as plt
import seaborn as sns
# iris = sns.load_dataset("iris")
# g =  sns.PairGrid(iris, hue='species', size=2)

# def f(x, **kwargs):
#     kwargs.pop("color")
#     col = next(plt.gca()._get_lines.prop_cycler)['color']
#     sns.kdeplot(x, color=col, **kwargs)

# g.map_diag(f)
# g.map_offdiag(plt.scatter)
# g.add_legend()
# plt.show()

g = sns.PairGrid(dat2, hue="label",corner=True)

g.map_diag(sns.histplot, multiple="stack", element="step")
g.map_offdiag(sns.scatterplot)
g.add_legend()

# g.map_diag(plt.hist)
# g.map_offdiag(plt.scatter)

# g = sns.PairGrid(dat2, diag_sharey=False)
# g.map_upper(sns.scatterplot)
# g.map_lower(sns.kdeplot)
# g.map_diag(sns.kdeplot)

plt.savefig(pt+'coner_fig.jpg',dpi=400)
