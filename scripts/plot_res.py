'''
Author       : Scallions
Date         : 2020-10-10 22:11:57
LastEditors  : Scallions
LastEditTime : 2020-11-20 17:33:05
FilePath     : /gps-ts/scripts/plot_res.py
Description  : 
'''

import os
import sys
sys.path.append('./')
from loguru import logger
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import ts.fill as fill

# plt.rcParams['figure.figsize'] = (8.0, 4.0) 
def mean(x):
    xc = x.mean(axis=0)
    return x - xc

fillers = [
    ### scipy
    # fill.SLinearFiller, # 一阶样条插值
    fill.SplineFiller, # 三次样条
    # fill.AkimaFiller,
    fill.PolyFiller, # 二阶多项式插值
    # fill.PiecewisePolynomialFiller, 
    # fill.KroghFiller, # overflow
    # fill.QuadraticFiller, # 二次
    # fill.BarycentricFiller, # overflow
    # fill.FromDerivativesFiller,
    fill.PchipFiller, # 三阶 hermite 插值

    ### matlab
    fill.RegEMFiller, 
    
    ### fancyimpute
    # fill.KNNFiller, # **
    # fill.SoftImputeFiller,
    # fill.IterativeSVDFiller,
    # fill.IterativeImputerFiller, # **
    # fill.MatrixFactorizationFiller, # **
    # fill.BiScalerFiller,
    # fill.NuclearNormMinimizationFiller,

    ### imputena
    # fill.MICEFiller, # **

    ### missingpy
    fill.MissForestFiller, # **

    ### miceforest
    # fill.MiceForestFiller, # **

    ### para
    # fill.RandomForestFiller,

    ### magic
    # fill.MagicFiller,

    ### self
    # fill.BritsFiller,
    # fill.MLPFiller,
    # fill.GainFiller,
    # fill.ConvFiller,
    # fill.SSAFiller,
]
names = ['raw'] + [filler.name for filler in fillers]
cnnames = ["原始"] + [filler.cnname for filler in fillers]
# names += ['mice']
# names += ['r']

ind = pd.read_csv("./res/raw.csv", index_col="jd",parse_dates=True).index
data = {}
for name in names:
    data[name] = pd.read_csv(f"./res/{name}.csv").to_numpy()[:,1:].astype(np.float64)

a = np.abs((data["raw"]-data[names[-1]])) > 1e-8
b = np.where(np.any(a, axis=1))
c = np.where(np.any(a, axis=0))

# print(b)
# print(c)


# # ind = ind[b[0][0]:b[0][-1]]
# ind = ind[b[0]]

# for k,v in data.items():
#     # data[k] = v[b[0][0]:b[0][-1],:]
#     data[k] = v[b[0],:]

dd = -3
pltsize = 1
cache = 100

plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.figure(figsize=(6,8))
fig, subs = plt.subplots(len(names)-1,1, sharex=True, figsize=(10,6))

i = 0
names = list(data.keys())
for k, x in data.items():

    # 绘图

    if k == 'raw':
        continue
    subs[i].scatter(ind[b[0][0]-cache:b[0][-1]+cache], data["raw"][b[0][0]-cache:b[0][-1]+cache,c[0][dd]], s=pltsize, c="black")
    # subs[i].scatter(ind, data["raw"][:,c[0][dd]], s=pltsize, c="black")
    subs[i].scatter(ind[b[0]], x[b[0],c[0][dd]], s=pltsize, c="red", label=k)
    # subs[i].legend([s],[k],loc='center left')
    # subs[i].set_title(k)
    subs[i].set_ylabel("U(mm)")
    # subs[i].set_xlabel(cnnames[i+1])
    subs[i].text(0.9, 0.1,f'{cnnames[i+1]}',
     horizontalalignment='center',
     verticalalignment='center',
     transform = subs[i].transAxes,
    #  bbox=dict(boxstyle="round",ec=(1., 0.5, 0.5),fc=(1., 0.8, 0.8),)
    )
    i += 1
plt.show()