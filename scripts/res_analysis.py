'''
Author       : Scallions
Date         : 2020-08-29 16:05:38
LastEditors  : Scallions
LastEditTime : 2020-11-20 14:45:17
FilePath     : /gps-ts/scripts/res_analysis.py
Description  : 
'''
import os
from os import stat
import sys
sys.path.append('./')
from loguru import logger
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import ts.fill as fill
from scipy import stats
from matplotlib import gridspec
import matplotlib.ticker as ticker
from pandas import Series

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

ttt = pd.read_csv("./res/raw.csv", index_col="jd",parse_dates=True)
print(ttt.columns)
ind = ttt.index
data = {}
for name in names:
    data[name] = pd.read_csv(f"./res/{name}.csv").to_numpy()[:,1:].astype(np.float64)

a = np.abs((data["raw"]-data[names[-1]])) > 0.00001
b = np.where(np.any(a, axis=1))
c = np.where(np.any(a, axis=0))

print(c[0])
# print(c)


# ind = ind[b[0][0]:b[0][-1]]
ind = ind[b[0]]

for k,v in data.items():
    # data[k] = v[b[0][0]:b[0][-1],:]
    data[k] = v[b[0],:]

dd = -3

## 相关系数
raw_s = pd.Series(data["raw"][:,c[0][dd]])
method = "pearson"
print("相关系数: ")
for name in names[1:]:
    ts_s = pd.Series(data[name][:,c[0][dd]])
    r = raw_s.corr(ts_s, method=method)
    # r2 = raw_s.corr(ts_s,method='spearman')
    print(f"{name}: ", r)
# regem_s = pd.Series(regem[:,c[0][dd]])
# mlp_s = pd.Series(mlp[:,c[0][dd]])
# r1 = raw_s.corr(regem_s,method=method)
# r2 = raw_s.corr(mlp_s, method=method)
# print("相关系数",r1,r2)
plt.rcParams['font.sans-serif'] = ['SimHei']
# fig, subs = plt.subplots((len(names)-1)//3,3, sharex=True)
# fig = plt.figure(figsize=(8, 12))
fig, subs = plt.subplots(2, 3, sharex=True,figsize=(12,8))
subs[-1,-1].axis("off")
# gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1])
i = 0
for k, x in data.items():


    # 残差

    d = x[:,c[0][dd]]-data["raw"][:,c[0][dd]]
    # if i == 0:
    #     subs[i].scatter(ind, data["raw"][:,c[0][dd]], s=1)
    #     subs[i].set_ylabel("Raw"+"(mm)")
    #     i += 1
    #     continue
    print(f"{names[i]} mean, std, mae: ",d.mean(), d.std(), np.abs(d).mean())
    
    # 绘图
    if k == 'raw':
        i += 1
        continue
    p = 2 * 100 + 3 * 10 + i
    # plt.subplot(p)
    # ax = plt.subplot(gs[i])
    # sns.jointplot(data["raw"][:,c[0][dd]], x[:,c[0][dd]], kind='reg', ax=ax).annotate(stats.pearsonr,template='r: {val:.2f}')
    # subs[(i-1)//3][(i-1)%3].scatter(data["raw"][:,c[0][dd]], x[:,c[0][dd]], s=1)
    sns.regplot(data["raw"][:,c[0][dd]], x[:,c[0][dd]],ax=subs[(i-1)//3][(i-1)%3],color='steelblue',scatter_kws={'s':2},line_kws={"linewidth":1})
    var = stats.pearsonr(data["raw"][:,c[0][dd]], x[:,c[0][dd]])[0]
    subs[(i-1)//3][(i-1)%3].set_xlabel(cnnames[i])
    subs[(i-1)//3][(i-1)%3].text(0.8, 0.8,f'r:{var:.2f}',
     horizontalalignment='center',
     verticalalignment='center',
     transform = subs[(i-1)//3][(i-1)%3].transAxes,
     bbox=dict(boxstyle="round",ec=(1., 0.5, 0.5),fc=(1., 0.8, 0.8),))
    
    i += 1


plt.show()