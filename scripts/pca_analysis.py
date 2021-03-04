'''
Author       : Scallions
Date         : 2020-08-29 16:05:38
LastEditors  : Scallions
LastEditTime : 2020-12-15 14:24:47
FilePath     : /gps-ts/scripts/pca_analysis.py
Description  : 
'''
import os
import sys
sys.path.append('./')
from loguru import logger
import pandas as pd
import time
import matplotlib.pyplot as plt
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

ind = pd.read_csv("./res/raw.csv", index_col="jd",parse_dates=True).index
data = {}
for name in names:
    data[name] = pd.read_csv(f"./res/{name}.csv").to_numpy()[:,1:].astype(np.float64)

a = np.abs((data["raw"]-data["RegEM"])) > 0.00001
b = np.where(np.any(a, axis=1))
c = np.where(np.any(a, axis=0))

# print(b)
# print(c)


# ind = ind[b[0][0]:b[0][-1]]
# ind = ind[b[0]]

for k,v in data.items():
    # data[k] = v[b[0][0]:b[0][-1],:]
    # data[k] = v[b[0],:]
    data[k] = v

dd = 0

# fig, subs = plt.subplots(len(names),1, sharex=True)
i = 0

# row pca
X = mean(data["raw"][:,:]).T
n, m = X.shape 
C = np.dot(X, X.T) / (n-1)
print(C.shape)
eigen_vals, eigen_vecs = np.linalg.eig(C)

for k, x in data.items():
    # if k == "raw":
    #     continue
    # PCA
    X = mean(x[:,:]).T
    n, m = X.shape 
    C = np.dot(X, X.T) / (n-1)
    _, vecs = np.linalg.eig(C)
    # s = eigen_vals.sum()

    V =  np.dot(eigen_vecs.T, np.dot(C, eigen_vecs))
    vs = np.diagonal(V)
    s = np.sum(vs)
    
    # print(s)
    res = vs/s * 100
    r = res.cumsum()
    print(k)
    print(res.real[:3])
    print(r.real[:3])
    if k == 'raw':
        raw = np.dot(X.T,eigen_vecs)[:,0].real
        print("length:", np.sqrt(np.dot(raw,raw.T)))
    else:
        xs = np.dot(X.T,eigen_vecs)[:,0].real
        angle = np.arccos(sum(raw*xs)/ np.sqrt(sum(raw*raw)*sum(xs*xs)))
        if angle > 3.14/2:
            xs = -1*np.dot(X.T,eigen_vecs)[:,0].real
            angle = np.arccos(sum(raw*xs)/ np.sqrt(sum(raw*raw)*sum(xs*xs)))
        print('angle', angle*180/np.pi)
        print("length:", np.sqrt(np.dot(xs,xs.T)))
        print("distance:",np.sqrt(np.dot((raw-xs),(raw-xs).T)))
    # 绘图
    # if k in ['Spline', 'Akima']:
    #     continue
    if k == 'raw':
        # subs[i].scatter(np.arange(20), res[:20])
        print('rec: ', eigen_vecs[:,0])
        plt.plot(ind, raw[:], label=k)
    else:
        # subs[i].scatter(np.arange(20), xs[:20])
        plt.plot(ind, xs[:], label=k)
        print('rec: ', vecs[:,0])
    # subs[i].set_ylim(-ymax,ymax)
    # subs[i].set_ylabel(k)
    i += 1
plt.legend()
plt.show()
# fig, subs = plt.subplots(3,1, sharex=True)
# for i in range(3):
#     subs[i].scatter(ind, eigen_vecs[:,i],c='black',s =1)
# plt.show()