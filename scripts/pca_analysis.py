'''
Author       : Scallions
Date         : 2020-08-29 16:05:38
LastEditors  : Scallions
LastEditTime : 2020-10-10 22:58:59
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
    fill.SLinearFiller, # 一阶样条插值
    fill.SplineFiller, # 三次样条
    fill.AkimaFiller,
    fill.RegEMFiller, 
    fill.MLPFiller,
    # fill.ConvFiller,
    # fill.PolyFiller, # 二阶多项式插值÷
    # fill.PiecewisePolynomialFiller, 
    # fill.KroghFiller, # overflow
    # fill.QuadraticFiller, # 二次
    # fill.BarycentricFiller, # overflow
    # fill.FromDerivativesFiller,
    # fill.PchipFiller, # 三阶 hermite 插值
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
ind = ind[b[0]]

for k,v in data.items():
    # data[k] = v[b[0][0]:b[0][-1],:]
    data[k] = v[b[0],:]

dd = 1

# fig, subs = plt.subplots(len(names),1, sharex=True)
i = 0
for k, x in data.items():
    # PCA
    X = mean(x[:,:3])
    n, m = X.shape 
    C = np.dot(X, X.T) / (n-1)
    eigen_vals, eigen_vecs = np.linalg.eig(C)
    s = eigen_vals.sum()
    # print(s)
    res = eigen_vals/s * 100
    r = res.cumsum()
    print(k)
    print(res.real[:3])
    print(r.real[:3])
    if k == 'raw':
        raw = np.dot(X.T,eigen_vecs)[2,:20].real
    else:
        xs = np.dot(X.T,eigen_vecs)[2,:20].real
        angle = np.arccos(sum(raw*xs)/ np.sqrt(sum(raw*raw)*sum(xs*xs)))
        if angle > 3.14/2:
            xs = -1*np.dot(X.T,eigen_vecs)[2,:20].real
            angle = np.arccos(sum(raw*xs)/ np.sqrt(sum(raw*raw)*sum(xs*xs)))
        print('angle', angle)
    # 绘图
    if k in ['Spline', 'Akima']:
        continue
    if k == 'raw':
        # subs[i].scatter(np.arange(20), res[:20])
        plt.plot(res[:20], label=k)
    else:
        # subs[i].scatter(np.arange(20), xs[:20])
        plt.plot( xs[:20], label=k)
    # subs[i].set_ylim(-ymax,ymax)
    # subs[i].set_ylabel(k)
    i += 1
plt.legend()
plt.show()