'''
Author       : Scallions
Date         : 2020-08-29 16:05:38
LastEditors  : Scallions
LastEditTime : 2020-10-05 21:05:05
FilePath     : /gps-ts/scripts/space_reflect.py
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
    fill.RegEMFiller, 
    fill.MLPFiller,
    fill.PolyFiller, # 二阶多项式插值
    # fill.PiecewisePolynomialFiller, 
    # fill.KroghFiller, # overflow
    # fill.QuadraticFiller, # 二次
    fill.AkimaFiller,
    fill.SplineFiller, # 三次样条
    # fill.BarycentricFiller, # overflow
    # fill.FromDerivativesFiller,
    fill.PchipFiller, # 三阶 hermite 插值
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


ind = ind[b[0][0]:b[0][-1]]
# ind = ind[b[0]]

for k,v in data.items():
    data[k] = v[b[0][0]:b[0][-1],:]
    # data[k] = v[b[0],:]

dd = 2

i = 0

X = mean(data["raw"][:,:3])
n, m = X.shape 
C = np.dot(X, X.T) / (n-1)
eigen_vals, eigen_vecs = np.linalg.eig(C)
plt.plot(eigen_vecs[:3,:].T)

for k, x in data.items():
    xs = x[:,dd].T * eigen_vecs
    print(k, xs[0,:3].real)

plt.show()

