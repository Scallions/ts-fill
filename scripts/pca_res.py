'''
Author       : Scallions
Date         : 2020-08-29 16:05:38
LastEditors  : Scallions
LastEditTime : 2020-08-29 17:21:41
FilePath     : /gps-ts/scripts/pca_res.py
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


def mean(x):
    xc = x.mean(axis=0)
    return x - xc

ind = pd.read_csv("./res/raw.csv", index_col="jd",parse_dates=True).index[270:469]
raw = pd.read_csv("./res/raw.csv").to_numpy()[270:469,1:].astype(np.float64)
regem = pd.read_csv("./res/RegEM.csv").to_numpy()[270:469,1:].astype(np.float64)
mlp = pd.read_csv("./res/MLP.csv").to_numpy()[270:469,1:].astype(np.float64)

# a = np.abs((raw-regem)) > 0.00001
# b = np.where(np.any(a, axis=1))

# print(b)

names= ["RAW", "RegEM", "MLP"]
fig, subs = plt.subplots(3,1, sharex=True)
i = 0
for x in [raw, regem, mlp]:
    X = mean(x)
    n, m = X.shape 
    C = np.dot(X.T, X) / (n-1)
    eigen_vals, eigen_vecs = np.linalg.eig(C)
    s = eigen_vals.sum()
    # print(s)
    res = eigen_vals/s * 100
    r = res.cumsum()
    print(res)
    print(r)
    y = np.dot(X,eigen_vecs)[:,2]
    # if i == 2:
    #     y = -1 * y
    subs[i].scatter(ind, y, s=1)
    subs[i].set_ylabel(names[i])
    i += 1

plt.show()