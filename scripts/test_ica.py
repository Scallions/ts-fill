'''
Author       : Scallions
Date         : 2020-11-12 17:15:06
LastEditors  : Scallions
LastEditTime : 2020-11-14 16:41:06
FilePath     : /gps-ts/scripts/test_ica.py
Description  : 
'''
import os
import sys
sys.path.append('./')
from ts.timeseries import MulTs as Mts
import ts.fill as fill
import ts.data as data
import ts.tool as tool
import pandas as pd
import numpy as np
from sklearn.decomposition import FastICA, PCA
import matplotlib.pyplot as plt

SITES = [
    ['VNAD.AN.tenv3', 'PALM.AN.tenv3', 'PAL2.AN.tenv3', 'PRPT.AN.tenv3', 'DUPT.AN.tenv3', 'OHI3.AN.tenv3']
    # # ['CAS1.AN.tenv3', 'MCM4.AN.tenv3', 'DAV1.AN.tenv3'],
    # # ['CAS1.AN.tenv3', 'CRAR.AN.tenv3', 'MCM4.AN.tenv3'],
    # ['CAPF.AN.tenv3', 'PRPT.AN.tenv3', 'PAL2.AN.tenv3'], # an
    # ['CAPF.AN.tenv3', 'PRPT.AN.tenv3', 'ROBN.AN.tenv3', 'PAL2.AN.tenv3'],
    # ['LEIJ.EU.tenv3', 'WARN.EU.tenv3', 'POTS.EU.tenv3', 'PTBB.EU.tenv3'], # eu
]

def load_data():
    """load data
    
    Args:
        lengths (int, optional): nums of mul ts. Defaults to 3.
    
    Returns:
        [mts]: mts s
    """
    dir_path = './data/an/'
    rtss = []
    for site in SITES:
        tss = []
        for file_ in site:
            tss.append(Mts(dir_path + file_, data.FileType.Ngl))
        mts = tool.concat_or_multss(tss)
        rtss.append(mts)
    return rtss

def trend_of_ts(ts):
    length = len(ts)
    x = np.array(list(range(length))).reshape((length,1))
    sinx = np.sin(x*np.pi*2/365)
    cosx = np.cos(x*np.pi*2/365)
    sin2x = np.sin(2*x*np.pi*2/365)
    cos2x = np.cos(2*x*np.pi*2/365)
    ones = np.ones((length,1))
    data = np.hstack((ones, x, sinx, cosx, sin2x, cos2x))
    b = np.dot(np.dot(np.linalg.inv(np.dot(data.transpose(), data)), data.transpose()), ts)
    ts_hat = np.dot(data, b)
    noise = ts - ts_hat 
    return ts_hat, noise


X = pd.read_csv("./res/raw.csv", index_col="jd",parse_dates=True)
X = load_data()[0]
X = X.get_or_longest()
X = fill.MissForestFiller.fill(X)
ind = X.index
data = X.to_numpy()
noises, data = trend_of_ts(data)

transformer = FastICA(n_components=7, random_state=0)
ptran = PCA(n_components=7, random_state=0)


X_tran = transformer.fit_transform(data)
X_ptran = ptran.fit_transform(data)
# print(X_ptran.shape)
A_ = transformer.mixing_
A_ = ptran.components_
A_ = X_tran

# print(ptran.explained_variance_)
# print(ptran.explained_variance_ratio_)
# print(ptran.singular_values_)
# print(A_[:,0])
# print(ptran.components_[0,:])


## show data

# for i in range(12):
#     plt.subplot(12,1,i+1)
#     plt.plot(ind,data[:,i])

# plt.subplot(9,1,1)
# plt.plot(ind, data)
# plt.subplot(9,1,2)
# plt.plot(ind,noises)

for i in range(7):
    plt.subplot(7,1,i+1)
    plt.plot(ind, A_[:,i], label=f"ica{i}")
# # plt.plot(ind, X_ptran[:,0], label="pca")
# plt.legend()
plt.show()