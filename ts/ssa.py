'''
@Author       : Scallions
@Date         : 2020-02-22 10:40:10
@LastEditors  : Scallions
@LastEditTime : 2020-02-22 10:40:18
@FilePath     : /gps-ts/ts/ssa.py
@Description  : 
'''

import numpy as np 
from numpy.linalg import eig
import matplotlib.pyplot as plt
import TS.tool as tool
from loguru import logger


def SSA(X, K, M, callback=None,plotpc=False):
    """
    This function can analyze X by SSA with arguments K and M and return the rebuild version. K is the number of EOFs used to rebuild the X. M is the length of the sub series.The `callback` argument is a function which input EOFs and can process EOFs by what way u want such as plot the EOFs.The `plotpc` is a boolean that control whether plot the PCs time series.

    args:
        X: List
        K: Int
        M: Int
        callback: Function(Matrix)
        plotpc: Boolean 

    """
    N = X.shape[0]
    N_ = N-M+1
    D = np.zeros([M,N-M+1])
    for i in range(N_):
        D[:,i] = X[i:i+M]
    Cx = np.dot(D, D.T) / (N - M + 1)
    eigs, E = eig(Cx)

    # plot sth with E
    if callback != None:
        # print(eigs)
        callback(E)
    T = np.dot(E[:,0:K].T, D) # PC times series
    D_ = np.dot(E[:,0:K], T)
    if plotpc:
        # plot top-n pc-ts
        # plt.plot(T[0,:],label='0')
        plt.plot(T[1,:],label='1')
        plt.plot(T[2,:],label='2')
        plt.plot(T[3,:],label='3')
        plt.title("pc-ts")
        plt.legend()
        plt.show()
    #print(T[2,100:110])
    X_ = np.zeros(X.shape)
    Dr = np.fliplr(D_)
    for i in range(X.shape[0]):
        X_[i] = np.mean(np.diag(Dr,N_-i-1))
    return X_

def iter_SSA_inner(X, sigma, K, M):
    """
    X 是缺失时间序列，对于给定的sigma,K,M 可以进行迭代插值，返回插值后的时间序列
    args:
        X::Ts
        sigama::Float
        K::Int
        M::Int
    return:
        _::Ts
    """
    #X = X.to_numpy(copy=True)
    print(K)
    X_m = X[~np.isnan(X)].mean()
    Xt = X.copy()
    Xt[np.isnan(X)] = X_m
    Xs = Xt - X_m
    #plt.plot(Xs)
    n = 0
    while True:
        Xn = SSA(Xs, K, M)
        Xn[~np.isnan(X)] = X[~np.isnan(X)] - X_m
        d = np.abs(Xn[np.isnan(X)]-Xs[np.isnan(X)]).max()
        if n > 200:
            sigma += 0.001
            logger.debug("SSA inner sigma iter: {}",n+1)
            logger.debug("delta : {}",d)
        if d < sigma:
            #print(Xs[125],Xn[125])
            break
        Xs[np.isnan(X)] = Xn[np.isnan(X)]
        n = n+1
    return Xn + X_m

def iter_SSA(X, indexs_cv, Mmin=1, Mmax=365, sigma=0.01):
    """
    X 是缺失时间序列，indexs_cv 是验证集位置，Mmax是M的最大值,sigma为收敛阈值，验证损失值最小的五组（K,M）
    args:
        X::Ts
        Mmax::Int
        sigma::Float
        indexs_cv::[Int]
    return:
        _::[(Int,Int)]
    """
    Deltas = np.zeros((Mmax+1,Mmax+1))
    Xc = X.copy()
    Xc[indexs_cv] = None
    # dd = []
    # plt.plot(Xc)
    # plt.show()
    for M in range(Mmin,Mmax):
        K0 = 1
        K4 = M
        print("M :", M)
        while True:
            K0 = K0
            K4 = K4
            K2 = (K0 + K4)//2
            K1 = (K0 + K2)//2
            K3 = (K2 + K4)//2
            # print(K0,K1,K2,K3,K4)
            if Deltas[M,K0] == 0:
                X0 = iter_SSA_inner(Xc,sigma,K0,M)
                Deltas[M,K0] = tool.delta(X[indexs_cv], X0[indexs_cv])
            if Deltas[M,K1] == 0:
                X1 = iter_SSA_inner(Xc,sigma,K1,M)
                Deltas[M,K1] = tool.delta(X[indexs_cv], X1[indexs_cv])
            if Deltas[M,K2] == 0:
                X2 = iter_SSA_inner(Xc,sigma,K2,M)
                Deltas[M,K2] = tool.delta(X[indexs_cv], X2[indexs_cv])
            if Deltas[M,K3] == 0:
                X3 = iter_SSA_inner(Xc,sigma,K3,M)
                Deltas[M,K3] = tool.delta(X[indexs_cv], X3[indexs_cv])
            if Deltas[M,K4] == 0:
                X4 = iter_SSA_inner(Xc,sigma,K4,M)
                Deltas[M,K4] = tool.delta(X[indexs_cv], X4[indexs_cv])
            deltas = [Deltas[M,K0],Deltas[M,K1],Deltas[M,K2],Deltas[M,K3],Deltas[M,K4]]
            # print(deltas)
            if deltas[1] <= deltas[2] and deltas[1] <= deltas[0]:
                K4 = K2
            elif deltas[2] <= deltas[3] and deltas[2] <= deltas[1]:
                K0 = K1
                K4 = K3
            elif deltas[3] <= deltas[4] and deltas[3] <= deltas[2]:
                K0 = K2
                K4 = K4
            elif deltas[1] < deltas[2]:
                K4 = K1
            elif deltas[2] > deltas[3]:
                K0 = K3
            if K0 == K4 or K0 == K4 - 1:
                print(M,K0,K4)
                break
    return Deltas
