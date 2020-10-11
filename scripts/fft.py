'''
Author       : Scallions
Date         : 2020-09-16 15:25:04
LastEditors  : Scallions
LastEditTime : 2020-10-10 22:49:16
FilePath     : /gps-ts/scripts/fft.py
Description  : 
'''
import os
import sys
sys.path.append('./')
import ts.tool as tool
import ts.data as data
from ts.timeseries import SingleTs as Sts
from ts.timeseries import MulTs as Mts
import ts.fill as fill
from loguru import logger
import pandas as pd
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft, ifft


if __name__ == "__main__":
    # tss = load_data()
    axis_name = ["N", "E", "U"]

    plt.rcParams['figure.figsize'] = (8.0, 4.0) 
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


    ind = ind[b[0][0]:b[0][-1]]
    # ind = ind[b[0]]

    for k,v in data.items():
        data[k] = v[b[0][0]:b[0][-1],:]
        # data[k] = v[b[0],:]

    dd = 0

    for i, name in enumerate(names):
        x = data[name][:365*2,0]
        x = mean(x)
        xf = np.fft.rfft(x)
        xfp = np.abs(xf)/(len(x)/2)
        freq = np.fft.rfftfreq(len(x), d=1.0/365)
        print(name)
        print(freq[:10])
        plt.subplot(100*len(names)+10+i+1)
        plt.plot(freq[:20],xfp[:20])
        ax = plt.gca()
        print(xfp[:10])
        from matplotlib.pyplot import MultipleLocator
        ax.xaxis.set_major_locator(MultipleLocator(1))
        plt.ylabel(name)
        # plt.ylim(0,2)



    plt.show()