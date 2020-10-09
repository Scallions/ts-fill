'''
Author       : Scallions
Date         : 2020-08-23 09:16:21
LastEditors  : Scallions
LastEditTime : 2020-10-09 14:56:15
FilePath     : /gps-ts/scripts/site_exp.py
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

SITES = [
    ['CAS1.AN.tenv3', 'CRAR.AN.tenv3', 'MCM4.AN.tenv3'],
    ['CRAR.AN.tenv3', 'CAS1.AN.tenv3', 'DAV1.AN.tenv3'],
]

def load_data(lengths=3, epoch=6):
    """load data
    
    Args:
        lengths (int, optional): nums of mul ts. Defaults to 3.
    
    Returns:
        [mts]: mts s
    """
    dir_path = './data/igs/'
    rtss = []
    for site in SITES:
        tss = []
        for file_ in site:
            tss.append(Mts(dir_path + file_, data.FileType.Ngl))
        mts = tool.concat_multss(tss)
        rtss.append(mts)
    return rtss


if __name__ == "__main__":
    tss = load_data()
    axis_name = ["N", "E", "U"]

    # 绘制两个站的原始图
    # fig,subs=plt.subplots(3,1, sharex=True)
    # for i in range(3):
    #     subs[i].scatter(tss[1].index, tss[1].iloc[:,i], s=1)
    #     subs[i].set_ylabel(axis_name[i]+'/mm')
    # fig.suptitle("CAS1")
    # fig,subs=plt.subplots(3,1, sharex=True)
    # for i in range(3):
    #     subs[i].scatter(tss[0].index, tss[0].iloc[:,i], s=1)
    #     subs[i].set_ylabel(axis_name[i]+'/mm')
    # fig.suptitle("CRAR")
    # plt.show()

    # 插值结果
    gap_size = 30
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

    # 点的大小
    pltsize = 2

    ltss = [ts.get_longest() for ts in tss]
    val_tss = [(ts, *ts.make_gap(gap_size,cache_size=30, cper=0.5, c_i=False, c_ii=['n0,e0,u0'])) for ts in ltss]
    for tsl, tsg, gidx, gridx in val_tss:
        # 去趋势
        trends, noises = tool.remove_trend(tsl)
        
        # set up gidx
        gidx = list(pd.date_range('2018/10/01','2018/11/01'))
        gidx = gidx + list(pd.date_range('2018/7/1', '2018/8/1'))
        
        tsg = tsl.copy()
        tsg.loc[gidx, gridx] = None
        noises.loc[gidx, gridx] = None
        noises = Mts(datas=noises,indexs=noises.index,columns=noises.columns)
        
        fig, subs = plt.subplots(len(fillers)+1,1, sharex=True)
        subs[0].scatter(tsl.index, tsl[gridx[2]], label="raw", s=pltsize)
        subs[0].scatter(tsl.index, tsg[gridx[2]], s=pltsize, c="black") 
        subs[0].set_ylabel("raw")
        tsl.to_csv("res/raw.csv")
        for i, filler in enumerate(fillers):
            tsc = filler.fill(noises)
            tsc = trends + tsc
            subs[i+1].scatter(tsl.index, tsc[gridx[2]], s=pltsize)
            subs[i+1].scatter(tsl.index, tsg[gridx[2]], s=pltsize, c="black") 
            subs[i+1].set_ylabel(filler.name)
            tsc.to_csv("res/"+filler.name+".csv")
        # plt.scatter(tsl.index, tsg[gridx[0]], s=pltsize)
        # plt.legend()
        plt.show()
        break