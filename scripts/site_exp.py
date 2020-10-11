'''
Author       : Scallions
Date         : 2020-08-23 09:16:21
LastEditors  : Scallions
LastEditTime : 2020-10-11 14:15:27
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
    ['CAS1.AN.tenv3', 'MCM4.AN.tenv3', 'DAV1.AN.tenv3'],
    ['CAS1.AN.tenv3', 'CRAR.AN.tenv3', 'MCM4.AN.tenv3'],
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

def load_data2(lengths=6, epoch=6):
    """load data
    
    Args:
        lengths (int, optional): nums of mul ts. Defaults to 3.
    
    Returns:
        [mts]: mts s
    """
    dir_path = './data/'
    tss = []
    files = os.listdir(dir_path)
    for file_ in files:
        if '.cwu.igs14.csv' in file_:
            tss.append(Mts(dir_path + file_, data.FileType.Cwu))
    nums = len(tss)
    rtss = []
    import random
    random.seed(0)
    for j in range(epoch):
        random.shuffle(tss)
        for i in range(0, nums - lengths, lengths):
            mts = tool.concat_multss(tss[i:i + lengths])
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
    fillers = [
        ### imputena
        fill.MICEFiller,

        ### missingpy
        fill.MissForestFiller,

        ### miceforest
        fill.MiceForestFiller,

        ### para
        # fill.RandomForestFiller,

        ### magic
        # fill.MagicFiller,
        
        ### fancyimpute
        fill.KNNFiller,
        # fill.SoftImputeFiller,
        # fill.IterativeSVDFiller,
        fill.IterativeImputerFiller,
        fill.MatrixFactorizationFiller,
        # fill.BiScalerFiller,
        # fill.NuclearNormMinimizationFiller,

        ### scipy
        # fill.SLinearFiller, # 一阶样条插值
        # fill.SplineFiller, # 三次样条
        # fill.AkimaFiller,
        # fill.PolyFiller, # 二阶多项式插值÷
        # fill.PiecewisePolynomialFiller, 
        # fill.KroghFiller, # overflow
        # fill.QuadraticFiller, # 二次
        # fill.BarycentricFiller, # overflow
        # fill.FromDerivativesFiller,
        # fill.PchipFiller, # 三阶 hermite 插值

        ### matlab
        fill.RegEMFiller, 

        ### self
        fill.MLPFiller,
        # fill.ConvFiller,
        # fill.SSAFiller,
    ]

    # 点的大小
    pltsize = 2

    gap_size = 30
    ltss = [ts.get_longest() for ts in tss]
    val_tss = [(ts, *ts.make_gap(gap_size,cache_size=100, cper=0.5, c_i=False, c_ii=['n0,e0,u0'], gmax=gap_size*5)) for ts in ltss]
    for tsl, tsg, gidx, gridx in val_tss:
        # set up gidx
        # gidx = list(pd.date_range('2018/10/01','2018/11/01'))
        # gidx = gidx + list(pd.date_range('2018/7/1', '2018/8/1'))
        # gidx = gidx + list(pd.date_range('2017/7/20', '2017/8/20'))
        # gidx = gidx + list(pd.date_range('2017/5/1', '2017/6/1'))
        # gidx = gidx + list(pd.date_range('2018/2/1', '2018/3/1'))
        
        # 去趋势
        # trends, noises = tool.remove_trend(tsl)
        # tsg = tsl.copy()
        # tsg.loc[gidx, gridx] = None
        # noises.loc[gidx, gridx] = None
        # noises = Mts(datas=noises,indexs=noises.index,columns=noises.columns)
        
        fig, subs = plt.subplots(len(fillers)+1,1, sharex=True)
        subs[0].scatter(tsl.index, tsl[gridx[2]], label="raw", s=pltsize)
        subs[0].scatter(tsl.index, tsg[gridx[2]], s=pltsize, c="black") 
        subs[0].set_ylabel("raw")
        tsg.to_csv("res/gap.csv")
        tsl.to_csv("res/raw.csv")
        for i, filler in enumerate(fillers):
            # tsc = filler.fill(noises)
            # tsc = trends + tsc
            tsc = filler.fill(tsg)
            subs[i+1].scatter(tsl.index, tsc[gridx[2]], s=pltsize)
            subs[i+1].scatter(tsl.index, tsg[gridx[2]], s=pltsize, c="black") 
            subs[i+1].set_ylabel(filler.name)
            tsc.to_csv("res/"+filler.name+".csv")
        # plt.scatter(tsl.index, tsg[gridx[0]], s=pltsize)
        # plt.legend()
        plt.show()
        break