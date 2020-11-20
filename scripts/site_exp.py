'''
Author       : Scallions
Date         : 2020-08-23 09:16:21
LastEditors  : Scallions
LastEditTime : 2020-11-20 20:38:20
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
import numpy as np
import matplotlib.pyplot as plt

SITES = [
    # ['FONP.AN.tenv3', 'OHI3.AN.tenv3', 'PAL2.AN.tenv3', 'PALM.AN.tenv3', 'PALV.AN.tenv3'],
    ['BLAS.NA.tenv3', 'DGJG.NA.tenv3', 'DKSG.NA.tenv3', 'HJOR.NA.tenv3', 'HMBG.NA.tenv3', 'HRDG.NA.tenv3', 'JGBL.NA.tenv3', 'JWLF.NA.tenv3', 'KMJP.NA.tenv3', 'KMOR.NA.tenv3', 'KUAQ.NA.tenv3', 'KULL.NA.tenv3', 'LBIB.NA.tenv3', 'LEFN.NA.tenv3', 'MARG.NA.tenv3', 'MSVG.NA.tenv3', 'NRSK.NA.tenv3', 'QAAR.NA.tenv3', 'UTMG.NA.tenv3', 'YMER.NA.tenv3'],
    # ['VNAD.AN.tenv3', 'PALM.AN.tenv3', 'PAL2.AN.tenv3', 'PRPT.AN.tenv3', 'DUPT.AN.tenv3'],
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
    dir_path = './data/greenland/'
    rtss = []
    for site in SITES:
        tss = []
        for file_ in site:
            tss.append(Mts(dir_path + file_, data.FileType.Ngl))
        mts = tool.concat_u_multss(tss)
        rtss.append(mts)
    return rtss

def load_data2(lengths=3, epoch=6):
    """load data
    
    Args:
        lengths (int, optional): nums of mul ts. Defaults to 3.
    
    Returns:
        [mts]: mts s
    """
    dir_path = './data/greenland/'
    tss = []
    files = os.listdir(dir_path)
    # for file_ in files:
    for file_ in ['BLAS.NA.tenv3', 'DGJG.NA.tenv3', 'DKSG.NA.tenv3', 'HJOR.NA.tenv3', 'HMBG.NA.tenv3', 'HRDG.NA.tenv3', 'JGBL.NA.tenv3', 'JWLF.NA.tenv3', 'KMJP.NA.tenv3', 'KMOR.NA.tenv3', 'KUAQ.NA.tenv3', 'KULL.NA.tenv3', 'LBIB.NA.tenv3', 'LEFN.NA.tenv3', 'MARG.NA.tenv3', 'MSVG.NA.tenv3', 'NRSK.NA.tenv3', 'QAAR.NA.tenv3', 'UTMG.NA.tenv3', 'YMER.NA.tenv3']:
        if '.tenv3' in file_:
            tss.append((file_, Mts(dir_path + file_, data.FileType.Ngl)))
    nums = len(tss)
    rtss = []
    import random
    for j in range(epoch):
        random.shuffle(tss)
        for i in range(0, nums - 1, lengths):
            try:
                if (i+lengths) > nums:
                    continue
                data1 = tss[i:i + lengths]
                names = [d[0] for d in data1]
                ttss  =[d[1] for d in data1]
                mts = tool.concat_u_multss(ttss)
            except:
                continue
            rtss.append((names,mts))
    return rtss


if __name__ == "__main__":
    tss = load_data2(lengths=20, epoch=300)
    
    # load from csv

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
        # fill.MICEFiller, # **

        ### missingpy
        fill.MissForestFiller, # **

        ### miceforest
        # fill.MiceForestFiller, # **

        ### para
        # fill.RandomForestFiller,

        ### magic
        # fill.MagicFiller,
        
        ### fancyimpute
        fill.KNNFiller, # **
        # fill.SoftImputeFiller,
        # fill.IterativeSVDFiller,
        # fill.IterativeImputerFiller, # **
        # fill.MatrixFactorizationFiller, # **
        # fill.BiScalerFiller,
        # fill.NuclearNormMinimizationFiller,

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

        ### self
        # fill.MLPFiller,
        # fill.BritsFiller,
        # fill.GainFiller,
        # fill.ConvFiller,
        # fill.SSAFiller,
    ]

    # 点的大小
    pltsize = 2

    gap_size = 7
    ltss = [[names,ts.get_longest()] for names,ts in tss if len(ts.get_longest()) > 400]
    # val_tss = [(ts, *ts.make_gap(gap_size,cache_size=100, cper=0.5, c_i=False, c_ii=['n0,e0,u0'], gmax=gap_size)) for ts in ltss]
    val_tss = [(names,ts, *ts.make_gap(gap_size,cache_size=100, cper=0.5, gmax=gap_size* 25)) for names, ts in ltss]
    mm = []
    for names, tsl, tsg, gidx, gridx in val_tss:
        # set up gidx
        # gidx = list(pd.date_range('2017/01/01','2017/07/01'))
        # gidx = gidx + list(pd.date_range('2018/7/1', '2018/8/1'))
        # gidx = gidx + list(pd.date_range('2017/7/20', '2017/8/20'))
        # gidx = gidx + list(pd.date_range('2017/5/1', '2017/6/1'))
        # gidx = gidx + list(pd.date_range('2018/2/1', '2018/3/1'))
        
        # 去趋势
        trends, noises = tool.remove_trend(tsl)
        tsg = tsl.copy()
        tsg.loc[gidx, gridx] = None
        # noises.to_csv("res/raw.csv")
        noises.loc[gidx, gridx] = None
        noises = Mts(datas=noises,indexs=noises.index,columns=noises.columns)
        
        # fig, subs = plt.subplots(len(fillers)+1,1, sharex=True)
        # subs[0].scatter(tsl.index, tsl[gridx[2]], label="raw", s=pltsize)
        # subs[0].scatter(tsl.index, tsg[gridx[2]], s=pltsize, c="black") 
        # subs[0].set_ylabel("raw")
        ## compare to find 
        raw = tsl.loc[gidx,gridx]
        reg = fill.RegEMFiller().fill(noises)
        rf = fill.MissForestFiller().fill(noises)
        reg = trends + reg
        rf = trends + rf
        reg = reg.loc[gidx,gridx]
        rf = rf.loc[gidx,gridx]
        reg = reg.sub(raw).std()
        rf = rf.sub(raw).std()
        # rf = rf[2] +rf[5]
        # reg = reg[2] + reg[5]
        dd = reg - rf 
        s = dd.sum()
        if len(mm) == 0 or (s > 0 and s > mm[-1][0]):
            mm.append([s,dd,names])
        else:
            continue

        tsg.to_csv("res/gap.csv")
        tsl.to_csv("res/raw.csv")
        for i, filler in enumerate(fillers):
            tsc = filler.fill(noises)
            tsc = trends + tsc


            # tsc = filler.fill(tsg)
            # subs[i+1].scatter(tsl.index, tsc[gridx[2]], s=pltsize)
            # subs[i+1].scatter(tsl.index, tsg[gridx[2]], s=pltsize, c="black") 
            # subs[i+1].set_ylabel(filler.name)
            tsc.to_csv("res/"+filler.name+".csv")

        # break

        # plt.scatter(tsl.index, tsg[gridx[0]], s=pltsize)
        # plt.legend()
        # plt.show()

        # # for nni valid
        # # gidx = list(pd.date_range('2017/01/01','2017/07/01')) 
        # # trends, noises = tool.remove_trend(tsl)
        # # tsg = tsl.copy()
        # # tsg.loc[gidx, gridx] = None
        # # noises.loc[gidx, gridx] = None
        # # noises = Mts(datas=noises,indexs=noises.index,columns=noises.columns)
        # raw = tsl.to_numpy().astype(np.float64)
        # aft = tsc.to_numpy().astype(np.float64)
        # a = np.abs((raw-aft)) > 0.00001
        # b = np.where(np.any(a, axis=1))
        # c = np.where(np.any(a, axis=0))
        # dd = 2
        # d = raw[:,c[0][dd]]-aft[:,c[0][dd]]
        # # print(f"mean, std, mae: ",d.mean(), d.std(), np.abs(d).mean())
        # """@nni.report_final_result((d*d).mean())"""


    print(len(mm),mm[-1])

    