'''
@Author       : Scallions
@Date         : 2020-08-01 14:58:15
LastEditors  : Scallions
LastEditTime : 2020-08-22 11:43:02
FilePath     : /gps-ts/scripts/fill_view.py
@Description  : 
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


def load_data(lengths=3, epoch=6):
    """load data
    
    Args:
        lengths (int, optional): nums of mul ts. Defaults to 3.
    
    Returns:
        [mts]: mts s
    """
    dir_path = './data/igs/'
    tss = []
    files = os.listdir(dir_path)
    for file_ in files:
        if '.AN.tenv3' in file_:
            tss.append(Mts(dir_path + file_, data.FileType.Ngl))
    nums = len(tss)
    rtss = []
    import random
    for j in range(epoch):
        random.shuffle(tss)
        for i in range(0, nums - lengths, lengths):
            try:
                mts = tool.concat_multss(tss[i:i + lengths])
            except:
                continue
            rtss.append(mts)
    return rtss

if __name__ == '__main__':
    """genrel code for compare
    """
    result = {}

    # 定义 数据集
    clip_length = 800
    tss = load_data(lengths=3, epoch=50)
    tsls = [ts.get_longest() for ts in tss if len(ts) > clip_length]
    tsls = [tsl for tsl in tsls if len(tsl) > clip_length]
    if len(tsls) == 0:
        logger.error("No data")
        sys.exit()

    # 定义比较的 gap size
    gap_sizes = [
        # 50, 
        # 100, 
        10
        ]

    # 定义比较的filler 种类
    fillers = [
        fill.SLinearFiller, 
        # fill.CubicFiller,
        # fill.PiecewisePolynomialFiller,
        # fill.FromDerivativesFiller,
        # fill.FromDerivativesFiller,
        # fill.QuadraticFiller,
        # fill.AkimaFiller,
        # fill.SplineFiller,
        # fill.BarycentricFiller,
        # fill.KroghFiller,
        fill.PchipFiller,
        fill.RegEMFiller, 
        fill.MLPFiller
        ]
    

    pltsize = 2

    for gap_size in gap_sizes:
        val_tss = [(ts, *ts.make_gap(gap_size, cache_size=200, cper=0.5, c_i
            =False, )) for ts in tsls]
        for tsl, tsg, gidx, gridx in val_tss:
            plt.scatter(tsl.index, tsl[gridx[0]], label="raw", s=pltsize)
            for i, filler in enumerate(fillers):
                if len(tsl) < 380:
                    continue
                tsc = filler.fill(tsg)
                plt.scatter(tsl.index, tsc[gridx[0]], label=filler.name, s=pltsize)
            plt.scatter(tsl.index, tsg[gridx[0]], s=pltsize)
            plt.legend()
            plt.show()
            break