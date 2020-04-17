'''
@Author       : Scallions
@Date         : 2020-03-25 08:39:45
@LastEditors  : Scallions
@LastEditTime : 2020-04-17 08:42:37
@FilePath     : /gps-ts/scripts/compare-mults-fill.py
@Description  : 
'''

import os
import sys

sys.path.append("./")

import ts.tool as tool
import ts.data as data
from ts.timeseries import SingleTs as Sts
from ts.timeseries import MulTs as Mts
import ts.fill as fill
from loguru import logger
import pandas as pd
import time 

def load_data(lengths=3,epoch=6):
    """load data
    
    Args:
        lengths (int, optional): nums of mul ts. Defaults to 3.
    
    Returns:
        [mts]: mts s
    """
    dir_path = "./data/"
    tss = []
    files = os.listdir(dir_path)
    for file_ in files:
        if ".cwu.igs14.csv" in file_:
            tss.append(Mts(dir_path + file_,data.FileType.Cwu))
    nums = len(tss)
    rtss = []

    # data increase
    import random
    for j in range(epoch):
        random.shuffle(tss)
        for i in range(0,nums-lengths, lengths):
            mts = tool.concat_multss(tss[i:i+lengths])
            rtss.append(mts)
    return rtss


if __name__ == "__main__":
    """genrel code for compare
    """
    result = {}    # result record different between gap sizes and filler functions
    tss = load_data(epoch=10)
    gap_sizes = [
        1,
        2,
        3,
        4,
        5,
        6,
        10,
        15,
        30,
        50
        ]

    fillers = [
        # fill.MeanFiller,
        # fill.MedianFiller,
        # fill.LinearFiller, 
        # fill.TimeFiller,
        # fill.QuadraticFiller,
        # fill.CubicFiller,
        # fill.SLinearFiller,
        # fill.PolyFiller,
        # # fill.BarycentricFiller,
        # fill.SplineFiller,
        # fill.PchipFiller,
        # # fill.KroghFiller,
        # fill.PiecewisePolynomialFiller,
        # fill.FromDerivativesFiller,
        # fill.AkimaFiller,
        # fill.RegEMFiller, 
        fill.MSSAFiller,
        ]

    result = pd.DataFrame(columns=[filler.name for filler in fillers],index=gap_sizes+['time','gap_count','count'])
    result.loc['time'] = 0
    for gap_size in gap_sizes:
        val_tss = [(ts.get_longest(), *ts.get_longest().make_gap()) for ts in tss]
        
        for i,filler in enumerate(fillers):
            res = None
            start = time.time()
            count = 0
            gap_count = 0
            for tsl, tsg, gidx, gridx in val_tss:
                if len(tsl) < 380: continue
                tsc = filler.fill(tsg)
                tsl.columns = tsc.columns
                this_res = tool.fill_res(tsc,tsl,gidx, gridx)
                if not isinstance(res, pd.DataFrame):
                    res = this_res
                else:
                    res = pd.concat([res,this_res], axis=1)

                count += len(tsl)
                gap_count += len(gidx)
            
            counts = res.loc['count'].values
            means = res.loc['mean'].values
            res_mean = sum(counts * means) / sum(counts)
            
            end  = time.time()
            result.loc[gap_size,filler.name] = res_mean
            result.loc['time', filler.name] = (end-start) + result.loc['time', filler.name]
            result.loc['count', filler.name] = count
            result.loc['gap_count', filler.name] = gap_count
            
            logger.info(f"{filler.name} mean: {res_mean} time: {end-start:0.4f} gap: {gap_size}")
            

    result.to_csv("res/fill_mul_gapsize.csv")
