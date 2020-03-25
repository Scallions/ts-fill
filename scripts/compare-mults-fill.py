'''
@Author       : Scallions
@Date         : 2020-03-25 08:39:45
@LastEditors  : Scallions
@LastEditTime : 2020-03-25 18:06:44
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
    tss = []
    files = os.listdir("./data")
    for file_ in files:
        if ".cwu.igs14.csv" in file_:
            tss.append(Mts("./data/" + file_,data.FileType.Cwu))
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
    tss = load_data(epoch=20)
    gap_sizes = [3,5,10,15,30,50]
    fillers = [fill.RegEMFiller, fill.MSSAFiller]
    fillernames = ["RegEM","MSSA"]
    result = pd.DataFrame(columns=fillernames,index=gap_sizes+['time','gap_count','count'])
    result.loc['time'] = 0
    for gap_size in gap_sizes:
        for i,filler in enumerate(fillers):
            res = None
            start = time.time()
            count = 0
            gap_count = 0
            for ts in tss:
                tsl = ts.get_longest()
                if len(tsl) < 380: break
                tsg, gidx, gridx = tsl.make_gap(gap_size)
                tsc = filler.fill(tsg)
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
            result.loc[gap_size,fillernames[i]] = res_mean
            result.loc['time', fillernames[i]] = (end-start) + result.loc['time', fillernames[i]]
            result.loc['count', fillernames[i]] = count
            result.loc['gap_count', fillernames[i]] = gap_count
            
            logger.info(f"{fillernames[i]} mean: {res_mean} time: {end-start:0.4f} gap: {gap_size}")
            

    result.to_csv("res/fill_mul_gapsize.csv")
