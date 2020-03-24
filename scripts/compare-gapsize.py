'''
@Author       : Scallions
@Date         : 2020-02-28 19:50:25
@LastEditors  : Scallions
@LastEditTime : 2020-03-24 12:15:23
@FilePath     : /gps-ts/scripts/compare-gapsize.py
@Description  : gap size compare
'''
import os
import sys

sys.path.append("./")

import ts.tool as tool
import ts.data as data
from ts.timeseries import SingleTs as Sts
import ts.fill as fill
from loguru import logger
import pandas as pd
import time 

def load_data():
    """Load data in dir data 
    
    Returns:
        List[TimeSeries]: a list of ts
    """
    tss = []
    files = os.listdir("./data")
    for file_ in files:
        if ".cwu.igs14.csv" in file_:
            tss.append(Sts("./data/" + file_,data.FileType.Cwu))
    return tss


if __name__ == "__main__":
    """genrel code for compare
    """
    result = {}    # result record different between gap sizes and filler functions
    tss = load_data()
    gap_sizes = [3,4,5,6,10,15]
    fillers = [fill.MeanFiller, fill.LinearFiller, fill.MedianFiller,fill.SSAFiller]
    fillernames = ["Mean", "Linear", "Median", "SSA"]
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
                tsg, gidx = tsl.make_gap(gap_size)
                tsc = filler.fill(tsg)
                this_res = tool.fill_res(tsc,tsl,gidx)
                #logger.debug(this_res)
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
            

    result.to_csv("res/fill_gapsize.csv")
