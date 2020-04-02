'''
@Author       : Scallions
@Date         : 2020-02-28 19:50:25
@LastEditors  : Scallions
@LastEditTime : 2020-04-02 18:10:17
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
        fill.MeanFiller, 
        fill.MedianFiller,
        fill.RollingMeanFiller,
        fill.RollingMedianFiller,
        fill.LinearFiller, 
        fill.TimeFiller,
        fill.QuadraticFiller,
        fill.CubicFiller,
        fill.SLinearFiller,
        fill.AkimaFiller,
        fill.PolyFiller,
        fill.SplineFiller,
        fill.FbFiller,
        fill.SSAFiller
        ]

    fillernames = [
        "Mean", 
        "Median", 
        "RollingMean",
        "RollingMedian",
        "Linear", 
        "Time",
        "Quadratic",
        "Cubic",
        "SLinear",
        "Akima",
        "Poly",
        "Spline",
        "fbprophet",
        "SSA"
        ]


    result = pd.DataFrame(columns=fillernames,index=gap_sizes+['time','gap_count','count'])
    result.loc['time'] = 0
    for gap_size in gap_sizes:
        val_tss = [(ts.get_longest(), *ts.get_longest().make_gap(gap_size)) for ts in tss]
        
        for i,filler in enumerate(fillers):
            logger.info(f"gap size: {gap_size}, filler: {fillernames[i]}")
            res = None
            start = time.time()
            count = 0
            gap_count = 0
            for tsl, tsg, gidx in val_tss:
                if len(tsl) < 380: continue
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
