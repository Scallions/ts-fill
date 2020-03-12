'''
@Author       : Scallions
@Date         : 2020-02-28 19:50:25
@LastEditors  : Scallions
@LastEditTime : 2020-03-12 15:34:00
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
    gap_sizes = [3,4,5,6]
    fillers = [fill.MeanFiller, fill.LinearFiller, fill.MedianFiller]
    fillernames = ["Mean", "Linear", "Median"]
    result = pd.DataFrame(columns=fillernames,index=gap_sizes)
    for gap_size in gap_sizes:
        for i,filler in enumerate(fillers):
            res = None
            for ts in tss[0:1]:
                tsl = ts.get_longest()
                tsg, gidx = tsl.make_gap(gap_size)
                tsc = filler.fill(tsg)
                this_res = tool.fill_res(tsc,tsl,gidx)
                logger.debug(this_res)
                if not isinstance(res, pd.DataFrame):
                    res = this_res
                else:
                    res = pd.concat([res,this_res])
            result.loc[gap_size,fillernames[i]] = res.loc["mean"].item()

    result.to_csv("res/fill_gapsize.csv")
