'''
@Author       : Scallions
@Date         : 2020-02-28 19:50:25
@LastEditors  : Scallions
@LastEditTime : 2020-03-09 21:01:33
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
    gap_sizes = [1,2,3,4,5]
    fillers = [fill.FbFiller, fill.SSAFiller]
    for gap_size in gap_sizes:
        for filler in fillers:
            tss_g = [tool.make_gap(ts, gap_size) for ts in tss] # make gap in ts
            tss_c = list(map(filler, tss_g)) # fill gap
            status = tool.get_status_between(tss, tss_c) # compare to raw data
            result[gap_size][filler.name] = status