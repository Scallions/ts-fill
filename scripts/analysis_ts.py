'''
@Author       : Scallions
@Date         : 2020-02-09 19:39:40
@LastEditors  : Scallions
@LastEditTime : 2020-02-09 21:21:48
@FilePath     : /gps-ts/scripts/analysis_ts.py
@Description  : Analysis gap size of gps time series
'''
import os
import sys

from loguru import logger

import ts.data as data
from ts.ts import SingleTs as Sts

sys.path.append("./")



def load_data() -> List[TimeSeries]:
    """Load data in dir data 
    
    Returns:
        List[TimeSeries]: a list of ts
    """
    tss = []
    files = os.listdir("./data")
    for file_ in files:
        if ".cwu.igs14.csv" in files_:
            tss.append(Sts("./data/" + file_))
    return tss


    


if __name__ == "__main__":
    tss = load_data()
    lengths = map(lambda x: x.gap_status(), tss)