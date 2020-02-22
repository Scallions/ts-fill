'''
@Author       : Scallions
@Date         : 2020-02-09 19:39:40
@LastEditors  : Scallions
@LastEditTime : 2020-02-22 11:39:23
@FilePath     : /gps-ts/scripts/analysis_ts.py
@Description  : Analysis gap size of gps time series
'''
import os
import sys

sys.path.append("./")

from loguru import logger

import ts.data as data
from ts.ts import SingleTs as Sts




def load_data():
    """Load data in dir data 
    
    Returns:
        List[TimeSeries]: a list of ts
    """
    tss = []
    files = os.listdir("./data")
    for file_ in files:
        if ".cwu.igs14.csv" in file_:
            tss.append(Sts("./data/" + file_))
    return tss

def get_lens(tss):
    """get gap lens form tss
    
    Args:
        tss ([Ts]): a list of ts
    """
    return list(map(lambda  x: x.gap_status(), tss))





if __name__ == "__main__":
    tss = load_data()
    lengths = list(map(lambda x: x.gap_status(), tss))
    print(lengths[1])