'''
@Author       : Scallions
@Date         : 2020-06-06 17:11:31
@LastEditors  : Scallions
@LastEditTime : 2020-06-06 17:20:37
@FilePath     : /gps-ts/scripts/analysis_singlets.py
@Description  : 
'''
import os
import sys

sys.path.append("./")

from loguru import logger
import matplotlib.pyplot as plt

import ts.data as data
from ts.timeseries import SingleTs as Sts

import matplotlib.pyplot as plt

import statsmodels.tsa.stattools as stattools




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
            break
    return tss[0]


if __name__ == "__main__":
    ts = load_data()
    # ts.plot()  
    
    acf = stattools.acf(ts)
    # plt.plot(acf)
    plt.plot(ts.rolling(window=20, center=False))
    plt.show()
    

