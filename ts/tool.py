'''
@Author       : Scallions
@Date         : 2020-02-05 13:06:55
@LastEditors  : Scallions
@LastEditTime : 2020-02-29 21:39:27
@FilePath     : /gps-ts/ts/tool.py
@Description  : 
'''
from __future__ import division, print_function

import datetime as dt

import numpy as np
import pandas as pd
from loguru import logger
from PyAstronomy import pyasl


def dy2jd(dy):
    """convert float year to jd year
    
    Args:
        dy (Float): float year
    
    Returns:
        Int: jd days
    """
    gd = pyasl.decimalYearGregorianDate(dy, "tuple")
    ddt = dt.datetime(*gd)
    return round(pyasl.jdcnv(ddt))
            
def count_tss(tss):
    """Count ts and days in tss
    
    Args:
        tss (List[TimeSeries]): list of time series needed to count
    
    Returns:
        Tuple[Int, Int]: numbers of time series and days in tss
    """
    return len(tss), sum(map(lambda x: x.shape[0], tss))

def delta(X, X_):
    """
    求两时间序列X, X_ 的loss, 目前为 MSE
    args:
        X,X_::np.ndarray
    return:
        _::Float
    """
    return np.mean(np.square(X - X_))

def make_gap(ts, gap_size=3):
    """make gap in ts return copy
    
    Args:
        ts (timeseries): ts without gap
        gap_size (int, optional): gap size will in the ts. Defaults to 3.
    """
    length = len(ts)
    gap_count = int(length * 0.2) # 20% gap
    batch_count = length // gap_size

def get_status_between(ts1, ts2):
    """description for delta between ts1 and ts2
    
    Args:
        ts1 (ts): timeseries
        ts2 (ts): timeseries
    """
    delta = pd.Series(np.abs(ts1 - ts2))
    return delta