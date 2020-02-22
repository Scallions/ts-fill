'''
@Author       : Scallions
@Date         : 2020-02-05 13:06:55
@LastEditors  : Scallions
@LastEditTime : 2020-02-09 21:18:54
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