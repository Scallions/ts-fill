'''
@Author       : Scallions
@Date         : 2020-02-05 13:06:55
@LastEditors  : Scallions
@LastEditTime : 2020-03-24 19:49:55
@FilePath     : /gps-ts/ts/tool.py
@Description  : 
'''
from __future__ import division, print_function

import datetime as dt

import numpy as np
import julian
import pandas as pd
from loguru import logger
from PyAstronomy import pyasl
import random


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

def jd2datetime(jd):
    return pd.Timestamp(julian.from_jd(jd, fmt='jd'))
            
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

def make_gap(ts, gap_size=3, per = 0.2):
    """make gap in ts return copy
    
    Args:
        ts (df): ts without gap
        gap_size (int, optional): gap size will in the ts. Defaults to 3.
    """
    # TODO: make gap not neighbor @scallions
    length = len(ts)
    gap_count = int(length * per) # 20% gap
    gap_index = []
    while len(gap_index)*gap_size < gap_count:
        r_index = random.randint(0,length-gap_size)
        flag = False
        for gap in gap_index:
            if abs(gap - r_index) <= gap_size:
                flag = True
                break
        if flag:
            continue
        gap_index.append(r_index)
    
    logger.debug(f"gap_count: {gap_count}, gap_index len: {len(gap_index)}")
    tsc = ts.copy()
    gap_indexs = []
    
    for ind in gap_index:
        for i in range(gap_size):
            g_ind = tsc.index[ind] + pd.Timedelta(days=i)
            gap_indexs.append(g_ind)
            tsc.loc[g_ind] = None
    return tsc,gap_indexs

def get_status_between(ts1, ts2):
    """description for delta between ts1 and ts2
    
    Args:
        ts1 (ts): timeseries
        ts2 (ts): timeseries
    """
    delta = pd.Series(np.abs(ts1 - ts2))
    return delta

def get_longest(ts):
    """get a longest sub ts in ts without gap
    
    Args:
        ts (df): ts with gap
    """
    gap_size = ts.gap_status()
    max_i = gap_size.lengths.index(max(gap_size.lengths))
    tsl = ts.loc[gap_size.starts[max_i]:gap_size.starts[max_i]+pd.Timedelta(days=gap_size.lengths[max_i]-1)]
    return tsl

def get_all_cts(ts):
    """get all continue sub ts in ts
    
    Args:
        ts (ts): ts with gap
    """
    gap_size = ts.gap_status()
    sub_ts = []
    for i, length in enumerate(gap_size.lengths):
        if length < 0: continue
        sub_ts.append(ts.loc[gap_size.starts[i]:gap_size.starts[i]+pd.Timedelta(days=length-1)])
    return sub_ts

def fill_res(ts,tsf,gidx):
    """result status between tsg and tsf
    
    Args:
        ts (ts): true ts
        tsf (ts): ts after imputation
        gidx (list): gap idx list
    """
    ts_g = tsf.loc[gidx]
    ts_t = ts.loc[gidx]

    delta = ts_g.sub(ts_t).abs()

    status = delta.describe()

    return status


def concat_multss(tss):
    res = tss[0]
    for ts in tss[1:]:
        r_idx = res.index
        t_idx = ts.index
        a_idx = ts.index & r_idx 
        res = pd.concat([res.loc[a_idx], ts.loc[a_idx]],axis=1)
    return type(tss[0]).from_df(res)