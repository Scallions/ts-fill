'''
@Author       : Scallions
@Date         : 2020-02-05 13:06:55
LastEditors  : Scallions
LastEditTime : 2020-11-19 09:58:47
FilePath     : /gps-ts/ts/tool.py
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
import os 
import torch

def set_seed(seed = 1):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)


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
    return pd.Timestamp(julian.from_jd(jd, fmt='jd').date())
    # return pd.Timestamp(julian.from_jd(jd, fmt='jd'))

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

def make_gap(ts, gmax=None, gap_size=3, per = 0.2, cache_size=0):
    """make gap idx
    
    Args:
        gap_size (int, optional): gap size will in the ts. Defaults to 3.
    """
    length = len(ts)
    if length == 0:
        raise Exception("ts length is zero")
    if per:
        gap_count = int(length * per) # 20% gap
    if gmax:
        gap_count = gmax
    gap_index = []
    while len(gap_index)*gap_size < gap_count:
        r_index = random.randint(cache_size,length-gap_size-cache_size)
        flag = False
        for gap in gap_index:
            if abs(gap - r_index) <= gap_size:
                flag = True
                break
        if flag:
            continue
        gap_index.append(r_index)
    
    logger.debug(f"gap_count: {gap_count}, gap_index len: {len(gap_index)}")
    gap_indexs = []
    
    for ind in gap_index:
        for i in range(gap_size):
            g_ind = ts.index[ind] + pd.Timedelta(days=i)
            gap_indexs.append(g_ind)
    return gap_indexs

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
    try:
        max_i = gap_size.lengths.index(max(gap_size.lengths))
    except:
        raise Exception("no enough")
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

def fill_res(ts,tsf,gidx,c_idx=None):
    """result status between tsg and tsf
    
    Args:
        ts (ts): true ts
        tsf (ts): ts after imputation
        gidx (list): gap idx list
    """
    if c_idx == None:
        ts_g = tsf.loc[gidx]
        ts_t = ts.loc[gidx]
    else:
        ts_g = tsf.loc[gidx,c_idx]
        ts_t = ts.loc[gidx,c_idx]
    delta = ts_g.sub(ts_t).abs()
    status = delta.describe()

    return status


def concat_multss(tss):
    res = tss[0]
    for ts in tss[1:]:
        r_idx = res.index
        t_idx = ts.index
        a_idx = t_idx & r_idx
        if len(a_idx) == 0:
            raise Exception("non common idx") 
        res = pd.concat([res.loc[a_idx], ts.loc[a_idx]],axis=1)
    return type(tss[0]).from_df(res)

def concat_u_multss(tss):
    res = tss[0].iloc[:,2]
    for ts in tss[1:]:
        r_idx = res.index
        t_idx = ts.index
        a_idx = t_idx & r_idx
        if len(a_idx) == 0:
            raise Exception("non common idx") 
        res = pd.concat([res.loc[a_idx], ts.loc[a_idx].iloc[:,2]],axis=1)
    return type(tss[0]).from_df(res)

def concat_or_multss(tss):
    res = tss[0]
    for ts in tss[1:]:
        r_idx = res.index
        t_idx = ts.index
        a_idx = t_idx & r_idx
        if len(a_idx) == 0:
            raise Exception("non common idx") 
        res = pd.concat([res, ts],axis=1)
    return type(tss[0]).from_df(res)

def remove_trend(mts):
    """从时间序列中提取周年项，和残差

    Args:
        mts ([type]): [description]

    Returns:
        [type]: [description]
    """
    def trend_of_ts(ts):
        length = len(ts)
        x = np.array(list(range(length))).reshape((length,1))
        # sinx = np.sin(x*np.pi*2/365)
        # cosx = np.cos(x*np.pi*2/365)
        # sin2x = np.sin(2*x*np.pi*2/365)
        # cos2x = np.cos(2*x*np.pi*2/365)
        ones = np.ones((length,1))
        # data = np.hstack((ones, x, sinx, cosx, sin2x, cos2x))
        data = np.hstack((ones, x))
        b = np.dot(np.dot(np.linalg.inv(np.dot(data.transpose(), data)), data.transpose()), ts)
        ts_hat = np.dot(data, b)
        noise = ts - ts_hat 
        trend = pd.DataFrame(data=ts_hat,index=mts.index, columns=[ts.name])
        noise = pd.DataFrame(data=noise,index=mts.index, columns=[ts.name])
        return trend, noise
    trends = None
    noise = None 
    for ts in mts:
        trend, noise = trend_of_ts(mts[ts])
        if trends is None:
            trends = trend
            noises = noise
        else:
            trends = pd.concat([trends, trend],axis=1)
            noises = pd.concat([noises, noise],axis=1)
    
    return trends, noises