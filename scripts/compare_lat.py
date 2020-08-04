"""
@Author       : Scallions
@Date         : 2020-06-02 15:08:15
@LastEditors  : Scallions
@LastEditTime : 2020-06-02 15:45:39
@FilePath     : /gps-ts/scripts/compare_lat.py
@Description  : 
"""
import os
import sys
sys.path.append('./')
import ts.tool as tool
import ts.data as data
from ts.timeseries import SingleTs as Sts
from ts.timeseries import MulTs as Mts
import ts.fill as fill
from loguru import logger
import pandas as pd
import time
lat_ranges = []
gap_lens = []
sites = []
res = {}
for site in sites:
    ts = Sts(site)
    tsl = ts.get_longest()
    for gapsize in gap_lens:
        tsg, gidx = ts.make_gap(gapsize, cache_size=int, per=float)
        tsf = fill.RegEMFiller().fill(tsg)
        res[lat, gapsize] = tool.fill_res(tsg, tsf, gidx)
res.to_csv('./res/compare_lat.csv')
