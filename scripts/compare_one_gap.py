'''
@Author       : Scallions
@Date         : 2020-04-18 08:25:52
@LastEditors  : Scallions
@LastEditTime : 2020-04-23 17:57:26
@FilePath     : /gps-ts/scripts/compare_one_gap.py
@Description  : 
'''
import os
import sys

sys.path.append("./")

import ts.tool as tool
import ts.data as data
import ts.timeseries as TS
from ts.timeseries import SingleTs as Sts
import ts.fill as fill
from loguru import logger
import pandas as pd
import time 

from ts.data import cwu_loader
from ts.data import FileType as FileType
from ts.timeseries import SingleTs as Sts
from ts.timeseries import MulTs as Mts
import ts.tcnfill as tf
import matplotlib.pyplot as plt

def load_data(lengths=3,epoch=6):
    """load data
    
    Args:
        lengths (int, optional): nums of mul ts. Defaults to 3.
    
    Returns:
        [mts]: mts s
    """
    dir_path = "./data/"
    tss = []
    files = os.listdir(dir_path)
    for file_ in files:
        if ".cwu.igs14.csv" in file_:
            tss.append(Mts(dir_path + file_,data.FileType.Cwu))
    nums = len(tss)
    rtss = []

    # data increase
    import random
    for j in range(epoch):
        random.shuffle(tss)
        for i in range(0,nums-lengths, lengths):
            mts = tool.concat_multss(tss[i:i+lengths])
            rtss.append(mts)
    return rtss

def compare_one_gap():
    filepath2 = "./data/BACK.cwu.igs14.csv"
    filepath = "./data/BLAS.cwu.igs14.csv"
    ts = Sts(filepath2,filetype=FileType.Cwu)
    logger.debug(ts.head())
    ts2 = ts.get_longest()
    ts2 = TS.SingleTs(datas=ts2[:1024], indexs=ts2.index[:1024])
    # ts2.plot()
    gapsize = 30
    ts3,gidx = ts2.make_gap(gapsize=gapsize,cache_size=40)
    # ts3.plot()
    plt.plot(ts2.loc[gidx[:gapsize]], 'o', label='raw', markersize=3)
    # ts4 = tf.tcn_fill(ts3)
    fillers = [
        fill.PolyFiller,
        fill.SLinearFiller, 
        # fill.SSAFiller,
        # fill.FbFiller
        fill.GANFiller
        ]
    for filler in fillers:
        ts4 = filler().fill(ts3)
        plt.plot(ts4.loc[gidx[:gapsize]],'o', label=filler.name, markersize=3)
        # ts4 = fill.PiecewisePolynomialFiller.fill(ts3)
        res = tool.fill_res(ts2,ts4,gidx)
        logger.debug(filler.name)
        logger.debug(res)
    plt.ylabel("mm")
    plt.xlabel('time')
    plt.legend()
    plt.show()

def compare_mul_gap():
    ts = load_data()[0]
    # plt.plot(ts,'o',markersize=1)
    # plt.show()
    # return None
    logger.debug(ts.head())
    ts2 = ts.get_longest()
    # ts2.plot()
    gapsize = 30
    ts3,gidx,cidx = ts2.make_gap(gapsize=gapsize,cache_size=40)
    # ts3.plot()
    plt.plot(ts2.loc[gidx[:gapsize],cidx[0]], 'o', label='raw', markersize=3)
    # ts4 = tf.tcn_fill(ts3)
    fillers = [
        fill.PolyFiller,
        fill.SLinearFiller, 
        # fill.RegEMFiller
        # fill.SSAFiller,
        # fill.FbFiller
        # fill.TCNFiller
        ]
    for filler in fillers:
        ts4 = filler().fill(ts3)
        plt.plot(ts4.loc[gidx[:gapsize], cidx[0]],'o', label=filler.name, markersize=3)
        # ts4 = fill.PiecewisePolynomialFiller.fill(ts3)
        res = tool.fill_res(ts2,ts4,gidx)
        logger.debug(filler.name)
        logger.debug(res)
    plt.ylabel("mm")
    plt.xlabel('time')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    compare_one_gap()