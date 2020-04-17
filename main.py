'''
@Author: your name
@Date: 2020-02-05 14:26:13
@LastEditTime : 2020-04-18 00:25:49
@LastEditors  : Scallions
@Description: In User Settings Edit
@FilePath     : /gps-ts/main.py
'''
from loguru import logger
from ts.data import cwu_loader
from ts.data import FileType as FileType
from ts.timeseries import SingleTs as Sts
from ts.timeseries import MulTs as Mts
import ts.fill as fill
import ts.tcnfill as tf
import ts.tool as tool
import matplotlib.pyplot as plt

if __name__ == "__main__":
    filepath2 = "./data/BACK.cwu.igs14.csv"
    filepath = "./data/BLAS.cwu.igs14.csv"
    ts = Sts(filepath2,filetype=FileType.Cwu)
    logger.debug(ts.head())
    ts2 = ts.get_longest()
    # ts2.plot()
    gapsize = 30
    ts3,gidx = ts2.make_gap(gapsize=gapsize,cache_size=40)
    # ts3.plot()
    plt.plot(ts2.loc[gidx[:gapsize]], 'o', label='raw', markersize=3)
    # ts4 = tf.tcn_fill(ts3)
    fillers = [
        fill.PolyFiller,
        fill.SLinearFiller, 
        fill.SSAFiller,
        # fill.FbFiller
        # fill.TCNFiller
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