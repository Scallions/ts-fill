'''
@Author: your name
@Date: 2020-02-05 14:26:13
@LastEditTime : 2020-03-25 18:06:21
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
import ts.tool as tool
import matplotlib.pyplot as plt

if __name__ == "__main__":
    filepath = "./data/BACK.cwu.igs14.csv"
    filepath2 = "./data/BLAS.cwu.igs14.csv"
    ts = Mts(filepath,filetype=FileType.Cwu)
    ts2 = Mts(filepath2, filetype=FileType.Cwu)
    ts3 = tool.concat_multss([ts,ts2])
    logger.debug(ts3.head())
    ts4 = ts3.get_longest()
    ts5,gidx,cidx = ts4.make_gap()
    ts6 = fill.MSSAFiller.fill(ts5)
    res = tool.fill_res(ts6,ts4,gidx,cidx)
    logger.debug(res)