'''
@Author: your name
@Date: 2020-02-05 14:26:13
@LastEditTime : 2020-03-24 20:11:37
@LastEditors  : Scallions
@Description: In User Settings Edit
@FilePath     : /gps-ts/main.py
'''
from loguru import logger
from ts.data import cwu_loader
from ts.data import FileType as FileType
from ts.timeseries import SingleTs as Sts
from ts.timeseries import MulTs as Mts
from ts.regem import fill as regem
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
    logger.debug(ts4.shape)
    regem(ts4)