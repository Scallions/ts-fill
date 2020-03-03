'''
@Author: your name
@Date: 2020-02-05 14:26:13
@LastEditTime : 2020-03-03 20:19:14
@LastEditors  : Scallions
@Description: In User Settings Edit
@FilePath     : /gps-ts/main.py
'''
from loguru import logger
from ts.data import cwu_loader
from ts.data import FileType as FileType
from ts.timeseries import SingleTs as Sts
import ts.fill as fill
import matplotlib.pyplot as plt

if __name__ == "__main__":
    filepath = "./data/BACK.cwu.igs14.csv"
    ts = Sts(filepath,filetype=FileType.Cwu)
    gaps = ts.gap_status()
    logger.debug(gaps.starts)