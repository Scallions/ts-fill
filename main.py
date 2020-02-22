'''
@Author: your name
@Date: 2020-02-05 14:26:13
@LastEditTime : 2020-02-22 12:25:59
@LastEditors  : Scallions
@Description: In User Settings Edit
@FilePath     : /gps-ts/main.py
'''
from loguru import logger
from ts.data import cwu_loader
from ts.timeseries import SingleTs as Sts
import ts.fill as fill

if __name__ == "__main__":
    filepath = "./data/BACK.cwu.igs14.csv"
    ts = Sts(filepath)
    ts.plot_gap()