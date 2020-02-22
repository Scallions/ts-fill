'''
@Author: your name
@Date: 2020-02-05 14:26:13
@LastEditTime : 2020-02-22 21:55:54
@LastEditors  : Scallions
@Description: In User Settings Edit
@FilePath     : /gps-ts/main.py
'''
from loguru import logger
from ts.data import cwu_loader
from ts.timeseries import SingleTs as Sts
import ts.fill as fill
import matplotlib.pyplot as plt

if __name__ == "__main__":
    filepath = "./data/BACK.cwu.igs14.csv"
    ts = Sts(filepath)
    fts = fill.FbFiller.fill(ts)
    plt.plot(fts)
    plt.show()