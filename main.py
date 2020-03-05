'''
@Author: your name
@Date: 2020-02-05 14:26:13
@LastEditTime : 2020-03-05 17:37:40
@LastEditors  : Scallions
@Description: In User Settings Edit
@FilePath     : /gps-ts/main.py
'''
from loguru import logger
from ts.data import cwu_loader
from ts.data import FileType as FileType
from ts.timeseries import SingleTs as Sts
import ts.fill as fill
import ts.tool as tool
import matplotlib.pyplot as plt

if __name__ == "__main__":
    filepath = "./data/BACK.cwu.igs14.csv"
    ts = Sts(filepath,filetype=FileType.Cwu)
    tsl = ts.get_longest()
    # tsl.plot()
    tsg = tsl.make_gap(20,0.3)
    tsc = fill.SSAFiller.fill(tsg)
    tsc.plot()
    tsg.plot()
    plt.show()
    print(id(tsc),id(tsg))