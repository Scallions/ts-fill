'''
@Author       : Scallions
@Date         : 2020-05-27 18:12:47
@LastEditors  : Scallions
@LastEditTime : 2020-06-29 08:32:09
@FilePath     : /gps-ts/test.py
@Description  : 
'''

import ts.data as data
import ts.tool as tool
import ts.timeseries as tss
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    filepath = "./data/gps/CAS1.AN.tenv3"
    fp2 = "./data/BLAS.cwu.igs14.csv"
    # df = pd.read_csv(filepath, sep="\s+")
    # print(df["YYMMMDD"].astype('double').dtype)

    ts = tss.SingleTs(filepath, filetype=data.FileType.Ngl)
    ts = ts.get_longest()
    ts2 = tss.SingleTs(fp2, data.FileType.Cwu)
    ts2 = ts2.get_longest()
    # ts = data.ngl_loader(filepath)
    # print(ts.head())
    # print(ts2.head())
    print(ts.index.duplicated().sum())
    # # # # tsg, gidx = ts.make_gap(gapsize=30)
    # # # print(tsg.head())
    # # tsg.plot()
    # plt.show()