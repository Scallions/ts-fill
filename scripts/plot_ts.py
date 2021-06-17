'''
@Author       : Scallions
@Date         : 2020-03-25 08:39:45
LastEditors  : Scallions
LastEditTime : 2020-11-21 19:19:06
FilePath     : /gps-ts/scripts/compare-mults-fill.py
@Description  : 
'''
import os
import sys

from numpy.core.fromnumeric import size
sys.path.append('./')
import ts.tool as tool
import ts.data as data
from ts.timeseries import SingleTs as Sts
from ts.timeseries import MulTs as Mts
import ts.fill as fill
from loguru import logger
import pandas as pd
import time
import matplotlib.pyplot as plt


def load_data(lengths=3, epoch=6):
    """load data
    
    Args:
        lengths (int, optional): nums of mul ts. Defaults to 3.
    
    Returns:
        [mts]: mts s
    """
    dir_path = './data/greenland/'
    tss = []
    files = os.listdir(dir_path)
    for file_ in files:
        if '.tenv3' in file_:
            tss.append(Mts(dir_path + file_, data.FileType.Ngl))
    nums = len(tss)
    rtss = []
    import random
    for j in range(epoch):
        random.shuffle(tss)
        for i in range(0, nums - lengths, lengths):
            try:
                mts = tool.concat_u_multss(tss[i:i + lengths])
            except:
                continue
            rtss.append(mts)
    return rtss


if __name__ == '__main__':
    """genrel code for compare
    """
    result = {}

    # 定义 数据集
    gap_size = 30
    j = 1
    raw = pd.read_csv(f"res/gap/raw-{gap_size}-{j}.csv",index_col="jd",parse_dates=True)
    gap = pd.read_csv(f"res/gap/gap-{gap_size}-{j}.csv",index_col="jd",parse_dates=True)
    h, w = raw.shape
    gap_idxs = [3,4,6,7,11,12,14,15,16,18]
    gap_idxs = [3,4,7,12,16]
    plt.figure(figsize=(10,5))
    plt.yticks([])
    for i, idx in enumerate(gap_idxs):
        plt.plot(raw.iloc[:,idx]*2+i)
    # raw.plot()
    plt.savefig("fig/ga.png")
    plt.figure(figsize=(10,5))
    plt.yticks([])
    for i, idx in enumerate(gap_idxs):
        plt.plot(gap.iloc[:,idx]*2+i)
    plt.savefig("fig/ga1.png")