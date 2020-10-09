'''
@Author       : Scallions
@Date         : 2020-06-05 15:06:13
LastEditors  : Scallions
LastEditTime : 2020-10-09 14:23:13
FilePath     : /gps-ts/scripts/test_mpl.py
@Description  : 
'''
import os
import sys
sys.path.append('./')
import ts.mlp as mlp
from ts.timeseries import MulTs as Mts
import ts.data as data
import ts.tool as tool
import matplotlib.pyplot as plt
from loguru import logger


def load_data(lengths=6, epoch=6):
    """load data
    
    Args:
        lengths (int, optional): nums of mul ts. Defaults to 3.
    
    Returns:
        [mts]: mts s
    """
    dir_path = './data/'
    tss = []
    files = os.listdir(dir_path)
    for file_ in files:
        if '.cwu.igs14.csv' in file_:
            tss.append(Mts(dir_path + file_, data.FileType.Cwu))
    nums = len(tss)
    rtss = []
    import random
    random.seed(0)
    for j in range(epoch):
        random.shuffle(tss)
        for i in range(0, nums - lengths, lengths):
            mts = tool.concat_multss(tss[i:i + lengths])
            rtss.append(mts)
    return rtss


if __name__ == '__main__':
    tss = load_data(lengths=3, epoch=1)
    mts = tss[0]
    tsl = mts.get_longest()
    tsg, gidx, gridx = tsl.make_gap(gapsize=30, cache_size=200, per=0.03, c_i=False, cper=0.5)
    trends, noises = tool.remove_trend(tsl)
    noises.loc[gidx, gridx] = None
    noises = Mts(datas=noises,indexs=noises.index,columns=noises.columns)
                                
    tsc = mlp.fill(noises)
    tsc = trends + tsc
    tsg.plot()
    tsc.plot()
    plt.show()