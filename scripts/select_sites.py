'''
Author       : Scallions
Date         : 2020-08-23 09:00:55
LastEditors  : Scallions
LastEditTime : 2020-11-15 15:36:20
FilePath     : /gps-ts/scripts/select_sites.py
Description  : 
'''

import os
import sys
sys.path.append('./')
import ts.tool as tool
import ts.data as data
from ts.timeseries import SingleTs as Sts
from ts.timeseries import MulTs as Mts
import ts.fill as fill
from loguru import logger
import pandas as pd
import time
import numpy as np
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
            tss.append((file_, Mts(dir_path + file_, data.FileType.Ngl)))
    nums = len(tss)
    rtss = []
    import random
    for j in range(epoch):
        random.shuffle(tss)
        for i in range(0, nums - 1, lengths):
            try:
                if (i+lengths) > nums:
                    continue
                data1 = tss[i:i + lengths]
                names = [d[0] for d in data1]
                ttss  =[d[1] for d in data1]
                mts = tool.concat_multss(ttss)
            except:
                continue
            rtss.append((names,mts))
    return rtss

def load_datas(lengths=3, epoch=6):
    """load data
    
    Args:
        lengths (int, optional): nums of mul ts. Defaults to 3.
    
    Returns:
        [mts]: mts s
    """
    dir_path = './data/an/'
    tss = []
    files = os.listdir(dir_path)
    # for file_ in files:
    for file_ in ['OHI2.AN.tenv3', 'FONP.AN.tenv3', 'PALM.AN.tenv3', 'TRVE.AN.tenv3', 'SPGT.AN.tenv3', 'ROBN.AN.tenv3', 'PAL2.AN.tenv3', 'MBIO.AN.tenv3', 'VNAD.AN.tenv3', 'CAPF.AN.tenv3', 'ROTH.AN.tenv3', 'OHI3.AN.tenv3']:
        if '.tenv3' in file_:
            tss.append((file_, Mts(dir_path + file_, data.FileType.Ngl)))
    # nums = len(tss)
    # rtss = []
    # import random
    # for j in range(epoch):
    #     random.shuffle(tss)
    #     for i in range(0, nums - 1, lengths):
    #         try:
    #             if (i+lengths) > nums:
    #                 continue
    #             data1 = tss[i:i + lengths]
    #             names = [d[0] for d in data1]
    #             ttss  =[d[1] for d in data1]
    #             mts = tool.concat_multss(ttss)
    #         except:
    #             continue
    #         rtss.append((names,mts))
    return tss

if __name__ == "__main__":
    """genrel code for compare
    """
    result = {}

    ## 定义 数据集
    clip_length = 400
    tss = load_data(lengths=20, epoch=1000)
    s = set()
    mm = []
    m = 0
    for names, ts in tss:
        ss = tuple(sorted(names))
        if ss in s:
            continue
        ll = len(ts.get_longest())
        if  ll > clip_length:
                print(names, ll)
                s.add(ss)
                if ll > m:
                    mm = [names,ll]
    print("max: \n", mm)

    # # 长度 图
    # tss = load_datas()
    # i = 0
    # names = []
    # for name,ts in tss:
    #     plt.scatter(ts.index, [i]*len(ts),s = 3)
    #     names.append(name.split('.')[0])
    #     i += 1
    #     # break
    # plt.yticks(range(len(names)), names)
    # plt.show()