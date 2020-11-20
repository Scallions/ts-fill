'''
Author       : Scallions
Date         : 2020-08-23 09:00:55
LastEditors  : Scallions
LastEditTime : 2020-11-17 21:15:56
FilePath     : /gps-ts/scripts/select_sites_compare.py
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
                mts = tool.concat_u_multss(ttss)
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
    for file_ in files:
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
    clip_length = 700
    tss = load_data(lengths=20, epoch=10000)
    s = set()
    mm = []
    for names, ts in tss:
        ss = tuple(sorted(names))
        if ss in s:
            continue
        ll = len(ts.get_longest())
        if  ll > clip_length:
            print(names, ll)
            s.add(ss)
            ts = ts.get_longest()
            tsg, gidx, gridx = ts.make_gap(30, cache_size=30, cper=0.5, c_i
        =False)
            trends, noises = tool.remove_trend(ts)
            # tsg = tsl.copy()
            # tsg.loc[gidx, gridx] = None
            # noises.to_csv("res/raw.csv")
            noises.loc[gidx, gridx] = None
            noises = Mts(datas=noises,indexs=noises.index,columns=noises.columns)
            raw = ts.loc[gidx,gridx]
            reg = fill.RegEMFiller().fill(noises)
            rf = fill.MissForestFiller().fill(noises)
            reg = trends + reg
            rf = trends + rf
            reg = reg.loc[gidx,gridx]
            rf = rf.loc[gidx,gridx]
            reg = reg.sub(raw).abs().mean().sum()
            rf = rf.sub(raw).abs().mean().sum()
            # rf = rf[2] +rf[5]
            # reg = reg[2] + reg[5]
            dd = reg - rf 
            if len(mm) == 0 or dd > mm[-1][0]:
                mm.append([dd,ll,ss,gidx, gridx])
                
    print(len(mm),mm[-1])        


    ## 长度 图
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