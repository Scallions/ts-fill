'''
Author       : Scallions
Date         : 2020-08-23 09:00:55
LastEditors  : Scallions
LastEditTime : 2020-10-19 10:54:34
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
import matplotlib.pyplot as plt


def load_data(lengths=3, epoch=6):
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


if __name__ == "__main__":
    """genrel code for compare
    """
    result = {}

    # 定义 数据集
    clip_length = 400
    tss = load_data(lengths=4, epoch=50)
    s = set()
    for names, ts in tss:
        if len(ts.get_longest()) > clip_length:
            ss = tuple(sorted(names))
            if ss not in s:
                print(names)
                s.add(ss)
    # tsls = [ts.get_longest() for ts in tss if len(ts) > clip_length]
    # tsls = [tsl for tsl in tsls if len(tsl) > clip_length]
    # if len(tsls) == 0:
    #     logger.error("No data")
    #     sys.exit()
    # for ts in tsls:
    #     print(ts.site_names)