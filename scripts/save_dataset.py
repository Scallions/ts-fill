"""
@Author       : Scallions
@Date         : 2020-04-25 16:31:29
@LastEditors  : Scallions
@LastEditTime : 2020-04-25 16:42:52
@FilePath     : /gps-ts/scripts/save_dataset.py
@Description  : 
"""
import os
import sys
sys.path.append('./')
import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn.utils import weight_norm
import numpy as np
from mgln import MGLN
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
    for j in range(epoch):
        random.shuffle(tss)
        for i in range(0, nums - lengths, lengths):
            mts = tool.concat_multss(tss[i:i + lengths])
            rtss.append(mts)
    return rtss


length = 1024
"""
ts dataset
取30天为input，之后一天为output
"""


class TssDataset(torch.utils.data.Dataset):

    def __init__(self, ts):
        super().__init__()
        self.data = ts.get_longest()
        try:
            self.gap, _, _ = self.data.make_gap(30, cache_size=100)
        except:
            self.len = -1
            return
        self.gap = self.gap.fillna(self.gap.interpolate(method='slinear'))
        self.data = self.data.to_numpy()
        self.gap = self.gap.to_numpy()
        self.len = self.data.shape[0] - length - 1

    def __getitem__(self, index):
        ts = self.data[index:index + length]
        gap = self.data[index:index + length]
        mean = np.mean(ts, axis=0)
        std = np.std(ts, axis=0)
        ts = (ts - mean) / std
        mean = np.mean(gap, axis=0)
        std = np.std(gap, axis=0)
        gap = (gap - mean) / std
        return ts, gap

    def __len__(self):
        return self.len


if __name__ == '__main__':
    tss = load_data(lengths=3, epoch=20)
    out_dir = './data/dataset/'
    for i, ts in enumerate(tss):
        ts.to_csv(out_dir + str(i) + '.csv')
