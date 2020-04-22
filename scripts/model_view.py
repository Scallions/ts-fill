'''
@Author       : Scallions
@Date         : 2020-04-21 20:48:38
@LastEditors  : Scallions
@LastEditTime : 2020-04-22 14:20:53
@FilePath     : /gps-ts/scripts/model_view.py
@Description  : 
'''

import os
import sys

sys.path.append("./")

from loguru import logger
import matplotlib.pyplot as plt

import ts.data as data
from ts.timeseries import SingleTs as Sts
import ts.fill as fill

import torch 
from tcn import TemporalConvNet
from gln import GLN
import numpy as np


def load_data():
    """Load data in dir data 
    
    Returns:
        List[TimeSeries]: a list of ts
    """
    tss = []
    dir_path = "./data/"
    files = os.listdir(dir_path)
    for file_ in files:
        if ".cwu.igs14.csv" in file_:
            tss.append(Sts(dir_path + file_,data.FileType.Cwu))
    return tss

def test(tss, net):
    for ts in tss:
        tsl = ts.get_longest()
        tsg,gidx = tsl.make_gap(30, cache_size=100)
        ts2 = fill.SSAFiller().fill(tsg)
        ts_numpy = ts2.to_numpy()[:1024]
        t_mu = np.mean(ts_numpy)
        t_std = np.std(ts_numpy)
        ts_numpy = (ts_numpy - t_mu) / t_std
        ts_t = torch.from_numpy(ts_numpy).float()
        ts_t.resize_(1,1,1024)
        for i in range(1):
            ts_tt = net(ts_t)
            ts_tt = net(ts_t).detach()
            ts_t = ts_tt
        ts_t = ts_t * t_std + t_mu
        ts_t = ts_t.numpy()
        ts_t.resize(1024)
        ts_res = Sts(indexs=ts2.index[:1024], datas=ts_t)
            

        plt.plot(tsl[:1024],label='raw')
        plt.plot(ts2[:1024],label='gap')
        plt.plot(ts_res,label='res') 
        plt.legend()
        plt.show()       
        
        break 

if __name__ == "__main__":
    logger.add("log/view_{time}.log", rotation="500MB", encoding="utf-8", enqueue=True, compression="zip", retention="10 days", level="INFO")

    tss = load_data()

    PATH = "models/gln/39.tar"
    model = GLN()
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval() 
    test(tss, model)


