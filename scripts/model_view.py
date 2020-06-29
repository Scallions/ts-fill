'''
@Author       : Scallions
@Date         : 2020-04-21 20:48:38
@LastEditors  : Scallions
@LastEditTime : 2020-06-29 20:37:11
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
from ts.timeseries import MulTs as Mts
import ts.fill as fill
import ts.tool as tool

import torch 
from tcn import TemporalConvNet
from gln import GLN
from mgln import MGLN
import numpy as np

def load_mdata(lengths=3,epoch=6):
    """load data
    
    Args:
        lengths (int, optional): nums of mul ts. Defaults to 3.
    
    Returns:
        [mts]: mts s
    """
    dir_path = "./data/"
    tss = []
    files = os.listdir(dir_path)
    for file_ in files:
        if ".cwu.igs14.csv" in file_:
            tss.append(Mts(dir_path + file_,data.FileType.Cwu))
    # dir_path = "./data/gps/"
    # tss = []
    # files = os.listdir(dir_path)
    # for file_ in files:
    #     if ".AN.tenv3" in file_:
    #         tss.append(Mts(dir_path + file_,data.FileType.Ngl))
    nums = len(tss)
    rtss = []

    # data increase
    import random
    for j in range(epoch):
        random.shuffle(tss)
        for i in range(0,nums-lengths, lengths):
            try:
                mts = tool.concat_multss(tss[i:i+lengths])
            except:
                continue
            rtss.append(mts)
    return rtss


def testm(tss, net):
    for ts in tss:
        try:
            tsl = ts.get_longest()
        except:
            continue
        if tsl.shape[0] < 1050: continue
        ts_numpy = tsl.to_numpy()[:1024]
        ts = Mts(datas=ts_numpy, indexs=tsl.index[:1024], columns=tsl.columns)
        tsg, gidx, cidx = ts.make_gap(100, cache_size=300, per=100/1024, c_i=False)
        ts.columns = tsg.columns
        tssl = fill.SLinearFiller().fill(tsg)
        tsreg = fill.RegEMFiller().fill(tsg)
        tsmlp = fill.MLPFiller().fill(tsg)
        ts2 = tssl
        ts_numpy = ts2.to_numpy()
        t_mu = np.mean(ts_numpy, axis=0)
        t_std = np.std(ts_numpy, axis=0)
        ts_numpy = (ts_numpy - t_mu) / t_std
        ts_t = torch.from_numpy(ts_numpy).float()
        ts_t.resize_(1,1,1024,9)
        for i in range(1):
            ts_tt = net(ts_t)
            ts_tt = net(ts_t).detach()
            ts_t = ts_tt
            ts_t = ts_t * t_std + t_mu
            ts_t = ts_t.numpy()
            ts_t.resize(1024,9)
            ts_res = Mts(indexs=ts2.index, datas=ts_t, columns=tsg.columns)
            ts_r = ts.copy()
            ts_r.loc[gidx, cidx] = ts_res.loc[gidx, cidx]
            ts_t = ts_r.to_numpy()
            ts_numpy = ts_t
            t_mu = np.mean(ts_numpy, axis=0)
            t_std = np.std(ts_numpy, axis=0)
            ts_numpy = (ts_numpy - t_mu) / t_std
            ts_t = torch.from_numpy(ts_numpy).float()
            ts_t.resize_(1,1,1024,9)
        ts_t = ts_t * t_std + t_mu
        ts_t = ts_t.numpy()
        ts_t.resize(1024,9)
        ts_res = Mts(indexs=ts2.index, datas=ts_t, columns=tsg.columns)
        ts_r = ts.copy()
        ts_r.loc[gidx, cidx] = ts_res.loc[gidx, cidx] 
            
        res = tool.fill_res(ts_res,ts, gidx, cidx)
        res2 = tool.fill_res(tsreg, ts, gidx, cidx)
        logger.info(res)
        logger.info(res2)
        plt.plot(ts.loc[:,cidx[2]], label='gap')
        plt.plot(tssl.loc[:,cidx[2]], label="slinear")
        plt.plot(tsreg.loc[:,cidx[2]], label='reg')
        plt.plot(tsmlp.loc[:,cidx[2]], label="mpl")
        plt.plot(ts_r.loc[:,cidx[2]],label='gan')
        plt.plot(tsg.loc[:,cidx[2]])   
        plt.legend()
        plt.show()
               
        
        break  

def test(tss, net):
    for ts in tss:
        tsl = ts.get_longest()
        ts_numpy = tsl.to_numpy()[:1024]
        ts = Sts(datas=ts_numpy, indexs=tsl.index[:1024])
        tsg,gidx = ts.make_gap(30, cache_size=300, per = 0.03)
        ts2 = fill.SLinearFiller().fill(tsg)
        ts_numpy = ts2.to_numpy()
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
            ts_res = Sts(indexs=ts2.index, datas=ts_t)
            ts_r = ts.copy()
            ts_r.loc[gidx] = ts_res.loc[gidx]
            ts_t = ts_r.to_numpy()
            ts_numpy = ts_t
            t_mu = np.mean(ts_numpy)
            t_std = np.std(ts_numpy)
            ts_numpy = (ts_numpy - t_mu) / t_std
            ts_t = torch.from_numpy(ts_numpy).float()
            ts_t.resize_(1,1,1024)
        ts_t = ts_t * t_std + t_mu
        ts_t = ts_t.numpy()
        ts_t.resize(1024)
        ts_res = Sts(indexs=ts2.index, datas=ts_t)
        ts_r = ts.copy()
        ts_r.loc[gidx] = ts_res.loc[gidx] 
            
        res = tool.fill_res(ts_res,ts, gidx)
        res2 = tool.fill_res(ts2, ts, gidx)
        logger.info(res)
        logger.info(res2)
        plt.plot(ts, label='gap')
        plt.plot(ts2, label="slinear")
        plt.plot(ts_r,label='res')
        plt.plot(tsg)   
        plt.legend()
        plt.show()
               
        
        break 

if __name__ == "__main__":
    logger.add("log/view_{time}.log", rotation="500MB", encoding="utf-8", enqueue=True, compression="zip", retention="10 days", level="INFO")

    tss = load_mdata()

    PATH = "models/mgan/93-G.tar"
    model = MGLN()
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval() 
    testm(tss, model)


