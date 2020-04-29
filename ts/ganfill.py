'''
@Author       : Scallions
@Date         : 2020-04-23 17:32:49
@LastEditors  : Scallions
@LastEditTime : 2020-04-25 12:22:35
@FilePath     : /gps-ts/ts/ganfill.py
@Description  : 
'''



from loguru import logger

import ts.data as data
import ts.timeseries as Ts
import ts.fill as fill
import ts.tool as tool

import torch 
from ts.gln import GLN
import numpy as np


net = None

def load_model():
    global net
    if net == None:
        PATH = "models/gan/189-G.tar"
        net = GLN()
        checkpoint = torch.load(PATH)
        net.load_state_dict(checkpoint['model_state_dict'])
        net.eval() 

def ganfill(ts):
    global net
    load_model()
    ts2 = fill.SLinearFiller().fill(ts)
    ts_numpy = ts2.to_numpy()
    t_mu = np.mean(ts_numpy)
    t_std = np.std(ts_numpy)
    ts_numpy = (ts_numpy - t_mu) / t_std
    ts_t = torch.from_numpy(ts_numpy).float()
    ts_t.resize_(1,1,1024)
    ts_tt = net(ts_t)
    ts_tt = net(ts_t).detach()
    ts_t = ts_tt
    ts_t = ts_t * t_std + t_mu
    ts_t = ts_t.numpy()
    ts_t.resize(1024)
    ts_res = Ts.SingleTs(indexs=ts2.index, datas=ts_t)
    return ts_res