"""
@Author       : Scallions
@Date         : 2020-06-30 07:36:29
@LastEditors  : Scallions
@LastEditTime : 2020-06-30 07:41:26
@FilePath     : /gps-ts/scripts/test_mgan.py
@Description  : 
"""
import os
import sys
sys.path.append('./')
from loguru import logger
import ts.data as data
import ts.timeseries as Ts
import ts.fill as fill
import ts.tool as tool
import torch
from ts.gln import GLN
import numpy as np



def load_model():
    PATH = 'models/gan/189-G.tar'
    net = GLN()
    checkpoint = torch.load(PATH)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()
    return net


if __name__ == '__main__':
    net = load_model()
    from torchsummary import summary
    summary(net, input_size=(2, 1024))
