'''
@Author       : Scallions
@Date         : 2020-03-04 21:06:57
@LastEditors  : Scallions
@LastEditTime : 2020-04-17 19:34:08
@FilePath     : /gps-ts/ts/rnnfill.py
@Description  : use rnn to fill gap in ts
'''

import torch 
import sys
from loguru import logger
import pandas as pd
import numpy as np

model_path = sys.path[0]+'/models/lstm/99.tar'

class LSTM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = torch.nn.LSTM(30,10,num_layers=3)
        self.output = torch.nn.Linear(10,1)
    
    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        return self.output(out)



def load_model(net, save_path):
    checkpoint = torch.load(save_path)
    model = net()
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


# net = load_model(LSTM,model_path)
net = None
def fill(ts):
    l = len(ts)
    length = 365
    ts = ts.copy()
    # logger.debug(model_path)
    for i in range(l):
        if pd.isna(ts.iloc[i]).item():
            x = torch.from_numpy(ts.iloc[i-length:i].to_numpy())
            x.resize_(1,1,length)
            x = x.float()
            y_hat = net(x)
            ts.iloc[i] = y_hat.item()
    return t


def lstm_fill(ts):
    return fill(ts)

