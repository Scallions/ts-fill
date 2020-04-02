'''
@Author       : Scallions
@Date         : 2020-03-04 21:06:57
@LastEditors  : Scallions
@LastEditTime : 2020-04-02 19:13:35
@FilePath     : /gps-ts/ts/rnnfill.py
@Description  : use rnn to fill gap in ts
'''

import torch 
import sys

model_path = sys.path[0]+'../models/'

class LSTM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = torch.nn.LSTM(30,10,num_layers=3)
        self.output = torch.nn.Linear(10,1)
    
    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        return self.output(out)


class TCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = 1
        self.decoder = 2

    def forward(self, x):
        x = self.conv(x)
        x = self.decoder(x)


def load_model(net, save_path):
    checkpoint = torch.load(save_path)
    model = net().load_state_dict(checkpoint['model_state_dict'])
    return model

def fill(net, ts):
    l = len(ts)
    net = load_model(net,model_path)
    for i in range(l):
        if ts.iloc[i] == None:
            x = ts.iloc[i-30:i].to_numpy()
            y_hat = net(x)
            ts.iloc[i] = y_hat 
    return ts


def lstm_fill(ts):
    return fill(LSTM, ts)

def tcn_fill(ts):
    return fill(TCN, ts)