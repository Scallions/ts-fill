'''
@Author       : Scallions
@Date         : 2020-03-04 21:06:57
@LastEditors  : Scallions
@LastEditTime : 2020-03-12 15:43:13
@FilePath     : /gps-ts/ts/rnnfill.py
@Description  : use rnn to fill gap in ts
'''

import torch 
import sys

model_path = sys.path[0]+'../models/'

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = torch.nn.LSTM(30,10,num_layers=3)
        self.output = torch.nn.Linear(10,1)
    
    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        return self.output(out)


def load_model(net, save_path):
    checkpoint = torch.load(save_path)
    model = net().load_state_dict(checkpoint['model_state_dict'])
    return model

