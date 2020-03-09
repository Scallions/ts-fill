'''
@Author       : Scallions
@Date         : 2020-03-09 20:54:10
@LastEditors  : Scallions
@LastEditTime : 2020-03-09 22:07:29
@FilePath     : /gps-ts/scripts/train.py
@Description  : 
'''
import os
import sys

sys.path.append("./")

from loguru import logger
import matplotlib.pyplot as plt

import ts.data as data
from ts.timeseries import SingleTs as Sts

import torch 


def load_data():
    """Load data in dir data 
    
    Returns:
        List[TimeSeries]: a list of ts
    """
    tss = []
    files = os.listdir("./data")
    for file_ in files:
        if ".cwu.igs14.csv" in file_:
            tss.append(Sts("./data/" + file_,data.FileType.Cwu))
    return tss


"""
ts dataset
取30天为input，之后一天为output
"""
class TsDataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        self.data = load_data()[0].get_longest()
        self.len = self.data.shape[0] - 31

    def __getitem__(self, index):
        return self.data.iloc[index:index+30].values, self.data.iloc[index+30].values

    def __len__(self):
        return self.len

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = torch.nn.LSTM(30,10,num_layers=3)
        self.output = torch.nn.Linear(10,1)
    
    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        return self.output(out)


def train(trainset, net, loss):
    """train framework
    
    Args:
        trainset (dataset): trainset
        net (nn.Module): lstm net needed to train
        loss (func): loss function
    """
    if not loss:
        loss = torch.nn.MSELoss()
    
    opt = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=0.1)

    epochs = 100

    for epoch in range(epochs):
        for i, data in enumerate(trainset):
            x,y = data
            y = y.float()
            x = x.float()
            x = x.permute(0,2,1)
            out_p = net(x).squeeze(-1)
            l = loss(out_p,y)
            opt.zero_grad()
            l.backward()
            opt.step()
            if i % 99 == 0:
                print(f"Epoch: {epoch}, Batch: {i}, Loss: {l.data.item()}")


if __name__ == "__main__":
    dataset = TsDataset()
    train_loader = torch.utils.data.DataLoader(dataset,batch_size=12)

    net = Net()

    train(train_loader,net,torch.nn.MSELoss())