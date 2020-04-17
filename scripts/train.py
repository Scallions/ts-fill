'''
@Author       : Scallions
@Date         : 2020-03-09 20:54:10
@LastEditors  : Scallions
@LastEditTime : 2020-04-17 19:40:38
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
from tcn import TemporalConvNet


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


length = 365

"""
ts dataset
取30天为input，之后一天为output
"""
class TsDataset(torch.utils.data.Dataset):
    def __init__(self, ts):
        super().__init__()
        self.data = ts.get_longest()
        self.len = self.data.shape[0] - length - 1

    def __getitem__(self, index):
        return self.data.iloc[index:index+length].values, self.data.iloc[index+length].values

    def __len__(self):
        return self.len

class LSTM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = torch.nn.LSTM(length,30,2)
        self.output = torch.nn.Linear(30,1)
    
    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        return self.output(out)


class TCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tcn = TemporalConvNet(1,[10,10,10,10,10])
        self.conv1 = torch.nn.Conv1d(10,1,5)
        self.fc = torch.nn.Linear(length-4,1)
        self.relu = torch.nn.ReLU()
    
    def forward(self, x):
        x = self.tcn(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.relu(x)
        return self.fc(x)

def train(tss, net, loss):
    """train framework
    
    Args:
        trainset (dataset): trainset
        net (nn.Module): lstm net needed to train
        loss (func): loss function
    """
    if not loss:
        loss = torch.nn.MSELoss()
    
    opt = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=0.1)

    epochs = 200

    checkpoint = True

    for epoch in range(epochs):
        lm = 0
        for ts in tss:
            if ts.shape[0] < 800: continue
            dataset = TsDataset(ts)
            train_loader = torch.utils.data.DataLoader(dataset,batch_size=30)
            for i, data in enumerate(train_loader):
                x,y = data
                y = y.float()
                x = x.float()
                x = x.permute(0,2,1)
                out_p = net(x).squeeze(-1)
                l = loss(out_p,y)
                opt.zero_grad()
                l.backward()
                opt.step()
                if lm < l.data.item():
                    lm = l.data.item()
                if i % 30 == 29:
                    print(f"Epoch: {epoch}, Batch: {i}, Loss: {l.data.item()}")
        print(f"Epoch: {epoch}, Loss: {lm}")
        if checkpoint and epoch % 2 == 1:
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'opt_state_dict' : opt.state_dict(),
                'loss': l,
            }, f"models/tcn/{epoch}.tar")                    

if __name__ == "__main__":
    tss = load_data()
    tcn = TCN()

    nets = [tcn]
    for net in nets:
        train(tss,net,torch.nn.MSELoss())



""" some snippets
## state_dict 
# save model
torch.save(model.state_dict(), PATH)
# load model
model = Net() # model that u saved
model.load_state_dict(torch.load(PATH))
model.eval()

## whole model
# save
torch.save(model, PATH)
# load
model = torch.load(PATH)
model.eval()

## checkpoint
# save
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'opt_state_dict' : opt.state_dict(),
    'loss': loss,
    ...
}, PATH)
# load
model = Model()
opt = Opt()
checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
opt.loader_state_dict(checkpoint['opt_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
model.eval()
"""