'''
@Author       : Scallions
@Date         : 2020-05-17 11:19:12
LastEditors  : Scallions
LastEditTime : 2020-10-20 20:25:01
FilePath     : /gps-ts/ts/mlp.py
@Description  : 
'''
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import random

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data, range_):
        self.data = data
        self.range_ = range_
        self.gap_idx = self.data.isna().any()
        if range_ != 0:
            self.idxs = data.index[data.isna().T.any()==False][2*range_:-2*range_] 
        else:
            self.idxs = data.index[data.isna().T.any()==False]
        self.tsg = self.data.loc[:, self.gap_idx == True] # na ts
        self.tsd = self.data.loc[:, self.gap_idx == False] # no na ts
        
    def __len__(self):
        return len(self.idxs)
        
    def __getitem__(self, index):
        idx = self.idxs[index]
    
        return self.tsd.loc[idx-pd.Timedelta(days=self.range_):idx+pd.Timedelta(days=self.range_),:].to_numpy().flatten(), self.tsg.loc[idx, :].to_numpy()
        # return np.expand_dims(self.tsd.loc[idx-pd.Timedelta(days=self.range_):idx+pd.Timedelta(days=self.range_),:].to_numpy().transpose(),0), self.tsg.loc[idx, :].to_numpy()
class SELayer(torch.nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.fc = nn.Sequential( 
            torch.nn.Linear(in_channel, in_channel//2, bias=False),
            nn.ReLU(inplace=True),
            torch.nn.Linear(in_channel//2, in_channel, bias=False),
            nn.Sigmoid()
        )
        self.avgpool = nn.AdaptiveMaxPool1d(1)

    def forward(self,x):
        b,l = x.size()
        y = self.avgpool(x.T.view(l,1,b)).view(1,l)
        y = self.fc(y)
        return x * y.expand_as(x)
        
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
# 设置随机数种子
# setup_seed(20)

def fill(ts, range_=0):
    """[summary]

    Args:
        ts ([type]): [description]
        range_ (int, optional): [前后范围大小]. Defaults to 5.

    Returns:
        [type]: [description]
    """
    setup_seed(76)
    """@nni.variable(nni.choice(0,1,3,5,10,15), name=trange_)"""
    trange_ = range_
    tsc = ts.copy()
    xm = ts.mean()
    xs = ts.std()
    tsc = (tsc - xm) / xs
    ds = MyDataset(tsc, trange_)
    train_size = int(0.8 * len(ds))
    test_size = len(ds) - train_size
    train_db, val_db = torch.utils.data.random_split(ds, [train_size, test_size])
    dataloader = torch.utils.data.DataLoader(dataset=train_db,batch_size = 256, shuffle=True, drop_last=False)
    val_dl = torch.utils.data.DataLoader(dataset=val_db,batch_size=test_size)
    model = train(tsc, dataloader, trange_, val_dl)
    complete(tsc, model, trange_)
    tsc = tsc * xs + xm
    return tsc


def train(tsc, dataloader,range_, val_dl):
    gap_idx = tsc.isna().any()
    net = make_net(gap_idx, range_) # use gap_idx make mlp net
    # from torchsummary import summary
    # len_gap = gap_idx.sum()
    # len_data = len(gap_idx) - len_gap
    # summary(net, input_size=(6,len_data*(2*range_+1)))
    train_net(net, dataloader, range_, val_dl)
    return net 

    

def make_net(gap_idx, range_):
    len_gap = gap_idx.sum() # 有缺失值的维数
    len_data = len(gap_idx) - len_gap # 没有缺失值的维数

    # nni
    """@nni.function_choice(torch.nn.ReLU(),torch.nn.LeakyReLU(),torch.nn.Tanh(),torch.nn.Sigmoid(), name=r)"""
    r = torch.nn.Tanh()
    # r = torch.nn.Tanh()

    # nni hidden
    # """@nni.variable(nni.choice(0.5,1,2), name=hidden)"""
    hidden = 0.5
    """@nni.variable(nni.choice(1,2,3,4), name=hidden2)"""
    hidden2 = 4

    # hidden_num = int(hidden*(2*range_+1))
    hidden_num = int(hidden2*len_gap)

    model = torch.nn.Sequential()
    ## 2d conv
    # model.add_module('conv1', nn.Conv2d(1,2,3, padding=1))
    # model.add_module('cac1', r)
    # model.add_module('conv2', nn.Conv2d(2,4,3, padding=1))
    # model.add_module('cac2', r)
    # model.add_module('cbn_1', nn.BatchNorm2d(4))
    # model.add_module('conv3', nn.Conv2d(4,8,3, padding=1))
    # model.add_module('cac3', r)
    # model.add_module('conv4', nn.Conv2d(8,16,3, padding=1))
    # model.add_module('cac4', r)
    # model.add_module('cbn_2', nn.BatchNorm2d(16))

    ## 1d conv
    # time conv
    # model.add_module('1conv1', nn.Conv2d(1,2, kernel_size=(3,1), padding=(1,0)))
    # model.add_module('1cac1', r)
    # site conv
    # model.add_module('1conv2', nn.Conv2d(2,4, kernel_size=(1,3), padding=(0,1)))
    # model.add_module('1cac2', r)
    # model.add_module('1bn_1', nn.BatchNorm2d(4))
    # model.add_module('pool', nn.AvgPool2d(kernel_size=(3,1)))

    ## head 
    # model.add_module('pool', nn.AdaptiveAvgPool2d(1))
    # model.add_module('flatten', nn.Flatten())
    # model.add_module('l1',torch.nn.Linear(64, hidden_num))
    # model.add_module('att1', SELayer(len_data*(range_*2+1)))
    model.add_module('l1',torch.nn.Linear(len_data*(range_*2+1), hidden_num))
    model.add_module('ac1',r)
    # model.add_module('bn1', torch.nn.BatchNorm1d(hidden_num))
    # model.add_module('l2', torch.nn.Linear(hidden_num,hidden_num2))
    model.add_module('att2', SELayer(hidden_num))
    model.add_module('l2', torch.nn.Linear(hidden_num,len_gap))
    # model.add_module('ac2',r) 
    # model.add_module('bn2', torch.nn.BatchNorm1d(hidden_num2))
    # model.add_module('l3', torch.nn.Linear(hidden_num2,len_gap))
    return model


def train_net(net, dataloader, range_,val_dl):
    """@nni.variable(nni.choice(10,30,50,100), name=epochs)"""
    epochs = 100
    # opt = torch.optim.Adam(net.parameters(), lr=0.001)
    """@nni.function_choice(torch.optim.SGD(net.parameters(), lr=0.1, weight_decay=0.001, momentum=0.9),torch.optim.Adam(net.parameters(), lr=0.001), name=opt)"""
    opt = torch.optim.SGD(net.parameters(), lr=0.1, weight_decay=0.001, momentum=0.9)

    l1 = torch.nn.MSELoss()
    # l1 = torch.nn.L1Loss()

    for epoch in range(epochs):
        for i, (x, y) in enumerate(dataloader):
            net.train()
            x = x.float() 
            y = y.float()
            y_hat = net(x)
            l11 = l1(y, y_hat)
            l12 = (y - y_hat).sum().abs() / y.shape[0] / y.shape[1]
            l = l11*1# + 0.3*l12
            opt.zero_grad()
            l.backward()
            opt.step()
        # if (epoch % 5) == 0:
        for x,y in val_dl:
            net.eval()
            x = x.float() 
            y = y.float()
            y_hat = net(x)
            l11 = l1(y, y_hat)
            l12 = (y - y_hat).sum().abs() / y.shape[0] / y.shape[1] 
            l = l11*1 + 0.3*l12
            print(f"\repoch: {epoch}, batch: {i}, loss: {l.item()}, l1: {l11.item()}, l2: {l12.item()}", end="")
            """@nni.report_intermediate_result(l.item())"""
    """@nni.report_final_result(l.item())"""


def complete(tsc, model, range_):
    gap_idx = tsc.isna().any()
    for i in range(len(tsc)):
        if tsc.iloc[i,:].isna().any() == False: 
            continue        
        x = tsc.iloc[i-range_:i+range_+1,:].loc[:,gap_idx == False]
        x = torch.from_numpy(x.to_numpy()).float().flatten().unsqueeze(0)
        # x = torch.from_numpy(x.to_numpy().transpose()).float().unsqueeze(0).unsqueeze(0)
        model.eval()
        out = model(x)
        tsc.iloc[i,:].loc[gap_idx==True] = out.detach().numpy().squeeze(0)
        
    