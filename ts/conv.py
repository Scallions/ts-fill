'''
@Author       : Scallions
@Date         : 2020-05-17 11:19:12
LastEditors  : Scallions
LastEditTime : 2020-10-20 17:00:49
FilePath     : /gps-ts/ts/conv.py
@Description  : 
'''
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data, range_):
        self.data = data
        self.range_ = range_
        self.gap_idx = self.data.isna().any()
        self.idxs = data.index[data.isna().T.any()==False][2*range_:-2*range_] 
        self.tsg = self.data.loc[:, self.gap_idx == True] # na ts
        self.tsd = self.data.loc[:, self.gap_idx == False] # no na ts
        
    def __len__(self):
        return len(self.idxs)
        
    def __getitem__(self, index):
        idx = self.idxs[index]
    
        # return self.tsd.loc[idx-pd.Timedelta(days=self.range_):idx+pd.Timedelta(days=self.range_),:].to_numpy().flatten(), self.tsg.loc[idx, :].to_numpy()
        return np.expand_dims(self.tsd.loc[idx-pd.Timedelta(days=self.range_):idx+pd.Timedelta(days=self.range_),:].to_numpy().transpose(),2), self.tsg.loc[idx, :].to_numpy()
        
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
        y = self.avgpool(x.view(l,1,b)).view(1,l)
        y = self.fc(y)
        return x * y.expand_as(x)
        

class sSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__() 
        self.Conv1x1 = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False) 
        self.norm = nn.Sigmoid()

    def forward(self, U):
        q = self.Conv1x1(U)  # U:[bs,c,h,w] to q:[bs,1,h,w]
        return U * q 

class cSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.Conv_Squeeze = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, bias=False)
        self.Conv_Excitation = nn.Conv2d(in_channels // 2, in_channels, kernel_size=1, bias=False) 
        self.norm = nn.Sigmoid()

    def forward(self, U):
        z = self.avgpool(U) # shape: [bs, c, h, w] => [bs, c, 1, 1]
        z = self.Conv_Squeeze(z) # shape: [bs, c/2]
        z = self.Conv_Excitation(z) # shape: [bs, c]
        z = self.norm(z)
        return U * z.expand_as(U)

class scSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__() 
        self.cSE = cSE(in_channels)
        self.sSE = sSE(in_channels) 
    
    def forward(self, U):
        U_sse = self.sSE(U)
        U_cse = self.cSE(U)
        return U_cse+U_sse

class DWConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.depth_conv = nn.Conv2d(in_channels, in_channels, kernel_size=(3,1), padding=(1,0), groups=in_channels)
        self.point_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1,padding=0,groups=1)

    def forward(self, x):
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x

class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # self.tConv = nn.Conv2d(in_channels, out_channels, kernel_size=(3,1), padding=(1,0))
        # self.sConv = nn.Conv2d(in_channels, out_channels, kernel_size=(1,3), padding=(0,1))
        # self.conv = nn.Conv2d(out_channels*2, out_channels*2, kernel_size=1)
        self.conv = DWConv(in_channels, out_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.ac = nn.Tanh()

    def forward(self, x):
        # tconv = self.tConv(x)
        # sconv = self.sConv(x)
        # y = torch.cat((tconv, sconv), 1) 
        # y = self.conv(y)
        # t = y.shape[1] // x.shape[1]
        # x = self.bn(self.ac(y)) + x.repeat(1,t,1,1)
        x = self.conv(x)
        x = self.ac(x)
        x = self.bn(x)
        return x


def fill(ts, range_=10):
    """[summary]

    Args:
        ts ([type]): [description]
        range_ (int, optional): [前后范围大小]. Defaults to 5.

    Returns:
        [type]: [description]
    """
    # """@nni.variable(nni.choice(1,3,5,10), name=trange_)"""
    trange_ = range_
    tsc = ts.copy()
    xm = ts.mean()
    xs = ts.std()
    tsc = (tsc - xm) / xs
    ds = MyDataset(tsc, trange_)
    dataloader = torch.utils.data.DataLoader(dataset=ds,batch_size = 256, shuffle=True, drop_last=True)
    model = train(tsc, dataloader, trange_)
    complete(tsc, model, trange_)
    tsc = tsc * xs + xm
    return tsc


def train(tsc, dataloader,range_):
    gap_idx = tsc.isna().any()
    net = make_net(gap_idx, range_) # use gap_idx make mlp net
    # from torchsummary import summary
    # len_gap = gap_idx.sum()
    # len_data = len(gap_idx) - len_gap
    # summary(net, input_size=(6,len_data*(2*range_+1)))
    train_net(net, dataloader, range_)
    return net 

    

def make_net(gap_idx, range_):
    len_gap = int(gap_idx.sum()) # 有缺失值的维数
    len_data = len(gap_idx) - len_gap # 没有缺失值的维数

    # nni
    # """@nni.function_choice(torch.nn.ReLU(),torch.nn.LeakyReLU(), torch.nn.Tanh(), torch.nn.Sigmoid(), name=torch.nn.ReLU)"""
    r = torch.nn.Tanh()
    # r = nn.LeakyReLU()

    # nni hidden
    # """@nni.variable(nni.choice(0.5,1,2), name=hidden)"""
    hidden = 0.5
    # """@nni.variable(nni.choice(1,2,3,4), name=hidden2)"""
    hidden2 = 4

    hidden_num = int(hidden*(2*range_+1))
    hidden_num2 = int(hidden2*len_gap)

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
    # # time conv
    # model.add_module('1conv1', nn.Conv2d(1,2, kernel_size=(3,1), padding=(1,0)))
    # model.add_module('1cac1', r)
    # # site conv
    # model.add_module('1conv2', nn.Conv2d(2,4, kernel_size=(1,3), padding=(0,1)))
    # model.add_module('1cac2', r)
    # model.add_module('1bn_1', nn.BatchNorm2d(4))
    model.add_module('block1', Block(len_data, 2*len_data)) 
    model.add_module('block2', Block(2*len_data,4*len_data))
    model.add_module('block3', Block(4*len_data,8*len_data))
    model.add_module('block4', Block(8*len_data,16*len_data))
    model.add_module('block5', Block(16*len_data,32*len_data))
    # att
    model.add_module('1att1', scSE(32*len_data))
    # model.add_module('pool', nn.AvgPool2d(kernel_size=(3,1)))

    ## head 
    model.add_module('pool', nn.AdaptiveAvgPool2d(1))
    model.add_module('flatten', nn.Flatten())
    # model.add_module('l1',torch.nn.Linear(64, hidden_num))
    # model.add_module('l1',torch.nn.Linear(16*len_data*(range_*2+1), hidden_num))
    # model.add_module('att1', SELayer(168))
    # model.add_module('bn1', torch.nn.BatchNorm1d(hidden_num))
    # model.add_module('ac1',r)
    # model.add_module('l2', torch.nn.Linear(hidden_num,len_gap))
    model.add_module('fc', torch.nn.Linear(32*len_data,len_gap))
    model.add_module('ac', nn.Tanh())
    return model


def train_net(net, dataloader, range_):
    # """@nni.variable(nni.choice(10,20,30,50), name=epochs)"""
    epochs = 200
    opt = torch.optim.Adam(net.parameters(), lr=0.01)

    l1 = torch.nn.MSELoss()
    # l1 = torch.nn.L1Loss()

    for epoch in range(epochs):
        for i, (x, y) in enumerate(dataloader):
            x = x.float() 
            y = y.float()
            y_hat = net(x)
            l11 = l1(y, y_hat)
            l12 = (y - y_hat).sum().abs() / y.shape[0] / y.shape[1]
            l = l11 + l12
            opt.zero_grad()
            l.backward()
            opt.step()
            # if i % 10 == 9:
            print(f"\repoch: {epoch}, batch: {i}, loss: {l.item()}, l1: {l11.item()}, l2: {l12.item()}", end="")
            # """@nni.report_intermediate_result(l.item())"""
    print()
    # """@nni.report_final_result(l.item())"""


def complete(tsc, model, range_):
    gap_idx = tsc.isna().any()
    for i in range(len(tsc)):
        if tsc.iloc[i,:].isna().any() == False: 
            continue        
        x = tsc.iloc[i-range_:i+range_+1,:].loc[:,gap_idx == False]
        # x = torch.from_numpy(x.to_numpy()).float().flatten().unsqueeze(0)
        x = torch.from_numpy(x.to_numpy().transpose()).float().unsqueeze(0).unsqueeze(3)
        model.eval()
        out = model(x)
        tsc.iloc[i,:].loc[gap_idx==True] = out.detach().numpy().squeeze(0)
        
    