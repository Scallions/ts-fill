'''
@Author       : Scallions
@Date         : 2020-05-17 11:19:12
LastEditors  : Scallions
LastEditTime : 2020-08-04 15:12:40
FilePath     : /gps-ts/ts/mlp.py
@Description  : 
'''
import torch
import pandas as pd

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
        return self.tsd.loc[idx-pd.Timedelta(days=self.range_):idx+pd.Timedelta(days=self.range_),:].to_numpy().flatten(), self.tsg.loc[idx, :].to_numpy()
    

def fill(ts, range_=10):
    """[summary]

    Args:
        ts ([type]): [description]
        range_ (int, optional): [前后范围大小]. Defaults to 5.

    Returns:
        [type]: [description]
    """
    """@nni.variable(nni.choice(1,3,5,10), name=trange_)"""
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
    len_gap = gap_idx.sum() # 有缺失值的维数
    len_data = len(gap_idx) - len_gap # 没有缺失值的维数

    # nni
    """@nni.function_choice(torch.nn.ReLU(),torch.nn.LeakyReLU(), torch.nn.Tanh(), torch.nn.Sigmoid(), name=torch.nn.ReLU)"""
    r = torch.nn.Tanh()

    # nni hidden
    """@nni.variable(nni.choice(0.5,1,2), name=hidden)"""
    hidden = 0.5
    """@nni.variable(nni.choice(1,2,3,4), name=hidden2)"""
    hidden2 = 4

    hidden_num = int(hidden*(2*range_+1))
    hidden_num2 = int(hidden2*len_gap)

    model = torch.nn.Sequential()
    model.add_module('l1',torch.nn.Linear(len_data*(2*range_+1), hidden_num))
    model.add_module('bn1', torch.nn.BatchNorm1d(hidden_num))
    model.add_module('ac1',r)
    model.add_module('l2',torch.nn.Linear(hidden_num, hidden_num2))
    model.add_module('bn2', torch.nn.BatchNorm1d(hidden_num2))
    model.add_module('ac2',r)
    model.add_module('l3', torch.nn.Linear(hidden_num2,len_gap))
    return model


def train_net(net, dataloader, range_):
    # """@nni.variable(nni.choice(10,20,30,50), name=epochs)"""
    epochs = 80
    opt = torch.optim.Adam(net.parameters(), lr=0.001)

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
            print(f"\repoch: {epoch}, batch: {i}, loss: {l.item()}, l1: {l11.item()}", end="")
            """@nni.report_intermediate_result(l.item())"""
    print()
    """@nni.report_final_result(l.item())"""


def complete(tsc, model, range_):
    gap_idx = tsc.isna().any()
    for i in range(len(tsc)):
        if tsc.iloc[i,:].isna().any() == False: 
            continue        
        x = tsc.iloc[i-range_:i+range_+1,:].loc[:,gap_idx == False]
        x = torch.from_numpy(x.to_numpy()).float().flatten().unsqueeze(0)
        model.eval()
        out = model(x)
        tsc.iloc[i,:].loc[gap_idx==True] = out.detach().numpy().squeeze(0)
        
    