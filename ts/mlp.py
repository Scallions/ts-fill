'''
@Author       : Scallions
@Date         : 2020-05-17 11:19:12
@LastEditors  : Scallions
@LastEditTime : 2020-06-29 20:28:56
@FilePath     : /gps-ts/ts/mlp.py
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
    

def fill(ts, range_=5):
    """[summary]

    Args:
        ts ([type]): [description]
        range_ (int, optional): [前后范围大小]. Defaults to 5.

    Returns:
        [type]: [description]
    """
    tsc = ts.copy()
    ds = MyDataset(tsc, range_)
    dataloader = torch.utils.data.DataLoader(dataset=ds,batch_size = 32, shuffle=True, drop_last=True)
    model = train(tsc, dataloader, range_)
    complete(tsc, model, range_)
    return tsc


def train(tsc, dataloader,range_):
    gap_idx = tsc.isna().any()
    net = make_net(gap_idx, range_) # use gap_idx make mlp net
    train_net(net, dataloader, range_)
    return net 

    

def make_net(gap_idx, range_):
    len_gap = gap_idx.sum()
    len_data = len(gap_idx) - len_gap
    model = torch.nn.Sequential()
    model.add_module('l1',torch.nn.Linear(len_data*(2*range_+1), len_data*(2*range_+1)))
    # model.add_module('ac1',torch.nn.ReLU())
    model.add_module('l2',torch.nn.Linear(len_data*(2*range_+1), 2*len_gap))
    # model.add_module('ac2',torch.nn.ReLU())
    model.add_module('l3', torch.nn.Linear(2*len_gap,len_gap))
    return model


def train_net(net, dataloader, range_):
    epochs = 30
    opt = torch.optim.Adam(net.parameters(), lr=0.001)

    l1 = torch.nn.MSELoss()
    # l1 = torch.nn.L1Loss()

    for epoch in range(epochs):
        for i, (x, y) in enumerate(dataloader):
            x = x.float() 
            y = y.float()
            y_hat = net(x)
            l11 = l1(y, y_hat)
            # l12 = (y - y_hat).sum().abs() / y.shape[0] / y.shape[1]
            l = l11 # + l12
            opt.zero_grad()
            l.backward()
            opt.step()
            if i % 10 == 9:
                print(f"\repoch: {epoch}, batch: {i}, loss: {l.item()}, l1: {l11.item()}", end="")
    print()


def complete(tsc, model, range_):
    # TODO
    gap_idx = tsc.isna().any()
    for i in range(len(tsc)):
        if tsc.iloc[i,:].isna().any() == False: 
            continue        
        x = tsc.iloc[i-range_:i+range_+1,:].loc[:,gap_idx == False]
        x = torch.from_numpy(x.to_numpy()).float().flatten()
        out = model(x)
        tsc.iloc[i,:].loc[gap_idx==True] = out.detach().numpy()
        
    