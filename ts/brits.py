'''
Author       : Scallions
Date         : 2020-10-13 09:56:23
LastEditors  : Scallions
LastEditTime : 2020-10-13 15:54:45
FilePath     : /gps-ts/ts/brits.py
Description  : 
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable
from torch.nn.parameter import Parameter

from tqdm import tqdm
import numpy as np

import math
from sklearn import metrics

SEQ_LEN = 36
RNN_HID_SIZE = 64

def binary_cross_entropy_with_logits(input, target, weight=None, size_average=True, reduce=True):
    if not (target.size() == input.size()):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

    max_val = (-input).clamp(min=0)
    loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()

    if weight is not None:
        loss = loss * weight

    if not reduce:
        return loss
    elif size_average:
        return loss.mean()
    else:
        return loss.sum()

class FeatureRegression(nn.Module):
    def __init__(self, input_size):
        super(FeatureRegression, self).__init__()
        self.build(input_size)

    def build(self, input_size):
        self.W = Parameter(torch.Tensor(input_size, input_size))
        self.b = Parameter(torch.Tensor(input_size))

        m = torch.ones(input_size, input_size) - torch.eye(input_size, input_size)
        self.register_buffer('m', m)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, x):
        z_h = F.linear(x, self.W * Variable(self.m), self.b)
        return z_h

class TemporalDecay(nn.Module):
    def __init__(self, input_size, output_size, diag = False):
        super(TemporalDecay, self).__init__()
        self.diag = diag

        self.build(input_size, output_size)

    def build(self, input_size, output_size):
        self.W = Parameter(torch.Tensor(output_size, input_size))
        self.b = Parameter(torch.Tensor(output_size))

        if self.diag == True:
            assert(input_size == output_size)
            m = torch.eye(input_size, input_size)
            self.register_buffer('m', m)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, d):
        if self.diag == True:
            gamma = F.relu(F.linear(d, self.W * Variable(self.m), self.b))
        else:
            gamma = F.relu(F.linear(d, self.W, self.b))
        gamma = torch.exp(-gamma)
        return gamma

class Rits(nn.Module):
    def __init__(self):
        super().__init__()
        self.build()

    def build(self):
        self.rnn_cell = nn.LSTMCell(9 * 2, RNN_HID_SIZE)

        self.temp_decay_h = TemporalDecay(input_size = 9, output_size = RNN_HID_SIZE, diag = False)
        self.temp_decay_x = TemporalDecay(input_size = 9, output_size = 9, diag = True)

        self.hist_reg = nn.Linear(RNN_HID_SIZE, 9)
        self.feat_reg = FeatureRegression(9)

        self.weight_combine = nn.Linear(9 * 2, 9)

        self.dropout = nn.Dropout(p = 0.25)
        self.out = nn.Linear(RNN_HID_SIZE, 1)

    def forward(self, data, direct):
        # Original sequence with 24 time steps
        # values = data[direct]['values']
        # masks = data[direct]['masks']
        # deltas = data[direct]['deltas']

        if direct == 'forward':
            values = data[0].float()
            masks = data[1].float()
            deltas = data[2].float()
        else:
            values = data[3].float()
            masks = data[4].float()
            deltas = data[5].float()

        # evals = data[direct]['evals']
        # eval_masks = data[direct]['eval_masks']

        # labels = data['labels'].view(-1, 1)
        # is_train = data['is_train'].view(-1, 1)

        h = Variable(torch.zeros((values.size()[0], RNN_HID_SIZE)))
        c = Variable(torch.zeros((values.size()[0], RNN_HID_SIZE)))

        if torch.cuda.is_available():
            h, c = h.cuda(), c.cuda()

        x_loss = 0.0
        # y_loss = 0.0

        imputations = []

        for t in range(SEQ_LEN):
            x = values[:, t, :]
            if torch.isnan(x).any():
                print("hhh")
            m = masks[:, t, :]
            d = deltas[:, t, :]

            gamma_h = self.temp_decay_h(d)
            gamma_x = self.temp_decay_x(d)

            h = h * gamma_h

            x_h = self.hist_reg(h)
            x_loss += torch.sum(torch.abs(x - x_h) * m) / (torch.sum(m) + 1e-5)

            x_c =  m * x +  (1 - m) * x_h

            z_h = self.feat_reg(x_c)
            x_loss += torch.sum(torch.abs(x - z_h) * m) / (torch.sum(m) + 1e-5)

            alpha = self.weight_combine(torch.cat([gamma_x, m], dim = 1))

            c_h = alpha * z_h + (1 - alpha) * x_h
            x_loss += torch.sum(torch.abs(x - c_h) * m) / (torch.sum(m) + 1e-5)

            c_c = m * x + (1 - m) * c_h

            inputs = torch.cat([c_c, m], dim = 1)

            h, c = self.rnn_cell(inputs, (h, c))

            imputations.append(c_c.unsqueeze(dim = 1))

        imputations = torch.cat(imputations, dim = 1)

        # y_h = self.out(h)
        # y_loss = binary_cross_entropy_with_logits(y_h, labels, reduce = False)
        # y_loss = torch.sum(y_loss * is_train) / (torch.sum(is_train) + 1e-5)

        # y_h = F.sigmoid(y_h)

        return {'loss': x_loss / SEQ_LEN ,\
                'imputations': imputations, }

    def run_on_batch(self, data, optimizer):
        ret = self(data, direct = 'forward')

        if optimizer is not None:
            optimizer.zero_grad()
            ret['loss'].backward()
            optimizer.step()

        return ret

class Brits(nn.Module):
    def __init__(self):
        super().__init__()
        self.build()

    def build(self):
        self.rits_f = Rits()
        self.rits_b = Rits()

    def forward(self, data):
        ret_f = self.rits_f(data, 'forward')
        ret_b = self.reverse(self.rits_b(data, 'backward'))

        ret = self.merge_ret(ret_f, ret_b)

        return ret

    def merge_ret(self, ret_f, ret_b):
        loss_f = ret_f['loss']
        loss_b = ret_b['loss']
        loss_c = self.get_consistency_loss(ret_f['imputations'], ret_b['imputations'])

        loss = loss_f + loss_b + loss_c

        # predictions = (ret_f['predictions'] + ret_b['predictions']) / 2
        imputations = (ret_f['imputations'] + ret_b['imputations']) / 2

        ret_f['loss'] = loss
        # ret_f['predictions'] = predictions
        ret_f['imputations'] = imputations

        return ret_f

    def get_consistency_loss(self, pred_f, pred_b):
        loss = torch.pow(pred_f - pred_b, 2.0).mean()
        return loss

    def reverse(self, ret):
        def reverse_tensor(tensor_):
            if tensor_.dim() <= 1:
                return tensor_
            indices = range(tensor_.size()[1])[::-1]
            indices = Variable(torch.LongTensor(indices), requires_grad = False)

            if torch.cuda.is_available():
                indices = indices.cuda()

            return tensor_.index_select(1, indices)

        for key in ret:
            ret[key] = reverse_tensor(ret[key])

        return ret

    def run_on_batch(self, data, optimizer):
        ret = self(data)

        if optimizer is not None:
            optimizer.zero_grad()
            ret['loss'].backward()
            optimizer.step()

        return ret

def to_var(var):
    if torch.is_tensor(var):
        var = Variable(var)
        if torch.cuda.is_available():
            var = var.cuda()
        return var
    if isinstance(var, int) or isinstance(var, float) or isinstance(var, str):
        return var
    if isinstance(var, dict):
        for key in var:
            var[key] = to_var(var[key])
        return var
    if isinstance(var, list):
        var = list(map(lambda x: to_var(x), var))
        return var

def stop_gradient(x):
    if isinstance(x, float):
        return x
    if isinstance(x, tuple):
        return tuple(map(lambda y: Variable(y.data), x))
    return Variable(x.data)

def zero_var(sz):
    x = Variable(torch.zeros(sz))
    if torch.cuda.is_available():
        x = x.cuda()
    return x

def fill(ts):
    tsc = ts.copy()
    global SEQ_LEN 
    SEQ_LEN = ts.shape[0]
    # x_min = ts.min()
    # x_max = ts.max()
    # tsc = (tsc - x_min) / (x_max - x_min)
    xm = ts.mean()
    xs = ts.std()
    tsc = (tsc - xm) / xs
    ds = MyDataset(tsc)
    dataloader = torch.utils.data.DataLoader(dataset=ds,batch_size = 1, drop_last=True)
    imputations = train(tsc, dataloader)
    complete(tsc, imputations)
    # tsc = tsc * (x_max - x_min) + x_min
    tsc = tsc * xs + xm
    return tsc

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        # self.gap_idx = self.data.isna().any()
        # self.idxs = data.index[data.isna().T.any()==False]
        # self.tsg = self.data.loc[:, self.gap_idx == True] # na ts
        # self.tsd = self.data.loc[:, self.gap_idx == False] # no na ts
        
    def __len__(self):
        return 1
        
    def __getitem__(self, index):
        masks = 1 - ((np.random.rand(*self.data.shape) < 0.20).astype(np.int) | self.data.isna().to_numpy().astype(np.int))
        values = self.data.to_numpy().copy()
        values[np.isnan(values)] = 0
        masks[:5,:] = 1
        masks[-5:,:] = 1
        for i in range(self.data.shape[0]):
            if np.isnan(values[i,:]).any():
                print("hh")
        deltas = np.zeros_like(self.data)
        lastobs = np.zeros(9)
        for i in range(self.data.shape[0]):
            deltas[i,:] = i - lastobs
            lastobs = lastobs * (1 - masks[i]) + i * masks[i]
        bvalues = values[::-1,:].copy()
        bmasks = masks[::-1,:].copy()
        bdeltas = np.zeros_like(self.data)
        lastobs = np.zeros(9)
        for i in range(self.data.shape[0]):
            bdeltas[i,:] = i - lastobs
            lastobs = lastobs * (1 - bmasks[i]) + i * bmasks[i]        
        return values, masks, deltas, bvalues, bmasks, bdeltas
        # return self.tsd.loc[idx-pd.Timedelta(days=self.range_):idx+pd.Timedelta(days=self.range_),:].to_numpy().flatten(), self.tsg.loc[idx, :].to_numpy()

def train(tsc, dataloader):
    # make generator discriminator
    # train all
    
    ### make model
    model = Brits()

    ### train
    epochs = 1000
    opt = torch.optim.Adam(model.parameters(), lr=0.0001)

    model.train()
    for epoch in tqdm(range(epochs)):
        run_loss = 0.0

        for i, x in enumerate(dataloader):
            # x = x.float() 
            x = to_var(x)
            ret = model.run_on_batch(x, opt)
            run_loss += ret['loss'].item()
            imputations = ret['imputations']


    return imputations


def complete(tsc, imputations):
    for i in range(tsc.shape[0]):
        if tsc.iloc[i,:].isna().any():
            idx = tsc.iloc[i,:].isna()
            tsc.iloc[i,:][idx] = imputations[0,i,:].detach().numpy()[idx]
        

