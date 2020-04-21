'''
@Author       : Scallions
@Date         : 2020-04-18 11:00:57
@LastEditors  : Scallions
@LastEditTime : 2020-04-21 22:27:24
@FilePath     : /gps-ts/scripts/gln.py
@Description  : 
'''
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm



class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2, encode=True):
        """
        相当于一个Residual block

        :param n_inputs: int, 输入通道数
        :param n_outputs: int, 输出通道数
        :param kernel_size: int, 卷积核尺寸
        :param stride: int, 步长，一般为1
        :param dilation: int, 膨胀系数
        :param padding: int, 填充系数
        :param dropout: float, dropout比率
        """
        super(TemporalBlock, self).__init__()
        if encode:
            self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                            stride=stride, padding=padding, dilation=dilation))
        else:
            self.conv1 = weight_norm(nn.ConvTranspose1d(n_inputs, n_outputs,
                                            kernel_size, padding=dilation,
                                            stride=stride, dilation=dilation)) 
        
        
        self.batch1 = nn.BatchNorm1d(n_outputs)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        if encode:
            self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs,        
                                            kernel_size,
                                            stride=stride, padding=padding, dilation=dilation))
        else:
            self.conv2 = weight_norm(nn.ConvTranspose1d(n_outputs, n_outputs,
                                            kernel_size, padding=dilation,
                                            stride=stride, dilation=dilation))  
    
        self.batch2 = nn.BatchNorm1d(n_outputs)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.relu1, self.batch1, self.dropout1,
                                 self.conv2, self.relu2, self.batch2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        """
        参数初始化

        :return:
        """
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """
        :param x: size of (Batch, input_channel, seq_len)
        :return:
        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2, encode=True):
        """
        TCN，目前paper给出的TCN结构很好的支持每个时刻为一个数的情况，即sequence结构，
        对于每个时刻为一个向量这种一维结构，勉强可以把向量拆成若干该时刻的输入通道，
        对于每个时刻为一个矩阵或更高维图像的情况，就不太好办。

        :param num_inputs: int， 输入通道数
        :param num_channels: list，每层的hidden_channel数，例如[25,25,25,25]表示有4个隐层，每层hidden_channel数为25
        :param kernel_size: int, 卷积核尺寸
        :param dropout: float, drop_out比率
        """
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i   # 膨胀系数：1，2，4，8……
            in_channels = num_inputs if i == 0 else num_channels[i-1]  # 确定每一层的输入通道数
            out_channels = num_channels[i]  # 确定每一层的输出通道数
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=dilation_size, dropout=dropout, encode=encode)]
            if encode:
                layers += [nn.MaxPool1d(2)]
            else:
                layers += [nn.ConvTranspose1d(out_channels,out_channels,kernel_size,stride=2, padding=1,output_padding=1)]

        self.network = nn.Sequential(*layers)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x, midin=None):
        """
        输入x的结构不同于RNN，一般RNN的size为(Batch, seq_len, channels)或者(seq_len, Batch, channels)，
        这里把seq_len放在channels后面，把所有时间步的数据拼起来，当做Conv1d的输入尺寸，实现卷积跨时间步的操作，
        很巧妙的设计。
        
        :param x: size of (Batch, input_channel, seq_len)
        :return: size of (Batch, output_channel, seq_len)
        """
        midout = []
        for name, midlayer in self.network._modules.items():
            
            x = midlayer(x)
            if int(name) % 4 == 3 and midin == None:
                midout += [x]
                # print(x.shape)
            if midin != None and int(name) % 4 == 1 and name !='9':
                t = midin[- int(name)//4 ]
                x += t
        if midin == None:
            return x, midout
        # x = self.sigmoid(x)
        return x

class GLN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = TemporalConvNet(1,[4,8,16,8,4])
        self.decoder = TemporalConvNet(4,[8,16,8,4,1], encode=False)

    def forward(self, x):
        x, midout = self.encoder(x)
        return self.decoder(x, midout)