'''
@Author       : Scallions
@Date         : 2020-05-27 18:12:47
@LastEditors  : Scallions
@LastEditTime : 2020-07-21 16:41:45
@FilePath     : /gps-ts/test.py
@Description  : 
'''

import torch
import matplotlib.pyplot as plt

data = torch.randn(1,64,512,512)

import torch.nn as nn

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        # self.avgpool = nn.Conv2d(channel, 1, 1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel//reduction,bias=False), 
            nn.ReLU(inplace=True),
            nn.Linear(channel//reduction,channel, bias=False),
            nn.Sigmoid()
        )
        self.bn = nn.BatchNorm2d(channel)

    def forward(self, x):
        b,c,h,w = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.fc(y).view(b,c,1,1)
        return self.bn(x * y.expand_as(x))

senet = SELayer(channel=64)

out = senet(data)

print(data.shape, data.mean().item(), data.std().item())
plt.hist(data.flatten())

print(out.shape, out.mean().item(), out.std().item())

plt.hist(out.detach().flatten())

out = senet(out)
print(out.shape, out.mean().item(), out.std().item())

plt.hist(out.detach().flatten())

plt.show()