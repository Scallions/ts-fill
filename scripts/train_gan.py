"""
@Author       : Scallions
@Date         : 2020-04-20 18:02:02
@LastEditors  : Scallions
@LastEditTime : 2020-04-24 09:36:16
@FilePath     : /gps-ts/scripts/train_gan.py
@Description  : 
"""
import os
import sys
sys.path.append('./')
import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn.utils import weight_norm
import numpy as np
from gln import GLN
from ts.timeseries import SingleTs as Sts
import ts.data as data
import matplotlib.pyplot as plt
from loguru import logger


def load_data():
    """Load data in dir data 
    
    Returns:
        List[TimeSeries]: a list of ts
    """
    tss = []
    dir_path = './data/'
    files = os.listdir(dir_path)
    for file_ in files:
        if '.cwu.igs14.csv' in file_:
            tss.append(Sts(dir_path + file_, data.FileType.Cwu))
    return tss


length = 1024
"""
ts dataset
取30天为input，之后一天为output
"""


class TssDataset(torch.utils.data.Dataset):

    def __init__(self, ts):
        super().__init__()
        self.data = ts.get_longest()
        self.gap, _ = self.data.make_gap(30, cache_size=100)
        self.gap = self.gap.fillna(self.gap.interpolate(method='slinear'))
        self.data = self.data.to_numpy()
        self.gap = self.gap.to_numpy()
        self.len = self.data.shape[0] - length - 1

    def __getitem__(self, index):
        ts = self.data[index:index + length]
        gap = self.data[index:index + length]
        mean = np.mean(ts)
        std = np.std(ts)
        ts = (ts - mean) / std
        mean = np.mean(gap)
        std = np.std(gap)
        gap = (gap - mean) / std
        return ts, gap

    def __len__(self):
        return self.len


class DBlock(nn.Module):

    def __init__(self, n_inputs, n_outputs, kernel_size=3, stride=2,
        dilation=2, dropout=0.2, padding=2):
        super().__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation))
        self.relu1 = nn.LeakyReLU()
        self.batch1 = nn.BatchNorm1d(n_outputs)
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs,
            kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.relu2 = nn.LeakyReLU()
        self.batch2 = nn.BatchNorm1d(n_outputs)
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.relu1, self.batch1, self.
            dropout1, self.conv2, self.relu2, self.batch2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1, stride=4
            ) if n_inputs != n_outputs else None
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        """
        :param x: size of (Batch, input_channel, seq_len)
        :return:
        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class Dnet(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.block1 = DBlock(1, 4)
        self.block2 = DBlock(4, 8)
        self.block3 = DBlock(8, 16)
        self.block4 = DBlock(16, 8)
        self.net = nn.Sequential(self.block1, self.block2, self.block3,
            self.block4)
        self.conv1 = nn.Conv1d(8, 1, 4)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        x = self.net(x)
        return self.sigmod(self.conv1(x))


def train(netD, netG, tss):
    num_epochs = 200
    lr = 0.0001
    beta1 = 0.5
    criterion = torch.nn.BCELoss()
    mse = torch.nn.MSELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
    checkpoint = True
    device = 'cpu'
    real_label = 1
    fake_label = 0
    G_losses = []
    D_losses = []
    iters = 0
    print('Starting Training Loop...')
    for epoch in range(num_epochs):
        datasets = []
        for ts in tss:
            tsdataset = TssDataset(ts)
            if tsdataset.len > 100:
                datasets.append(tsdataset)
        dataset = torch.utils.data.ConcatDataset(datasets)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=30,
            drop_last=True, shuffle=True)
        for i, (ts, gap) in enumerate(dataloader):
            netD.zero_grad()
            real_cpu = ts.to(device).float()
            real_cpu = real_cpu.permute(0, 2, 1)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, device=device).float()
            output = netD(real_cpu).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()
            noise = gap.to(device).float()
            noise = noise.permute(0, 2, 1)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()
            netG.zero_grad()
            label.fill_(real_label)
            output = netD(fake).view(-1)
            errG = criterion(output, label)
            l = mse(fake, real_cpu)
            errG = errG + 0.1 * l
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()
            if i % 20 == 0:
                logger.info(
                    '[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                     % (epoch, num_epochs, i, len(dataloader), errD.item(),
                    errG.item(), D_x, D_G_z1, D_G_z2))
            G_losses.append(errG.item())
            D_losses.append(errD.item())
            iters += 1
        if checkpoint and epoch % 2 == 1:
            torch.save({'epoch': epoch, 'model_state_dict': netG.state_dict
                ()}, f'models/gan/{epoch}-G.tar')
            torch.save({'epoch': epoch, 'model_state_dict': netD.state_dict
                ()}, f'models/gan/{epoch}-D.tar')


if __name__ == '__main__':
    logger.add('log/train_gan_{time}.log', rotation='500MB', encoding=
        'utf-8', enqueue=True, compression='zip', retention='10 days',
        level='INFO')
    netg = GLN()
    netd = Dnet()
    tss = load_data()
    train(netd, netg, tss)
