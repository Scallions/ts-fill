"""
@Author       : Scallions
@Date         : 2020-04-29 10:44:07
@LastEditors  : Scallions
@LastEditTime : 2020-07-20 09:50:41
@FilePath     : /gps-ts/scripts/model_summary.py
@Description  : 
"""
from torchsummary import summary
from mgln import MGLN
net = MGLN()
summary(net, input_size=(1, 1024, 9))
