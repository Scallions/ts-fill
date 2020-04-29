'''
@Author       : Scallions
@Date         : 2020-04-29 10:44:07
@LastEditors  : Scallions
@LastEditTime : 2020-04-29 10:47:30
@FilePath     : /gps-ts/scripts/model_summary.py
@Description  : 
'''

from torchsummary import summary 

from gln import GLN


net = GLN()

summary(net, input_size=(1,1024))