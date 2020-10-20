'''
Author       : Scallions
Date         : 2020-10-16 14:53:37
LastEditors  : Scallions
LastEditTime : 2020-10-16 22:37:41
FilePath     : /gps-ts/ts/stmvl.py
Description  : 
'''

import numpy as np 


def ucf(ts):
    pass

def tcf(ts):
    pass 

def idw(ts):
    pass 

def ses(ts):
    pass

def mvl(ts):
    res_ucf = ucf(ts)
    res_tcf = tcf(ts)
    res_idw = idw(ts)
    res_ses = ses(ts)
    T = np.hstack((res_idw, res_ses, res_tcf, res_ucf))
    w = np.dot(np.dot(T.T, T), ts)
    res = np.dot(T, w)
    return res 
