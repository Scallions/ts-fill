'''
Author       : Scallions
Date         : 2020-10-16 22:40:26
LastEditors  : Scallions
LastEditTime : 2020-10-16 22:52:20
FilePath     : /gps-ts/ts/stvregem.py
Description  : 
'''

import ts.regem as regem
import pandas as pd

def fill(ts):

    ## add local time view
    len_local = 10
    len_ts = len(ts)
    tss = ts.iloc[:,0:ts-len_local]
    for i in range(1,len_local):
        ts_l = ts.iloc[:, i:len_ts-len_local+i]
        tss = pd.concat(tss,ts_l)   
    tss = regem.fill(tss)
    
    gidx = ts.isna().any()
    
    

