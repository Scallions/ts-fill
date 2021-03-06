'''
@Author       : Scallions
@Date         : 2020-03-23 18:23:29
LastEditors  : Scallions
LastEditTime : 2020-10-26 20:46:29
FilePath     : /gps-ts/ts/regem.py
@Description  : regem for imputation
use matlab connection pls open ur matlab and change dir to where reg em in.
'''

import numpy as np
import pandas as pd
from loguru import logger
import matlab.engine






def fill(ts):
    """fill by reg em
    
    Args:
        ts (ts): multiple ts with gap
    
    Returns:
        ts: ts after fill
    """
    fp = "/home/cyf/Code/gps-fill/RegEM/data/out.csv"
    fout = "/home/cyf/Code/gps-fill/RegEM/data/fill.csv"
    eng = matlab.engine.connect_matlab()
    eng.edit('regem', nargout=1)
    ts = ts.complete()
    # ts.iloc[:,[0,1,3,4]].to_csv(fp)
    ts.to_csv(fp)

    # save fill to file to make reg em read

    # eng.regem()
    eng.pyfill(nargout=0)
    eng.quit()
    
    # read the res
    tss = pd.read_csv(fout,header=None)
    tss.index = ts.index
    tss.columns = ts.columns
    return tss