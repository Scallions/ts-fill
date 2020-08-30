'''
@Author       : Scallions
@Date         : 2020-02-05 12:58:11
LastEditors  : Scallions
LastEditTime : 2020-08-23 09:42:04
FilePath     : /gps-ts/ts/data.py
@Description  : some func used to process time series

ts data is a dataframe contained columns such as [time:Datatime,jd:Float,N:Float,E:Float,U:Float,]
'''
import pandas as pd
from loguru import logger
from enum import Enum
import ts.tool as tool

class FileType(Enum):
    Cwu     = 1
    Sopac   = 2
    Df      = 3
    Ngl     = 4

def cwu_loader(filepath):
    """load data from cwu csv file

    load gps ts data from cuw
    """
    df = pd.read_csv(filepath,skiprows=list(range(11)),index_col="Date",parse_dates=True)
    df['jd'] = df.index.to_julian_date()
    ts = df.iloc[:,[0,1,2,-1]]
    return ts


def sopac_loader(filepath):
    """load data from sopac

    load gps ts data from sopac
    """
    logger.info("Read SOPAC data {}", filepath)
    df = pd.read_csv(filepath, header=None,sep=r"\s+", skiprows=15)
    df['jd'] = df[0].apply(lambda x: tool.jd2datetime(tool.dy2jd(x)))
    ts = df.iloc[:, [1,2,3,-1]]
    return ts

def ngl_loader(filepath):
    logger.info("Read NGL data {}", filepath)
    df = pd.read_csv(filepath, sep="\s+")
    df['jd'] = df["yyyy.yyyy"].apply(lambda x: tool.jd2datetime(tool.dy2jd(x)))
    mask = ~df['jd'].duplicated()
    df.iloc[:, [10,8,12]] *= 1000
    ts = df.iloc[:, [10,8,12, -1]].loc[mask,:] # N E U
    return ts 
