'''
@Author       : Scallions
@Date         : 2020-02-05 14:30:53
@LastEditors  : Scallions
@LastEditTime : 2020-03-24 20:22:32
@FilePath     : /gps-ts/ts/timeseries.py
@Description  :Single Variant and multiple variant time series datatype
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import ts.data as data
import ts.tool as tool


class GapStatus:
    def __init__(self):
        super().__init__()
        self.starts = []
        self.lengths = []

class TimeSeries(pd.DataFrame):
    """
    基类，规定一些接口
    """

    def plot_gap(self):
        """plot gap
        """
        gap_sizes = self.gap_status().lengths
        plt.hist(gap_sizes, bins=20)
        plt.show()
        

class SingleTs(TimeSeries):    
    def __init__(self, filepath="", filetype=data.FileType.Df, datas=None, indexs = None):
        # load cwu
        if filepath != "" and filetype == data.FileType.Cwu:
            ts = data.cwu_loader(filepath)
            _data = ts.iloc[:,2].to_numpy()
            index = ts.index
            columns = ['x']
        # load sopac
        if filepath != "" and filetype == data.FileType.Sopac:
            ts = data.sopac_loader(filepath)
            _data = ts.iloc[:,1].to_numpy()
            index = ts.index
            columns = ['x']
        # load custom data
        if filetype == data.FileType.Df and (isinstance(datas, SingleTs) or isinstance(datas,pd.DataFrame)):
            _data = datas.x
            index = datas.index
            columns = ['x']
        if filetype == data.FileType.Df and (isinstance(datas, np.ndarray) or isinstance(datas,pd.Series)):
            _data = datas
            index = indexs 
            columns = ['x']
        super().__init__(data=_data, index=index, columns=columns)

    def complete(self):
        """
        对空值填充NAN
        """
        start = self.index[0]
        end = self.index[-1]
        indexs = pd.date_range(start=start,end=end)
        for index in indexs:
            if not index in self.index:
                self.loc[index] = None
        self.sort_index(inplace=True)
        return SingleTs(datas=self.copy())

    def get_longest(self):
        """sub ts longest
        """
        tsl = tool.get_longest(self)
        return SingleTs(datas = tsl)
    
    def make_gap(self,gapsize=3, per = 0.2):
        """make gap in ts
        """
        tsg,gindex = tool.make_gap(self,gapsize, per)
        return SingleTs(datas = tsg),gindex


    def gap_status(self):
        """get status of ts no compelte
        
        Returns:
            List[Gap]: gap size of ts 
        """
        indexs = self.index.to_julian_date()
        start = indexs[0]
        gaps = GapStatus()
        for i in range(1,len(indexs)):
            if indexs[i] - indexs[i-1] == 1:
                pass 
            else:
                len_1 = indexs[i-1] - start + 1
                len_2 = indexs[i] - indexs[i-1] - 1
                gaps.starts.append(tool.jd2datetime(start))
                gaps.lengths.append(int(len_1))
                gaps.starts.append(tool.jd2datetime(indexs[i-1] + 1))
                gaps.lengths.append(-1*int(len_2))
                start = indexs[i]
        return gaps

class MulTs(TimeSeries):
    def __init__(self, filepath="", filetype=data.FileType.Df, datas=None, indexs = None):
        if filepath != "" and filetype == data.FileType.Cwu:
            ts = data.cwu_loader(filepath)
            _data = ts.iloc[:,[0,1,2]].to_numpy()
            index = ts.index
            columns = ['n','e','v']

        # load custom data
        if filetype == data.FileType.Df and (isinstance(datas, MulTs) or isinstance(datas,pd.DataFrame)):
            _data = datas.values
            index = datas.index
            columns = datas.columns
        
        super().__init__(data=_data, index=index, columns=columns)

    def complete(self):
        """
        对空值填充NAN
        """
        start = self.index[0]
        end = self.index[-1]
        indexs = pd.date_range(start=start,end=end)
        for index in indexs:
            if not index in self.index:
                self.loc[index] = None
        self.sort_index(inplace=True)
        return MulTs(datas=self.copy())

    def get_longest(self):
        """sub ts longest
        """
        tsl = tool.get_longest(self)
        return MulTs(datas=tsl)

    def gap_status(self):
        """get status of ts no compelte
        
        Returns:
            List[Gap]: gap size of ts 
        """
        indexs = self.index.to_julian_date()
        start = indexs[0]
        gaps = GapStatus()
        for i in range(1,len(indexs)):
            if indexs[i] - indexs[i-1] == 1:
                pass 
            else:
                len_1 = indexs[i-1] - start + 1
                len_2 = indexs[i] - indexs[i-1] - 1
                gaps.starts.append(tool.jd2datetime(start))
                gaps.lengths.append(int(len_1))
                gaps.starts.append(tool.jd2datetime(indexs[i-1] + 1))
                gaps.lengths.append(-1*int(len_2))
                start = indexs[i]
        return gaps

    def make_gap(self,gapsize=3, per = 0.2):
        """make gap in ts
        """
        tsg,gindex = tool.make_gap(self,gapsize, per)
        return SingleTs(datas = tsg),gindex

    @staticmethod
    def from_df(df):
        return MulTs(datas=df, filetype=data.FileType.Df)

def df2ts(df):
    return SingleTs(datas=df)