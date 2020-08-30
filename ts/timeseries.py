'''
@Author       : Scallions
@Date         : 2020-02-05 14:30:53
LastEditors  : Scallions
LastEditTime : 2020-08-23 10:16:37
FilePath     : /gps-ts/ts/timeseries.py
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
        # load ngl:
        if filepath != "" and filetype == data.FileType.Ngl:
            ts = data.ngl_loader(filepath)
            _data = ts.iloc[:,2].to_numpy()
            index = ts['jd']
            columns = ['x']
        # load sopac
        if filepath != "" and filetype == data.FileType.Sopac:
            ts = data.sopac_loader(filepath)
            _data = ts.iloc[:,1].to_numpy()
            index = ts.index
            columns = ['x']
        # load custom data
        if (isinstance(datas, SingleTs) or isinstance(datas,pd.DataFrame)):
            _data = datas.x
            index = datas.index
            columns = ['x']
        if (isinstance(datas, np.ndarray) or isinstance(datas,pd.Series)):
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
    
    def make_gap(self,gapsize=3, per = 0.2, cache_size=0):
        """make gap in ts
        """
        gindex = tool.make_gap(self,gapsize, per, cache_size)
        tsg = self.copy()
        tsg.loc[gindex] = None
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
    def __init__(self, filepath="", filetype=data.FileType.Df, datas=None, indexs = None, columns=None):
        if filepath != "" and filetype == data.FileType.Cwu:
            ts = data.cwu_loader(filepath)
            _data = ts.iloc[:,[0,1,2]].to_numpy()
            index = ts.index
            columns = ['n','e','u']

        if filepath != "" and filetype == data.FileType.Ngl:
            ts = data.ngl_loader(filepath)
            _data = ts.iloc[:,[0,1,2]].to_numpy()
            index = ts['jd']
            columns = ['n','e','u']

        # load custom data
        if filetype == data.FileType.Df and (isinstance(datas, MulTs) or isinstance(datas,pd.DataFrame)):
            _data = datas.values
            index = datas.index
            if columns == None:
                columns = datas.columns

        if filetype == data.FileType.Df and isinstance(datas, np.ndarray):
            _data = datas
            index = indexs
        
        
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

    def make_gap(self,gapsize=3, per = 0.2, cper = 0.5, cache_size = 0, c_i=True, c_ii=None):
        """make gap in ts

        c_i: Ture 随机取可能取到同一个站点的， False 在站点之间随机
        """
        gindex = tool.make_gap(self,gapsize, per, cache_size)
        nums_c = self.shape[1]
        import random
        c_idx = list(range(nums_c))
        c_s = self.columns 
        c_idx = list(map(lambda x: str(x[0])+str(x[1]//3),zip(c_s,c_idx)))
        self.columns = c_idx
        if c_i == True:
            random.shuffle(c_idx)
            c_idx = c_idx[0:(int(cper*nums_c))]
        else:
            ts_num = nums_c // 3
            t_num = int(ts_num * cper)
            cache = list(range(ts_num))
            random.shuffle(cache)
            t_get = cache[0:t_num]
            cache_idx = []
            for i in t_get:
                cache_idx.extend(c_idx[3*i:3*i+3])
            c_idx = cache_idx
        if c_ii != None:
            c_idx = self.columns[:3]
        tsg = self.copy()
        g = tsg.loc[gindex,c_idx] = None
        return MulTs(datas = tsg),gindex, c_idx

    @staticmethod
    def from_df(df):
        return MulTs(datas=df, filetype=data.FileType.Df)

def df2ts(df):
    return SingleTs(datas=df)