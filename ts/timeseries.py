'''
@Author       : Scallions
@Date         : 2020-02-05 14:30:53
@LastEditors  : Scallions
@LastEditTime : 2020-03-03 20:46:43
@FilePath     : /gps-ts/ts/timeseries.py
@Description  :Single Variant and multiple variant time series datatype
'''
import pandas as pd
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
    def __init__(self, filepath="", filetype=data.FileType.Raw, datas=None, indexs=None):
        # load cwu
        if filepath != "" and filetype == data.FileType.Cwu:
            ts = data.cwu_loader(filepath)
            _data = ts.iloc[:,1].to_numpy()
            index = ts.index
            columns = ['x']
        # load sopac
        if filepath != "" and filetype == data.FileType.Sopac:
            ts = data.sopac_loader(filepath)
            _data = ts.iloc[:,1].to_numpy()
            index = ts.index
            columns = ['x']
        # load custom data
        if filetype == data.FileType.Raw and not datas and not indexs:
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

class MulTs:
    pass