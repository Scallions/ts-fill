'''
@Author       : Scallions
@Date         : 2020-02-05 14:30:53
@LastEditors  : Scallions
@LastEditTime : 2020-02-22 12:18:34
@FilePath     : /gps-ts/ts/timeseries.py
@Description  :Single Variant and multiple variant time series datatype
'''
import pandas as pd
import matplotlib.pyplot as plt

import ts.data as data


class TimeSeries(pd.DataFrame):
    """
    基类，规定一些接口
    """

    def plot_gap(self):
        """plot gap
        """
        if not hasattr(self, "gap_size"):
            self.gap_status()
        plt.hist(self.gap_sizes)
        plt.show()
        

class SingleTs(TimeSeries):    
    def __init__(self, filepath):
        ts = data.cwu_loader(filepath)
        _data = ts.iloc[:,1].to_numpy()
        index = ts.index
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
        lengths = []
        indexs = self.index.to_julian_date()
        start = indexs[0]
        for i in range(1,len(indexs)):
            if indexs[i] - indexs[i-1] == 1:
                pass 
            else:
                len_1 = indexs[i-1] - start + 1
                len_2 = indexs[i] - indexs[i-1] - 1
                lengths.append(int(len_1))
                lengths.append(int(-1*len_2))
                start = indexs[i]
        self.gap_sizes = lengths
        return lengths

class MulTs:
    pass