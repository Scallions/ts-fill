'''
@Author       : Scallions
@Date         : 2020-02-05 14:30:53
@LastEditors  : Scallions
@LastEditTime : 2020-02-09 21:21:41
@FilePath     : /gps-ts/ts/ts.py
@Description  :Single Variant and multiple variant time series datatype
'''
import pandas as pd

import ts.data as data


class TimeSeries(pd.DataFrame):
    """
    基类，规定一些接口
    """

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


    def gap_status(self) -> List[Gap]:
        """get status of ts
        
        Returns:
            List[Gap]: gap size of ts 
        """
        lengths = []
        start = -1
        start_ = self.index[0]
        for i in self.index:
            if self[i] == None and start_ != -1:
                lengths.append({
                    "len": i - start_,
                    "start": start_,
                    "end": i - 1
                })
                start_ = -1
                start = i 
            if self[i] != None and start != -1:
                lengths.append({
                    "len": i - start,
                    "start": start,
                    "end": i - 1
                })
                start = -1
                start_ = i
        return lengths

class MulTs:
    pass
