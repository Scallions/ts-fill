"""
@Author       : Scallions
@Date         : 2020-02-09 19:39:40
@LastEditors  : Scallions
@LastEditTime : 2020-03-24 12:21:16
@FilePath     : /gps-ts/scripts/analysis_ts.py
@Description  : Analysis gap size of gps time series
"""
import os
import sys
sys.path.append('./')
from loguru import logger
import matplotlib.pyplot as plt
import ts.data as data
from ts.timeseries import SingleTs as Sts


def load_data():
    """Load data in dir data 
    
    Returns:
        List[TimeSeries]: a list of ts
    """
    tss = []
    files = os.listdir('./data')
    for file_ in files:
        if '.cwu.igs14.csv' in file_:
            tss.append(Sts('./data/' + file_, data.FileType.Cwu))
    return tss


def get_lens(tss):
    """get gap lens form tss
    
    Args:
        tss ([Ts]): a list of ts
    """
    return list(map(lambda x: x.gap_status(), tss))


if __name__ == '__main__':
    """plot gap size distrbution
    """
    tss = load_data()
    lengths = list(map(lambda x: x.gap_status().lengths, tss))
    gap_sizes = []
    for lens in lengths:
        gap_sizes += lens
    from collections import Counter
    strick_sizes = [gap for gap in gap_sizes if gap > 0]
    gap_sizes = [gap for gap in gap_sizes if gap < 0 and gap > -365]
    result = Counter(gap_sizes)
    gap_sum = []
    k_sum = 0
    for k in sorted(result.keys(), reverse=True):
        k_sum += -k * result[k]
        gap_sum.append(k_sum)
    plt.subplot(1, 2, 1)
    plt.bar(result.keys(), result.values())
    plt.ylim(0, 50)
    plt.subplot(1, 2, 2)
    plt.plot(sorted(result.keys(), reverse=True), gap_sum)
    plt.show()
    con_res = Counter(strick_sizes)
    plt.bar(con_res.keys(), con_res.values())
    plt.show()
