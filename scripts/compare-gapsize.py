"""
@Author       : Scallions
@Date         : 2020-02-28 19:50:25
@LastEditors  : Scallions
@LastEditTime : 2020-04-25 12:38:36
@FilePath     : /gps-ts/scripts/compare-gapsize.py
@Description  : gap size compare
"""
import os
import sys
sys.path.append('./')
import ts.tool as tool
import ts.data as data
from ts.timeseries import SingleTs as Sts
import ts.fill as fill
from loguru import logger
import pandas as pd
import time


def load_data():
    """Load data in dir data 
    
    Returns:
        List[TimeSeries]: a list of ts
    """
    dir_path = './data/'
    tss = []
    files = os.listdir(dir_path)
    for file_ in files:
        if '.cwu.igs14.csv' in file_:
            tss.append(Sts(dir_path + file_, data.FileType.Cwu))
    return tss


if __name__ == '__main__':
    """genrel code for compare
    """
    result = {}
    tss = load_data()
    gap_sizes = [1, 2, 3, 4, 5, 6, 10, 15, 30, 50]
    fillers = [fill.LinearFiller, fill.SLinearFiller, fill.GANFiller]
    result = pd.DataFrame(columns=[filler.name for filler in fillers],
        index=gap_sizes + ['time', 'gap_count', 'count'])
    result.loc['time'] = 0
    for gap_size in gap_sizes:
        val_tss = []
        for ts in tss:
            ts2 = ts.get_longest()
            if ts2.shape[0] < 1030:
                continue
            ts2 = Sts(datas=ts2[:1024], indexs=ts2.index[:1024])
            gapsize = 30
            ts3, gidx = ts2.make_gap(gapsize=gapsize, cache_size=300, per=0.03)
            val_tss.append((ts2, ts3, gidx))
        for i, filler in enumerate(fillers):
            logger.info(f'gap size: {gap_size}, filler: {filler.name}')
            res = None
            start = time.time()
            count = 0
            gap_count = 0
            for tsl, tsg, gidx in val_tss:
                if len(tsl) < 1000:
                    continue
                tsc = filler.fill(tsg)
                this_res = tool.fill_res(tsc, tsl, gidx)
                if not isinstance(res, pd.DataFrame):
                    res = this_res
                else:
                    res = pd.concat([res, this_res], axis=1)
                count += len(tsl)
                gap_count += len(gidx)
            counts = res.loc['count'].values
            means = res.loc['mean'].values
            res_mean = sum(counts * means) / sum(counts)
            end = time.time()
            result.loc[gap_size, filler.name] = res_mean
            result.loc['time', filler.name] = end - start + result.loc[
                'time', filler.name]
            result.loc['count', filler.name] = count
            result.loc['gap_count', filler.name] = gap_count
            logger.info(
                f'{filler.name} mean: {res_mean} time: {end - start:0.4f} gap: {gap_size}'
                )
    result.to_csv('res/fill_gapsize.csv')
