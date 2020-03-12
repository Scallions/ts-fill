'''
@Author       : Scallions
@Date         : 2020-02-05 14:01:53
@LastEditors  : Scallions
@LastEditTime : 2020-03-12 14:42:55
@FilePath     : /gps-ts/tests/test_tool.py
@Description  : 
'''
import pytest 
import ts.tool as tool
import ts.timeseries as Ts
import pandas as pd
import ts.data as data
import ts.fill as fill

@pytest.fixture()
def get_a_ts():
    ts = Ts.SingleTs("data/ASKY.cwu.igs14.csv", filetype=data.FileType.Cwu)
    return ts

@pytest.fixture()
def get_a_continue_ts(get_a_ts):
    ts = get_a_ts
    return ts.get_longest()

def test_dy2jd():
    dy = 2020.123
    assert 2458895 == tool.dy2jd(dy)

def test_jd2datetime():
    pd_datetime = pd.Timestamp("20180809")
    jd = pd_datetime.to_julian_date()
    datetime = tool.jd2datetime(jd)
    assert pd_datetime == datetime

def test_pytest():
    assert 1 == 1

def test_make_gap(get_a_continue_ts):
    ts = get_a_continue_ts
    gap_size = 3
    ts_g = tool.make_gap(ts,gap_size)
    for ind in ts_g.index:
        if ts_g.loc[ind].item() == None:
            for i in range(gap_size):
                assert ts_g.loc[ind + pd.Timedelta(days=i)] == None
    
   
    
def test_get_longest(get_a_ts):
    ts =  get_a_ts
    gap_status = ts.gap_status()
    longest = tool.get_longest(ts)
    assert longest.shape[0] == max(gap_status.lengths)

def test_all_sub_ts(get_a_ts):
    ts = get_a_ts
    sub_ts = tool.get_all_cts(ts)
    gap_size = ts.gap_status()
    j = 0
    # test for length and start
    for i,length in enumerate(gap_size.lengths):
        if length < 0: continue
        assert sub_ts[j].shape[0] == length
        assert sub_ts[j].index[0] == gap_size.starts[i]
        j += 1
        index = sub_ts[j-1].index
        for k in range(length - 1):
            assert index[k+1] == index[k] + pd.Timedelta(days=1)

def test_comparets(get_a_continue_ts):
    ts = get_a_continue_ts
    tsg, gdix = ts.make_gap()
    tsc = fill.LinearFiller().fill(tsg)
    tool.fill_res(ts,tsc,gdix)