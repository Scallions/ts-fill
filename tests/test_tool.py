'''
@Author       : Scallions
@Date         : 2020-02-05 14:01:53
@LastEditors  : Scallions
@LastEditTime : 2020-03-03 21:11:48
@FilePath     : /gps-ts/tests/test_tool.py
@Description  : 
'''
import pytest 
import ts.tool as tool
import ts.timeseries as Ts
import pandas as pd
import ts.data as data

@pytest.fixture()
def get_a_ts():
    ts = Ts.SingleTs("data/ASKY.cwu.igs14.csv", filetype=data.FileType.Cwu)
    return ts

@pytest.fixture()
def get_a_continue_ts(get_a_ts):
    ts = get_a_ts
    return ts[100:200]

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
    # TODO: test for make gap @scallions
    # find index of gap, then count gap size, then whither size equal true size
    ts = get_a_continue_ts
    ts_g = tool.make_gap(ts)
   
    
def test_get_longest(get_a_ts):
    ts =  get_a_ts
    gap_status = ts.gap_status()
    longest = tool.get_longest(ts)
    assert longest.shape[0] == max(gap_status.lengths)