'''
@Author       : Scallions
@Date         : 2020-02-22 11:40:55
@LastEditors  : Scallions
@LastEditTime : 2020-02-22 12:17:54
@FilePath     : /gps-ts/tests/test_ts.py
@Description  : 
'''
import pytest

import ts.timeseries as Ts

@pytest.fixture()
def get_a_ts():
    ts = Ts.SingleTs("data/ASKY.cwu.igs14.csv")
    return ts

def test_gap_status(get_a_ts):
    ts = get_a_ts
    assert 1 == 1


def test_gap_plot(get_a_ts):
    ts = get_a_ts
    ts.plot_gap()