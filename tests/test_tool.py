'''
@Author       : Scallions
@Date         : 2020-02-05 14:01:53
@LastEditors  : Scallions
@LastEditTime : 2020-02-22 10:37:58
@FilePath     : /gps-ts/tests/test_tool.py
@Description  : 
'''
import pytest 
import ts.tool as tool

def test_dy2jd():
    dy = 2020.123
    assert 2458895 == tool.dy2jd(dy)


def test_pytest():
    assert 1 == 1