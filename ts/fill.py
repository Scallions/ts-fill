'''
@Author: Scallions
@Date: 2020-02-07 13:51:31
@LastEditTime : 2020-03-24 17:31:37
@LastEditors  : Scallions
@FilePath     : /gps-ts/ts/fill.py
@Description: gap fill functions and return a new ts
'''
from fbprophet import Prophet
from loguru import logger
import matplotlib.pyplot as plt
import ts.ssa as ssa 
from ts.timeseries import SingleTs as STs
import ts.rnnfill as rnn
import pandas as pd
import sys
import warnings
import ts.regem as regem


class Filler:
    pass


class RegEMFiller(Filler):
    @staticmethod
    def fill(ts):
        tss = regem.fill(ts)
        return MTs(datas = tss.values, indexs = ts.index)

class FbFiller(Filler):
    @staticmethod
    def fill(ts):
        def customwarn(message, category, filename, lineno, file=None,  line=None):
            sys.stdout.write(warnings.formatwarning(message, category, filename, lineno))
        warnings.showwarning = customwarn
        f = open("./res/fb.log","w")
        savestdout = sys.stdout
        savestderr = sys.stderr
        sys.stdout = f
        sys.stderr = f
        ts.complete()
        ts.columns = ['y']
        ts['ds'] = ts.index
        m = Prophet()
        m.fit(ts)
        future = m.make_future_dataframe(periods=365)
        forecast = m.predict(future)
        fig1 = m.plot(forecast)
        fig1.savefig("./fig/forecast.png")
        fig2 = m.plot_components(forecast)
        fig2.savefig("./fig/components.png")
        plt.close("all")
        fts = forecast["yhat"][:-365]
        fts.index = ts.index 
        tss = STs(datas = fts.values, indexs=ts.index)
        sys.stdout = savestdout
        sys.stderr = savestderr
        f.close()
        return tss

class SSAFiller(Filler):
    @staticmethod
    def fill(ts):
        tsc = ts.complete()
        tc = ssa.iter_SSA_inner(tsc, 0.01, 4, 365)
        tss = STs(datas = tc, indexs=tsc.index)
        return tss


class MeanFiller(Filler):
    @staticmethod
    def fill(ts):
        tsc = ts.complete()
        tc = ts.x.fillna(tsc.x.mean())
        tss = STs(datas = tc, indexs=tsc.index)
        return tss

class MedianFiller(Filler):
    @staticmethod
    def fill(ts):
        tsc = ts.complete()
        tc = ts.x.fillna(tsc.x.median())
        tss = STs(datas = tc, indexs=tsc.index)
        return tss

class RollingMeanFiller(Filler):
    @staticmethod
    def fill(ts,steps=24):
        tsc = ts.complete()
        tc = ts.x.fillna(tsc.x.rolling(steps,min_periods=1,).mean())
        tss = STs(datas = tc, indexs=tsc.index)
        return tss

class RollingMedianFiller(Filler):
    @staticmethod
    def fill(ts,steps=24):
        tsc = ts.complete()
        tc = ts.x.fillna(tsc.x.rolling(steps,min_periods=1,).median())
        tss = STs(datas = tc, indexs=tsc.index)
        return tss

class LinearFiller(Filler):
    @staticmethod
    def fill(ts):
        tsc = ts.complete()
        tc = ts.x.fillna(tsc.x.interpolate(method='linear'))
        tss = STs(datas = tc, indexs=tsc.index)
        return tss

class TimeFiller(Filler):
    @staticmethod
    def fill(ts):
        tsc = ts.complete()
        tc = ts.x.fillna(tsc.x.interpolate(method='time'))
        tss = STs(datas = tc, indexs=tsc.index)
        return tss

class QuadraticFiller(Filler):
    @staticmethod
    def fill(ts):
        tsc = ts.complete()
        tc = ts.x.fillna(tsc.x.interpolate(method='quadratic'))
        tss = STs(datas = tc, indexs=tsc.index)
        return tss

class CubicFiller(Filler):
    @staticmethod
    def fill(ts):
        tsc = ts.complete()
        tc = ts.x.fillna(tsc.x.interpolate(method='cubic'))
        tss = STs(datas = tc, indexs=tsc.index)
        return tss

class SLinearFiller(Filler):
    @staticmethod
    def fill(ts):
        tsc = ts.complete()
        tc = tsc.x.fillna(tsc.x.interpolate(method='slinear'))
        tss = STs(datas = tc, indexs=tsc.index)
        return tss

class AkimaFiller(Filler):
    @staticmethod
    def fill(ts):
        tsc = ts.complete()
        tc = ts.x.fillna(tsc.x.interpolate(method='akima'))
        tss = STs(datas = tc, indexs=tsc.index)
        return tss

class PolyFiller(Filler):
    @staticmethod
    def fill(ts, order=3):
        tsc = ts.complete()
        tc = ts.x.fillna(tsc.x.interpolate(method='polynomial', order=order))
        tss = STs(datas = tc, indexs=tsc.index)
        return tss

class SplineFiller(Filler):
    @staticmethod
    def fill(ts, order=3):
        tsc = ts.complete()
        tc = ts.x.fillna(tsc.x.interpolate(method='spline', order=order))
        tss = STs(datas = tc, indexs=tsc.index)
        return tss


class LstmFiller(Filler):
    @staticmethod
    def fill(ts):
        net = rnn.load_model(rnn.Net,'lstm')
        tsc = ts.complete()
        tsf = tsc.copy()
        gap_index = tsc.isnull()
        for gap in gap_index:
            data = torch.zeros((1,30))
            for i in range(30):
                data_idx = gap-pd.Timedelta(days=(30-i))
                data[0,i] = tsc.loc[data_idx]
            out = net.predict(data).item()
            tsf.loc[gap] = out
        return STs(datas = tsf, indexs=tsc.index)