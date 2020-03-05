'''
@Author: Scallions
@Date: 2020-02-07 13:51:31
@LastEditTime : 2020-03-05 17:32:07
@LastEditors  : Scallions
@FilePath     : /gps-ts/ts/fill.py
@Description: gap fill functions and return a new ts
'''
from fbprophet import Prophet
from loguru import logger
import matplotlib.pyplot as plt
import ts.ssa as ssa 
from ts.timeseries import SingleTs as STs


class Filler:
    pass


class FbFiller(Filler):
    @staticmethod
    def fill(ts):
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
        return fts

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
        return tc 

class MedianFiller(Filler):
    @staticmethod
    def fill(ts):
        tsc = ts.complete()
        tc = ts.x.fillna(tsc.x.median())

class RollingMeanFiller(Filler):
    @staticmethod
    def fill(ts,steps=24):
        tsc = ts.complete()
        tc = ts.x.fillna(tsc.x.rolling(steps,min_periods=1,).mean())
        return tc 

class RollingMedianFiller(Filler):
    @staticmethod
    def fill(ts,steps=24):
        tsc = ts.complete()
        tc = ts.x.fillna(tsc.x.rolling(steps,min_periods=1,).median())

class LinearFiller(Filler):
    @staticmethod
    def fill(ts):
        tsc = ts.complete()
        tc = ts.x.fillna(tsc.x.interpolate(method='linear'))
        return tc

class TimeFiller(Filler):
    @staticmethod
    def fill(ts):
        tsc = ts.complete()
        tc = ts.x.fillna(tsc.x.interpolate(method='time'))
        return tc

class QuadraticFiller(Filler):
    @staticmethod
    def fill(ts):
        tsc = ts.complete()
        tc = ts.x.fillna(tsc.x.interpolate(method='quadratic'))
        return tc

class CubicFiller(Filler):
    @staticmethod
    def fill(ts):
        tsc = ts.complete()
        tc = ts.x.fillna(tsc.x.interpolate(method='cubic'))
        return tc

class SLinearFiller(Filler):
    @staticmethod
    def fill(ts):
        tsc = ts.complete()
        tc = tsc.x.fillna(tsc.x.interpolate(method='slinear'))
        return tc

class AkimaFiller(Filler):
    @staticmethod
    def fill(ts):
        tsc = ts.complete()
        tc = ts.x.fillna(tsc.x.interpolate(method='akima'))
        return tc

class PolyFiller(Filler):
    @staticmethod
    def fill(ts, order):
        tsc = ts.complete()
        tc = ts.x.fillna(tsc.x.interpolate(method='polynomial', order=order))
        return tc

class SplineFiller(Filler):
    @staticmethod
    def fill(ts, order):
        tsc = ts.complete()
        tc = ts.x.fillna(tsc.x.interpolate(method='spline', order=order))
        return tc