'''
@Author: Scallions
@Date: 2020-02-07 13:51:31
LastEditTime : 2020-10-04 10:37:27
LastEditors  : Scallions
FilePath     : /gps-ts/ts/fill.py
@Description: gap fill functions and return a new ts
'''
from fbprophet import Prophet
from loguru import logger
import matplotlib.pyplot as plt
import ts.ssa as ssa 
from ts.timeseries import SingleTs as STs
from ts.timeseries import MulTs as MTs
import ts.rnnfill as rnn
import ts.tcnfill as tcn
import pandas as pd
import sys
import warnings
import ts.regem as regem


class Filler:
    pass


class RegEMFiller(Filler):
    name = "RegEM"
    
    @staticmethod
    def fill(ts):
        tss = regem.fill(ts)
        return MTs(datas = tss)

class FbFiller(Filler):
    name = "fbprophet"
    
    @staticmethod
    def fill(ts, plot=False):
        def customwarn(message, category, filename, lineno, file=None,  line=None):
            sys.stdout.write(warnings.formatwarning(message, category, filename, lineno))
        warnings.showwarning = customwarn
        f = open("./res/fb.log","w")
        savestdout = sys.stdout
        savestderr = sys.stderr
        sys.stdout = f
        sys.stderr = f
        ts.complete()
        ts = ts.copy()
        ts.columns = ['y']
        ts['ds'] = ts.index
        m = Prophet()
        m.fit(ts)
        future = m.make_future_dataframe(periods=365)
        forecast = m.predict(future)
        if plot:
            fig1 = m.plot(forecast)
            fig1.savefig("./fig/forecast.png")
            fig2 = m.plot_components(forecast)
            fig2.savefig("./fig/components.png")
            plt.close("all")
        fts = forecast["yhat"][:-365]
        fts.index = ts.index 
        tss = type(ts)(datas = fts.values, indexs=ts.index)
        sys.stdout = savestdout
        sys.stderr = savestderr
        f.close()
        return tss

class SSAFiller(Filler):
    name = "SSA"
    
    @staticmethod
    def fill(ts):
        tsc = ts.complete()
        tc = ssa.iter_SSA_inner(tsc, 0.01, 4, 365)
        tss = type(ts)(datas = tc, indexs=tsc.index)
        return tss

class MSSAFiller(Filler):
    name = "MSSA"

    @staticmethod
    def fill(ts):
        tsc = ts.complete()
        tc = ssa.iter_MSSA_inner(tsc, 0.01, 4, 365)
        tss = MTs(datas = tc, indexs=tsc.index, columns=tsc.columns)
        return tss

class MeanFiller(Filler):
    name = "Mean"
    @staticmethod
    def fill(ts):
        tsc = ts.complete()
        tc = tsc.fillna(tsc.mean())
        tss = type(ts)(datas = tc, indexs=tsc.index)
        return tss

class MedianFiller(Filler):
    name="Median"
    @staticmethod
    def fill(ts):
        tsc = ts.complete()
        tc = tsc.fillna(tsc.median())
        tss = type(ts)(datas = tc, indexs=tsc.index)
        return tss

class RollingMeanFiller(Filler):
    name="RollingMean"
    @staticmethod
    def fill(ts,steps=24):
        tsc = ts.complete()
        tc = tsc.fillna(tsc.x.rolling(steps,min_periods=1,).mean())
        tss = type(ts)(datas = tc, indexs=tsc.index)
        return tss

class RollingMedianFiller(Filler):
    name = "RollingMedian"
    @staticmethod
    def fill(ts,steps=24):
        tsc = ts.complete()
        tc = tsc.fillna(tsc.x.rolling(steps,min_periods=1,).median())
        tss = type(ts)(datas = tc, indexs=tsc.index)
        return tss

class LinearFiller(Filler):
    name = "Linear"
    @staticmethod
    def fill(ts):
        tsc = ts.complete()
        tc = tsc.fillna(tsc.interpolate(method='linear'))
        tss = type(ts)(datas = tc, indexs=tsc.index)
        return tss

class TimeFiller(Filler):
    name = "Time"
    @staticmethod
    def fill(ts):
        tsc = ts.complete()
        tc = tsc.fillna(tsc.interpolate(method='time'))
        tss = type(ts)(datas = tc, indexs=tsc.index)
        return tss

class QuadraticFiller(Filler):
    # 二次
    name = "Quadratic"
    @staticmethod
    def fill(ts):
        tsc = ts.complete()
        tc = tsc.fillna(tsc.interpolate(method='quadratic'))
        tss = type(ts)(datas = tc, indexs=tsc.index)
        return tss

class CubicFiller(Filler):
    name = "Cubic"
    @staticmethod
    def fill(ts):
        tsc = ts.complete()
        tc = tsc.fillna(tsc.interpolate(method='cubic'))
        tss = type(ts)(datas = tc, indexs=tsc.index)
        return tss

class SLinearFiller(Filler):
    name = "SLinear"
    @staticmethod
    def fill(ts):
        tsc = ts.complete()
        tc = tsc.fillna(tsc.interpolate(method='slinear'))
        tss = type(ts)(datas = tc, indexs=tsc.index)
        return tss

class AkimaFiller(Filler):
    name = "Akima"
    @staticmethod
    def fill(ts):
        tsc = ts.complete()
        tc = tsc.fillna(tsc.interpolate(method='akima'))
        tss = type(ts)(datas = tc, indexs=tsc.index)
        return tss

class PolyFiller(Filler):
    name = "Poly"
    @staticmethod
    def fill(ts, order=2):
        tsc = ts.complete()
        tc = tsc.fillna(tsc.interpolate(method='polynomial', order=order))
        tss = type(ts)(datas = tc, indexs=tsc.index)
        return tss

class SplineFiller(Filler):
    name = "Spline"
    @staticmethod
    def fill(ts, order=3):
        tsc = ts.complete()
        tc = tsc.fillna(tsc.interpolate(method='spline', order=order))
        tss = type(ts)(datas = tc, indexs=tsc.index)
        return tss

class BarycentricFiller(Filler):
    name = "Barycentric"
    @staticmethod
    def fill(ts):
        tsc = ts.complete()
        tc = tsc.fillna(tsc.interpolate(method='barycentric'))
        tss = type(ts)(datas = tc, indexs=tsc.index)
        return tss

class KroghFiller(Filler):
    name = "Krogh"
    @staticmethod
    def fill(ts):
        tsc = ts.complete()
        tc = tsc.fillna(tsc.interpolate(method='krogh'))
        tss = type(ts)(datas = tc, indexs=tsc.index)
        return tss

class PiecewisePolynomialFiller(Filler):
    name = "PiecewisePolynominal"
    @staticmethod
    def fill(ts):
        tsc = ts.complete()
        tc = tsc.fillna(tsc.interpolate(method='piecewise_polynomial'))
        tss = type(ts)(datas = tc, indexs=tsc.index)
        return tss

        
class FromDerivativesFiller(Filler):
    name = 'FromDerivatives'
    @staticmethod
    def fill(ts):
        tsc = ts.complete()
        tc = tsc.fillna(tsc.interpolate(method='from_derivatives'))
        tss = type(ts)(datas = tc, indexs=tsc.index)
        return tss


class PchipFiller(Filler):
    name = "Pchip"
    @staticmethod
    def fill(ts):
        tsc = ts.complete()
        tc = tsc.fillna(tsc.interpolate(method='pchip'))
        tss = type(ts)(datas = tc, indexs=tsc.index)
        return tss

class LSTMFiller(Filler):
    name = "LSTM"
    @staticmethod
    def fill(ts):
        tsc = ts.complete()
        tsc = tsc.copy()
        tc = rnn.lstm_fill(tsc)
        return type(ts)(datas = tc, indexs=tsc.index)


class TCNFiller(Filler):
    name = "TCN"
    @staticmethod
    def fill(ts):
        tsc = ts.complete()
        tsc = tsc.copy()
        tc = tcn.tcn_fill(tsc)
        return type(ts)(datas = tc, indexs=tsc.index)


class GANFiller(Filler):
    name = "GAN"
    @staticmethod
    def fill(ts):
        import ts.ganfill as gan
        tsc = ts.complete()
        tc = gan.ganfill(tsc)
        return tc


class MLPFiller(Filler):
    name = "MLP"
    @staticmethod 
    def fill(ts):
        import ts.mlp as mlp
        tsc = ts.complete()
        tc = mlp.fill(tsc)
        return tc