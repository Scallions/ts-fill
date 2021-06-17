'''
@Author: Scallions
@Date: 2020-02-07 13:51:31
LastEditTime : 2021-03-11 19:41:48
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
# import ts.tcnfill as tcn
import pandas as pd
import sys
import warnings
import ts.regem as regem


class Filler:
    pass


class RegEMFiller(Filler):
    name = "RegEM"
    fname = "RegEM"
    cnname = "RegEM"
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
    fname = "Orthogonal Polynomial"
    cnname = "正交多项式"
    @staticmethod
    def fill(ts, order=2):
        tsc = ts.complete()
        tc = tsc.fillna(tsc.interpolate(method='polynomial', order=order))
        tss = type(ts)(datas = tc, indexs=tsc.index)
        return tss

class SplineFiller(Filler):
    fname = "Cubic Spline"
    name = "Spline"
    cnname = "三次样条"
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
    fname = "Hermite"
    cnname = "埃尔米特多项式"
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


# class TCNFiller(Filler):
#     name = "TCN"
#     @staticmethod
#     def fill(ts):
#         tsc = ts.complete()
#         tsc = tsc.copy()
#         tc = tcn.tcn_fill(tsc)
#         return type(ts)(datas = tc, indexs=tsc.index)


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

class ConvFiller(Filler):
    name = "Conv"
    @staticmethod
    def fill(ts):
        import ts.conv as conv 
        tsc = ts.complete()
        tc = conv.fill(tsc)
        return tc


class KNNFiller(Filler):
    name = "KNN"
    @staticmethod
    def fill(ts):
        from fancyimpute import KNN 
        tsc = ts.complete()
        tc = KNN(k=3).fit_transform(tsc)
        return type(ts)(datas = tc, indexs=tsc.index, columns=tsc.columns)


class SoftImputeFiller(Filler):
    name = "SoftImpute"
    @staticmethod
    def fill(ts):
        from fancyimpute import SoftImpute
        tsc = ts.complete()
        tc = SoftImpute().fit_transform(tsc)
        return type(ts)(datas = tc, indexs=tsc.index, columns=tsc.columns)

class IterativeSVDFiller(Filler):
    name = "IterativeSVD"
    @staticmethod
    def fill(ts):
        from fancyimpute import IterativeSVD
        tsc = ts.complete()
        tc = IterativeSVD().fit_transform(tsc)
        return type(ts)(datas = tc, indexs=tsc.index, columns=tsc.columns)

class IterativeImputerFiller(Filler):
    name = "IterativeImputer"
    @staticmethod
    def fill(ts):
        from fancyimpute import IterativeImputer
        tsc = ts.complete()
        tc = IterativeImputer().fit_transform(tsc)
        return type(ts)(datas = tc, indexs=tsc.index, columns=tsc.columns)


class MatrixFactorizationFiller(Filler):
    name = "MF"
    @staticmethod
    def fill(ts):
        from fancyimpute import MatrixFactorization
        tsc = ts.complete()
        tc = MatrixFactorization().fit_transform(tsc)
        return type(ts)(datas = tc, indexs=tsc.index, columns=tsc.columns)

class BiScalerFiller(Filler):
    name = "BiScaler"
    @staticmethod
    def fill(ts):
        from fancyimpute import BiScaler
        tsc = ts.complete()
        tc = BiScaler().fit_transform(tsc.to_numpy())
        return type(ts)(datas = tc, indexs=tsc.index, columns=tsc.columns)
class NuclearNormMinimizationFiller(Filler):
    name = "NuclearNormMinimization"
    @staticmethod
    def fill(ts):
        from fancyimpute import NuclearNormMinimization
        tsc = ts.complete()
        tc = NuclearNormMinimization().fit_transform(tsc)
        return type(ts)(datas = tc, indexs=tsc.index, columns=tsc.columns)

class SVMFiller(Filler):
    name = "SVM"
    cnname = "支持向量机回归"
    @staticmethod
    def fill(ts):
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer
        from sklearn import svm
        # from sklearn import linear_model
        tsc = ts.complete()
        tc = IterativeImputer(estimator=svm.SVR()).fit_transform(tsc)
        return type(ts)(datas = tc, indexs=tsc.index, columns=tsc.columns)

class MissForestFiller(Filler):
    name = "MissForest"
    fname = "MissForest"
    cnname = "随机森林回归"
    @staticmethod
    def fill(ts, oob=False):
        from missingpy import MissForest
        tsc = ts.complete()
        if not oob:
            tc = MissForest(decreasing=True, max_iter=5, n_estimators=50).fit_transform(tsc)
            return type(ts)(datas = tc, indexs=tsc.index, columns=tsc.columns)
        else:
            tc, oobs, oobp = MissForest(oob_score=oob).fit_transform(tsc) 
            return type(ts)(datas = tc, indexs=tsc.index, columns=tsc.columns), oobs, oobp

class MICEFiller(Filler):
    name = "MICE"
    @staticmethod
    def fill(ts):
        import imputena
        tsc = ts.complete()
        tc = imputena.mice(data=tsc)
        return type(ts)(datas = tc[2], indexs=tsc.index, columns=tsc.columns)

class RandomForestFiller(Filler):
    name = "RF"
    @staticmethod
    def fill(ts):
        from pimpute import RFImputer
        tsc = ts.complete()
        tc = RFImputer.impute(tsc)
        return type(ts)(datas = tc, indexs=tsc.index, columns=tsc.columns)    

class MagicFiller(Filler):
    name = "Magic"
    @staticmethod
    def fill(ts):
        import magic 
        tsc = ts.complete()
        tc = magic.MAGIC().fit_transform(tsc)
        return type(ts)(datas = tc, indexs=tsc.index, columns=tsc.columns)    

class MiceForestFiller(Filler):
    name = "Miceforest"
    @staticmethod
    def fill(ts):
        import miceforest as mf
        tsc = ts.complete()
        kernel = mf.MultipleImputedKernel(
            tsc,
            datasets=5,
            save_all_iterations=True,
            random_state=1991
        )
        kernel.mice(5)
        tc = kernel.complete_data(4)
        return type(ts)(datas = tc, indexs=tsc.index, columns=tsc.columns)    

class GainFiller(Filler):
    name = "GAIN"
    @staticmethod
    def fill(ts):
        import ts.gain as gain
        tsc = ts.complete()
        tc = gain.fill(tsc)
        return tc
 

class BritsFiller(Filler):
    name = "Brits"
    @staticmethod
    def fill(ts):
        import ts.brits as brits
        tsc = ts.complete()
        tc = brits.fill(tsc)
        return tc
 
 
class GBDT(Filler):
    name = "GBDT"
    @staticmethod
    def fill(ts):
        from sklearn.ensemble import GradientBoostingRegressor
        reg = GradientBoostingRegressor(random_state=0)
        import numpy as np
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer
        imp_mean = IterativeImputer(estimator=reg)
        # imp_mean = IterativeImputer(random_state=0)
        tsc = ts.complete()
        tc = imp_mean.fit_transform(ts)
        return type(ts)(datas = tc, indexs=tsc.index, columns=tsc.columns)    