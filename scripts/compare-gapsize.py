"""compare different gap rate
"""

import os
import sys
sys.path.append('./')
import ts.tool as tool
import ts.data as data
from ts.timeseries import SingleTs as Sts
from ts.timeseries import MulTs as Mts
import ts.fill as fill
from loguru import logger
import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt

SITES = [
    # ['FONP.AN.tenv3', 'OHI3.AN.tenv3', 'PAL2.AN.tenv3', 'PALM.AN.tenv3', 'PALV.AN.tenv3'],
    ['BLAS.NA.tenv3', 'DGJG.NA.tenv3', 'DKSG.NA.tenv3', 'HJOR.NA.tenv3', 'HMBG.NA.tenv3', 'HRDG.NA.tenv3', 'JGBL.NA.tenv3', 'JWLF.NA.tenv3', 'KMJP.NA.tenv3', 'KMOR.NA.tenv3', 'KUAQ.NA.tenv3', 'KULL.NA.tenv3', 'LBIB.NA.tenv3', 'LEFN.NA.tenv3', 'MARG.NA.tenv3', 'MSVG.NA.tenv3', 'NRSK.NA.tenv3', 'QAAR.NA.tenv3', 'UTMG.NA.tenv3', 'YMER.NA.tenv3'],
    # ['VNAD.AN.tenv3', 'PALM.AN.tenv3', 'PAL2.AN.tenv3', 'PRPT.AN.tenv3', 'DUPT.AN.tenv3'],
    # # ['CAS1.AN.tenv3', 'MCM4.AN.tenv3', 'DAV1.AN.tenv3'],
    # # ['CAS1.AN.tenv3', 'CRAR.AN.tenv3', 'MCM4.AN.tenv3'],
    # ['CAPF.AN.tenv3', 'PRPT.AN.tenv3', 'PAL2.AN.tenv3'], # an
    # ['CAPF.AN.tenv3', 'PRPT.AN.tenv3', 'ROBN.AN.tenv3', 'PAL2.AN.tenv3'],
    # ['LEIJ.EU.tenv3', 'WARN.EU.tenv3', 'POTS.EU.tenv3', 'PTBB.EU.tenv3'], # eu
]

def load_data():
    """load data
    
    Args:
        lengths (int, optional): nums of mul ts. Defaults to 3.
    
    Returns:
        [mts]: mts s
    """
    dir_path = './data/greenland/'
    rtss = []
    for site in SITES:
        tss = []
        for file_ in site:
            tss.append(Mts(dir_path + file_, data.FileType.Ngl))
        mts = tool.concat_u_multss(tss)
        rtss.append(mts)
    return rtss
TSS = None
def load_data2(lengths=3, epoch=6, length=None, minlen=None):
    """load data
    
    Args:
        lengths (int, optional): nums of mul ts. Defaults to 3.
    
    Returns:
        [mts]: mts s
    """
    global TSS
    dir_path = './data/greenland/'
    files = os.listdir(dir_path)
    # for file_ in files:
    if TSS is None:
        TSS = []
        for file_ in ['BLAS.NA.tenv3', 'DGJG.NA.tenv3', 'DKSG.NA.tenv3', 'HJOR.NA.tenv3', 'HMBG.NA.tenv3', 'HRDG.NA.tenv3', 'JGBL.NA.tenv3', 'JWLF.NA.tenv3', 'KMJP.NA.tenv3', 'KMOR.NA.tenv3', 'KUAQ.NA.tenv3', 'KULL.NA.tenv3', 'LBIB.NA.tenv3', 'LEFN.NA.tenv3', 'MARG.NA.tenv3', 'MSVG.NA.tenv3', 'NRSK.NA.tenv3', 'QAAR.NA.tenv3', 'UTMG.NA.tenv3', 'YMER.NA.tenv3']:
            if '.tenv3' in file_:
                ts_temp = Mts(dir_path + file_, data.FileType.Ngl)
                ts_temp = (ts_temp - ts_temp.min()) / (ts_temp.max()-ts_temp.min())
                ts_temp = Mts.from_df(ts_temp)
                TSS.append((file_, ts_temp))
    nums = len(TSS)
    rtss = []
    import random
    if length != None:
        j = 0
        while j < length:
            random.shuffle(TSS)
            for i in range(0, nums - 1, lengths):
                try:
                    if (i+lengths) > nums:
                        continue
                    data1 = TSS[i:i + lengths]
                    names = [d[0] for d in data1]
                    ttss  =[d[1] for d in data1]
                    mts = tool.concat_u_multss(ttss)
                    rtss.append((names,mts))
                except:
                    continue
            j += 1
            print(f"\r{j}/{length}",end="\r")
        return rtss
    for j in range(epoch):
        random.shuffle(tss)
        for i in range(0, nums - 1, lengths):
            try:
                if (i+lengths) > nums:
                    continue
                data1 = tss[i:i + lengths]
                names = [d[0] for d in data1]
                ttss  =[d[1] for d in data1]
                mts = tool.concat_u_multss(ttss)
                rtss.append((names,mts))
            except:
                continue
    return rtss

if __name__ == '__main__':
    """genrel code for compare
    """


    # 定义 插值方法 gaprates

    fillers = [
        ### imputena
        # fill.MICEFiller,

        ### missingpy
        fill.MissForestFiller,

        ### miceforest
        # fill.MiceForestFiller,

        ### para
        # fill.RandomForestFiller,

        ### magic
        # fill.MagicFiller,
        
        ### fancyimpute
        # fill.KNNFiller,
        # fill.SoftImputeFiller,
        # fill.IterativeSVDFiller,
        # fill.IterativeImputerFiller,
        # fill.MatrixFactorizationFiller,
        # fill.BiScalerFiller,
        # fill.NuclearNormMinimizationFiller,

        ### scipy
        # fill.SLinearFiller, # 一阶样条插值
        fill.SplineFiller, # 三次样条
        # fill.AkimaFiller,
        fill.PolyFiller, # 二阶多项式插值÷
        # fill.PiecewisePolynomialFiller, 
        # fill.KroghFiller, # overflow
        # fill.QuadraticFiller, # 二次
        # fill.BarycentricFiller, # overflow
        # fill.FromDerivativesFiller,
        fill.PchipFiller, # 三阶 hermite 插值

        ### matlab
        fill.RegEMFiller, 

        ### self
        # fill.BritsFiller,
        # fill.MLPFiller,
        # fill.ConvFiller,
        # fill.SSAFiller,
        fill.GBDT,
    ]
    gap_sizes = [
        2,
        7,
        30,
        180
    ]

    gap_rate = 0.2

    ltss = []
    while len(ltss) == 0:
        tss = load_data2(lengths=20, length=200)
        ltss = [[names,ts.get_longest()] for names,ts in tss if len(ts.get_longest()) > 400]
        # val_tss = [(ts, *ts.make_gap(gap_size,cache_size=100, cper=0.5, c_i=False, c_ii=['n0,e0,u0'], gmax=gap_size)) for ts in ltss]
        # val_tss = [(names,ts, *ts.make_gap(gap_size, per=0.2, cper=0.5)) for names, ts in ltss]
        ltss = ltss[:1]
    # 遍历gap rates
    res = pd.DataFrame(columns=['gap_rate']+list(map(lambda x:x.name, fillers)))
    for gap_size in gap_sizes:
        ## 生成 数据集
        # - 筛选数据
        ltss = [[names,ts.get_longest()] for names,ts in tss if len(ts.get_longest()) > 300]
        # val_tss = [(ts, *ts.make_gap(gap_size,cache_size=100, cper=0.5, c_i=False, c_ii=['n0,e0,u0'], gmax=gap_size)) for ts in ltss]
        ltss = ltss[:1]
        res = res.append(
            {
                'gap_size':gap_size,
            },
            ignore_index=True
        )
        rr = {}
        j = 0
        while j < 200:
            val_tss = [(names,ts, *ts.make_gap(gap_size,per=gap_rate, cper=0.5)) for names, ts in ltss]
            for names, tsl, tsg, gidx, gridx in val_tss:
                ### 单个数据集插值
                # - 去除趋势项
                # trends, noises = tool.remove_trend(tsl)
                # tsg = tsl.copy()
                # tsg.loc[gidx, gridx] = None
                # noises.to_csv("res/raw.csv")
                # noises.loc[gidx, gridx] = None
                # noises = Mts(datas=noises,indexs=noises.index,columns=noises.columns)
                for i, filler in enumerate(fillers):
                    tsc = filler.fill(tsg)
                    # tsc = trends + tsc
                    # tsc.to_csv(f"res/{gap_size}/"+filler.name+".csv")
                    # print(filler.name, (tsl.loc[gidx,gridx] - tsc.loc[gidx,gridx]).std().mean())
                    if filler.name in rr:
                        temp = (tsl.loc[gidx,gridx] - tsc.loc[gidx,gridx]).to_numpy().reshape(-1)
                        rr[filler.name] = np.concatenate([rr[filler.name],temp])
                    else:
                        rr[filler.name] = (tsl.loc[gidx,gridx] - tsc.loc[gidx,gridx]).to_numpy().reshape(-1)
                ### 统计插值方法结果
            j+=1
            print(f"\r{j}/200", end="")
        for filler in fillers:
            res[filler.name].iloc[-1] = rr[filler.name].std()
            pd.DataFrame(rr[filler.name]).to_csv(f"res/gap/{filler.name}-{gap_rate}.csv")
    print(res)
    res.to_csv("res/gap.csv")



