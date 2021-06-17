import os
import sys
sys.path.append('./')
import pandas as pd
import ts.fill as fill
import numpy as np
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.ticker as ticker
from pandas import Series
from scipy import stats

if __name__ == '__main__':

    fillers = [
        fill.SplineFiller,
        fill.PolyFiller,
        fill.PchipFiller,
        fill.RegEMFiller,
        fill.MissForestFiller,
    ]
    gap_sizes = [
        2,
        7,
        30,
        180,
    ]

    gap_rates = [
        0.1,
        0.2,
        0.3,
        0.4,
    ]

    plt.rcParams['font.sans-serif'] = ['SimHei']
    fig, subs = plt.subplots(4, 5, sharex=True,figsize=(12,8))
    for k, gap_size in enumerate(gap_sizes):
        print(f"+++++ gap size: {gap_size} +++++")

        for i, filler in enumerate(fillers):
            raw_s = None
            fills = None
            for j in range(20):
                raw = pd.read_csv(f"res/gap/raw-{gap_size}-{j}.csv",index_col="jd",parse_dates=True)
                gap = pd.read_csv(f"res/gap/gap-{gap_size}-{j}.csv",index_col="jd",parse_dates=True)
                # find cidx and ridx
                a = gap.isna()
                r_idx = np.where(np.any(a, axis=1))[0]
                c_idx = np.where(np.any(a, axis=0))[0]
                filldata = pd.read_csv(f"res/gap/{filler.name}-{gap_size}-fill-{j}.csv",index_col="jd",parse_dates=True) 
                if raw_s is None:
                    raw_s = raw.iloc[r_idx,c_idx].to_numpy()
                    fills = filldata.iloc[r_idx,c_idx].to_numpy()
                else:
                    raw_s = np.concatenate([raw.iloc[r_idx,c_idx].to_numpy(),raw_s])
                    fills = np.concatenate([filldata.iloc[r_idx,c_idx].to_numpy(), fills])
            raw_s = raw_s.reshape(-1)
            fills = fills.reshape(-1)
            sns.regplot(raw_s.reshape(-1), fills.reshape(-1),ax=subs[k][i],color='steelblue',scatter_kws={'s':2},line_kws={"linewidth":1})
            var = stats.pearsonr(raw_s, fills)[0]
            if k == 3:
                subs[k][i].set_xlabel(filler.fname, fontsize=12)
            if i == 0:
                if gap_size == 7:
                    subs[k][i].set_ylabel(f"20 by {gap_size}-day",fontsize=12)
                else:
                    subs[k][i].set_ylabel(f"{180//gap_size} by {gap_size}-day",fontsize=12)
            subs[k][i].text(0.2, 0.8,f'r:{var:.2f}',
            horizontalalignment='center',
            verticalalignment='center',
            transform = subs[k][i].transAxes,
            bbox=dict(boxstyle="round",ec=(1., 0.5, 0.5),fc=(1., 0.8, 0.8),))
            r = pearsonr(raw_s, fills)
            print(filler.name, r[0])


    # subs[-1,-1].axis("off")
    plt.savefig(f"fig/gap-rp.png")
            