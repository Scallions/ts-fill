import os
import sys
sys.path.append('./')
import pandas as pd
import ts.fill as fill
import numpy as np


def mean(x):
    xc = x.mean(axis=0)
    return x - xc

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
        0.5
    ]

    for gap_size in gap_rates:

        print(f"+++++ gap size: {gap_size} +++++")
        for i, filler in enumerate(fillers):
            ress = None
            lss = None
            rss = None
            angs = None
            for j in range(20):
                raw = pd.read_csv(f"res/rate/raw-{gap_size}-{j}.csv",index_col="jd",parse_dates=True)
                gap = pd.read_csv(f"res/rate/gap-{gap_size}-{j}.csv",index_col="jd",parse_dates=True)
                # find cidx and ridx
                a = gap.isna()
                r_idx = np.where(np.any(a, axis=1))[0]
                c_idx = np.where(np.any(a, axis=0))[0]
                # row pcaX
                X = mean(raw).T
                n, m = X.shape 
                C = np.dot(X, X.T) / (n-1)
                eigen_vals, eigen_vecs = np.linalg.eig(C)
                s = np.sum(eigen_vals)
                res = eigen_vals / s * 100
                r = res.cumsum()
                if j == 0 and i == 0:
                    print("raw",res[:3],r[2])
                # raw1 = np.dot(X.T,eigen_vecs)[:,0].real
                raw1 = eigen_vecs[:,0]

                filldata = pd.read_csv(f"res/rate/{filler.name}-{gap_size}-fill-{j}.csv",index_col="jd",parse_dates=True) 
                X = mean(filldata).T
                n, m = X.shape 
                C = np.dot(X, X.T) / (n-1)
                _, vecs = np.linalg.eig(C)
                V =  np.dot(eigen_vecs.T, np.dot(C, eigen_vecs))
                vs = np.diagonal(V)
                s = np.sum(vs)
                res = vs/s * 100
                r = res.cumsum()
                # xs = np.dot(X.T,eigen_vecs)[:,0].real
                xs = vecs[:,0]
                angle = np.arccos(sum(raw1*xs)/ np.sqrt(sum(raw1*raw1)*sum(xs*xs)))
                if angle > 3.14/2:
                    # xs = -1*np.dot(X.T,eigen_vecs)[:,0].real
                    xs = -1*xs
                    angle = np.arccos(sum(raw1*xs)/ np.sqrt(sum(raw1*raw1)*sum(xs*xs)))
                ang =  angle*180/np.pi
                length =  np.sqrt(np.dot(xs,xs.T))
                distance = np.sqrt(np.dot((raw1-xs),(raw1-xs).T))
                if ress is None:
                    ress = res.real[:3]
                    lss = [distance]
                    rss = [r[2]]
                    angs = [np.cos(angle)]
                else:
                    ress = np.vstack([ress,res.real[:3]])
                    lss.append(distance)
                    rss.append(r[2])
                    angs.append(np.cos(angle))
            print(filler.name, ress.mean(axis=0),np.mean(rss),np.mean(lss),np.mean(angs))



