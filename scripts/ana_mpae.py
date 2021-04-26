import os
import sys
sys.path.append('./')
import pandas as pd
import ts.fill as fill
import numpy as np

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

    for gap_size in gap_rates:

        print(f"+++++ gap size: {gap_size} +++++")
        for filler in fillers:
            filldata = pd.read_csv(f"res/rate/{filler.name}-{gap_size}.csv") 
            # print(filler.name,filldata.abs().mean()*100)
            print(filler.name,filldata.abs().mean())
            


