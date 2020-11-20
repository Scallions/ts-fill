'''
Author       : Scallions
Date         : 2020-11-14 10:15:45
LastEditors  : Scallions
LastEditTime : 2020-11-18 09:56:40
FilePath     : /gps-ts/scripts/plot_site_pos.py
Description  : 
'''

import requests
from requests.api import get
# from bs4 import Beautifulsoup
from lxml import etree
import os
import pandas as pd
import time
import pygmt

def sites_pos():
    """load data
    
    Args:
        lengths (int, optional): nums of mul ts. Defaults to 3.
    
    Returns:
        [mts]: mts s
    """
    # dir_path = './data/greenland/'
    tss = []
    # files = os.listdir(dir_path)
    sites = []
    df = pd.DataFrame(columns=["Name","Latitude","Longitude","Height(m)"])
    # for file_ in files:
    for file_ in ['BLAS.NA.tenv3', 'DGJG.NA.tenv3', 'DKSG.NA.tenv3', 'HJOR.NA.tenv3', 'HMBG.NA.tenv3', 'HRDG.NA.tenv3', 'JGBL.NA.tenv3', 'JWLF.NA.tenv3', 'KMJP.NA.tenv3', 'KMOR.NA.tenv3', 'KUAQ.NA.tenv3', 'KULL.NA.tenv3', 'LBIB.NA.tenv3', 'LEFN.NA.tenv3', 'MARG.NA.tenv3', 'MSVG.NA.tenv3', 'NRSK.NA.tenv3', 'QAAR.NA.tenv3', 'UTMG.NA.tenv3', 'YMER.NA.tenv3']:
        if '.tenv3' in file_:
            site = file_.split('.')[0]
            pos = get_site_pos(site)
            while len(pos) == 0:
                time.sleep(1)
                pos = get_site_pos(site)
            s = pd.Series([site] + pos, index=df.columns)
            df = df.append(s, ignore_index=True)
    # df.to_csv(dir_path+"pos.csv")
    return df


def get_site_pos(site):
    # site = "OHI3"
    url = f"http://geodesy.unr.edu/NGLStationPages/stations/{site}.sta"
    rep = requests.get(url)
    # html = lxml.html.fromstring(rep.text)
    html = etree.HTML(rep.text)
    pos = html.xpath('/html/body/div[3]/table[1]/tr[1]/td/table/tr[last()]/td//h4')
    pp = []
    for p in pos:
        pp.append(float(p.text.split()[1]))
    print(f"{site}: {pp}")
    return pp


if __name__ == "__main__":
    df = sites_pos()
    # get_site_pos("CAPF")
    fig = pygmt.Figure()

    # fig.basemap(region=[-109,-45,-78,-59], projection="S-77/-90/5i", frame=True)
    print(df.head())
    fig.basemap(region="-75/-10/58/86", projection="L-40/30/35/25/5i", frame=True)
    fig.coast(shorelines=True, water="lightblue")
    fig.plot(x=df.iloc[:,2],y=df.iloc[:,1], style="c0.2c", color="red", pen="black")
    # fig.text(text=df.iloc[:,1].to_list(),x=df.iloc[:,3],y=df.iloc[:,2])
    fig.savefig("greenland.jpg")