'''
Author       : Scallions
Date         : 2020-11-14 10:15:45
LastEditors  : Scallions
LastEditTime : 2020-11-17 21:00:10
FilePath     : /gps-ts/scripts/site_pos.py
Description  : 
'''

import requests
from requests.api import get
# from bs4 import Beautifulsoup
from lxml import etree
import os
import pandas as pd
import time

def sites_pos():
    """load data
    
    Args:
        lengths (int, optional): nums of mul ts. Defaults to 3.
    
    Returns:
        [mts]: mts s
    """
    dir_path = './data/greenland/'
    tss = []
    files = os.listdir(dir_path)
    sites = []
    df = pd.DataFrame(columns=["Name","Latitude","Longitude","Height(m)"])
    for file_ in files:
    # for file_ in ['ATQK.NA.tenv3', 'BLAS.NA.tenv3', 'DKSG.NA.tenv3', 'GROK.NA.tenv3', 'HEL2.NA.tenv3', 'HJOR.NA.tenv3', 'HRDG.NA.tenv3', 'KBUG.NA.tenv3', 'KMJP.NA.tenv3', 'KMOR.NA.tenv3', 'KSNB.NA.tenv3', 'KUAQ.NA.tenv3', 'KULL.NA.tenv3', 'NRSK.NA.tenv3', 'QAAR.NA.tenv3', 'RINK.NA.tenv3', 'SCBY.NA.tenv3', 'TREO.NA.tenv3', 'UTMG.NA.tenv3', 'YMER.NA.tenv3']:
        if '.tenv3' in file_:
            site = file_.split('.')[0]
            pos = get_site_pos(site)
            while len(pos) == 0:
                time.sleep(1)
                pos = get_site_pos(site)
            s = pd.Series([site] + pos, index=df.columns)
            df = df.append(s, ignore_index=True)
    df.to_csv(dir_path+"pos.csv")


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
    sites_pos()
    # get_site_pos("CAPF")