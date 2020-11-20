'''
Author       : Scallions
Date         : 2020-11-15 16:03:18
LastEditors  : Scallions
LastEditTime : 2020-11-15 16:17:40
FilePath     : /gps-ts/scripts/plot_site_greenland.py
Description  : 
'''
import pygmt 
import pandas as pd


sites = ['BLAS.NA.tenv3', 'VFDG.NA.tenv3', 'JWLF.NA.tenv3', 'TIMM.NA.tenv3', 'RINK.NA.tenv3', 'HEL2.NA.tenv3', 'BARO.NA.tenv3', 'KMJP.NA.tenv3', 'TREO.NA.tenv3', 'YMER.NA.tenv3', 'SCBY.NA.tenv3', 'KBUG.NA.tenv3', 'MSVG.NA.tenv3', 'KUAQ.NA.tenv3', 'KMOR.NA.tenv3', 'DKSG.NA.tenv3', 'ATQK.NA.tenv3', 'LBIB.NA.tenv3', 'SRMP.NA.tenv3', 'NRSK.NA.tenv3']
sites = [site.split('.')[0] for site in sites]

df = pd.read_csv("../data/greenland/pos.csv")

# print(sites)
# print(df.Name)

df = df[df.Name.isin(sites)]

print(df.head())

fig = pygmt.Figure()

# fig.basemap(region=[-109,-45,-78,-59], projection="S-77/-90/5i", frame=True)
fig.basemap(region="-75/-10/58/86", projection="L-40/30/35/25/6i", frame=True)
fig.coast(shorelines=True, water="lightblue")
fig.plot(x=df.iloc[:,3],y=df.iloc[:,2], style="c0.1c", color="red", pen="black")
# fig.text(text=df.iloc[:,1].to_list(),x=df.iloc[:,3],y=df.iloc[:,2])
fig.savefig("greenland.jpg")