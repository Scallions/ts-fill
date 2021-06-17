'''
Author       : Scallions
Date         : 2020-11-15 16:03:18
LastEditors  : Scallions
LastEditTime : 2021-03-03 15:10:44
FilePath     : /gps-ts/scripts/plot_site_greenland.py
Description  : 
'''
import pygmt 
import pandas as pd


# sites = ['BLAS.NA.tenv3', 'VFDG.NA.tenv3', 'JWLF.NA.tenv3', 'TIMM.NA.tenv3', 'RINK.NA.tenv3', 'HEL2.NA.tenv3', 'BARO.NA.tenv3', 'KMJP.NA.tenv3', 'TREO.NA.tenv3', 'YMER.NA.tenv3', 'SCBY.NA.tenv3', 'KBUG.NA.tenv3', 'MSVG.NA.tenv3', 'KUAQ.NA.tenv3', 'KMOR.NA.tenv3', 'DKSG.NA.tenv3', 'ATQK.NA.tenv3', 'LBIB.NA.tenv3', 'SRMP.NA.tenv3', 'NRSK.NA.tenv3']
sites = ['BLAS.NA.tenv3', 'DGJG.NA.tenv3', 'DKSG.NA.tenv3', 'HJOR.NA.tenv3', 'HMBG.NA.tenv3', 'HRDG.NA.tenv3', 'JGBL.NA.tenv3', 'JWLF.NA.tenv3', 'KMJP.NA.tenv3', 'KMOR.NA.tenv3', 'KUAQ.NA.tenv3', 'KULL.NA.tenv3', 'LBIB.NA.tenv3', 'LEFN.NA.tenv3', 'MARG.NA.tenv3', 'MSVG.NA.tenv3', 'NRSK.NA.tenv3', 'QAAR.NA.tenv3', 'UTMG.NA.tenv3', 'YMER.NA.tenv3']
sites = [site.split('.')[0] for site in sites]

df = pd.read_csv("../data/greenland/pos.csv")

# print(sites)
# print(df.Name)

df = df[df.Name.isin(sites)]

print(df)
print(df.shape)

fig = pygmt.Figure()

# fig.basemap(region=[-109,-45,-78,-59], projection="S-77/-90/5i", frame=True)
fig.basemap(region="-75/-10/58/86", projection="L-40/30/35/25/6i", frame=True)
fig.coast(shorelines=True, water="lightblue", resolution="l")
fig.plot(x=df.iloc[:,3],y=df.iloc[:,2], style="c0.3c", color="red", pen="black")
df.loc[31,'Longitude'] -= 3
df.loc[31,'Latitude'] -= 0.5
# df.loc[31,'Longitude'] -= 3
df.loc[41,'Latitude'] -= 0.5
fig.text(text=df.iloc[:,1].to_list(),x=df.iloc[:,3]+1,y=df.iloc[:,2]+0.1, justify="ML", font="16p")
fig.savefig("greenland.jpg")