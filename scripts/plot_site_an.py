'''
Author       : Scallions
Date         : 2020-11-14 10:58:03
LastEditors  : Scallions
LastEditTime : 2020-11-14 15:29:06
FilePath     : /gps-ts/scripts/plot_site_an.py
Description  : 
'''
import pygmt 
import pandas as pd


df = pd.read_csv("../data/an/pos.csv")
print(df.head())

fig = pygmt.Figure()

# fig.basemap(region=[-109,-45,-78,-59], projection="S-77/-90/5i", frame=True)
fig.basemap(region=[-80,-45,-71,-59], projection="S-62/-90/5i", frame=True)
fig.coast(shorelines=True, water="lightblue")
fig.plot(x=df.iloc[:,3],y=df.iloc[:,2], style="c0.1c", color="red", pen="black")
# fig.text(text=df.iloc[:,1].to_list(),x=df.iloc[:,3],y=df.iloc[:,2])
fig.savefig("test.jpg")