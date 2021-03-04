
import matplotlib.pyplot as plt 
import pandas as pd 



df = pd.read_csv("./results/2d.csv", sep=r"\s+")
df["Other(%)"] = 100 - df["前三和(%)"]
df = df.iloc[:,[0,1,2,3,-1]]

# print(df.head())
plt.rcParams['font.sans-serif'] = ['SimHei']

for file in ["2d", "7d", "30d", "180d"]:
    df = pd.read_csv(f"./results/{file}.csv", sep=r"\s+")
    df["Other(%)"] = 100 - df["前三和(%)"]
    df = df.iloc[:,[0,1,2,3,-1]]
    fig, subs = plt.subplots(2, 3, sharex=True,figsize=(9,6))
    for i in range(6):
        subs[i//3][i%3].set_title(df.iloc[i,0])
        subs[i//3][i%3].pie(df.iloc[i,1:],
            labels=df.columns[1:],
            # autopct='%1.1f%%',
            # explode = (0.1,0.1,0.1,0.1),
            )
    fig.legend(df.columns[1:])
    fig.savefig(f"./results/{file}.png")

# plt.show()