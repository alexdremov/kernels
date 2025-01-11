import sys

import itertools
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

file_path = sys.argv[1]
df = pd.read_csv(file_path, index_col=0)

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
colors = itertools.cycle(colors)

plt.figure(figsize=(11, 4))

for col, color in zip(df.columns, colors):
    df_plot = df[df[col] > 0]
    sns.lineplot(x=df_plot.index, y=df_plot[col], label=col, color=color)
    plt.hlines(0, df_plot.index.max(), df.index.max(), color=color, linestyles=":")

plt.ylabel("ms")
plt.legend()
plt.savefig("plot.svg")
plt.savefig("plot.png")
