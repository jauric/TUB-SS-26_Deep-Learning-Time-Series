from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Project root = two levels up from this file (src/tub_ss_26_deep_learning_time_series/etth-1.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ETTh1_link = PROJECT_ROOT / "ETTh1_data" / "ETTh1.csv"

data_ETTh1 = pd.read_csv(ETTh1_link)
data_ETTh1.head()

# Parse the date column to make it more readable
dates = pd.to_datetime(data_ETTh1["date"])

data_ETTh1 = data_ETTh1.set_index("date")
print("Dataset Shape", data_ETTh1.shape)
print("Dataset Description", data_ETTh1.describe())

# check for missing values per column
print(data_ETTh1.isna().sum())
print('-' * 50)

# Visualize all 7 features
fig, axes = plt.subplots(7, 1, figsize=(14, 16), sharex=True)
for ax, col in zip(axes, data_ETTh1.columns):
    ax.plot(dates, data_ETTh1[col], linewidth=0.6)
    ax.set_ylabel(col)
    ax.grid(alpha=0.3)

axes[-1].set_xlabel("date")
axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
for label in axes[-1].get_xticklabels():
    label.set_rotation(45)
    label.set_ha("right")

fig.suptitle("ETTh1 — All variables", y=1.01)
plt.tight_layout()
plt.show()