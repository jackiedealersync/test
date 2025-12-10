#!/usr/bin/env python3
"""
Import Packages
"""

import yfinance as yf
import pandas as pd
import numpy as np
import mplfinance as mpf
import plotly.express as px
import matplotlib
# Try to set a backend that works for displaying plots
try:
    matplotlib.use('TkAgg')
except:
    try:
        matplotlib.use('Qt5Agg')
    except:
        pass  # Use default backend
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec

# Set pandas display options to show all columns without truncation
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', None)  # Auto-detect terminal width
pd.set_option('display.max_colwidth', None)  # Show full content of each column
pd.set_option('display.max_rows', None)  # Show all rows (optional, can be set to a number if needed)

"""
1.1
"""

DJI = yf.download("^DJI", start="2015-11-01", end="2025-10-31")
print(DJI.head(10))

DJI_MthEnd_Close = DJI['Close'].resample('M').last()
print(type(DJI_MthEnd_Close))
DJI_MthEnd_Close = DJI_MthEnd_Close.rename(columns={'^DJI': 'DJI_MthEnd_Close'})
print(DJI_MthEnd_Close.head(10))

DJI_MthEnd_PctgChg = DJI_MthEnd_Close['DJI_MthEnd_Close'].pct_change() * 100
print(type(DJI_MthEnd_PctgChg))
DJI_MthEnd_PctgChg = pd.DataFrame(DJI_MthEnd_PctgChg).rename(columns={'DJI_MthEnd_Close': 'DJI_MthEnd_PctgChg'})
print(DJI_MthEnd_PctgChg.head(10))

DJI_monthly_stats = DJI_MthEnd_Close.join(DJI_MthEnd_PctgChg)
print(DJI_monthly_stats.shape)

r = (DJI_monthly_stats['DJI_MthEnd_Close'].iloc[-1] / DJI_monthly_stats['DJI_MthEnd_Close'].iloc[0] ) ** (1/120)-1
growth_factor = (1+r)**12-1
print(f"Growth factor : {growth_factor:.2%}")

log_return = np.log(DJI_monthly_stats['DJI_MthEnd_Close'] / DJI_monthly_stats['DJI_MthEnd_Close'].shift(1)).dropna()
annualized_log_return = log_return.mean() * 12
print(f"Annaulized Log return : {annualized_log_return:.2%}")

df_log_return = pd.DataFrame(log_return*100).rename(columns={'DJI_MthEnd_Close': 'Log_Return_Pctg'})
DJI_monthly_stats = DJI_monthly_stats.join(df_log_return)
print(DJI_monthly_stats.head(6))
print(DJI_monthly_stats.tail(6))



"""
1.2
"""
VIX = yf.download("^VIX", start="2005-01-01", end="2025-10-31")
#DJI = yf.download("^DJI", start="2005-01-01", end="2025-10-31")
RUT = yf.download("^DJI", start="2005-01-01", end="2025-10-31")
#TNX = yf.download("^TNX", start="2005-01-01", end="2025-10-31")

print(RUT.head(10))
RUT.columns = RUT.columns.get_level_values(0)
print(RUT.info())
print(RUT.head(10))


RUT['SMA_6m'] = RUT['Close'].rolling(window=125).mean() ## roughly based on no. of trading days in 6 months
RUT['SMA_36m'] = RUT['Close'].rolling(window=750).mean() ## roughly based on no. of trading days in 36 months
print(RUT.info())

# Prepare VIX data - align with RUT dates
#VIX.columns = VIX.columns.get_level_values(0)
#VIX_Close = VIX['Close'].reindex(RUT.index)

# SMA plot with VIX on secondary axis
SMA_plt = [
        mpf.make_addplot(VIX['Close'], color='green', linestyle='-', label='VIX', secondary_y=True),
        mpf.make_addplot(RUT['SMA_6m'], color='red', linestyle='--', label='SMA 6m (125-day)'),
        mpf.make_addplot(RUT['SMA_36m'], color='blue', linestyle='--', label='SMA 36m (750-day)')
        ]

# Stock prices
fig, axes = mpf.plot(RUT, 
        type='candle', 
        style='charles', 
        volume=True, 
        title='Russell 2000 Index (^RUT) with 6-month (125-day) and 36-month (750-day) SMA', 
        ylabel='Price',
        addplot=SMA_plt,
        returnfig=True
        )

axes[0].legend(loc='upper left')
mpf.show()


"""
1.3
"""
VIX = yf.download("^VIX", start="2005-01-01", end="2025-10-31")
DJI = yf.download("^DJI", start="2005-01-01", end="2025-10-31")
RUT = yf.download("^RUT", start="2005-01-01", end="2025-10-31")
TNX = yf.download("^TNX", start="2005-01-01", end="2025-10-31")

ALL = VIX.join(DJI).join(RUT).join(TNX)

ALL_dropna = ALL.dropna()

ALL_Close = ALL.loc[:, ALL.columns.get_level_values(0) == 'Close']
ALL_Close.columns = ALL_Close.columns.get_level_values(1)
print(ALL_Close.info())
print(ALL_Close.dropna().info())
print(ALL_Close.dropna().head(10))

ALL_Close = ALL_Close.dropna()
print(ALL_Close.dropna().corr())

ALL_Close_delta = ALL_Close - ALL_Close.shift(1)
print(ALL_Close_delta.head(10))

ALL_Close_deltaPctg = (ALL_Close / ALL_Close.shift(1) - 1) *100
ALL_Close_deltaPctg = ALL_Close_deltaPctg.drop('^TNX', axis=1)
print(ALL_Close_deltaPctg.head(10))

ALL_stat = ALL_Close_delta.add_suffix('_delta').join(ALL_Close_deltaPctg.add_suffix('_deltaPctg'))
ALL_stat['^VIX_lag1'] = ALL_stat['^VIX_delta'].shift(1) 
ALL_stat['^VIX_lag10'] = ALL_stat['^VIX_delta'].shift(10) 
print(ALL_stat.head(10))
print(ALL_stat.dropna().corr())



"""
1.4
"""
VIX = yf.download("^VIX", start="2005-01-01", end="2025-10-31")
DJI = yf.download("^DJI", start="2005-01-01", end="2025-10-31")
RUT = yf.download("^RUT", start="2005-01-01", end="2025-10-31")
TNX = yf.download("^TNX", start="2005-01-01", end="2025-10-31")

VIX_std = VIX['Close'].std()
VIX_avg = VIX['Close'].mean()
print( VIX.head(10) )
print(f"VIX_std : {VIX_std}")
print(f"VIX_avg : {VIX_avg}")

VIX_zscore = (VIX['Close'] - VIX_avg) / VIX_std
VIX_zscore = pd.DataFrame(VIX_zscore).rename(columns={'^VIX': 'VIX_zscore'})
print(VIX_zscore.info())
print(VIX_zscore.head(10))

DJI['DJI_SMA6m'] = DJI['Close'].rolling(window=125).mean() ## roughly based on no. of trading days in 6 months
DJI['DJI_SMA36m'] = DJI['Close'].rolling(window=750).mean() ## roughly based on no. of trading days in 36 months
DJI['DJI_SMA_idx'] = DJI['DJI_SMA6m'] / DJI['DJI_SMA36m']
TNX['TNX_SMA30d'] = TNX['Close'].rolling(window=30).mean() 
TNX['TNX_SMA360d'] = TNX['Close'].rolling(window=360).mean() 
TNX['TNX_diff'] = TNX['TNX_SMA30d'] - TNX['TNX_SMA360d']

print( type(VIX_zscore))

ALL = VIX_zscore.join(DJI['DJI_SMA_idx']).join(TNX['TNX_diff']).join(TNX['TNX_SMA30d'])
print(ALL.dropna().head(10))




from hmmlearn import hmm
# config states
n_states = 2

# config features
features = ['DJI_SMA_idx', 'TNX_diff', 'VIX_zscore','TNX_SMA30d']
data = ALL[features].dropna()

# split
split_index = int(len(data) * 0.65)
train_data = data[features].iloc[:split_index].values
test_data = data[features].iloc[split_index:].values

# train HMM
print(f"Training HMM model with {len(data)} data points...")
model = hmm.GaussianHMM(n_components=n_states, 
                    covariance_type="full", 
                    n_iter=1000, 
                    random_state=42)
model.fit(data)
print("HMM model training completed.")

# predict states
train_states = model.predict(train_data)
test_states = model.predict(test_data)
all_states = np.concatenate([train_states, test_states])
data['state'] = all_states

# predict certainties
probs = model.predict_proba(data[features].values)
state_certainty = pd.Series(probs.max(axis=1), index=data.index)















"""
TEST
"""

# Plot line chart with TNX as primary y-axis and VIX as secondary y-axis
fig, ax1 = plt.subplots(figsize=(12, 6))

# Primary y-axis: TNX
color1 = 'tab:blue'
ax1.set_xlabel('Date')
ax1.set_ylabel('TNX (10-Year Treasury Yield)', color=color1)
line1 = ax1.plot(ALL_Close.index, ALL_Close['^TNX'], color=color1, label='TNX', linewidth=2)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.grid(True, alpha=0.3)

# Secondary y-axis: VIX
ax2 = ax1.twinx()
color2 = 'tab:red'
ax2.set_ylabel('VIX (Volatility Index)', color=color2)
line2 = ax2.plot(ALL_Close.index, ALL_Close['^VIX'], color=color2, label='VIX', linewidth=2)
ax2.tick_params(axis='y', labelcolor=color2)

# Add title and legend
plt.title('TNX vs VIX Over Time', fontsize=14, fontweight='bold')
fig.tight_layout()
plt.show()