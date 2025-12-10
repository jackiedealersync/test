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
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# Set pandas display options to show all columns without truncation
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', None)  # Auto-detect terminal width
pd.set_option('display.max_colwidth', None)  # Show full content of each column
pd.set_option('display.max_rows', None)  # Show all rows (optional, can be set to a number if needed)


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

VIX_zscore = np.abs( (VIX['Close'] - VIX_avg) / VIX_std )
VIX_zscore = pd.DataFrame(VIX_zscore).rename(columns={'^VIX': 'VIX_zscore'})
VIX_zscore['VIX_zscore'] = VIX_zscore['VIX_zscore'].rolling(window=30).mean()
print(VIX_zscore.info())
print(VIX_zscore.head(10))

DJI['DJI_SMA6m'] = DJI['Close'].rolling(window=125).mean() ## roughly based on no. of trading days in 6 months
DJI['DJI_SMA36m'] = DJI['Close'].rolling(window=750).mean() ## roughly based on no. of trading days in 36 months
DJI['DJI_SMA_idx'] = DJI['DJI_SMA6m'] / DJI['DJI_SMA36m']
TNX['TNX_SMA30d'] = TNX['Close'].rolling(window=30).mean() 
TNX['TNX_SMA360d'] = TNX['Close'].rolling(window=360).mean() 
TNX['TNX_diff'] = TNX['TNX_SMA30d'] - TNX['TNX_SMA360d']

print( type(VIX_zscore))

# Create TEMP dataframe with all raw data (handling MultiIndex columns from yfinance)
TEMP = VIX_zscore.join(DJI['DJI_SMA_idx']).join(TNX['TNX_diff']).join(TNX['TNX_SMA30d']).join(DJI['Close']).join(TNX['Close'])
csv_filename = 'ALL_DATA.csv'
TEMP.to_csv(csv_filename)
print(f"✓ Raw data exported to: {csv_filename}")


ALL = VIX_zscore.join(DJI['DJI_SMA_idx']).join(TNX['TNX_diff']).join(TNX['TNX_SMA30d'])
ALL = ALL.dropna()

"""
GMM Market Regime Classification
Using 4 features to classify two market regimes
"""
print("\n" + "="*60)
print("GMM Market Regime Classification")
print("="*60)

# Extract the 4 features for GMM
features = ALL[['VIX_zscore', 'DJI_SMA_idx', 'TNX_diff', 'TNX_SMA30d']].copy()
print(f"\nData shape: {features.shape}")
print(f"Features: {list(features.columns)}")
print(f"\nFeatures statistics:")
print(features.describe())

# Standardize features for better GMM performance
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
features_scaled_df = pd.DataFrame(features_scaled, 
                                   index=features.index, 
                                   columns=features.columns)

# Fit GMM with 2 components (two market regimes)
print("\nFitting GMM with 2 components...")
gmm = GaussianMixture(n_components=2, 
                      covariance_type='full', 
                      random_state=42,
                      max_iter=200,
                      n_init=10)
gmm.fit(features_scaled_df)

# Predict regime labels
regime_labels = gmm.predict(features_scaled_df)
regime_probs = gmm.predict_proba(features_scaled_df)

# Add regime information to ALL dataframe
ALL['Regime'] = regime_labels
ALL['Regime_0_Prob'] = regime_probs[:, 0]
ALL['Regime_1_Prob'] = regime_probs[:, 1]

# Analyze the regimes
print("\n" + "-"*60)
print("Regime Analysis:")
print("-"*60)
print(f"\nRegime 0: {np.sum(regime_labels == 0)} days ({np.sum(regime_labels == 0)/len(regime_labels)*100:.2f}%)")
print(f"Regime 1: {np.sum(regime_labels == 1)} days ({np.sum(regime_labels == 1)/len(regime_labels)*100:.2f}%)")

# Calculate mean feature values for each regime
print("\nMean feature values by regime:")
regime_stats = features.groupby(regime_labels).mean()
print(regime_stats)

# Calculate GMM parameters
print("\n" + "-"*60)
print("GMM Model Parameters:")
print("-"*60)
print(f"Converged: {gmm.converged_}")
print(f"Number of iterations: {gmm.n_iter_}")
print(f"Log-likelihood: {gmm.score(features_scaled_df):.4f}")
print(f"\nComponent weights:")
for i, weight in enumerate(gmm.weights_):
    print(f"  Regime {i}: {weight:.4f}")

print(f"\nComponent means (standardized):")
for i, mean in enumerate(gmm.means_):
    print(f"  Regime {i}: {mean}")

# Visualize regime classification over time
print("\nGenerating visualization...")
fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)

# Plot 1: VIX z-score
axes[0].plot(ALL.index, ALL['VIX_zscore'], label='VIX z-score', linewidth=0.8, alpha=0.7)
axes[0].axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
axes[0].set_ylabel('VIX z-score')
axes[0].set_title('Market Regime Classification using GMM', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# Plot 2: DJI SMA Index
axes[1].plot(ALL.index, ALL['DJI_SMA_idx'], label='DJI SMA Index', linewidth=0.8, alpha=0.7, color='green')
axes[1].axhline(y=1, color='gray', linestyle='--', linewidth=0.5)
axes[1].set_ylabel('DJI SMA Index')
axes[1].grid(True, alpha=0.3)
axes[1].legend()

# Plot 3: TNX Difference
axes[2].plot(ALL.index, ALL['TNX_diff'], label='TNX Diff', linewidth=0.8, alpha=0.7, color='orange')
axes[2].axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
axes[2].set_ylabel('TNX Difference')
axes[2].grid(True, alpha=0.3)
axes[2].legend()

# Plot 4: TNX SMA30d
axes[3].plot(ALL.index, ALL['TNX_SMA30d'], label='TNX SMA30d', linewidth=0.8, alpha=0.7, color='purple')
axes[3].set_ylabel('TNX SMA30d')
axes[3].grid(True, alpha=0.3)
axes[3].legend()

# Plot 5: Regime classification with color coding
colors_regime = ['red' if r == 0 else 'blue' for r in ALL['Regime']]
axes[4].scatter(ALL.index, ALL['Regime'], c=colors_regime, s=10, alpha=0.6, label='Regime')
axes[4].set_ylabel('Regime\n(0=Red, 1=Blue)')
axes[4].set_xlabel('Date')
axes[4].set_ylim(-0.5, 1.5)
axes[4].set_yticks([0, 1])
axes[4].grid(True, alpha=0.3)
axes[4].legend()

# Add regime shading to all plots
for ax in axes[:-1]:
    for i, date in enumerate(ALL.index):
        if i < len(ALL.index) - 1:
            if ALL.loc[date, 'Regime'] == 0:
                ax.axvspan(date, ALL.index[i+1] if i+1 < len(ALL.index) else date, 
                          alpha=0.1, color='red', zorder=0)
            else:
                ax.axvspan(date, ALL.index[i+1] if i+1 < len(ALL.index) else date, 
                          alpha=0.1, color='blue', zorder=0)

plt.tight_layout()
plt.savefig('market_regime_gmm.png', dpi=300, bbox_inches='tight')
print("Visualization saved as 'market_regime_gmm.png'")

# Display sample of results
print("\n" + "-"*60)
print("Sample Results (last 10 days):")
print("-"*60)
print(ALL[['VIX_zscore', 'DJI_SMA_idx', 'TNX_diff', 'TNX_SMA30d', 
           'Regime', 'Regime_0_Prob', 'Regime_1_Prob']].tail(10))

print("\n" + "="*60)
print("GMM Classification Complete!")
print("="*60)

"""
Export Final Model Data
"""
print("\n" + "-"*60)
print("Exporting Final Model Data...")
print("-"*60)

# Export to CSV
csv_filename = 'market_regime_data.csv'
ALL.to_csv(csv_filename)
print(f"✓ CSV file saved: {csv_filename}")
print(f"  Rows: {len(ALL)}, Columns: {len(ALL.columns)}")

# Export to Excel (requires openpyxl package)
try:
    excel_filename = 'market_regime_data.xlsx'
    ALL.to_excel(excel_filename, index=True, engine='openpyxl')
    print(f"✓ Excel file saved: {excel_filename}")
except ImportError:
    print("⚠ Excel export skipped: 'openpyxl' package not installed")
    print("  Install with: pip install openpyxl")
except Exception as e:
    print(f"⚠ Excel export failed: {e}")
    print("  CSV file is still available")

# Display column information
print(f"\nExported columns:")
for i, col in enumerate(ALL.columns, 1):
    print(f"  {i}. {col}")

print(f"\nData export complete!")
print("="*60)





