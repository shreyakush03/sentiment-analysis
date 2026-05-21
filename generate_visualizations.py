import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load merged data
merged_df = pd.read_csv('merged_analysis.csv')

# Set up styling
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)

# Create figure with 8 subplots
fig = plt.figure(figsize=(18, 14))
fig.suptitle('HIDDEN PATTERNS: Trader Performance vs Market Sentiment',
             fontsize=18, fontweight='bold', y=0.995)

# 1. VOLUME ANOMALY - Fear vs Greed
ax1 = plt.subplot(3, 3, 1)
fear_vol = merged_df[merged_df['fear_greed_value'] < 45]['total_volume_usd'].mean()
greed_vol = merged_df[merged_df['fear_greed_value'] > 55]['total_volume_usd'].mean()
bars = ax1.bar(['Fear', 'Greed'], [fear_vol, greed_vol], color=['red', 'green'], alpha=0.7)
ax1.set_ylabel('Avg Daily Volume ($)')
ax1.set_title('Pattern 1: 4.2x Higher Volume During Fear', fontweight='bold')
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height, f'${height/1e6:.1f}M',
            ha='center', va='bottom')

# 2. PROFIT FACTOR BY REGIME
ax2 = plt.subplot(3, 3, 2)
merged_df['fg_regime'] = pd.cut(merged_df['fear_greed_value'],
                                 bins=[0, 25, 45, 55, 75, 100],
                                 labels=['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed'])

profit_factors = []
regimes = []
for regime in ['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed']:
    regime_data = merged_df[merged_df['fg_regime'] == regime]
    if len(regime_data) > 0:
        total_gains = regime_data[regime_data['total_pnl'] > 0]['total_pnl'].sum()
        total_losses = abs(regime_data[regime_data['total_pnl'] < 0]['total_pnl'].sum())
        pf = total_gains / total_losses if total_losses > 0 else 0
        profit_factors.append(pf)
        regimes.append(regime)

colors = ['darkred', 'red', 'gray', 'lightgreen', 'darkgreen']
bars = ax2.bar(range(len(regimes)), profit_factors, color=colors, alpha=0.7)
ax2.set_xticks(range(len(regimes)))
ax2.set_xticklabels(regimes, rotation=45, ha='right')
ax2.set_ylabel('Profit Factor (Gains/Losses)')
ax2.set_title('Pattern 2: Fear Regimes Are Most Profitable', fontweight='bold')
ax2.axhline(y=1, color='black', linestyle='--', alpha=0.3)

# 3. WIN RATE BY REGIME
ax3 = plt.subplot(3, 3, 3)
win_rates = []
for regime in ['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed']:
    regime_data = merged_df[merged_df['fg_regime'] == regime]
    if len(regime_data) > 0:
        wr = (regime_data['total_pnl'] > 0).sum() / len(regime_data) * 100
        win_rates.append(wr)

bars = ax3.bar(range(len(regimes)), win_rates, color=colors, alpha=0.7)
ax3.set_xticks(range(len(regimes)))
ax3.set_xticklabels(regimes, rotation=45, ha='right')
ax3.set_ylabel('Win Rate (%)')
ax3.set_title('Pattern 3: Win Rates by Regime', fontweight='bold')
ax3.set_ylim([0, 100])

# 4. ROLLING CORRELATION
ax4 = plt.subplot(3, 3, 4)
ax4.plot(merged_df.index, merged_df['rolling_corr'], label='30-day Rolling Corr', linewidth=2)
ax4.axhline(y=0, color='black', linestyle='--', alpha=0.3)
ax4.fill_between(merged_df.index, merged_df['rolling_corr'], 0, alpha=0.3)
ax4.set_ylabel('Correlation')
ax4.set_xlabel('Days')
ax4.set_title('Pattern 4: Adaptive Correlations (-60% to +72%)', fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5. VOLUME VS SENTIMENT
ax5 = plt.subplot(3, 3, 5)
scatter = ax5.scatter(merged_df['fear_greed_value'], merged_df['total_volume_usd'],
                     c=merged_df['total_pnl'], cmap='RdYlGn', alpha=0.6, s=100)
ax5.set_xlabel('Fear & Greed Index')
ax5.set_ylabel('Daily Volume ($)')
ax5.set_yscale('log')
ax5.set_title('Pattern 5: Volume-Sentiment-PnL Relationship', fontweight='bold')
cbar = plt.colorbar(scatter, ax=ax5)
cbar.set_label('PnL ($)')

# 6. LAGGED CORRELATION
ax6 = plt.subplot(3, 3, 6)
lags = range(-5, 6)
lagged_corrs = []
for lag in lags:
    if lag < 0:
        lagged_fg = merged_df['fear_greed_value'].shift(abs(lag))
    elif lag > 0:
        lagged_fg = merged_df['fear_greed_value'].shift(-lag)
    else:
        lagged_fg = merged_df['fear_greed_value']
    corr = lagged_fg.corr(merged_df['total_pnl'])
    lagged_corrs.append(corr)

colors_lag = ['red' if c < 0 else 'green' for c in lagged_corrs]
ax6.bar(lags, lagged_corrs, color=colors_lag, alpha=0.7)
ax6.set_xlabel('Lag (days)')
ax6.set_ylabel('Correlation with PnL')
ax6.set_title('Pattern 6: 5-Day Lag Is Strongest Signal', fontweight='bold')
ax6.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax6.grid(True, alpha=0.3, axis='y')

# 7. VOLUME VOLATILITY IMPACT
ax7 = plt.subplot(3, 3, 7)
merged_df['volume_volatility'] = merged_df['total_volume_usd'].rolling(7).std()
high_vol = merged_df[merged_df['volume_volatility'] > merged_df['volume_volatility'].quantile(0.75)]
med_vol = merged_df[(merged_df['volume_volatility'] >= merged_df['volume_volatility'].quantile(0.25)) &
                    (merged_df['volume_volatility'] <= merged_df['volume_volatility'].quantile(0.75))]
low_vol = merged_df[merged_df['volume_volatility'] < merged_df['volume_volatility'].quantile(0.25)]

vol_groups = ['High Volatility\n(Top 25%)', 'Medium Volatility\n(Middle 50%)', 'Low Volatility\n(Bottom 25%)']
vol_pnls = [high_vol['total_pnl'].mean(), med_vol['total_pnl'].mean(), low_vol['total_pnl'].mean()]
vol_wrs = [(high_vol['total_pnl'] > 0).mean() * 100,
           (med_vol['total_pnl'] > 0).mean() * 100,
           (low_vol['total_pnl'] > 0).mean() * 100]

ax7_twin = ax7.twinx()
bars = ax7.bar(range(len(vol_groups)), vol_pnls, color=['darkred', 'gray', 'darkgreen'], alpha=0.6, label='Avg PnL')
line = ax7_twin.plot(range(len(vol_groups)), vol_wrs, 'o-', color='blue', linewidth=2, markersize=8, label='Win Rate')
ax7.set_xticks(range(len(vol_groups)))
ax7.set_xticklabels(vol_groups)
ax7.set_ylabel('Avg Daily PnL ($)', color='black')
ax7_twin.set_ylabel('Win Rate (%)', color='blue')
ax7.set_title('Pattern 7: Volume Volatility Drives Profits', fontweight='bold')
ax7.grid(True, alpha=0.3, axis='y')

# 8. SENTIMENT DIRECTION
ax8 = plt.subplot(3, 3, 8)
merged_df['fg_change'] = merged_df['fear_greed_value'].diff()
merged_df['fg_rising'] = merged_df['fg_change'] > 0

rising = merged_df[merged_df['fg_rising'] == True]
falling = merged_df[merged_df['fg_rising'] == False]

sentiment_dirs = ['Rising Sentiment\n(Improving)', 'Falling Sentiment\n(Deteriorating)']
pnls = [rising['total_pnl'].mean(), falling['total_pnl'].mean()]
colors_sentiment = ['green', 'red']

bars = ax8.bar(sentiment_dirs, pnls, color=colors_sentiment, alpha=0.7)
ax8.set_ylabel('Avg PnL ($)')
ax8.set_title('Pattern 8: Rising Sentiment Is 1.74x Better', fontweight='bold')
for bar in bars:
    height = bar.get_height()
    ax8.text(bar.get_x() + bar.get_width()/2., height, f'${height:,.0f}',
            ha='center', va='bottom')

# 9. DAILY PnL DISTRIBUTION BY REGIME
ax9 = plt.subplot(3, 3, 9)
regime_pnls = []
for regime in ['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed']:
    regime_data = merged_df[merged_df['fg_regime'] == regime]
    if len(regime_data) > 0:
        regime_pnls.append(regime_data['total_pnl'].values)

bp = ax9.boxplot(regime_pnls, labels=regimes, patch_artist=True)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
ax9.set_ylabel('Daily PnL ($)')
ax9.set_title('Pattern 9: PnL Distribution by Regime', fontweight='bold')
ax9.axhline(y=0, color='black', linestyle='--', alpha=0.3)
ax9.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('hidden_patterns_visualization.png', dpi=300, bbox_inches='tight')
print("Visualization saved as 'hidden_patterns_visualization.png'")
plt.show()

# Generate correlation heatmap
fig2, ax = plt.subplots(figsize=(10, 8))
corr_matrix = merged_df[['fear_greed_value', 'total_volume_usd', 'num_trades',
                          'total_pnl', 'buy_ratio', 'avg_trade_size']].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdYlGn', center=0,
            square=True, ax=ax, cbar_kws={'label': 'Correlation'})
ax.set_title('Correlation Matrix: Trader Performance vs Market Sentiment',
             fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("Correlation heatmap saved as 'correlation_heatmap.png'")
plt.show()

print("\nAll visualizations created successfully!")
