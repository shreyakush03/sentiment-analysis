import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
fear_greed_df = pd.read_csv("dataset/fear_greed_index.csv")
historical_data_df = pd.read_csv("dataset/historical_data.csv")

# Lowercase columns
fear_greed_df.columns = fear_greed_df.columns.str.lower()
historical_data_df.columns = historical_data_df.columns.str.lower()

# Parse dates
fear_greed_df["date"] = pd.to_datetime(fear_greed_df["date"]).dt.date
historical_data_df["timestamp ist"] = pd.to_datetime(historical_data_df["timestamp ist"], format="%d-%m-%Y %H:%M")
historical_data_df["Date"] = historical_data_df["timestamp ist"].dt.date

# Daily aggregation
daily_trades = historical_data_df.groupby('Date').agg({
    'size usd': ['sum', 'mean', 'count'],
    'closed pnl': ['sum', 'mean'],
    'side': lambda x: (x == 'BUY').sum()
}).reset_index()

daily_trades.columns = ['Date', 'total_volume_usd', 'avg_trade_size', 'num_trades',
                        'total_pnl', 'avg_pnl', 'buy_count']
daily_trades['sell_count'] = daily_trades['num_trades'] - daily_trades['buy_count']
daily_trades['buy_ratio'] = daily_trades['buy_count'] / daily_trades['num_trades']

# Merge datasets
merged_df = pd.merge(daily_trades, fear_greed_df[['date', 'value', 'classification']],
                     left_on='Date', right_on='date', how='inner')
merged_df = merged_df.drop('date', axis=1)
merged_df.columns = ['Date', 'total_volume_usd', 'avg_trade_size', 'num_trades',
                     'total_pnl', 'avg_pnl', 'buy_count', 'sell_count', 'buy_ratio',
                     'fear_greed_value', 'sentiment_class']

print("\n" + "="*70)
print("ADVANCED PATTERN DISCOVERY: TRADER PERFORMANCE vs MARKET SENTIMENT")
print("="*70)

# 1. LAGGED CORRELATION ANALYSIS
print("\n1. LAGGED CORRELATION ANALYSIS")
print("-" * 70)
print("Does today's sentiment predict tomorrow's trader performance?")

lagged_correlations = {}
for lag in range(-5, 6):
    if lag < 0:
        lagged_fg = merged_df['fear_greed_value'].shift(abs(lag))
        desc = f"FG {abs(lag)} days AHEAD predicts"
    elif lag > 0:
        lagged_fg = merged_df['fear_greed_value'].shift(-lag)
        desc = f"FG {lag} days BEHIND predicts"
    else:
        lagged_fg = merged_df['fear_greed_value']
        desc = "FG SAME DAY predicts"

    corr = lagged_fg.corr(merged_df['total_pnl'])
    lagged_correlations[lag] = corr
    print(f"  {desc:.<35} {corr:>7.4f}")

best_lag = max(lagged_correlations, key=lambda x: abs(lagged_correlations[x]))
print(f"\n  >> STRONGEST SIGNAL: Lag {best_lag} (r={lagged_correlations[best_lag]:.4f})")

# 2. ROLLING CORRELATION
print("\n2. ROLLING CORRELATION ANALYSIS (30-day windows)")
print("-" * 70)
merged_df['rolling_corr'] = merged_df['fear_greed_value'].rolling(30).corr(merged_df['total_pnl'])

rolling_stats = {
    'Mean': merged_df['rolling_corr'].mean(),
    'Std': merged_df['rolling_corr'].std(),
    'Min': merged_df['rolling_corr'].min(),
    'Max': merged_df['rolling_corr'].max()
}

for stat, value in rolling_stats.items():
    print(f"  {stat:.<30} {value:>7.4f}")

print(f"\n  >> Correlation varies from {rolling_stats['Min']:.3f} to {rolling_stats['Max']:.3f}")

# 3. MARKET REGIME ANALYSIS
print("\n3. MARKET REGIME ANALYSIS")
print("-" * 70)

merged_df['fg_regime'] = pd.cut(merged_df['fear_greed_value'],
                                 bins=[0, 25, 45, 55, 75, 100],
                                 labels=['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed'])

print("\nPerformance by Market Regime:")
regime_stats = merged_df.groupby('fg_regime', observed=True).agg({
    'total_pnl': ['mean', 'std', 'count'],
    'total_volume_usd': ['mean'],
    'num_trades': ['mean'],
    'buy_ratio': ['mean']
}).round(2)
print(regime_stats)

# Win rate by regime
merged_df['is_profitable'] = merged_df['total_pnl'] > 0
win_rate_by_regime = merged_df.groupby('fg_regime', observed=True)['is_profitable'].apply(lambda x: (x.sum() / len(x) * 100))

print("\nWin Rate by Regime:")
for regime, rate in win_rate_by_regime.items():
    print(f"  {regime:.<30} {rate:>6.1f}%")

# 4. VOLUME ANOMALY
print("\n4. ANOMALY: HIGH VOLUME DURING FEAR (Contrarian Pattern)")
print("-" * 70)

fear_data = merged_df[merged_df['fear_greed_value'] < 45]
greed_data = merged_df[merged_df['fear_greed_value'] > 55]

vol_ratio = fear_data['total_volume_usd'].mean() / greed_data['total_volume_usd'].mean()
print(f"  Fear Period Average Volume:  ${fear_data['total_volume_usd'].mean():,.0f}")
print(f"  Greed Period Average Volume: ${greed_data['total_volume_usd'].mean():,.0f}")
print(f"  Ratio (Fear / Greed):        {vol_ratio:.2f}x")
print(f"\n  >> KEY INSIGHT: Traders trade {vol_ratio:.1f}x MORE during fear!")
print(f"     Possible: Panic trading, liquidations, or contrarian buying")

# 5. PNL DISTRIBUTION
print("\n5. PnL DISTRIBUTION BY REGIME")
print("-" * 70)

for regime in ['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed']:
    regime_data = merged_df[merged_df['fg_regime'] == regime]
    if len(regime_data) > 0:
        mean_pnl = regime_data['total_pnl'].mean()
        wins = (regime_data['total_pnl'] > 0).sum()
        losses = (regime_data['total_pnl'] < 0).sum()
        total_gains = regime_data[regime_data['total_pnl'] > 0]['total_pnl'].sum()
        total_losses = abs(regime_data[regime_data['total_pnl'] < 0]['total_pnl'].sum())
        profit_factor = total_gains / total_losses if total_losses > 0 else 0

        print(f"\n  {regime}:")
        print(f"    Avg Daily PnL:       ${mean_pnl:>10,.0f}")
        print(f"    Win Days / Loss Days: {wins:>10} / {losses}")
        print(f"    Total Gains / Losses: ${total_gains:>10,.0f} / ${total_losses:>10,.0f}")
        print(f"    Profit Factor:       {profit_factor:>10.2f}x")

# 6. CORRELATION MATRIX
print("\n6. CORRELATION MATRIX: KEY VARIABLES")
print("-" * 70)

correlation_matrix = merged_df[['fear_greed_value', 'total_volume_usd', 'num_trades',
                                 'total_pnl', 'buy_ratio', 'avg_trade_size']].corr()
print(correlation_matrix.round(3).to_string())

# 7. ADVANCED: Volume volatility vs PnL
print("\n7. VOLUME-PNL RELATIONSHIP")
print("-" * 70)

merged_df['volume_volatility'] = merged_df['total_volume_usd'].rolling(7).std()
vol_pnl_corr = merged_df['volume_volatility'].corr(merged_df['total_pnl'])
print(f"  Volume Volatility <-> PnL Correlation: {vol_pnl_corr:.4f}")

high_vol_days = merged_df[merged_df['volume_volatility'] > merged_df['volume_volatility'].quantile(0.75)]
low_vol_days = merged_df[merged_df['volume_volatility'] < merged_df['volume_volatility'].quantile(0.25)]

print(f"\n  High Volatility Days (top 25%):")
print(f"    Avg PnL: ${high_vol_days['total_pnl'].mean():,.0f}")
print(f"    Win Rate: {(high_vol_days['total_pnl'] > 0).mean() * 100:.1f}%")

print(f"\n  Low Volatility Days (bottom 25%):")
print(f"    Avg PnL: ${low_vol_days['total_pnl'].mean():,.0f}")
print(f"    Win Rate: {(low_vol_days['total_pnl'] > 0).mean() * 100:.1f}%")

# 8. SENTIMENT CHANGE IMPACT
print("\n8. IMPACT OF SENTIMENT CHANGES")
print("-" * 70)

merged_df['fg_change'] = merged_df['fear_greed_value'].diff()
merged_df['fg_rising'] = merged_df['fg_change'] > 0

rising_sentiment = merged_df[merged_df['fg_rising'] == True]
falling_sentiment = merged_df[merged_df['fg_rising'] == False]

print(f"  Rising Sentiment Days: Avg PnL = ${rising_sentiment['total_pnl'].mean():,.0f}")
print(f"  Falling Sentiment Days: Avg PnL = ${falling_sentiment['total_pnl'].mean():,.0f}")
print(f"\n  >> Pattern: Traders perform better when sentiment is {'' if rising_sentiment['total_pnl'].mean() > falling_sentiment['total_pnl'].mean() else 'FALLING'}")

print("\n" + "="*70)
print("SUMMARY: HIDDEN PATTERNS FOR SMARTER TRADING STRATEGIES")
print("="*70)
print("""
1. CONTRARIAN OPPORTUNITY: Traders increase volume during fear but don't profit proportionally
   STRATEGY: Scale up QUALITY trades during fear periods, not just volume

2. REGIME SWITCHING: Sentiment doesn't predict PnL directly, but affects trading behavior
   STRATEGY: Adjust position sizing and risk management by market regime

3. VOLATILITY TRADES: High volume volatility correlates with different PnL profiles
   STRATEGY: Separate strategies for stable vs volatile trading periods

4. TIMING ANOMALY: Sentiment changes may precede or lag trader performance
   STRATEGY: Use lagged sentiment data for predictive signals
""")

# Export merged data for further analysis
merged_df.to_csv('merged_analysis.csv', index=False)
print("\nAnalysis saved to 'merged_analysis.csv'")
