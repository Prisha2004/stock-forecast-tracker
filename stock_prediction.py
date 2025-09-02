# -------------------------------
# Install (run once in VS Code terminal if not done yet):
# pip3 install yfinance statsmodels scikit-learn matplotlib pandas seaborn
# -------------------------------

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")   # ✅ Save plots as PNG on Mac
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import warnings, os
warnings.filterwarnings("ignore")

# -------------------------------
# 📌 Step 1: Define Tickers
# -------------------------------
tickers = ["INFY.NS", "TCS.NS", "RELIANCE.NS", "HDFCBANK.NS"]

# Prepare results storage
all_results = []

# -------------------------------
# 📌 Step 2: Loop over tickers
# -------------------------------
for ticker in tickers:
    print(f"\n🔹 Processing {ticker} ...")

    # Download stock data
    data = yf.download(ticker, start="2018-01-01", end="2024-01-01")
    if data.empty:
        print(f"⚠️ No data found for {ticker}, skipping...")
        continue

    data = data[['Close']].dropna()

    # Train-Test Split
    train_size = int(len(data) * 0.8)
    train, test = data.iloc[:train_size], data.iloc[train_size:]

    # -------------------------------
    # ARIMA Model
    # -------------------------------
    print("   Training ARIMA model...")
    arima_model = ARIMA(train, order=(5, 1, 0))
    arima_fit = arima_model.fit()
    arima_forecast = arima_fit.forecast(steps=len(test))
    arima_rmse = np.sqrt(mean_squared_error(test, arima_forecast))
    arima_mae = mean_absolute_error(test, arima_forecast)

    # -------------------------------
    # Random Forest Model
    # -------------------------------
    print("   Training Random Forest...")
    def create_features(df, lags=5):
        df_feat = df.copy()
        for i in range(1, lags+1):
            df_feat[f'lag_{i}'] = df_feat['Close'].shift(i)
        df_feat.dropna(inplace=True)
        return df_feat

    rf_data = create_features(data, lags=5)
    X = rf_data.drop('Close', axis=1)
    y = rf_data['Close']
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)

    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))
    rf_mae = mean_absolute_error(y_test, rf_preds)

    # -------------------------------
    # Backtesting Strategy (RF)
    # -------------------------------
    strategy_returns = []
    benchmark_returns = []

    test_prices = y_test.values
    preds = rf_preds

    for i in range(1, len(test_prices)):
        # RF signal: Buy if tomorrow’s prediction > today
        if preds[i] > test_prices[i-1]:
            # Long position return
            strategy_returns.append((test_prices[i] - test_prices[i-1]) / test_prices[i-1])
        else:
            # Short position return (inverse)
            strategy_returns.append(-(test_prices[i] - test_prices[i-1]) / test_prices[i-1])

        # Benchmark: Buy & Hold
        benchmark_returns.append((test_prices[i] - test_prices[i-1]) / test_prices[i-1])

    strategy_cum = (1 + np.array(strategy_returns)).cumprod()
    benchmark_cum = (1 + np.array(benchmark_returns)).cumprod()

    plt.figure(figsize=(10,6))
    plt.plot(range(len(strategy_cum)), strategy_cum, label="RF Strategy")
    plt.plot(range(len(benchmark_cum)), benchmark_cum, label="Buy & Hold")
    plt.title(f"Backtesting Strategy vs Benchmark - {ticker}")
    plt.xlabel("Days")
    plt.ylabel("Cumulative Returns")
    plt.legend()
    strat_file = f"{ticker.replace('.NS','')}_backtest.png"
    plt.savefig(strat_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   ✅ Backtest saved as {strat_file}")

    # -------------------------------
    # Volatility Spikes Chart
    # -------------------------------
    data['Returns'] = data['Close'].pct_change()
    threshold = 2 * data['Returns'].std()  # 2 standard deviations
    spikes = data[data['Returns'].abs() > threshold]

    plt.figure(figsize=(12,6))
    plt.plot(data.index, data['Returns'], label="Daily Returns")
    plt.scatter(spikes.index, spikes['Returns'], color="red", label="Spikes")
    plt.axhline(threshold, color="green", linestyle="--", label="±2 Std Dev")
    plt.axhline(-threshold, color="green", linestyle="--")
    plt.title(f"Volatility Spikes - {ticker}")
    plt.xlabel("Date")
    plt.ylabel("Daily Returns")
    plt.legend()
    spike_file = f"{ticker.replace('.NS','')}_spikes.png"
    plt.savefig(spike_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   ✅ Spikes chart saved as {spike_file}")

    # -------------------------------
    # Save results into list
    # -------------------------------
    all_results.append({
        "Ticker": ticker,
        "ARIMA_RMSE": round(arima_rmse, 2),
        "ARIMA_MAE": round(arima_mae, 2),
        "RF_RMSE": round(rf_rmse, 2),
        "RF_MAE": round(rf_mae, 2),
        "RF_Strategy_Final_Return": round(strategy_cum[-1], 3),
        "BuyHold_Final_Return": round(benchmark_cum[-1], 3)
    })

# -------------------------------
# 📌 Step 3: Save All Results to CSV
# -------------------------------
results_df = pd.DataFrame(all_results)

if os.path.exists("model_results.csv"):
    results_df.to_csv("model_results.csv", mode='a', header=False, index=False)
else:
    results_df.to_csv("model_results.csv", index=False)

print("\n📊 Final Results Saved to model_results.csv")
print(results_df)

# -------------------------------
# 📌 Step 4: Combined Comparison Charts
# -------------------------------
if not results_df.empty:
    sns.set(style="whitegrid")

    # RMSE Comparison
    results_df.plot(x="Ticker", y=["ARIMA_RMSE", "RF_RMSE"], kind="bar", figsize=(10,6))
    plt.title("RMSE Comparison: ARIMA vs Random Forest")
    plt.ylabel("RMSE")
    plt.xticks(rotation=0)
    plt.savefig("RMSE_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    # MAE Comparison
    results_df.plot(x="Ticker", y=["ARIMA_MAE", "RF_MAE"], kind="bar", figsize=(10,6))
    plt.title("MAE Comparison: ARIMA vs Random Forest")
    plt.ylabel("MAE")
    plt.xticks(rotation=0)
    plt.savefig("MAE_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Strategy Returns Comparison
    results_df.plot(x="Ticker", y=["RF_Strategy_Final_Return", "BuyHold_Final_Return"], kind="bar", figsize=(10,6))
    plt.title("Final Returns: RF Strategy vs Buy & Hold")
    plt.ylabel("Cumulative Return")
    plt.xticks(rotation=0)
    plt.savefig("Strategy_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("✅ Comparison charts saved: RMSE_comparison.png, MAE_comparison.png, Strategy_comparison.png")
