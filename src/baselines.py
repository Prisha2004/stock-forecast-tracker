"""
Benchmark Evaluation Harness: Buy & Hold, ARIMA Signal Rule, and SMA Crossover.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any

def calculate_metrics(portfolio_values: np.ndarray, returns: np.ndarray, fees_paid: float, num_trades: int):
    total_ret = (portfolio_values[-1] / portfolio_values[0]) - 1.0
    years = max(len(returns) / 252.0, 1.0 / 252.0)
    ann_ret = ((1.0 + total_ret) ** (1.0 / years)) - 1.0
    excess = returns - (0.065 / 252.0)
    std = np.std(returns)
    sharpe = float((np.mean(excess) / std) * np.sqrt(252)) if std > 1e-8 else 0.0
    peaks = np.maximum.accumulate(portfolio_values)
    drawdowns = (peaks - portfolio_values) / peaks
    max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0
    return {
        "Annualized Return (%)": float(ann_ret * 100.0),
        "Sharpe Ratio": float(sharpe),
        "Max Drawdown (%)": float(max_dd * 100.0),
        "Total Trades": int(num_trades),
        "Total Fees Paid (INR)": float(fees_paid)
    }

def evaluate_signals(df: pd.DataFrame, signals: np.ndarray, initial_capital=100_000.0, cost_rate=0.0012):
    n = len(df)
    vals = np.zeros(n)
    rets = np.zeros(n)
    val = initial_capital
    pos = 0.0
    fees = 0.0
    trades = 0
    for t in range(n):
        target = signals[t]
        delta = abs(target - pos)
        fee = (val * delta * cost_rate) if delta > 0 else 0.0
        fees += fee
        if delta > 0: trades += 1
        mkt_ret = df.loc[t, "Return"]
        pnl = val * (pos * mkt_ret)
        val = max(val + pnl - fee, 1.0)
        vals[t] = val
        rets[t] = (pnl - fee) / initial_capital
        pos = target
    return calculate_metrics(vals, rets, fees, trades)

def run_baselines(df: pd.DataFrame):
    n = len(df)
    bnh = np.ones(n, dtype=np.float32)
    arima = (df["ARIMA_Forecast"].values > 0.0).astype(np.float32) if "ARIMA_Forecast" in df.columns else np.zeros(n)
    sma = (df["SMA_20"].values > df["SMA_50"].values).astype(np.float32) if "SMA_20" in df.columns else np.zeros(n)
    return {
        "Buy & Hold": evaluate_signals(df, bnh),
        "ARIMA Signal Rule": evaluate_signals(df, arima),
        "SMA 20/50 Crossover": evaluate_signals(df, sma)
    }
