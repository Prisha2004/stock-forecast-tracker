"""
Data Loader for NSE Equities (RELIANCE, TCS, INFY, HDFCBANK).
Supports yfinance download, local CSV loading, and walk-forward rolling splits.
"""
import numpy as np
import pandas as pd
from typing import Generator, Tuple, Optional

STOCKS = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS"]

def load_stock_data(ticker: str = "RELIANCE.NS", period: str = "5y") -> pd.DataFrame:
    try:
        import yfinance as yf
        print(f"Fetching {ticker} via yfinance...")
        raw = yf.download(ticker, period=period, interval="1d", progress=False)
        if not raw.empty:
            df = raw.reset_index()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            df["Return"] = df["Close"].pct_change().fillna(0)
            df["Volatility"] = df["Return"].rolling(20).std().fillna(0.012) * np.sqrt(252)
            # Simulated ARIMA forecast proxy (short-term momentum + noise)
            df["ARIMA_Forecast"] = df["Return"].rolling(5).mean().fillna(0)
            df["SMA_20"] = df["Close"].rolling(20).mean().bfill()
            df["SMA_50"] = df["Close"].rolling(50).mean().bfill()
            return df
    except Exception as e:
        print(f"yfinance failed ({e}), generating synthetic series...")

    # Fallback synthetic series
    np.random.seed(42)
    dates = pd.date_range(end=pd.Timestamp.now(), periods=1260, freq="B")
    ret = np.random.normal(0.0005, 0.015, len(dates))
    price = 2500.0 * np.exp(np.cumsum(ret))
    df = pd.DataFrame({"Date": dates, "Close": price, "Return": ret})
    df["Volatility"] = df["Return"].rolling(20).std().fillna(0.012) * np.sqrt(252)
    df["ARIMA_Forecast"] = df["Return"].rolling(5).mean().fillna(0)
    df["SMA_20"] = df["Close"].rolling(20).mean().bfill()
    df["SMA_50"] = df["Close"].rolling(50).mean().bfill()
    return df

def walk_forward_splits(
    df: pd.DataFrame,
    train_window: int = 504,
    test_window: int = 126,
    step_size: int = 126
) -> Generator[Tuple[pd.DataFrame, pd.DataFrame, int], None, None]:
    start = 0
    split_idx = 0
    total_len = len(df)
    while start + train_window + test_window <= total_len:
        train_df = df.iloc[start : start + train_window].copy().reset_index(drop=True)
        test_df = df.iloc[start + train_window : start + train_window + test_window].copy().reset_index(drop=True)
        yield train_df, test_df, split_idx
        start += step_size
        split_idx += 1
