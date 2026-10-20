"""
Unit Tests for Custom Gymnasium Environment.
"""
import unittest
import numpy as np
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.abspath("src"))

from nse_trading_env import NSETradingEnv

class TestNSETradingEnv(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100)
        rets = np.random.normal(0, 0.01, 100)
        self.df = pd.DataFrame({
            "Date": dates,
            "Close": 1000 * np.exp(np.cumsum(rets)),
            "Return": rets,
            "Volatility": np.full(100, 0.15),
            "ARIMA_Forecast": np.roll(rets, -1)
        })

    def test_initialization(self):
        env = NSETradingEnv(self.df, lookback_window=5)
        self.assertEqual(env.portfolio_value, 100_000.0)
        self.assertEqual(env.action_space.n, 3)
        self.assertEqual(env.observation_space.shape, (8,))

    def test_fee_deduction(self):
        env = NSETradingEnv(self.df, transaction_cost_pct=0.0010, slippage_pct=0.0002)
        env.reset()
        obs, r, term, trunc, info = env.step(1)
        expected_fee = 100_000.0 * 1.0 * 0.0012
        self.assertAlmostEqual(env.total_fees_paid, expected_fee, places=3)
        self.assertEqual(env.trades_count, 1)

if __name__ == "__main__":
    unittest.main()
