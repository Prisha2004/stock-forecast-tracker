"""
Custom Gymnasium Environment for NSE Equities.
Action: 0 = Cash, 1 = Long (+1), 2 = Short (-1)
State: Lookback returns, ARIMA forecast, Volatility, Current Position
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

class NSETradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        lookback_window: int = 5,
        initial_capital: float = 100_000.0,
        transaction_cost_pct: float = 0.0010,
        slippage_pct: float = 0.0002,
        reward_type: str = "sharpe",
        include_forecast: bool = True,
        risk_free_rate: float = 0.065 / 252
    ):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.lookback_window = lookback_window
        self.initial_capital = initial_capital
        self.transaction_cost_pct = transaction_cost_pct
        self.slippage_pct = slippage_pct
        self.reward_type = reward_type
        self.include_forecast = include_forecast
        self.risk_free_rate = risk_free_rate

        self.num_steps = len(self.df) - 1
        self.action_space = spaces.Discrete(3)
        self.action_to_position = {0: 0.0, 1: 1.0, 2: -1.0}

        self.obs_dim = self.lookback_window + (1 if self.include_forecast else 0) + 2
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)

        self.current_step = 0
        self.position = 0.0
        self.portfolio_value = self.initial_capital
        self.peak_portfolio_value = self.initial_capital
        self.total_fees_paid = 0.0
        self.trades_count = 0

    def _get_obs(self):
        t = self.current_step
        start = t - self.lookback_window
        rets = self.df.loc[start : t - 1, "Return"].values
        elems = [rets]
        if self.include_forecast:
            f = self.df.loc[t, "ARIMA_Forecast"] if "ARIMA_Forecast" in self.df.columns else 0.0
            elems.append(np.array([f], dtype=np.float32))
        vol = self.df.loc[t, "Volatility"] if "Volatility" in self.df.columns else 0.015
        elems.append(np.array([vol, self.position], dtype=np.float32))
        return np.concatenate(elems).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed) if hasattr(super(), "reset") else None
        self.current_step = self.lookback_window
        self.position = 0.0
        self.portfolio_value = self.initial_capital
        self.peak_portfolio_value = self.initial_capital
        self.total_fees_paid = 0.0
        self.trades_count = 0
        return self._get_obs(), {"portfolio_value": self.portfolio_value}

    def step(self, action: int):
        target_pos = self.action_to_position[int(action)]
        delta = abs(target_pos - self.position)
        fee = 0.0
        if delta > 0.0:
            fee = self.portfolio_value * delta * (self.transaction_cost_pct + self.slippage_pct)
            self.total_fees_paid += fee
            self.trades_count += 1

        mkt_ret = self.df.loc[self.current_step, "Return"]
        pnl = self.portfolio_value * (self.position * mkt_ret)
        self.portfolio_value = max(self.portfolio_value + pnl - fee, 1.0)
        if self.portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = self.portfolio_value
        dd = (self.peak_portfolio_value - self.portfolio_value) / self.peak_portfolio_value

        net_ret = (pnl - fee) / self.initial_capital
        vol = self.df.loc[self.current_step, "Volatility"] if "Volatility" in self.df.columns else 0.015
        daily_vol = vol / np.sqrt(252) if vol > 0 else 0.01

        if self.reward_type == "pnl":
            reward = float(net_ret)
        elif self.reward_type == "sharpe":
            reward = float((net_ret - self.risk_free_rate) / (daily_vol + 1e-6))
        elif self.reward_type == "drawdown_penalty":
            reward = float(net_ret - 2.0 * (dd ** 2))
        else:
            reward = float(net_ret)

        self.position = target_pos
        self.current_step += 1
        term = self.current_step >= self.num_steps
        trunc = self.portfolio_value < (self.initial_capital * 0.5)
        obs = self._get_obs() if not term else np.zeros(self.obs_dim, dtype=np.float32)
        info = {
            "portfolio_value": self.portfolio_value,
            "net_return": net_ret,
            "drawdown": dd,
            "total_fees": self.total_fees_paid,
            "trades": self.trades_count
        }
        return obs, reward, term, trunc, info
