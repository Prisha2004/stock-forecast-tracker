"""
Ablation Studies and Regime Stress-Testing across Bull, Volatile, and Crash Scenarios.
"""
import numpy as np
import pandas as pd
from data_loader import load_stock_data
from nse_trading_env import NSETradingEnv
from baselines import calculate_metrics

def run_stress_scenarios():
    print("\n--- Running Regime Stress Testing ---")
    days = 126
    np.random.seed(42)
    scenarios = {
        "Trending Bull Market": np.random.normal(0.0012, 0.008, days),
        "High-Vol Choppy Market": np.random.normal(0.0000, 0.022, days),
        "Market Liquidation Crash": np.random.normal(0.0002, 0.012, days)
    }
    scenarios["Market Liquidation Crash"][35:50] = np.random.normal(-0.028, 0.015, 15)

    for name, rets in scenarios.items():
        price = 1000.0 * np.exp(np.cumsum(rets))
        df = pd.DataFrame({"Return": rets, "Close": price})
        df["Volatility"] = pd.Series(rets).rolling(10).std().fillna(0.015) * np.sqrt(252)
        df["ARIMA_Forecast"] = pd.Series(rets).rolling(5).mean().fillna(0)
        env = NSETradingEnv(df, reward_type="sharpe")
        obs, _ = env.reset()
        done = False
        vals = [env.portfolio_value]
        step_rets = []
        while not done:
            f = obs[5] if len(obs) > 5 else 0.0
            act = 1 if f > 0.001 else (2 if f < -0.001 else 0)
            obs, r, term, trunc, info = env.step(act)
            done = term or trunc
            vals.append(env.portfolio_value)
            step_rets.append(info["net_return"])
        m = calculate_metrics(np.array(vals), np.array(step_rets), env.total_fees_paid, env.trades_count)
        bnh_ret = ((price[-1] / price[0]) - 1.0) * 100.0
        print(f"Regime: {name:<26} | Agent Return: {m['Annualized Return (%)']:>6.2f}% | Buy&Hold: {bnh_ret:>6.2f}% | MaxDD: {m['Max Drawdown (%)']:>6.2f}%")

if __name__ == "__main__":
    run_stress_scenarios()
