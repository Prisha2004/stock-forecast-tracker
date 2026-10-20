"""
PPO Walk-Forward Training & Evaluation Pipeline with MLflow.
"""
import numpy as np
import pandas as pd
from data_loader import load_stock_data, walk_forward_splits
from nse_trading_env import NSETradingEnv
from baselines import calculate_metrics, run_baselines

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False

try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

def run_pipeline(ticker="RELIANCE.NS"):
    print(f"\n--- Running Walk-Forward PPO Pipeline for {ticker} ---")
    df = load_stock_data(ticker)
    splits = list(walk_forward_splits(df, train_window=504, test_window=126))
    print(f"Generated {len(splits)} walk-forward folds.")

    if HAS_MLFLOW:
        mlflow.set_experiment("NSE_Forecast_to_Decision_RL")
        mlflow.start_run(run_name=f"PPO_{ticker}")

    fold_metrics = []
    test_dfs = []
    for train_df, test_df, idx in splits:
        train_env = NSETradingEnv(train_df, reward_type="sharpe")
        test_env = NSETradingEnv(test_df, reward_type="sharpe")
        test_dfs.append(test_df)

        if HAS_SB3:
            vec_env = DummyVecEnv([lambda: train_env])
            model = PPO("MlpPolicy", vec_env, learning_rate=3e-4, n_steps=256, batch_size=64, n_epochs=10, verbose=0)
            model.learn(total_timesteps=15_000)
            obs, _ = test_env.reset()
            done = False
            vals = [test_env.portfolio_value]
            rets = []
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, r, term, trunc, info = test_env.step(action)
                done = term or trunc
                vals.append(test_env.portfolio_value)
                rets.append(info["net_return"])
        else:
            # Fallback heuristic policy if SB3 is not yet installed
            obs, _ = test_env.reset()
            done = False
            vals = [test_env.portfolio_value]
            rets = []
            while not done:
                f = obs[5] if len(obs) > 5 else 0.0
                act = 1 if f > 0.001 else (2 if f < -0.001 else 0)
                obs, r, term, trunc, info = test_env.step(act)
                done = term or trunc
                vals.append(test_env.portfolio_value)
                rets.append(info["net_return"])

        m = calculate_metrics(np.array(vals), np.array(rets), test_env.total_fees_paid, test_env.trades_count)
        fold_metrics.append(m)
        print(f"Fold {idx+1} Out-of-Sample -> Sharpe: {m['Sharpe Ratio']:.2f} | Return: {m['Annualized Return (%)']:.2f}% | MaxDD: {m['Max Drawdown (%)']:.2f}%")

    avg_sharpe = np.mean([m["Sharpe Ratio"] for m in fold_metrics])
    avg_ret = np.mean([m["Annualized Return (%)"] for m in fold_metrics])
    avg_dd = np.mean([m["Max Drawdown (%)"] for m in fold_metrics])

    print("\n========================================================")
    print(f"PPO AGENT OUT-OF-SAMPLE RESULTS ({ticker}):")
    print(f"Mean Sharpe: {avg_sharpe:.2f} | Mean Ann. Return: {avg_ret:.2f}% | Mean MaxDD: {avg_dd:.2f}%")
    print("========================================================")

    combined_test = pd.concat(test_dfs, ignore_index=True)
    baselines = run_baselines(combined_test)
    print("\nBENCHMARK BASELINES (Same Test Window):")
    for name, res in baselines.items():
        print(f"{name:<22}: Sharpe = {res['Sharpe Ratio']:>6.2f} | Return = {res['Annualized Return (%)']:>6.2f}% | MaxDD = {res['Max Drawdown (%)']:>6.2f}%")

    if HAS_MLFLOW:
        mlflow.log_metrics({"avg_sharpe": avg_sharpe, "avg_return": avg_ret, "avg_dd": avg_dd})
        mlflow.end_run()

if __name__ == "__main__":
    run_pipeline("RELIANCE.NS")
