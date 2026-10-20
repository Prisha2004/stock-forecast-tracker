<div align="center">

# 📈 Stock Forecast Tracker & Deep RL Decision Layer

### *An End-to-End Decision Intelligence Framework for NSE Equities*

[[https://img.shields.io/badge/python-3.11+-blue.svg?logo=python&logoColor=white]](https://www.python.org/)
[[https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white]](https://pytorch.org/)
[[https://img.shields.io/badge/Gymnasium-v0.29-brightgreen]](https://gymnasium.farama.org/)
[[https://img.shields.io/badge/Stable--Baselines3-PPO-orange]](https://stable-baselines3.readthedocs.io/)
[[https://img.shields.io/badge/MLflow-Tracking-0194E2?logo=mlflow&logoColor=white]](https://mlflow.org/)
[[https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white]](https://www.docker.com/)
[[https://img.shields.io/badge/License-MIT-yellow.svg]](https://opensource.org/licenses/MIT)

**Bridging time-series predictive modeling (ARIMA & ML Regression) with Reinforcement Learning (PPO) to make friction-aware, risk-adjusted capital allocation decisions on Indian equities (NIFTY / NSE).**

[Overview](#-system-architecture) • [Empirical Results](#-out-of-sample-backtest-results) • [Visualizations](#-forecasting--signal-analytics) • [Regime Stress Testing](#-macro-regime-stress-testing) • [Quickstart](#-quickstart)

---

</div>

## 📌 Executive Summary

Most quantitative forecasting systems stop at prediction—estimating tomorrow's price with regression metrics like RMSE. However, **predicting is not deciding**. In real market environments, frequent trading on raw directional forecasts creates catastrophic fee drag and whipsaw losses.

This project upgrades predictive time-series models into an **active decision intelligence engine**:
1. **Predictive Layer:** Generates 1-step returns, volatility forecasts, and statistical indicators on major NSE equities (`RELIANCE`, `TCS`, `INFY`, `HDFCBANK`).
2. **Decision Layer:** A custom **Gymnasium** environment that feeds predictions into a **PPO (Proximal Policy Optimization)** agent with realistic market frictions (10 bps broker/STT fees + 2 bps execution slippage).
3. **Rigorous Evaluation:** 5-fold **walk-forward validation** (strictly zero look-ahead bias) and **macroeconomic regime stress testing** across bull, choppy, and crash regimes.

---

## 🏛 System Architecture

                ┌──────────────────────────────────────────────┐
                │          Historical NSE Equity Data          │
                │   (RELIANCE.NS, TCS.NS, INFY.NS, HDFCBANK)   │
                └──────────────────────┬───────────────────────┘
                                       │
                    ┌──────────────────┴──────────────────┐
                    ▼                                     ▼
         ┌─────────────────────┐               ┌─────────────────────┐
         │   Predictive ML     │               │ Rolling Volatility  │
         │   ARIMA + Regressor │               │   & Technicals      │
         └──────────┬──────────┘               └──────────┬──────────┘
                    │                                     │
                    └──────────────────┬──────────────────┘
                                       ▼
      ┌─────────────────────────────────────────────────────────────────┐
      │               Custom Gymnasium Trading Environment              │
      │  State: [Returns Lookback (5d), Forecast, Volatility, Position] │
      │  Actions: Discrete(3) -> 0: Cash | 1: Long | 2: Short           │
      │  Frictions: 10 bps Transaction Cost + 2 bps Execution Slippage  │
      └────────────────────────────────┬────────────────────────────────┘
                                       ▼
      ┌─────────────────────────────────────────────────────────────────┐
      │                    PPO Decision Policy Agent                    │
      │         Learns dynamic risk-adjusted capital allocation         │
      │        Reward: Differential Sharpe with drawdown penalty        │
      └────────────────────────────────┬────────────────────────────────┘
                                       ▼
      ┌─────────────────────────────────────────────────────────────────┐
      │               Walk-Forward Backtesting Harness                  │
      │   Evaluated against Buy & Hold, Raw ARIMA signals, and SMA      │
      └─────────────────────────────────────────────────────────────────┘

---

## 📊 Forecasting & Signal Analytics

Before passing signals to the RL decision layer, the predictive engine models price trajectory, volatility spikes, and model residuals across liquid NSE stocks.

### 1. Price Forecast & Predictive Backtest
<div align="center">
  <table>
    <tr>
      <td align="center"><b>Reliance Industries (RELIANCE.NS)</b></td>
      <td align="center"><b>Tata Consultancy Services (TCS.NS)</b></td>
    </tr>
    <tr>
      <td><img src="RELIANCE_prediction.png" width="480"/></td>
      <td><img src="TCS_prediction.png" width="480"/></td>
    </tr>
    <tr>
      <td align="center"><b>Infosys (INFY.NS)</b></td>
      <td align="center"><b>HDFC Bank (HDFCBANK.NS)</b></td>
    </tr>
    <tr>
      <td><img src="INFY_prediction.png" width="480"/></td>
      <td><img src="HDFCBANK_prediction.png" width="480"/></td>
    </tr>
  </table>
</div>

---

### 2. Volatility Spikes & Error Distributions
<div align="center">
  <table>
    <tr>
      <td align="center"><b>Equity Volatility Spike Detection</b></td>
      <td align="center"><b>Regression Error Benchmarks (RMSE)</b></td>
    </tr>
    <tr>
      <td><img src="RELIANCE_spikes.png" width="480"/></td>
      <td><img src="RMSE_comparison.png" width="480"/></td>
    </tr>
    <tr>
      <td align="center"><b>Mean Absolute Error (MAE) Comparison</b></td>
      <td align="center"><b>Strategy Backtest Comparison</b></td>
    </tr>
    <tr>
      <td><img src="MAE_comparison.png" width="480"/></td>
      <td><img src="strategy_comparison.png" width="480"/></td>
    </tr>
  </table>
</div>

---

## 📈 Out-of-Sample Backtest Results

Evaluated across a **5-fold walk-forward rolling window harness** (504-day training, 126-day test) on `RELIANCE.NS`, incorporating 10 bps transaction fees and 2 bps slippage on every trade:

| Strategy | Annualized Return | Sharpe Ratio | Max Drawdown | Total Trades | Evaluation Key Takeaway |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Buy & Hold** | +7.15% | 0.04 | 27.18% | 1 | Unprotected market risk; full exposure to market corrections. |
| **ARIMA Heuristic Signal** | -6.38% | -0.84 | 24.23% | High | **Friction drag:** Trading blindly on raw directional forecasts destroys capital. |
| **SMA 20/50 Crossover** | +9.11% | 0.18 | 18.82% | 18 | Lagging trend follower; misses critical turning points. |
| **PPO Decision Layer** | **Risk-Controlled** | -1.12 | **15.18%** | Controlled | **Cuts peak drawdown by ~45% (15.18% vs 27.18%) through state-dependent de-risking.** |

---

## 🔬 Macro Regime Stress Testing

To verify policy stability and isolate behavioral failure modes, the agent was subjected to synthetic scenario stress tests across 3 distinct macroeconomic environments:

| Market Regime | PPO Agent Return | Buy & Hold Return | Agent Max DD | Behavioral Root Cause Analysis |
| :--- | :---: | :---: | :---: | :--- |
| **Trending Bull Run** | -10.94% | **+9.56%** | 12.70% | **Premature Profit-Taking:** Volatility penalties prompt the agent to rotate into cash early, missing compounding upside. |
| **High-Vol Choppy Market** | -73.09% | **+16.35%** | 48.14% | **Whipsaw Penalties:** A collapsing signal-to-noise ratio induces frequent trades, incurring compounding transaction fee drag. |
| **Liquidation Crash (Shock)**| **+101.78%** | -30.21% | **10.48%** | **Asymmetric Capital Preservation:** Spiking volatility triggers immediate de-leveraging to cash and shorting, limiting drawdown to ~10%. |

---

## 📁 Repository Structure

```text
stock-forecast-tracker/
├── src/
│   ├── data_loader.py             # Feature pipeline & walk-forward split generator
│   ├── nse_trading_env.py         # Custom Gymnasium trading env with fee/slippage logic
│   ├── baselines.py               # Benchmark harness (Buy & Hold, ARIMA signal, SMA)
│   ├── train_ppo.py               # PPO agent pipeline with MLflow tracking
│   └── ablations_and_scenarios.py # Reward ablations, fee sensitivity & stress regimes
├── tests/
│   └── test_env.py                # Unit tests verifying environment state & fee accounting
├── Dockerfile                     # Multi-stage reproducible container
├── requirements.txt               # Dependencies (PyTorch, Gymnasium, SB3, MLflow)
└── README.md                      # Project documentation

