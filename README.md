# stock-forecast-tracker
Multi-Model Stock Forecast &amp; Strategy Tracker (ARIMA vs Random Forest + Backtesting + Volatility Spikes)

📈 Stock Forecast & Strategy Tracker

A Streamlit-powered web app that forecasts stock prices using ARIMA & Random Forest, detects volatility spikes, and provides next-day predictions.

🔹 Built with Python, Streamlit, and machine learning libraries.
🔹 Visualizes real-world stock data (via Yahoo Finance).
🔹 Includes model comparison, backtesting, and risk analysis.

🚀 Features

✅ Stock Data Fetching – Real-time data from Yahoo Finance
✅ ARIMA Forecasting – Time-series based predictions
✅ Random Forest Forecasting – ML-based predictions with sliding window
✅ Volatility Spike Detection – Identify sudden ±3% daily moves
✅ Performance Metrics – RMSE & MAE for accuracy comparison
✅ Next-Day Prediction – Forecast tomorrow’s closing price


📂 Project Structure
📦 stock-forecast-tracker
 ┣ 📜 stock_prediction.py  # Backend training script
 ┣ 📜 requirements.txt     # Python dependencies
 ┣ 📜 README.md            # Documentation
 ┣ 📂 charts/              # Saved plots (backtests, spikes, etc.)
 ┗ 📜 model_results.csv    # Results summary

