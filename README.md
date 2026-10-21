# Odimm_v1

First generation of **Odimm**: an ML-driven system for 1-minute CFD / crypto markets that combines quantile forecasting, a regime-aware spike filter, backtesting, and a live signal bot.

> Research code only. Not financial advice. Data and trained models are not included. API credentials were removed (`REDACTED`); supply your own through environment variables before running anything.

## Contents

| File | Description |
| --- | --- |
| `get_data.py` | Downloads roughly 30 days of M1 candles from the OANDA v20 API and saves them as CSV. |
| `algobackup.py` | Tuning-ready spike-only backtester: GMM regime clustering plus ML probability filter, vectorised volume-spike detection and Optuna optimisation. |
| `Algo.py` | Inference-only backtester that loads pre-trained models from an artifact directory and runs on new data. |
| `backtest.py` | Event-driven, anomaly-centric trading system with train/execute flags (LightGBM, XGBoost, logistic regression meta-model). |
| `backteston3.1.py` | Runs the quantile model and writes forecasts (percentages) with OHLCV to a single CSV. |
| `ann_quntil.py` | Small helper that queries a list of model APIs with one prompt and collects the results. |
| `live_xau.py` | Live loop for gold: pre-backtest on the last 7 days, signal logic and notifications via Telegram (Pyrogram), optional order placement. |
| `live_NATGAS.py` | Live trading simulator for natural gas using OANDA candles. |

## Requirements

Python 3.9+, `numpy`, `pandas`, `scikit-learn`, `xgboost`, `lightgbm`, `torch`, `optuna`, `tqdm`, `joblib`, `oandapyV20`, `pyrogram`, `matplotlib`.

## Configuration

Set your own values for `OANDA_ACCESS_TOKEN`, `OANDA_ACCOUNT_ID` and Telegram credentials, ideally read from environment variables rather than edited into the source. Use an OANDA **practice** account first.
