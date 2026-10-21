# -*- coding: utf-8 -*-
"""
Single-file script:
- Fetches M1 candles from OANDA for the last 7 days (pre-backtest window),
- Runs feature generation, model prediction, spike-filter and backtest with PRECISION_PARAMS,
- Sends results via Telegram (Pyrogram) to TARGET_CHATS,
- If pre-backtest runs, starts a live loop that polls OANDA for new M1 candles and runs the same
  signal/confirmation logic to notify (and optionally place orders via place_order()).
Notes:
- Requires: oandapyV20, pandas, numpy, torch, joblib, pyrogram, matplotlib (plotting optional),
  ccxt (not used here), and your 'auto.place_order' helper if you want live order placement.
- Edit QUANT_META_DIR, OANDA_ACCESS_TOKEN, Telegram credentials, and place_order usage as needed.
"""
import os
import sys
import json
import asyncio
import logging
from pathlib import Path
from datetime import datetime, timezone, timedelta
import time
import traceback

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn

# plotting optional (used for saving backtest plot)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# OANDA REST client
import oandapyV20
import oandapyV20.endpoints.instruments as instruments

# Telegram (Pyrogram)
from pyrogram import Client
from pyrogram.enums import ParseMode

# (optional) order placement helper - if missing, script will warn and continue without placing orders
try:
    from auto import place_order
    _HAS_PLACE_ORDER = True
except Exception:
    _HAS_PLACE_ORDER = False

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("oanda-live-precision")

# ---------------- CONFIG / PATHS (edit if needed) ----------------
# quant artifacts path (the folder that contains ann_meta.json and artifacts)
QUANT_META_DIR = Path(r"D:\project\Odimm_v1\mhq_ann")
QUANT_META_PATH = QUANT_META_DIR / "ann_meta.json"

# OANDA
OANDA_ACCESS_TOKEN = "REDACTED"
OANDA_ACCOUNT_ID = "REDACTED"
OANDA_ENVIRONMENT = "practice"  # or "live"

# Instrument + timeframe
SYMBOL = "XAU_USD"   # as you set in the merged code
TIMEFRAME = "M1"

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_PARALLEL_STATS = True
SEED = 42
np.random.seed(SEED)

# Telegram credentials (kept in-file as requested)
API_TOKEN = "REDACTED"
API_ID = 20205931
API_HASH = "REDACTED"

# Allowed Telegram usernames (without @)
TARGET_CHATS = ["saeed_khosh_neshin", "Xio_Dr"]

# Pre-backtest window: 7 days
PREBACKTEST_DAYS = 7
PREBACKTEST_MINUTES = PREBACKTEST_DAYS * 24 * 60  # number of 1-min candles to use for pre-backtest

# Live history buffer size (how many minutes to retain for live signal generation)
HISTORY_SIZE = 10_000  # ~ 7 days is ~10080

# ---------------- PRECISION_PARAMS (the tuned dict you provided) ----------------
PRECISION_PARAMS = {
    'spike_window': 139, 'spike_z': 1.6270019396875117, 'confirm_horizon': 5, 'confirm_min_move_pct': 0.03580774657470852,
    'long_tp1': 0.11902596537958256, 'long_tp2': 0.3006214128280116, 'long_tp3': 0.9618522123853089,
    'long_sl': 0.09914003536744626, 'long_freeze': 3,
    'short_tp1': 0.08002482354349998, 'short_tp2': 0.26865753342033116, 'short_tp3': 1.405512622037665,
    'short_sl': 0.1621656356879882, 'short_freeze': 3,
    'long_str_h30': 0.6219016624735749, 'long_str_h60': 0.21448130782679337, 'long_str_h90': 0.5380700503199526, 'long_str_h180': 0.5871247305610957,
    'short_str_h30': 0.5138186523218197, 'short_str_h60': 0.20350491364514203, 'short_str_h90': 0.4109181574500486, 'short_str_h180': 0.11327481500268533
}

# ---------------- Telegram (Pyrogram) client ----------------
tg_app = Client(
    "trading_bot_session",
    api_id=API_ID,
    api_hash=API_HASH,
    bot_token=API_TOKEN,
    parse_mode=ParseMode.HTML,
    in_memory=True
)

async def tg_start():
    try:
        if not getattr(tg_app, "is_connected", False):
            await tg_app.start()
            logger.info("Telegram client started.")
    except Exception as e:
        logger.exception("Failed to start Telegram client: %s", e)
        raise

async def tg_stop():
    try:
        if getattr(tg_app, "is_connected", False):
            await tg_app.stop()
            logger.info("Telegram client stopped.")
    except Exception:
        logger.exception("Failed to stop Telegram client.")

async def send_to_allowed(text: str):
    """Send message to allowed usernames via Telegram."""
    for u in TARGET_CHATS:
        try:
            await tg_app.send_message(f"@{u}", text, disable_web_page_preview=True)
        except Exception:
            logger.exception("Failed to send message to %s", u)

# ---------------- Utilities / features / model loader ----------------
def ensure_epoch_seconds(series: pd.Series) -> np.ndarray:
    arr = pd.to_datetime(series, errors="coerce", utc=True)
    return (arr.astype("int64") // 10**9).to_numpy().astype(np.int64)

def compute_trends(df: pd.DataFrame) -> pd.DataFrame:
    up_cols = ['pred_up_h30_q0p9_pct','pred_up_h60_q0p9_pct','pred_up_h90_q0p9_pct','pred_up_h180_q0p9_pct']
    dn_cols = ['pred_dn_h30_q0p1_pct','pred_dn_h60_q0p1_pct','pred_dn_h90_q0p1_pct','pred_dn_h180_q0p1_pct']
    out = df.copy()
    up_exist = [c for c in up_cols if c in out.columns]
    dn_exist = [c for c in dn_cols if c in out.columns]
    out["trend_up"] = out[up_exist].mean(axis=1) if up_exist else 0.0
    out["trend_dn"] = out[dn_exist].mean(axis=1) if dn_exist else 0.0
    return out

def detect_volume_spikes(data: pd.DataFrame, window: int, z_thresh: float) -> pd.Index:
    import warnings
    if "Volume" not in data.columns:
        warnings.warn("Volume column not found; no spikes.")
        return data.iloc[0:0].index
    vol = data["Volume"].astype(float)
    rm = vol.rolling(window, min_periods=1).median().shift(1)
    abs_dev = (vol - rm).abs()
    mad = abs_dev.rolling(window, min_periods=1).median().shift(1)
    mad_safe = mad.replace(0, np.nan)
    robust_z = 0.6745 * (vol - rm) / mad_safe
    return data.index[robust_z > z_thresh]

def load_model_blob(model_path: Path):
    blob = None
    try:
        add_safe = getattr(torch.serialization, "add_safe_globals", None)
        safe_ctx = getattr(torch.serialization, "safe_globals", None)
        if callable(add_safe):
            import numpy as _np
            add_safe([_np._core.multiarray._reconstruct])
            blob = torch.load(model_path, map_location=DEVICE)
            return blob
        elif safe_ctx is not None:
            import numpy as _np
            with safe_ctx([_np._core.multiarray._reconstruct]):
                blob = torch.load(model_path, map_location=DEVICE)
                return blob
    except Exception:
        pass
    try:
        blob = torch.load(model_path, map_location=DEVICE, weights_only=False)
        return blob
    except TypeError:
        try:
            blob = torch.load(model_path, map_location=DEVICE)
            return blob
        except Exception:
            return None
    except Exception:
        return None

# ---------------- Backtester (identical semantics) ----------------
def backtest_fast_numpy(df: pd.DataFrame, long_params: dict, short_params: dict, confirm_params: dict):
    trade_cols = ["entry_time_s", "exit_time_s", "side", "entry", "exit", "pnl_pct", "reason"]
    if df.empty:
        return {"total_trades": 0, "long_trades": 0, "short_trades": 0, "win_rate": 0.0, "total_pnl": 0.0, "avg_pnl": 0.0}, pd.DataFrame(columns=trade_cols)
    if "timestamp" not in df.columns:
        raise KeyError("timestamp required for backtest")
    t = ensure_epoch_seconds(df["timestamp"])
    close = df["Close"].to_numpy(float)
    open_p = df["Open"].to_numpy(float)
    high = df["High"].to_numpy(float)
    low = df["Low"].to_numpy(float)

    def col_or_zeros(name):
        return df[name].to_numpy(float) if name in df.columns else np.zeros(len(close), dtype=float)

    up30, up60, up90, up180 = (col_or_zeros("pred_up_h30_q0p9_pct"),
                              col_or_zeros("pred_up_h60_q0p9_pct"),
                              col_or_zeros("pred_up_h90_q0p9_pct"),
                              col_or_zeros("pred_up_h180_q0p9_pct"))

    dn30, dn60, dn90, dn180 = (col_or_zeros("pred_dn_h30_q0p1_pct"),
                              col_or_zeros("pred_dn_h60_q0p1_pct"),
                              col_or_zeros("pred_dn_h90_q0p1_pct"),
                              col_or_zeros("pred_dn_h180_q0p1_pct"))

    tr_up = df["trend_up"].to_numpy(float) if "trend_up" in df.columns else np.zeros(len(close), dtype=float)
    tr_dn = df["trend_dn"].to_numpy(float) if "trend_dn" in df.columns else np.zeros(len(close), dtype=float)

    n = len(close)
    position, entry_price, entry_time_s = None, 0.0, 0
    tp_levels = np.zeros(3, dtype=float)
    stop_level, stage, freeze_until = np.nan, 0, -1
    trades, scheduled_entries = [], {}

    lp, sp = long_params.copy(), short_params.copy()
    freeze_long_s = int(lp.get("freeze_period_minutes", 20)) * 60
    freeze_short_s = int(sp.get("freeze_period_minutes", 20)) * 60

    H = int(confirm_params.get("horizon_bars", 10))
    min_move = float(confirm_params.get("min_confirm_move_pct", 0.0))

    def eval_long_sig(i):
        return ((up30[i] > lp["strength_h30"]) or (up60[i] > lp["strength_h60"]) or
                (up90[i] > lp["strength_h90"]) or (up180[i] > lp["strength_h180"])) and (tr_up[i] > 0)

    def eval_short_sig(i):
        return ((dn30[i] > sp["strength_h30"]) or (dn60[i] > sp["strength_h60"]) or
                (dn90[i] > sp["strength_h90"]) or (dn180[i] > sp["strength_h180"])) and (tr_dn[i] > 0)

    for i in range(n):
        ts = int(t[i])

        # Execute scheduled entries (use Open if present)
        if position is None and i in scheduled_entries and ts >= freeze_until:
            side = scheduled_entries.pop(i)
            entry_price = open_p[i] if (not np.isnan(open_p[i])) else float(close[i])
            entry_time_s = ts
            stage = 0
            if side == "long":
                position = "long"
                tp_levels = entry_price * (1.0 + np.array([lp["tp1_pct"], lp["tp2_pct"], lp["tp3_pct"]]) / 100.0)
                stop_level = entry_price * (1.0 - lp["sl_pct"] / 100.0)
            else:
                position = "short"
                tp_levels = entry_price * (1.0 - np.array([sp["tp1_pct"], sp["tp2_pct"], sp["tp3_pct"]]) / 100.0)
                stop_level = entry_price * (1.0 + sp["sl_pct"] / 100.0)

        if ts >= freeze_until and position is None:
            long_sig = eval_long_sig(i)
            short_sig = eval_short_sig(i)

            if long_sig or short_sig:
                chosen_side = None
                if long_sig and not short_sig:
                    chosen_side = "long"
                elif short_sig and not long_sig:
                    chosen_side = "short"
                elif long_sig and short_sig:
                    if tr_up[i] > tr_dn[i]:
                        chosen_side = "long"
                    elif tr_dn[i] > tr_up[i]:
                        chosen_side = "short"
                    else:
                        up_sum = up30[i] + up60[i] + up90[i] + up180[i]
                        dn_sum = dn30[i] + dn60[i] + dn90[i] + dn180[i]
                        if up_sum > dn_sum:
                            chosen_side = "long"
                        elif dn_sum > up_sum:
                            chosen_side = "short"

                if chosen_side and (j := i + H) < n:
                    future_ret = (close[j] - close[i]) / close[i] * 100.0
                    if (chosen_side == "long" and future_ret >= min_move) or \
                       (chosen_side == "short" and future_ret <= -min_move):
                        scheduled_entries[j] = chosen_side

        # Manage open position
        if position:
            if position == "long":
                if stage == 0:
                    if low[i] <= stop_level:
                        exit_price = low[i]
                        pnl = ((stop_level if not np.isnan(stop_level) else exit_price) - entry_price) / entry_price * 100.0
                        trades.append((entry_time_s, ts, "long", entry_price, exit_price, pnl, "SL"))
                        position, freeze_until = None, ts + freeze_long_s
                    elif high[i] >= tp_levels[0]:
                        stage, stop_level = 1, tp_levels[0]
                elif stage == 1:
                    if low[i] <= stop_level:
                        trades.append((entry_time_s, ts, "long", entry_price, stop_level, (stop_level - entry_price) / entry_price * 100.0, "BE@TP1"))
                        position, freeze_until = None, ts + freeze_long_s
                    elif high[i] >= tp_levels[1]:
                        stage, stop_level = 2, tp_levels[1]
                elif stage == 2:
                    if low[i] <= stop_level:
                        trades.append((entry_time_s, ts, "long", entry_price, stop_level, (stop_level - entry_price) / entry_price * 100.0, "BE@TP2"))
                        position, freeze_until = None, ts + freeze_long_s
                    elif high[i] >= tp_levels[2]:
                        trades.append((entry_time_s, ts, "long", entry_price, tp_levels[2], (tp_levels[2] - entry_price) / entry_price * 100.0, "TP3"))
                        position, freeze_until = None, ts + freeze_long_s
            else:
                if stage == 0:
                    if high[i] >= stop_level:
                        exit_price = high[i]
                        pnl = (entry_price - exit_price) / entry_price * 100.0
                        trades.append((entry_time_s, ts, "short", entry_price, exit_price, pnl, "SL"))
                        position, freeze_until = None, ts + freeze_short_s
                    elif low[i] <= tp_levels[0]:
                        stage, stop_level = 1, tp_levels[0]
                elif stage == 1:
                    if high[i] >= stop_level:
                        trades.append((entry_time_s, ts, "short", entry_price, stop_level, (entry_price - stop_level) / entry_price * 100.0, "BE@TP1"))
                        position, freeze_until = None, ts + freeze_short_s
                    elif low[i] <= tp_levels[1]:
                        stage, stop_level = 2, tp_levels[1]
                elif stage == 2:
                    if high[i] >= stop_level:
                        trades.append((entry_time_s, ts, "short", entry_price, stop_level, (entry_price - stop_level) / entry_price * 100.0, "BE@TP2"))
                        position, freeze_until = None, ts + freeze_short_s
                    elif low[i] <= tp_levels[2]:
                        trades.append((entry_time_s, ts, "short", entry_price, tp_levels[2], (entry_price - tp_levels[2]) / entry_price * 100.0, "TP3"))
                        position, freeze_until = None, ts + freeze_short_s

    if not trades:
        return {"total_trades": 0, "long_trades": 0, "short_trades": 0, "win_rate": 0.0, "total_pnl": 0.0, "avg_pnl": 0.0}, pd.DataFrame(columns=trade_cols)

    trades_df = pd.DataFrame(trades, columns=trade_cols)
    pnls = trades_df["pnl_pct"].astype(float)
    summary = {
        "total_trades": len(trades),
        "long_trades": (trades_df["side"] == "long").sum(),
        "short_trades": (trades_df["side"] == "short").sum(),
        "win_rate": (pnls > 0).mean() * 100.0 if len(pnls) > 0 else 0.0,
        "total_pnl": pnls.sum(),
        "avg_pnl": pnls.mean() if len(pnls) > 0 else 0.0
    }
    return summary, trades_df

# ---------------- Feature-engineering / model class / run_prediction ----------------
def _rolling_stats_np(x, w):
    n = x.size
    if n < w or w <= 1:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    c0 = np.empty(n + 1); c0[0] = 0; c0[1:] = np.cumsum(x)
    m_seg = (c0[w:] - c0[:-w]) / w; m = np.r_[np.full(w - 1, np.nan), m_seg]
    c2 = np.empty(n + 1); c2[0] = 0; c2[1:] = np.cumsum(x * x)
    mean2 = (c2[w:] - c2[:-w]) / w; var = mean2 - m_seg ** 2
    s = np.r_[np.full(w - 1, np.nan), np.sqrt(np.maximum(var, 0))]
    t = np.arange(w, dtype=float); tm = t.mean(); denom = np.sum((t - tm) ** 2) or 1
    sum_xt = np.convolve(x, t[::-1], mode="valid")
    sl = np.r_[np.full(w - 1, np.nan), (sum_xt - w * m_seg * tm) / denom]
    return m, s, sl

def create_all_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    eps = 1e-8
    new_cols = {}
    new_cols["oc_ret"] = np.log(df["Close"].to_numpy(float) / (df["Open"].to_numpy(float) + eps))
    new_cols["hl_range"] = (df["High"].to_numpy(float) - df["Low"].to_numpy(float)) / (df["Low"].to_numpy(float) + eps)
    close = df["Close"].values.astype(float)
    logc = np.log(np.maximum(close, eps))
    vol = df["Volume"].values.astype(float)

    R = [15, 30, 60, 90, 120]
    VW = [5, 15, 30, 60, 90, 180]

    tasks = [(new_cols[c], w, c) for c in ("oc_ret", "hl_range") for w in R]
    if USE_PARALLEL_STATS:
        res = joblib.Parallel(n_jobs=max(1, os.cpu_count() // 2), backend="threading")(joblib.delayed(_rolling_stats_np)(a, w) for a, w, _ in tasks)
    else:
        res = [_rolling_stats_np(a, w) for a, w, _ in tasks]

    for ((_, w, c), (m, s, sl)) in zip(tasks, res):
        new_cols[f"{c}_rmean_{w}"] = m
        new_cols[f"{c}_rstd_{w}"] = s
        new_cols[f"{c}_rslope_{w}"] = sl

    for w in R: new_cols[f"momentum_{w}"] = pd.Series(close).pct_change(w).values
    for w in VW: new_cols[f"ret_{w}"] = pd.Series(logc).diff(w).values
    for w in VW:
        vm = pd.Series(vol).rolling(w).mean().values
        vs = pd.Series(vol).rolling(w).std().values
        new_cols[f"vol_z_{w}"] = (vol - vm) / (np.abs(vs) + eps)

    d = pd.Series(close).diff().fillna(0); up = d.clip(lower=0); down = -d.clip(upper=0)
    rs = up.ewm(alpha=1/14, adjust=False).mean() / (down.ewm(alpha=1/14, adjust=False).mean() + eps)
    new_cols["rsi_14"] = (100 - 100 / (1 + rs)).clip(0, 100).values
    ma = pd.Series(close).rolling(20).mean(); sd = pd.Series(close).rolling(20).std()
    new_cols["bb_pos"] = ((pd.Series(close) - ma) / (2 * (sd + eps))).values

    wins = [1, 3, 5, 15, 30, 45, 60, 90]
    for w in wins:
        vm = pd.Series(df["Volume"].values).rolling(w, min_periods=1).mean().values
        new_cols[f"volume_mean_{w}"] = vm
        new_cols[f"volume_change_{w}"] = pd.Series(df["Volume"].values).diff(w).values
        new_cols[f"volume_ratio_{w}"] = df["Volume"].to_numpy(float) / (vm + eps)
        pm = pd.Series(df["Close"].values).rolling(w, min_periods=1).mean().values
        new_cols[f"price_mean_{w}"] = pm
        new_cols[f"price_change_past_{w}"] = pd.Series(df["Close"].values).diff(w).values
        new_cols[f"price_ratio_{w}"] = df["Close"].to_numpy(float) / (pm + eps)
        new_cols[f"volume_std_{w}"] = pd.Series(df["Volume"].values).rolling(w, min_periods=2).std().values if w > 1 else np.zeros(len(df))
        new_cols[f"price_std_{w}"] = pd.Series(df["Close"].values).rolling(w, min_periods=2).std().values if w > 1 else np.zeros(len(df))

    new_features_df = pd.DataFrame(new_cols, index=df.index)
    return pd.concat([df, new_features_df], axis=1).copy()

class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden=512, depth=3, dropout=0.1):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout)]
            d = hidden
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

def run_prediction(df: pd.DataFrame, artifacts_local: dict) -> pd.DataFrame:
    if df.empty: return df
    df_fe = create_all_features(df.copy())
    df_fe.replace([np.inf, -np.inf], np.nan, inplace=True)
    feats, scaler = artifacts_local['feats_quant'], artifacts_local['scaler_quant']
    for m in [f for f in feats if f not in df_fe.columns]: df_fe[m] = 0.0
    X = df_fe[feats].values.astype(np.float32)
    X_scaled = scaler.transform(X)
    with torch.no_grad():
        yhat_full = artifacts_local['model_quant'](torch.from_numpy(X_scaled).to(DEVICE)).cpu().numpy()
    out = df_fe.copy()
    for i, (side, h, q) in enumerate(artifacts_local['quant_spec']):
        out[f"pred_{side}_h{h}_q{str(q).replace('.', 'p')}_pct"] = yhat_full[:, i].astype(float)
    return compute_trends(out).dropna(subset=['Close'])

# ---------------- OANDA fetch helpers ----------------
def oanda_candles_to_dataframe(candles):
    rows = []
    for c in candles:
        t = c.get('time')
        volume = c.get('volume', 0)
        price_bucket = c.get('mid') or c.get('bid') or c.get('ask')
        if price_bucket is None:
            continue
        try:
            o = float(price_bucket.get('o', np.nan))
            h = float(price_bucket.get('h', np.nan))
            l = float(price_bucket.get('l', np.nan))
            cl = float(price_bucket.get('c', np.nan))
        except Exception:
            continue
        rows.append((t, o, h, l, cl, volume))
    df = pd.DataFrame(rows, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        for col in ['Open','High','Low','Close','Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    return df

def fetch_oanda_historical_blocking(days: int, instrument: str = SYMBOL, granularity: str = "M1"):
    api = oandapyV20.API(access_token=OANDA_ACCESS_TOKEN, environment=OANDA_ENVIRONMENT)
    end_date_aware = datetime.now(timezone.utc)
    start_date_aware = end_date_aware - timedelta(days=days)
    all_candles = []
    current_start_dt = start_date_aware
    max_candles = int(days) * 24 * 60
    attempts = 0
    print(f"Fetching ~{days} days ({max_candles} M1 candles) from OANDA for {instrument} ...")
    while len(all_candles) < max_candles and current_start_dt < end_date_aware:
        attempts += 1
        from_time_str = current_start_dt.isoformat().replace('+00:00', 'Z')
        params = {"granularity": granularity, "from": from_time_str, "count": 5000}
        r = instruments.InstrumentsCandles(instrument=instrument, params=params)
        try:
            api.request(r)
        except Exception as e:
            logger.exception("OANDA API Error during historical fetch: %s", e)
            break
        candles = r.response.get('candles', [])
        if not candles:
            break
        all_candles.extend(candles)
        last_candle_dt = pd.to_datetime(candles[-1]['time'])
        current_start_dt = last_candle_dt + timedelta(minutes=1)
        print(f"\rCollected: {len(all_candles):,}/{max_candles} candles (last: {last_candle_dt})", end="")
        if attempts > 2000:
            print("\nReached max attempts; stopping.")
            break
    print("\nFetch finished, building DataFrame...")
    df = oanda_candles_to_dataframe(all_candles)
    if len(df) > max_candles:
        df = df.tail(max_candles).reset_index(drop=True)
    return df

async def fetch_historical_data_oanda(days):
    loop = asyncio.get_running_loop()
    df = await loop.run_in_executor(None, fetch_oanda_historical_blocking, days, SYMBOL, TIMEFRAME)
    return df

def fetch_latest_oanda_since_blocking(since_ts: pd.Timestamp, instrument: str = SYMBOL, granularity: str = "M1"):
    """Blocking: fetch recent candles and return completed candles strictly after since_ts."""
    api = oandapyV20.API(access_token=OANDA_ACCESS_TOKEN, environment=OANDA_ENVIRONMENT)
    # request last 200 candles (should be enough); we'll filter by time > since_ts
    params = {"granularity": granularity, "count": 200}
    r = instruments.InstrumentsCandles(instrument=instrument, params=params)
    try:
        api.request(r)
    except Exception as e:
        logger.exception("OANDA API Error when fetching latest candles: %s", e)
        return pd.DataFrame()
    candles = r.response.get('candles', [])
    # keep only complete candles and those after since_ts
    completed = [c for c in candles if c.get('complete', True)]
    df = oanda_candles_to_dataframe(completed)
    if df.empty:
        return df
    # if since_ts is naive or None, return all
    if since_ts is None:
        return df
    # ensure since_ts is Timestamp with tz
    try:
        st = pd.to_datetime(since_ts)
        if st.tzinfo is None:
            st = st.tz_localize(timezone.utc)
    except Exception:
        st = pd.to_datetime(since_ts, utc=True)
    new_df = df[df['timestamp'] > st].reset_index(drop=True)
    return new_df

async def fetch_latest_oanda_since(since_ts):
    loop = asyncio.get_running_loop()
    df = await loop.run_in_executor(None, fetch_latest_oanda_since_blocking, since_ts, SYMBOL, TIMEFRAME)
    return df

# ---------------- Plotting helper (optional) ----------------
def plot_trades(price_df: pd.DataFrame, trades_df: pd.DataFrame, save_path: Path = None, show_plot: bool = True):
    if trades_df.empty:
        print("No trades to plot.")
        return
    price_df = price_df.sort_values("timestamp").reset_index(drop=True)
    dates, closes = pd.to_datetime(price_df["timestamp"]), price_df["Close"].astype(float).values
    t_sec = ensure_epoch_seconds(price_df["timestamp"])
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(dates, closes, label="Close", linewidth=0.7)
    for _, row in trades_df.iterrows():
        i_entry = np.searchsorted(t_sec, int(row["entry_time_s"]))
        i_exit = np.searchsorted(t_sec, int(row["exit_time_s"]))
        if i_entry >= len(t_sec): i_entry = len(t_sec) - 1
        if i_exit >= len(t_sec): i_exit = len(t_sec) - 1
        d_entry, p_entry = dates.iloc[i_entry], closes[i_entry]
        d_exit, p_exit = dates.iloc[i_exit], closes[i_exit]
        color = 'lime' if row['pnl_pct'] > 0 else 'red'
        ax.plot([d_entry, d_exit], [p_entry, p_exit], color=color, linestyle='-', linewidth=1)
        if row["side"] == "long":
            ax.scatter(d_entry, p_entry, marker="^", color='lime', zorder=5)
        else:
            ax.scatter(d_entry, p_entry, marker="v", color='red', zorder=5)
    ax.set_title(f"Backtest Trades for {SYMBOL} (UTC)"); ax.set_xlabel("Time (UTC)"); ax.set_ylabel("Price")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
    fig.autofmt_xdate(); plt.grid(True, linestyle='--', alpha=0.5); plt.tight_layout()
    if save_path: fig.savefig(save_path, bbox_inches="tight"); print(f"Saved plot -> {save_path}")
    if show_plot: plt.show()
    plt.close(fig)

# ---------------- Map precision to param dicts ----------------
def _map_precision_to_params(flat: dict):
    long_p = {
        "tp1_pct": flat["long_tp1"],
        "tp2_pct": flat["long_tp2"],
        "tp3_pct": flat["long_tp3"],
        "sl_pct": flat["long_sl"],
        "freeze_period_minutes": int(flat.get("long_freeze", 20)),
        "strength_h30": flat["long_str_h30"],
        "strength_h60": flat["long_str_h60"],
        "strength_h90": flat["long_str_h90"],
        "strength_h180": flat["long_str_h180"],
    }
    short_p = {
        "tp1_pct": flat["short_tp1"],
        "tp2_pct": flat["short_tp2"],
        "tp3_pct": flat["short_tp3"],
        "sl_pct": flat["short_sl"],
        "freeze_period_minutes": int(flat.get("short_freeze", 20)),
        "strength_h30": flat["short_str_h30"],
        "strength_h60": flat["short_str_h60"],
        "strength_h90": flat["short_str_h90"],
        "strength_h180": flat["short_str_h180"],
    }
    confirm = {
        "horizon_bars": int(flat["confirm_horizon"]),
        "min_confirm_move_pct": float(flat["confirm_min_move_pct"])
    }
    return long_p, short_p, confirm

# ---------------- Live loop ----------------
async def live_loop(df_history: pd.DataFrame, artifacts: dict):
    """
    Poll OANDA for new M1 candles and run prediction + live signal logic.
    Sends Telegram messages on entries/exits. Attempts to place orders via place_order() if available.
    """
    logger.info("Live loop started.")
    long_p, short_p, confirm = _map_precision_to_params(PRECISION_PARAMS)
    lp = long_p; sp = short_p
    spike_window = int(PRECISION_PARAMS["spike_window"]); spike_z = float(PRECISION_PARAMS["spike_z"])

    position = None
    entry_price = 0.0
    tp_levels = np.zeros(3)
    stop_level = np.nan
    stage = 0
    freeze_until = 0  # epoch seconds

    while True:
        try:
            # sleep until next minute boundary
            now = datetime.now(timezone.utc)
            wait_seconds = 60 - now.second
            await asyncio.sleep(max(1, wait_seconds))

            if df_history is None or df_history.empty:
                logger.warning("No history available; sleeping.")
                continue

            last_ts = df_history['timestamp'].iloc[-1]
            new_df = await fetch_latest_oanda_since(last_ts)
            if not new_df.empty:
                # append new candles
                df_history = pd.concat([df_history, new_df], ignore_index=True)
                if len(df_history) > HISTORY_SIZE:
                    df_history = df_history.iloc[-HISTORY_SIZE:].reset_index(drop=True)
                else:
                    df_history.reset_index(drop=True, inplace=True)
                logger.info("Appended %d new candles. History length: %d", len(new_df), len(df_history))

            # run prediction on the full history in background thread (heavy)
            loop = asyncio.get_running_loop()
            results_df = await loop.run_in_executor(None, run_prediction, df_history.copy(), artifacts)
            if results_df.empty:
                logger.debug("Prediction returned empty.")
                continue

            idx_check = len(results_df) - 1
            if idx_check < int(confirm["horizon_bars"]):
                continue

            candle = results_df.iloc[idx_check]
            current_price = float(candle['Close'])
            current_time = int(candle['timestamp'].timestamp())
            utc_time_str = datetime.fromtimestamp(current_time, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')

            # manage existing position (TP/SL progression)
            exit_reason = None
            if position:
                if position == "long":
                    if stage == 0:
                        if current_price >= tp_levels[0]:
                            stage = 1; stop_level = tp_levels[0]
                            await send_to_allowed(f"✅ TP1 HIT (LONG). SL -> BE @ {entry_price:.2f}\n🕔 {utc_time_str}")
                        elif current_price <= stop_level:
                            exit_reason = "SL"
                    elif stage == 1:
                        if current_price <= stop_level:
                            exit_reason = "BE@TP1"
                        elif current_price >= tp_levels[1]:
                            stage = 2; stop_level = tp_levels[1]
                            await send_to_allowed(f"✅ TP2 HIT (LONG). SL -> TP1 @ {tp_levels[0]:.2f}\n🕔 {utc_time_str}")
                    elif stage == 2:
                        if current_price <= stop_level:
                            exit_reason = "BE@TP2"
                        elif current_price >= tp_levels[2]:
                            exit_reason = "TP3"; exit_price = tp_levels[2]
                else:  # short
                    if stage == 0:
                        if current_price <= tp_levels[0]:
                            stage = 1; stop_level = tp_levels[0]
                            await send_to_allowed(f"✅ TP1 HIT (SHORT). SL -> BE @ {entry_price:.2f}\n🕔 {utc_time_str}")
                        elif current_price >= stop_level:
                            exit_reason = "SL"
                    elif stage == 1:
                        if current_price >= stop_level:
                            exit_reason = "BE@TP1"
                        elif current_price <= tp_levels[1]:
                            stage = 2; stop_level = tp_levels[1]
                            await send_to_allowed(f"✅ TP2 HIT (SHORT). SL -> TP1 @ {tp_levels[0]:.2f}\n🕔 {utc_time_str}")
                    elif stage == 2:
                        if current_price >= stop_level:
                            exit_reason = "BE@TP2"
                        elif current_price <= tp_levels[2]:
                            exit_reason = "TP3"; exit_price = tp_levels[2]

                if exit_reason:
                    exit_price = current_price if 'exit_price' not in locals() else exit_price
                    pnl = (exit_price - entry_price) / entry_price * 100.0 if position == "long" else (entry_price - exit_price) / entry_price * 100.0
                    icon = "🟢" if pnl >= 0 else "🔴"
                    await send_to_allowed(f"{icon} **EXIT {position.upper()}** at `{exit_price:.2f}` due to **{exit_reason}**\n**PNL:** `{pnl:.2f}%`\n🕔 {utc_time_str}")
                    freeze_mins = lp['freeze_period_minutes'] if position == 'long' else sp['freeze_period_minutes']
                    freeze_until = current_time + int(freeze_mins) * 60
                    position = None
                    stage = 0

            # check new signals (only if not in freeze and no open position)
            if position is None and current_time >= freeze_until:
                sig_idx = idx_check - int(confirm["horizon_bars"])
                if sig_idx < 0:
                    continue
                signal_candle = results_df.iloc[sig_idx]
                signal_index_label = results_df.index[sig_idx]

                # require signal candle to be a spike
                spike_idx = set(detect_volume_spikes(results_df, spike_window, spike_z))
                if signal_index_label not in spike_idx:
                    logger.debug("Signal candle at idx %s is not a spike; skip", signal_index_label)
                    continue

                long_sig = ((signal_candle.get("pred_up_h30_q0p9_pct", 0) > lp["strength_h30"]) or
                            (signal_candle.get("pred_up_h60_q0p9_pct", 0) > lp["strength_h60"]) or
                            (signal_candle.get("pred_up_h90_q0p9_pct", 0) > lp["strength_h90"]) or
                            (signal_candle.get("pred_up_h180_q0p9_pct", 0) > lp["strength_h180"])) and (signal_candle.get("trend_up", 0) > 0)

                short_sig = ((signal_candle.get("pred_dn_h30_q0p1_pct", 0) > sp["strength_h30"]) or
                             (signal_candle.get("pred_dn_h60_q0p1_pct", 0) > sp["strength_h60"]) or
                             (signal_candle.get("pred_dn_h90_q0p1_pct", 0) > sp["strength_h90"]) or
                             (signal_candle.get("pred_dn_h180_q0p1_pct", 0) > sp["strength_h180"])) and (signal_candle.get("trend_dn", 0) > 0)

                chosen_side = None
                if long_sig and not short_sig:
                    chosen_side = "long"
                elif short_sig and not long_sig:
                    chosen_side = "short"
                elif long_sig and short_sig:
                    chosen_side = "long" if signal_candle.get("trend_up", 0) > signal_candle.get("trend_dn", 0) else "short"

                if chosen_side:
                    # confirmation: movement from signal_candle close to current price over horizon
                    future_ret_pct = (current_price - signal_candle['Close']) / signal_candle['Close'] * 100.0
                    confirmed = False
                    if chosen_side == "long" and future_ret_pct >= confirm["min_confirm_move_pct"]:
                        confirmed = True
                    if chosen_side == "short" and future_ret_pct <= -confirm["min_confirm_move_pct"]:
                        confirmed = True

                    if confirmed:
                        position = chosen_side
                        entry_price = current_price
                        stage = 0
                        if position == "long":
                            tp_levels[0] = entry_price * (1 + lp["tp1_pct"] / 100)
                            tp_levels[1] = entry_price * (1 + lp["tp2_pct"] / 100)
                            tp_levels[2] = entry_price * (1 + lp["tp3_pct"] / 100)
                            stop_level = entry_price * (1 - lp["sl_pct"] / 100)
                        else:
                            tp_levels[0] = entry_price * (1 - sp["tp1_pct"] / 100)
                            tp_levels[1] = entry_price * (1 - sp["tp2_pct"] / 100)
                            tp_levels[2] = entry_price * (1 - sp["tp3_pct"] / 100)
                            stop_level = entry_price * (1 + sp["sl_pct"] / 100)

                        # attempt to place order if helper present
                        if _HAS_PLACE_ORDER:
                            try:
                                # NOTE: adapt args to your place_order signature
                                place_order(asset="XAUUSD", volume="0.1",
                                            sl=str(round(stop_level, 2)),
                                            tp="",
                                            is_buy=str(True if position == 'long' else False))
                                await send_to_allowed(f"🚀 **ENTER {position.upper()}** @ `{entry_price:.2f}` (order placed)\n**SL:** `{stop_level:.2f}`\n🕔 {utc_time_str}")
                            except Exception as e:
                                logger.exception("Order placement failed: %s", e)
                                await send_to_allowed(f"🚀 **ENTER {position.upper()}** @ `{entry_price:.2f}` (order placement FAILED - see logs)\n**SL:** `{stop_level:.2f}`\n🕔 {utc_time_str}")
                        else:
                            await send_to_allowed(f"🚀 **ENTER {position.upper()}** @ `{entry_price:.2f}`\n**SL:** `{stop_level:.2f}` | **TP1:** `{tp_levels[0]:.2f}`\n🕔 {utc_time_str} (order NOT placed - place_order not available)")
        except Exception:
            logger.exception("Exception in live loop")
            # keep running unless critical
            await asyncio.sleep(5)

# ---------------- Main: load artifacts, run 7-day pre-backtest, start live ----------------
async def main():
    global tg_app
    # start telegram
    try:
        await tg_start()
    except Exception:
        logger.exception("Telegram start failed; continuing without TG (messages will not be sent).")

    # load quant artifacts
    qm = QUANT_META_PATH
    if not qm.exists():
        msg = f"Quant meta JSON not found: {qm}"
        logger.error(msg)
        try:
            await send_to_allowed(f"❌ {msg}")
        except Exception:
            pass
        return

    try:
        with open(qm, "r") as f:
            meta = json.load(f)
    except Exception:
        logger.exception("Failed to read quant meta JSON.")
        await send_to_allowed("❌ Failed to read quant meta JSON.")
        return

    out_dir = qm.parent
    try:
        artifacts = {
            'scaler_quant': joblib.load(out_dir / meta["artifacts"]["scaler"]),
            'feats_quant': joblib.load(out_dir / meta["artifacts"]["features"])
        }
    except Exception:
        logger.exception("Failed to load scaler/features artifacts", exc_info=True)
        await send_to_allowed("❌ Failed to load scaler/features artifacts.")
        return

    model_path = out_dir / meta["artifacts"]["model"]
    model_blob = load_model_blob(model_path)
    if model_blob is None:
        await send_to_allowed(f"❌ Failed to load model checkpoint: {model_path}")
        return

    try:
        model_q = MLP(in_dim=model_blob['in_dim'], out_dim=model_blob['out_dim'],
                      hidden=model_blob.get('hidden', 512), depth=model_blob.get('depth', 3),
                      dropout=model_blob.get('dropout', 0.1)).to(DEVICE)
        if 'state_dict' in model_blob:
            model_q.load_state_dict(model_blob['state_dict'])
        else:
            try:
                model_q.load_state_dict(model_blob)
            except Exception:
                logger.warning("Model blob didn't contain loadable state_dict; attempting fallback.")
        artifacts['model_quant'] = model_q.eval()
        artifacts['quant_spec'] = model_blob.get('spec', getattr(model_blob, 'spec', None))
    except Exception:
        logger.exception("Failed to construct model from blob", exc_info=True)
        await send_to_allowed("❌ Failed to construct model from checkpoint.")
        return

    await send_to_allowed("✅ Models & artifacts loaded. Fetching 7 days of historical data for pre-backtest...")

    # fetch 7 days history (blocking in thread)
    try:
        df_history = await fetch_historical_data_oanda(PREBACKTEST_DAYS)
    except Exception:
        logger.exception("Failed to fetch historical data.")
        await send_to_allowed("❌ Failed to fetch historical data from OANDA.")
        return

    if df_history is None or df_history.empty:
        await send_to_allowed("❌ Historical fetch returned empty.")
        return

    await send_to_allowed(f"✅ Fetched {len(df_history)} rows (~{PREBACKTEST_DAYS} days). Running pre-backtest...")

    # prediction and spike filter
    loop = asyncio.get_running_loop()
    preds = await loop.run_in_executor(None, run_prediction, df_history.copy(), artifacts)
    if preds.empty:
        await send_to_allowed("❌ Prediction returned empty on pre-backtest data.")
        return

    spike_idx = detect_volume_spikes(preds, int(PRECISION_PARAMS['spike_window']), float(PRECISION_PARAMS['spike_z']))
    preds_used = preds.loc[spike_idx].reset_index(drop=True)
    await send_to_allowed(f"ℹ️ Pre-backtest: running on {len(preds_used)} spike rows out of {len(preds)} rows.")

    # map params and run backtest
    long_params = {
        "tp1_pct": PRECISION_PARAMS["long_tp1"],
        "tp2_pct": PRECISION_PARAMS["long_tp2"],
        "tp3_pct": PRECISION_PARAMS["long_tp3"],
        "sl_pct": PRECISION_PARAMS["long_sl"],
        "freeze_period_minutes": PRECISION_PARAMS["long_freeze"],
        "strength_h30": PRECISION_PARAMS["long_str_h30"],
        "strength_h60": PRECISION_PARAMS["long_str_h60"],
        "strength_h90": PRECISION_PARAMS["long_str_h90"],
        "strength_h180": PRECISION_PARAMS["long_str_h180"],
    }
    short_params = {
        "tp1_pct": PRECISION_PARAMS["short_tp1"],
        "tp2_pct": PRECISION_PARAMS["short_tp2"],
        "tp3_pct": PRECISION_PARAMS["short_tp3"],
        "sl_pct": PRECISION_PARAMS["short_sl"],
        "freeze_period_minutes": PRECISION_PARAMS["short_freeze"],
        "strength_h30": PRECISION_PARAMS["short_str_h30"],
        "strength_h60": PRECISION_PARAMS["short_str_h60"],
        "strength_h90": PRECISION_PARAMS["short_str_h90"],
        "strength_h180": PRECISION_PARAMS["short_str_h180"],
    }
    confirm = {
        "horizon_bars": int(PRECISION_PARAMS["confirm_horizon"]),
        "min_confirm_move_pct": float(PRECISION_PARAMS["confirm_min_move_pct"])
    }

    summary, trades = await loop.run_in_executor(None, backtest_fast_numpy, preds_used.copy(), long_params, short_params, confirm)

    # send summary
    text = (
        f"📊 **Pre-live Backtest Result** (last {PREBACKTEST_DAYS} days)\n\n"
        f"**Total Trades:** `{summary['total_trades']}`\n"
        f" - **Long:** `{summary['long_trades']}` | **Short:** `{summary['short_trades']}`\n"
        f"**Win Rate:** `{summary['win_rate']:.2f}%`\n"
        f"**Total PnL:** `{summary['total_pnl']:.2f}%`\n"
        f"**Avg PnL:** `{summary['avg_pnl']:.4f}%`\n"
    )
    await send_to_allowed(text)

    # save trades CSV
    out_dir = Path("output_trades"); out_dir.mkdir(exist_ok=True, parents=True)
    ts_name = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    fname = out_dir / f"prebacktest_trades_{ts_name}.csv"
    if not trades.empty:
        trades = trades.copy()
        trades['entry_time_utc'] = pd.to_datetime(trades['entry_time_s'], unit='s', utc=True).dt.strftime('%Y-%m-%dT%H:%M:%SZ')
        trades['exit_time_utc'] = pd.to_datetime(trades['exit_time_s'], unit='s', utc=True).dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    trades.to_csv(fname, index=False)
    await send_to_allowed(f"💾 Saved prebacktest trades CSV: `{fname}`")

    # optionally save plot
    try:
        plot_path = out_dir / f"prebacktest_trades_plot_{ts_name}.png"
        await loop.run_in_executor(None, plot_trades, preds_used, trades, plot_path, False)
    except Exception:
        logger.debug("Plotting prebacktest failed (non-fatal).")

    # start live loop
    await send_to_allowed("▶️ Starting live loop now (precision mode).")
    # run live_loop in background
    live_task = asyncio.create_task(live_loop(df_history.copy(), artifacts))

    # keep main alive; monitor live task
    try:
        while True:
            if live_task.done():
                logger.warning("Live task finished unexpectedly.")
                await send_to_allowed("⚠️ Live loop stopped unexpectedly.")
                break
            await asyncio.sleep(60)
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Shutdown requested by user.")
    finally:
        try:
            live_task.cancel()
            await tg_stop()
        except Exception:
            pass

# ---------------- Entrypoint ----------------
if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    except Exception:
        logger.exception("Fatal error in main run.")
