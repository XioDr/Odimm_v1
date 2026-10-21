# -*- coding: utf-8 -*-
"""
Spike-Only Backtester - INFERENCE-ONLY SCRIPT
---------------------------------------------
This script loads pre-trained models and parameters from an artifact directory
and runs a backtest on new data. It does NOT perform any training or tuning.
"""
import os
import warnings
import numpy as np
import pandas as pd
import json
from datetime import datetime

# Optional imports for loading models and plotting
try:
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    import joblib
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    MATPLOTLIB_OK = True
except ImportError:
    MATPLOTLIB_OK = False

# ======================================================================
#  >>>>>>>>>  تنظیمات اصلی: این دو مسیر را ویرایش کنید  <<<<<<<<<<<
# ======================================================================

# 1. مسیر فایل CSV جدیدی که می‌خواهید روی آن تست کنید
NEW_DATA_CSV_PATH = r"quantile_pct_results.csv"

# 2. مسیر پوشه‌ای که آرتیفکت‌های شما (مدل‌ها و پارامترها) در آن ذخیره شده
ARTIFACTS_PATH = r"D:\project\Odimm_v1\tuning_artifacts(NAS100USD_M1)\run_20250921_235806"# مثال: این مسیر را با مسیر واقعی جایگزین کنید

# 3. پوشه خروجی برای ذخیره نتایج معاملات جدید
OUTPUT_DIR = "output_trades_new_data"

# ======================================================================

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
SEED = 42
np.random.seed(SEED)

# =========================
# Utilities & Backtest Function (کپی شده از اسکریپت اصلی)
# =========================
def ensure_epoch_seconds(series: pd.Series) -> np.ndarray:
    s = series.copy()
    if np.issubdtype(s.dtype, np.number):
        arr = s.astype("int64").to_numpy()
        if arr.max() > 10**12:
            arr = arr // 1000
        return arr
    s_dt = pd.to_datetime(s, errors="coerce", utc=True)
    if s_dt.isna().all():
        raise ValueError("timestamp could not be parsed.")
    return (s_dt.astype('int64') // 10**9).to_numpy()

def compute_trends(df: pd.DataFrame) -> pd.DataFrame:
    up_cols = [f'pred_up_h{h}_q0p9_pct' for h in [30, 60, 90, 180]]
    dn_cols = [f'pred_dn_h{h}_q0p1_pct' for h in [30, 60, 90, 180]]
    if missing := [c for c in up_cols + dn_cols if c not in df.columns]:
        raise KeyError(f"Missing prediction columns: {missing}")
    out = df.copy()
    out["trend_up"] = out[up_cols].mean(axis=1)
    out["trend_dn"] = out[dn_cols].mean(axis=1)
    return out

def detect_volume_spikes(data: pd.DataFrame, window: int, z_thresh: float) -> pd.Index:
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

def build_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ret1"] = out["Close"].pct_change().fillna(0.0)
    out["hl_range_pct"] = (out["High"] - out["Low"]).div(out["Open"].replace(0, np.nan)).fillna(0.0)
    out["vol_roll"] = out["ret1"].rolling(48, min_periods=8).std().fillna(0.0)
    feat_cols = [
        "trend_up", "trend_dn", "hl_range_pct", "vol_roll",
        'pred_up_h30_q0p9_pct', 'pred_up_h60_q0p9_pct', 'pred_up_h90_q0p9_pct', 'pred_up_h180_q0p9_pct',
        'pred_dn_h30_q0p1_pct', 'pred_dn_h60_q0p1_pct', 'pred_dn_h90_q0p1_pct', 'pred_dn_h180_q0p1_pct'
    ]
    missing = [c for c in feat_cols if c not in out.columns]
    if missing:
        raise KeyError(f"Missing columns for regime features: {missing}")
    return out[feat_cols]

def make_ml_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in [
        'pred_up_h30_q0p9_pct', 'pred_up_h60_q0p9_pct', 'pred_up_h90_q0p9_pct', 'pred_up_h180_q0p9_pct',
        'pred_dn_h30_q0p1_pct', 'pred_dn_h60_q0p1_pct', 'pred_dn_h90_q0p1_pct', 'pred_dn_h180_q0p1_pct',
        'trend_up', 'trend_dn'
    ]:
        out[c] = df[c].astype(float)
    out["range_pct"] = (df["High"] - df["Low"]).div(df["Open"].replace(0, np.nan)).fillna(0.0)
    out["ret1"] = df["Close"].pct_change().fillna(0.0)
    out["ret3"] = df["Close"].pct_change(3).fillna(0.0)
    out["ret6"] = df["Close"].pct_change(6).fillna(0.0)
    return out

def backtest_fast_numpy(
    df: pd.DataFrame,
    long_params: dict,
    short_params: dict,
    confirm_params: dict,
    regime_params: dict | None = None,
    proba_filter: dict | None = None,
):
    # This function is unchanged from your original script
    required = ["timestamp", "Close", "Open", "High", "Low", "trend_up", "trend_dn"]
    up_cols = [f'pred_up_h{h}_q0p9_pct' for h in [30, 60, 90, 180]]
    dn_cols = [f'pred_dn_h{h}_q0p1_pct' for h in [30, 60, 90, 180]]
    required.extend(up_cols); required.extend(dn_cols)
    if miss := [c for c in required if c not in df.columns]:
        raise KeyError(f"Missing columns: {miss}")

    trade_cols = ["entry_time_s", "exit_time_s", "side", "entry", "exit", "pnl_pct", "reason", "regime"]
    if df.empty:
        return ({"total_trades": 0, "long_trades": 0, "short_trades": 0, "win_rate": 0.0, "total_pnl": 0.0, "avg_pnl": 0.0},
                pd.DataFrame(columns=trade_cols))

    t = ensure_epoch_seconds(df["timestamp"])
    close = df["Close"].to_numpy(float)
    open_p = df["Open"].to_numpy(float)
    high = df["High"].to_numpy(float)
    low = df["Low"].to_numpy(float)
    up30, up60, up90, up180 = (df[c].to_numpy(float) for c in up_cols)
    dn30, dn60, dn90, dn180 = (df[c].to_numpy(float) for c in dn_cols)
    tr_up, tr_dn = df["trend_up"].to_numpy(float), df["trend_dn"].to_numpy(float)
    regimes = df["regime"].to_numpy(int) if "regime" in df.columns else np.zeros(len(df), dtype=int)

    n = len(close)
    position, entry_price, entry_time_s, active_regime = None, 0.0, 0, None
    tp_levels, stop_level, stage, freeze_until = np.zeros(3, dtype=float), np.nan, 0, -1
    trades, scheduled_entries = [], {}

    base_lp = long_params.copy()
    base_sp = short_params.copy()
    H = int(confirm_params.get("horizon_bars", 10))
    min_move = float(confirm_params.get("min_confirm_move_pct", 0.0))

    p_up = proba_filter.get('p_up') if proba_filter else None
    p_dn = proba_filter.get('p_dn') if proba_filter else None
    min_prob_long = float(proba_filter.get('min_prob_long', 0.55)) if proba_filter else 0.0 # Default value
    min_prob_short = float(proba_filter.get('min_prob_short', 0.55)) if proba_filter else 0.0 # Default value

    def pick_params(reg_idx: int):
        if regime_params and reg_idx in regime_params:
            return regime_params[reg_idx]["long"], regime_params[reg_idx]["short"]
        return base_lp, base_sp

    def eval_long_sig(i, lp):
        return (
            (up30[i] > lp["strength_h30"]) or
            (up60[i] > lp["strength_h60"]) or
            (up90[i] > lp["strength_h90"]) or
            (up180[i] > lp["strength_h180"])
        ) and (tr_up[i] > 0)

    def eval_short_sig(i, sp):
        return (
            (dn30[i] > sp["strength_h30"]) or
            (dn60[i] > sp["strength_h60"]) or
            (dn90[i] > sp["strength_h90"]) or
            (dn180[i] > sp["strength_h180"])
        ) and (tr_dn[i] > 0)

    freeze_long_s_default = int(base_lp.get("freeze_period_minutes", 20)) * 60
    freeze_short_s_default = int(base_sp.get("freeze_period_minutes", 20)) * 60

    for i in range(n):
        ts = int(t[i])
        reg = int(regimes[i])
        lp, sp = pick_params(reg)
        freeze_long_s = int(lp.get("freeze_period_minutes", freeze_long_s_default)) * 60
        freeze_short_s = int(sp.get("freeze_period_minutes", freeze_short_s_default)) * 60

        if position is None and i in scheduled_entries and ts >= freeze_until:
            side = scheduled_entries.pop(i)
            entry_price = open_p[i] if not np.isnan(open_p[i]) else float(close[i])
            entry_time_s = ts
            stage = 0
            active_regime = reg
            if side == "long":
                position = "long"
                tp_levels = entry_price * (1.0 + np.array([lp["tp1_pct"], lp["tp2_pct"], lp["tp3_pct"]]) / 100.0)
                stop_level = entry_price * (1.0 - lp["sl_pct"] / 100.0)
            else:
                position = "short"
                tp_levels = entry_price * (1.0 - np.array([sp["tp1_pct"], sp["tp2_pct"], sp["tp3_pct"]]) / 100.0)
                stop_level = entry_price * (1.0 + sp["sl_pct"] / 100.0)

        if ts >= freeze_until and position is None:
            l_sig = eval_long_sig(i, lp)
            s_sig = eval_short_sig(i, sp)

            if proba_filter is not None:
                if l_sig and (p_up is not None) and not (p_up[i] >= min_prob_long):
                    l_sig = False
                if s_sig and (p_dn is not None) and not (p_dn[i] >= min_prob_short):
                    s_sig = False

            if l_sig or s_sig:
                chosen_side = None
                if l_sig and not s_sig:
                    chosen_side = "long"
                elif s_sig and not l_sig:
                    chosen_side = "short"
                elif l_sig and s_sig:
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

        if position:
            if position == "long":
                if low[i] <= stop_level:
                    exit_price = stop_level
                    reason = "SL"
                    if stage == 1: reason = "TSL@TP1"
                    if stage == 2: reason = "TSL@TP2"
                    pnl = (exit_price - entry_price) / entry_price * 100.0
                    trades.append((entry_time_s, ts, "long", entry_price, exit_price, pnl, reason, active_regime))
                    position, freeze_until = None, ts + freeze_long_s
                    continue
                if stage == 0 and high[i] >= tp_levels[0]:
                    stage = 1
                    stop_level = tp_levels[0]
                if stage == 1 and high[i] >= tp_levels[1]:
                    stage = 2
                    stop_level = tp_levels[1]
                if stage == 2 and high[i] >= tp_levels[2]:
                    exit_price = tp_levels[2]
                    pnl = (exit_price - entry_price) / entry_price * 100.0
                    trades.append((entry_time_s, ts, "long", entry_price, exit_price, pnl, "TP3", active_regime))
                    position, freeze_until = None, ts + freeze_long_s
                    continue
            else: # short
                if high[i] >= stop_level:
                    exit_price = stop_level
                    reason = "SL"
                    if stage == 1: reason = "TSL@TP1"
                    if stage == 2: reason = "TSL@TP2"
                    pnl = (entry_price - exit_price) / entry_price * 100.0
                    trades.append((entry_time_s, ts, "short", entry_price, exit_price, pnl, reason, active_regime))
                    position, freeze_until = None, ts + freeze_short_s
                    continue
                if stage == 0 and low[i] <= tp_levels[0]:
                    stage = 1
                    stop_level = tp_levels[0]
                if stage == 1 and low[i] <= tp_levels[1]:
                    stage = 2
                    stop_level = tp_levels[1]
                if stage == 2 and low[i] <= tp_levels[2]:
                    exit_price = tp_levels[2]
                    pnl = (entry_price - exit_price) / entry_price * 100.0
                    trades.append((entry_time_s, ts, "short", entry_price, exit_price, pnl, "TP3", active_regime))
                    position, freeze_until = None, ts + freeze_short_s
                    continue

    if not trades:
        return ({"total_trades": 0, "long_trades": 0, "short_trades": 0, "win_rate": 0.0, "total_pnl": 0.0, "avg_pnl": 0.0}, pd.DataFrame(columns=trade_cols))

    trades_df = pd.DataFrame(trades, columns=trade_cols)
    pnls = trades_df["pnl_pct"].astype(float)
    summary = {
        "total_trades": len(trades),
        "long_trades": (trades_df["side"] == "long").sum(),
        "short_trades": (trades_df["side"] == "short").sum(),
        "win_rate": (pnls > 0).mean() * 100.0 if len(trades) > 0 else 0.0,
        "total_pnl": pnls.sum(),
        "avg_pnl": pnls.mean() if len(trades) > 0 else 0.0,
    }
    return summary, trades_df

def create_param_dicts(params: dict):
    long_p = {k.replace('long_', ''): v for k, v in params.items() if k.startswith('long_')}
    short_p = {k.replace('short_', ''): v for k, v in params.items() if k.startswith('short_')}
    confirm = {"horizon_bars": params["confirm_horizon"], "min_confirm_move_pct": params["confirm_min_move_pct"]}
    for p_dict, prefix in [(long_p, "long_"), (short_p, "short_")]:
        p_dict['sl_pct'] = params[f'{prefix}sl']
        p_dict['tp1_pct'] = params[f'{prefix}tp1']
        p_dict['tp2_pct'] = params[f'{prefix}tp2']
        p_dict['tp3_pct'] = params[f'{prefix}tp3']
        p_dict['freeze_period_minutes'] = params[f'{prefix}freeze']
        for h in [30, 60, 90, 180]:
            p_dict[f'strength_h{h}'] = params[f'{prefix}str_h{h}']
    return long_p, short_p, confirm

def plot_daily_win_rate_from_trades(trades_df: pd.DataFrame, output_dir: str, suffix: str=""):
    if not MATPLOTLIB_OK:
        print("\nmatplotlib not found, skipping plot. Install with: pip install matplotlib")
        return
    if trades_df.empty or 'exit_time_utc' not in trades_df.columns:
        print(f"No trade data available to generate a daily win rate plot for '{suffix}'.")
        return
    df = trades_df.copy()
    df['exit_date'] = pd.to_datetime(df['exit_time_utc']).dt.date
    df['is_win'] = df['pnl_pct'] > 0
    daily_summary = df.groupby('exit_date').agg(total_trades=('is_win', 'count'), wins=('is_win','sum'))
    daily_summary['win_rate'] = (daily_summary['wins'] / daily_summary['total_trades']) * 100.0
    daily_summary.index = pd.to_datetime(daily_summary.index)
    plot_data = daily_summary.tail(30)
    if plot_data.empty:
        print("Not enough daily trade data to create a plot.")
        return
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(15,7))
    ax.plot(plot_data.index, plot_data['win_rate'], marker='o', linestyle='-', label='Daily Win Rate')
    for date, row in plot_data.iterrows():
        wr = row['win_rate']
        ax.text(date, wr + 1.5, f'{wr:.1f}%', ha='center', va='bottom', fontsize=9)
    title_suffix = f" ({suffix})" if suffix else ""
    ax.set_title(f'Daily Win Rate{title_suffix} (Last {len(plot_data)} Trading Days)', fontsize=16)
    ax.set_xlabel('Date', fontsize=12); ax.set_ylabel('Win Rate (%)', fontsize=12)
    ax.set_ylim(0,105); ax.axhline(y=50, color='r', linestyle='--', linewidth=0.8, label='50% Mark'); ax.legend(); ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    interval = max(1, len(plot_data) // 15)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=interval))
    fig.autofmt_xdate(rotation=45)
    plot_filename = os.path.join(output_dir, f'daily_win_rate_{suffix}.png')
    try:
        plt.savefig(plot_filename, bbox_inches='tight', dpi=150)
        print(f"\n📊 Plot of daily win rate saved to: {plot_filename}")
        plt.show()
    except Exception as e:
        print(f"Could not save or show plot. Error: {e}")
    plt.close(fig)


# =========================
# Main Execution (فقط بارگذاری و تست)
# =========================
def run_inference():
    """
    Loads artifacts and runs the backtest on new data.
    """
    if not os.path.isdir(ARTIFACTS_PATH):
        print(f"❌ Error: Artifact directory not found at: {ARTIFACTS_PATH}")
        return

    # --- 1. Load Data ---
    print(f"🔄 Loading new data from: {NEW_DATA_CSV_PATH}")
    try:
        df_full = pd.read_csv(NEW_DATA_CSV_PATH)
    except FileNotFoundError:
        print(f"❌ Error: Data file not found at: {NEW_DATA_CSV_PATH}")
        return
    print(f"✅ Loaded {len(df_full)} rows of new data.")

    # --- 2. Pre-process Data ---
    if "Close" not in df_full.columns and "close" in df_full.columns:
        df_full.rename(columns={"close": "Close"}, inplace=True)
    for c in ["Open", "High", "Low", "Volume"]:
        if c not in df_full.columns and c.lower() in df_full.columns:
            df_full.rename(columns={c.lower(): c}, inplace=True)
    df_full = compute_trends(df_full)

    # --- 3. Load Artifacts ---
    print(f"\n🔄 Loading artifacts from: {ARTIFACTS_PATH}")
    
    # Load global parameters
    params_file = os.path.join(ARTIFACTS_PATH, "global_params.json")
    try:
        with open(params_file, "r") as f:
            best_global_params = json.load(f)
        print("✅ Loaded global parameters.")
    except FileNotFoundError:
        print(f"❌ Error: 'global_params.json' not found in artifacts folder.")
        return

    # Load other artifacts (GMM, ML models, regime params) if they exist
    gmm_obj, gmm_scaler = None, None
    ml_models_tuple = None
    regime_overrides = None

    if SKLEARN_OK:
        try:
            gmm_obj = joblib.load(os.path.join(ARTIFACTS_PATH, "gmm_model.joblib"))
            gmm_scaler = joblib.load(os.path.join(ARTIFACTS_PATH, "gmm_scaler.joblib"))
            print("✅ Loaded GMM model and scaler.")
        except FileNotFoundError:
            print("- GMM model not found, will use a single regime.")

        try:
            up_model = joblib.load(os.path.join(ARTIFACTS_PATH, "ml_up_model.joblib"))
            dn_model = joblib.load(os.path.join(ARTIFACTS_PATH, "ml_dn_model.joblib"))
            scaler_ml = joblib.load(os.path.join(ARTIFACTS_PATH, "ml_scaler.joblib"))
            ml_models_tuple = (up_model, dn_model, scaler_ml)
            print("✅ Loaded ML filter models and scaler.")
        except FileNotFoundError:
            print("- ML filter models not found, will run without probability filter.")
    else:
        print("⚠️ Warning: scikit-learn/joblib not installed. Cannot load GMM or ML models.")
        
    try:
        with open(os.path.join(ARTIFACTS_PATH, "regime_params.json"), "r") as f:
            # Keys in JSON are strings, convert them back to integers
            regime_overrides = {int(k): v for k, v in json.load(f).items()}
        print("✅ Loaded per-regime parameters.")
    except FileNotFoundError:
        print("- Per-regime parameters not found, will use global params for all regimes.")

    # --- 4. Prepare Data for Backtest ---
    print("\n⚙️ Preparing data for backtest...")
    
    # Filter data based on volume spikes
    idx_spikes = detect_volume_spikes(df_full, best_global_params["spike_window"], best_global_params["spike_z"])
    df_used = df_full.loc[idx_spikes].reset_index(drop=True)
    print(f"Found {len(df_used)} volume spikes to test.")

    # Assign regimes using the loaded GMM model
    if gmm_obj and gmm_scaler:
        X_regime = build_regime_features(df_used)
        Xs_regime = gmm_scaler.transform(X_regime.values)
        df_used["regime"] = gmm_obj.predict(Xs_regime)
        print(f"Assigned data to {gmm_obj.n_components} regimes.")
    else:
        df_used["regime"] = 0 # Default to regime 0 if no model

    # Calculate probabilities using the loaded ML models
    proba_filter = None
    if ml_models_tuple:
        up_model, dn_model, scaler_ml = ml_models_tuple
        X_ml = make_ml_features(df_used)
        Xs_ml = scaler_ml.transform(X_ml.fillna(0.0).values)
        p_up = up_model.predict_proba(Xs_ml)[:, 1]
        p_dn = dn_model.predict_proba(Xs_ml)[:, 1]
        
        # Note: min_prob values are hardcoded here, but could be loaded from manifest.json if saved
        proba_filter = {
            'p_up': p_up, 'p_dn': p_dn,
            'min_prob_long': 0.55,
            'min_prob_short': 0.55,
        }
        print("Calculated trade probabilities using ML filter.")

    # --- 5. Run the Backtest ---
    print("\n🚀 Running backtest on new data...")
    long_p, short_p, confirm = create_param_dicts(best_global_params)
    
    summary, trades_df = backtest_fast_numpy(
        df=df_used,
        long_params=long_p,
        short_params=short_p,
        confirm_params=confirm,
        regime_params=regime_overrides,
        proba_filter=proba_filter,
    )

    # --- 6. Show and Save Results ---
    print("\n" + "="*50)
    print("🏁 BACKTEST COMPLETE - SUMMARY ON NEW DATA 🏁")
    print("="*50)
    print(f"  Total Trades: {summary['total_trades']}")
    print(f"  Long Trades:  {summary['long_trades']}")
    print(f"  Short Trades: {summary['short_trades']}")
    print(f"  Win Rate:     {summary['win_rate']:.2f}%")
    print(f"  Total PnL:    {summary['total_pnl']:.2f}%")
    print(f"  Average PnL:  {summary['avg_pnl']:.4f}%")
    print("="*50)

    if not trades_df.empty:
        trades_df["entry_time_utc"] = pd.to_datetime(trades_df["entry_time_s"], unit='s', utc=True)
        trades_df["exit_time_utc"] = pd.to_datetime(trades_df["exit_time_s"], unit='s', utc=True)
        
        # Save trades to a new CSV file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = os.path.join(OUTPUT_DIR, f"trades_new_data_{timestamp}.csv")
        trades_df.to_csv(output_filename, index=False)
        print(f"\n💾 Saved detailed trades to: {output_filename}")
        
        # Generate plot
        plot_daily_win_rate_from_trades(trades_df, OUTPUT_DIR, suffix="NEW_DATA")

if __name__ == "__main__":
    run_inference()