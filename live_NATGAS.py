# -*- coding: utf-8 -*-
# =========================================================================================
# LIVE TRADING SIMULATOR (Corrected & Simplified Version with Stats Output)
# This script combines data fetching, model inference, and trading logic into a single
# continuous loop to simulate live trading based on 1-minute candle closes.
#
# INTEGRATION: The data fetching function has been replaced to use the OANDA v20 API
# via the oandapyV20 library instead of ccxt.
# =========================================================================================
import os, warnings, joblib, numpy as np, pandas as pd, matplotlib.pyplot as plt, json, asyncio, time
from joblib import Parallel, delayed
import torch, torch.nn as nn
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm
from datetime import datetime, timezone

# --- OANDA API ---
import oandapyV20
import oandapyV20.endpoints.instruments as instruments

# Telegram (Pyrogram)
from pyrogram import Client
from pyrogram.enums import ParseMode

# Optional imports for plotting and models
try:
    from sklearn.mixture import GaussianMixture
    from sklearn.linear_model import LogisticRegression
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False
try:
    import matplotlib.dates as mdates
    MATPLOTLIB_OK = True
except ImportError:
    MATPLOTLIB_OK = False

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)

# ======================================================================
#  bölüm 1: اصلی تنظیمات
# ======================================================================

# --- Data Fetcher Settings (OANDA) ---
# OANDA Credentials
OANDA_ACCESS_TOKEN = "REDACTED" # Your OANDA API access token
OANDA_ACCOUNT_ID = "REDACTED"    # Your OANDA account ID

# Instrument and Timeframe
INSTRUMENT = "NATGAS_USD"      # OANDA instrument name (e.g., 'EUR_USD', 'XAU_USD', 'NATGAS_USD')
GRANULARITY = "M1"             # OANDA granularity (e.g., 'M1', 'M5', 'H1')
TOTAL_CANDLES_TO_FETCH = 7320  # حدود 3 روز داده‌ی دقیقه‌ای برای مهندسی ویژگی
HISTORY_CSV = 'live_history.csv' # فایل برای ذخیره داده‌های کندل

# --- Prediction Model Paths (Script 1) ---
PREDICTION_RESULTS_CSV = "live_combined_predictions.csv" # خروجی مدل پیش‌بینی
# Quantile Model Artifacts
QUANT_META_PATH = "mhq_ann/ann_meta.json"

# --- Trading Logic Paths (Script 2) ---
ARTIFACTS_PATH = r"C:\Users\Administrator\Desktop\Arcx-v3\tuning_artifacts(btc)\run_20250914_180148"
TRADES_OUTPUT_DIR = "live_trades_output"


# Telegram credentials (kept in-file as requested)
API_TOKEN = "REDACTED"
API_ID = 20205931
API_HASH = "REDACTED"

# Allowed Telegram usernames (without @)
TARGET_CHATS = ["saeed_khosh_neshin", "Xio_Dr"]

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
        await tg_app.start()
    except Exception as e:
        print("Failed to start Telegram client: ", e)
        raise

async def send_to_allowed(text: str):
    """Send message to allowed usernames via Telegram."""
    for u in TARGET_CHATS:
        try:
            await tg_app.send_message(f"@{u}", text, disable_web_page_preview=True)
        except Exception as e:
            print("Faild to send message: ",e)


# --- General Config ---
BATCH_SIZE = 1024
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_PARALLEL_STATS = True
os.makedirs(TRADES_OUTPUT_DIR, exist_ok=True)


# ======================================================================
# bölüm 2: توابع کمکی و مدل‌ها (کپی شده از اسکریپت‌های شما)
# ======================================================================

# region <<< بخش ۱: توابع واکشی داده (Data Fetching) - UPDATED FOR OANDA >>>
def fetch_latest_candles():
    """داده‌های کندل اخیر را از OANDA دریافت و در فایل CSV ذخیره می‌کند."""
    print(f"[{datetime.now():%H:%M:%S}] 📥 Fetching last {TOTAL_CANDLES_TO_FETCH} candles for {INSTRUMENT} from OANDA...")

    api = oandapyV20.API(access_token=OANDA_ACCESS_TOKEN, environment="practice")
    params = {
        "granularity": GRANULARITY,
        "count": TOTAL_CANDLES_TO_FETCH + 1,  # Fetch one extra to drop the current incomplete one
        "price": "M",  # Midpoint price
    }
    r = instruments.InstrumentsCandles(instrument=INSTRUMENT, params=params)

    try:
        api.request(r)
        candles_data = r.response.get('candles', [])
        if not candles_data:
            print("❌ No candle data returned from OANDA.")
            return False

        # --- Data Processing and Saving ---
        df = pd.DataFrame(candles_data)
        
        # Check if data is complete
        if 'mid' not in df.columns or 'volume' not in df.columns or 'time' not in df.columns:
            print("❌ OANDA response is missing expected columns ('mid', 'volume', 'time').")
            return False

        # Normalize price columns from 'mid' dictionary
        df = df.join(pd.json_normalize(df['mid'].tolist())).drop(columns=['mid'])

        # Rename and format columns to match the rest of the script
        df['time'] = pd.to_datetime(df['time'])
        df.rename(columns={
            'time': 'timestamp',
            'o': 'Open',
            'h': 'High',
            'l': 'Low',
            'c': 'Close',
            'volume': 'Volume'
        }, inplace=True)
        
        # Ensure correct data types
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Select final columns
        df = df[['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']]
        
        # Drop the last row, which is the current, incomplete candle
        df = df.iloc[:-1]

        df.to_csv(HISTORY_CSV, index=False)
        print(f"✅ Saved {len(df)} closed candles to {HISTORY_CSV} (up to {df.iloc[-1]['timestamp']})")
        return True

    except oandapyV20.exceptions.V20Error as err:
        print(f"❌ OANDA API Error: {err}")
        return False
    except Exception as e:
        print(f"❌ An unexpected error occurred during data fetching: {e}")
        return False
# endregion

# region <<< بخش ۲: توابع مدل پیش‌بینی (Prediction Model) >>>
def _rolling_stats_np(x, w):
    n=x.size
    if n<w or w<=1:
        nan=np.full(n,np.nan); return nan,nan,nan
    c0=np.empty(n+1); c0[0]=0; c0[1:]=np.cumsum(x)
    m_seg=(c0[w:]-c0[:-w])/w; m=np.r_[np.full(w-1,np.nan), m_seg]
    c2=np.empty(n+1); c2[0]=0; c2[1:]=np.cumsum(x*x)
    mean2=(c2[w:]-c2[:-w])/w; var=mean2-m_seg**2
    s=np.r_[np.full(w-1,np.nan), np.sqrt(np.maximum(var,0))]
    t=np.arange(w,dtype=float); tm=t.mean(); denom=np.sum((t-tm)**2) or 1
    sum_xt=np.convolve(x, t[::-1], mode="valid")
    sl=np.r_[np.full(w-1,np.nan), (sum_xt - w*m_seg*tm)/denom]
    return m,s,sl

def create_all_features(df: pd.DataFrame)->pd.DataFrame:
    df=df.copy()
    eps = 1e-8
    df["oc_ret"] = np.log(df["Close"] / (df["Open"] + eps))
    df["hl_range"] = (df["High"] - df["Low"]) / (df["Low"] + eps)
    close=df["Close"].values.astype(float)
    logc=np.log(np.maximum(close,eps)); vol=df["Volume"].values.astype(float)
    R=[15,30,60,90,120]; VW=[5,15,30,60,90,180]
    tasks=[(df[c].values.astype(float),w,c) for c in("oc_ret","hl_range") for w in R]
    if USE_PARALLEL_STATS:
        res=Parallel(n_jobs=max(1,os.cpu_count()//2))(delayed(_rolling_stats_np)(a,w) for a,w,_ in tasks)
    else:
        res=[_rolling_stats_np(a,w) for a,w,_ in tqdm(tasks, desc="FE: Rolling Stats", leave=False)]
    for ((_,w,c),(m,s,sl)) in zip(tasks,res):
        df[f"{c}_rmean_{w}"]=m; df[f"{c}_rstd_{w}"]=s; df[f"{c}_rslope_{w}"]=sl
    for w in R:  df[f"momentum_{w}"]=pd.Series(close).pct_change(w).values
    for w in VW: df[f"ret_{w}"]=pd.Series(logc).diff(w).values
    for w in VW:
        vm=pd.Series(vol).rolling(w).mean().values; vs=pd.Series(vol).rolling(w).std().values
        df[f"vol_z_{w}"]=(vol - vm)/((np.abs(vs))+eps)
    d=pd.Series(close).diff(); up=d.clip(lower=0); down=-d.clip(upper=0)
    rs=up.ewm(alpha=1/14,adjust=False).mean()/(down.ewm(alpha=1/14,adjust=False).mean()+eps)
    df["rsi_14"]=(100-100/(1+rs)).clip(0,100).values
    ma=pd.Series(close).rolling(20).mean(); sd=pd.Series(close).rolling(20).std()
    df["bb_pos"]=((pd.Series(close)-ma)/(2*(sd+eps))).values
    wins = [1, 3, 5, 15, 30, 45, 60, 90]
    for w in wins:
        vm = df["Volume"].rolling(w, min_periods=1).mean()
        df[f"volume_mean_{w}"] = vm; df[f"volume_change_{w}"] = df["Volume"].diff(w)
        df[f"volume_ratio_{w}"] = df["Volume"] / (vm + eps)
        pm = df["Close"].rolling(w, min_periods=1).mean()
        df[f"price_mean_{w}"] = pm; df[f"price_change_past_{w}"] = df["Close"].diff(w)
        df[f"price_ratio_{w}"] = df["Close"] / (pm + eps)
        if w > 1:
            df[f"volume_std_{w}"] = df["Volume"].rolling(w, min_periods=2).std()
            df[f"price_std_{w}"] = df["Close"].rolling(w, min_periods=2).std()
        else:
            df[f"volume_std_{w}"] = 0.0; df[f"price_std_{w}"] = 0.0
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df

class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden=512, depth=3, dropout=0.1):
        super().__init__()
        layers = []; d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout)]; d = hidden
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

def safe_load_state_dict(model, path, device):
    sd = torch.load(path, map_location=device, weights_only=False)
    if isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']
    try:
        model.load_state_dict(sd, strict=True)
    except Exception:
        remap = {k.replace("enc.","encoder.").replace("dec.","decoder."):v for k,v in sd.items()}
        model.load_state_dict(remap, strict=False)
    return model

def run_quantile_prediction(df_fe: pd.DataFrame):
    with open(QUANT_META_PATH, "r") as f: meta = json.load(f)
    out_dir = os.path.dirname(QUANT_META_PATH)
    scaler_quant = joblib.load(os.path.join(out_dir, meta["artifacts"]["scaler"]))
    feats_quant = joblib.load(os.path.join(out_dir, meta["artifacts"]["features"]))
    model_blob = torch.load(os.path.join(out_dir, meta["artifacts"]["model"]), map_location=DEVICE, weights_only=False)
    model_quant = MLP(in_dim=model_blob['in_dim'], out_dim=model_blob['out_dim'], hidden=model_blob['hidden'],
                      depth=model_blob['depth'], dropout=model_blob['dropout']).to(DEVICE)
    model_quant.load_state_dict(model_blob['state_dict']); model_quant.eval()
    missing_feats = set(feats_quant) - set(df_fe.columns)
    if missing_feats: raise ValueError(f"Quantile model missing features: {list(missing_feats)}")
    X_quant = df_fe[feats_quant].values.astype(np.float32)
    X_quant_scaled = scaler_quant.transform(X_quant)
    all_preds = []
    for i in range(0, len(X_quant_scaled), BATCH_SIZE):
        batch = torch.from_numpy(X_quant_scaled[i:i+BATCH_SIZE]).to(DEVICE)
        with torch.no_grad():
            preds_batch = model_quant(batch).cpu().numpy()
        all_preds.append(preds_batch)
    yhat_full = np.concatenate(all_preds, axis=0)
    return yhat_full, model_blob['spec']

def run_prediction_pipeline():
    """خط لوله کامل پیش‌بینی را اجرا کرده و نتایج را در فایل CSV ذخیره می‌کند."""
    print(f"[{datetime.now():%H:%M:%S}] 🚀 Running prediction pipeline...")
    if not os.path.exists(HISTORY_CSV):
        print(f"   - History file '{HISTORY_CSV}' not found. Skipping prediction.")
        return False
        
    df = pd.read_csv(HISTORY_CSV, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    
    print("   - Creating features...")
    df_fe = create_all_features(df)
    if df_fe.empty:
        print("❌ Feature set is empty after engineering. Skipping prediction.")
        return False
    
    print("   - Running Quantile Forecast Model...")
    quantile_preds_matrix, quantile_spec = run_quantile_prediction(df_fe)

    out = df_fe[["timestamp", "Open", "High", "Low", "Close", "Volume"]].copy()
    
    print("   - Merging quantile results...")
    for i, (side, h, q) in enumerate(quantile_spec):
        q_str = str(q).replace('.', 'p')
        col_name_pct = f"pred_{side}_h{h}_q{q_str}_pct"
        out[col_name_pct] = quantile_preds_matrix[:, i]
    
    out.to_csv(PREDICTION_RESULTS_CSV, index=False, float_format='%.4f')
    print(f"✅ Saved prediction results to: {PREDICTION_RESULTS_CSV}")
    return True
# endregion

# region <<< بخش ۳: توابع منطق معاملاتی (Trading Logic) >>>
def ensure_epoch_seconds(series: pd.Series) -> np.ndarray:
    s = series.copy()
    if np.issubdtype(s.dtype, np.number):
        arr = s.astype("int64").to_numpy()
        if arr.max() > 10**12: arr = arr // 1000
        return arr
    s_dt = pd.to_datetime(s, errors="coerce", utc=True)
    if s_dt.isna().all(): raise ValueError("timestamp could not be parsed.")
    return (s_dt.astype('int64') // 10**9).to_numpy()

def compute_trends(df: pd.DataFrame) -> pd.DataFrame:
    up_cols = [f'pred_up_h{h}_q0p9_pct' for h in [30, 60, 90, 180]]
    dn_cols = [f'pred_dn_h{h}_q0p1_pct' for h in [30, 60, 90, 180]]
    if missing := [c for c in up_cols + dn_cols if c not in df.columns]:
        raise KeyError(f"Missing prediction columns: {missing}")
    out = df.copy(); out["trend_up"] = out[up_cols].mean(axis=1); out["trend_dn"] = out[dn_cols].mean(axis=1)
    return out

def detect_volume_spikes(data: pd.DataFrame, window: int, z_thresh: float) -> pd.Index:
    if "Volume" not in data.columns: return data.iloc[0:0].index
    vol = data["Volume"].astype(float); rm = vol.rolling(window, min_periods=1).median().shift(1)
    abs_dev = (vol - rm).abs(); mad = abs_dev.rolling(window, min_periods=1).median().shift(1)
    mad_safe = mad.replace(0, np.nan); robust_z = 0.6745 * (vol - rm) / mad_safe
    return data.index[robust_z > z_thresh]

def build_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ret1"] = out["Close"].pct_change().fillna(0.0)
    out["hl_range_pct"] = (out["High"] - out["Low"]).div(out["Open"].replace(0, np.nan)).fillna(0.0)
    out["vol_roll"] = out["ret1"].rolling(48, min_periods=8).std().fillna(0.0)
    feat_cols = ["trend_up", "trend_dn", "hl_range_pct", "vol_roll"] + \
                [f'pred_up_h{h}_q0p9_pct' for h in [30, 60, 90, 180]] + \
                [f'pred_dn_h{h}_q0p1_pct' for h in [30, 60, 90, 180]]
    if missing := [c for c in feat_cols if c not in out.columns]:
        raise KeyError(f"Missing columns for regime features: {missing}")
    return out[feat_cols]

def make_ml_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    cols_to_copy = [f'pred_up_h{h}_q0p9_pct' for h in [30, 60, 90, 180]] + \
                   [f'pred_dn_h{h}_q0p1_pct' for h in [30, 60, 90, 180]] + \
                   ['trend_up', 'trend_dn']
    for c in cols_to_copy: out[c] = df[c].astype(float)
    out["range_pct"] = (df["High"] - df["Low"]).div(df["Open"].replace(0, np.nan)).fillna(0.0)
    out["ret1"] = df["Close"].pct_change().fillna(0.0)
    out["ret3"] = df["Close"].pct_change(3).fillna(0.0)
    out["ret6"] = df["Close"].pct_change(6).fillna(0.0)
    return out

def backtest_fast_numpy(df: pd.DataFrame, long_params: dict, short_params: dict, confirm_params: dict,
                        regime_params: dict | None = None, proba_filter: dict | None = None, print_details: bool = False):
    required = ["timestamp", "Close", "Open", "High", "Low", "trend_up", "trend_dn"]
    up_cols = [f'pred_up_h{h}_q0p9_pct' for h in [30, 60, 90, 180]]
    dn_cols = [f'pred_dn_h{h}_q0p1_pct' for h in [30, 60, 90, 180]]
    required.extend(up_cols); required.extend(dn_cols)
    if miss := [c for c in required if c not in df.columns]: raise KeyError(f"Missing columns: {miss}")
    trade_cols = ["entry_time_s", "exit_time_s", "side", "entry", "exit", "pnl_pct", "reason", "regime"]
    if df.empty: return ({"total_trades": 0, "win_rate": 0.0, "total_pnl": 0.0}, pd.DataFrame(columns=trade_cols), None)
    t = ensure_epoch_seconds(df["timestamp"])
    close, open_p, high, low = (df[c].to_numpy(float) for c in ["Close", "Open", "High", "Low"])
    up30, up60, up90, up180 = (df[c].to_numpy(float) for c in up_cols)
    dn30, dn60, dn90, dn180 = (df[c].to_numpy(float) for c in dn_cols)
    tr_up, tr_dn = df["trend_up"].to_numpy(float), df["trend_dn"].to_numpy(float)
    regimes = df["regime"].to_numpy(int) if "regime" in df.columns else np.zeros(len(df), dtype=int)
    n = len(close)
    position, entry_price, entry_time_s, active_regime = None, 0.0, 0, None
    tp_levels, stop_level, stage, freeze_until = np.zeros(3, dtype=float), np.nan, 0, -1
    trades, scheduled_entries = [], {}
    base_lp, base_sp = long_params.copy(), short_params.copy()
    H = int(confirm_params.get("horizon_bars", 10))
    min_move = float(confirm_params.get("min_confirm_move_pct", 0.0))
    p_up = proba_filter.get('p_up') if proba_filter else None
    p_dn = proba_filter.get('p_dn') if proba_filter else None
    min_prob_long = float(proba_filter.get('min_prob_long', 0.55)) if proba_filter else 0.0
    min_prob_short = float(proba_filter.get('min_prob_short', 0.55)) if proba_filter else 0.0
    def pick_params(reg_idx: int):
        if regime_params and str(reg_idx) in regime_params: return regime_params[str(reg_idx)]["long"], regime_params[str(reg_idx)]["short"]
        return base_lp, base_sp
    def eval_long_sig(i, lp):
        return ((up30[i] > lp["strength_h30"]) or (up60[i] > lp["strength_h60"]) or (up90[i] > lp["strength_h90"]) or (up180[i] > lp["strength_h180"])) and (tr_up[i] > 0)
    def eval_short_sig(i, sp):
        return ((dn30[i] > sp["strength_h30"]) or (dn60[i] > sp["strength_h60"]) or (dn90[i] > sp["strength_h90"]) or (dn180[i] > sp["strength_h180"])) and (tr_dn[i] > 0)
    freeze_long_s_default = int(base_lp.get("freeze_period_minutes", 20)) * 60
    freeze_short_s_default = int(base_sp.get("freeze_period_minutes", 20)) * 60
    for i in range(n):
        ts, reg = int(t[i]), int(regimes[i])
        lp, sp = pick_params(reg)
        freeze_long_s, freeze_short_s = int(lp.get("freeze_period_minutes", freeze_long_s_default)) * 60, int(sp.get("freeze_period_minutes", freeze_short_s_default)) * 60
        
        if position is None and i in scheduled_entries and ts >= freeze_until:
            side = scheduled_entries.pop(i)
            entry_price, entry_time_s, stage, active_regime = (open_p[i] if not np.isnan(open_p[i]) else float(close[i])), ts, 0, reg
            
            if side == "long":
                position = "long"
                tp_levels = entry_price * (1.0 + np.array([lp["tp1_pct"], lp["tp2_pct"], lp["tp3_pct"]]) / 100.0)
                stop_level = entry_price * (1.0 - lp["sl_pct"] / 100.0)
            else: 
                position = "short"
                tp_levels = entry_price * (1.0 - np.array([sp["tp1_pct"], sp["tp2_pct"], sp["tp3_pct"]]) / 100.0)
                stop_level = entry_price * (1.0 + sp["sl_pct"] / 100.0)

            if print_details:
                print("\n" + "="*45)
                print(f"🔥 NEW TRADE SIGNAL (Initial Run Details) 🔥")
                entry_time_str = pd.to_datetime(ts, unit='s', utc=True).strftime('%Y-%m-%d %H:%M:%S UTC')
                print(f"   - Time:          {entry_time_str}")
                print(f"   - Side:          {side.upper()}")
                print(f"   - Entry Price:   {entry_price:.2f}")
                print("-" * 45)
                print(f"   - Stop Loss (SL):  {stop_level:.2f}")
                print(f"   - Take Profit 1:   {tp_levels[0]:.2f}")
                print(f"   - Take Profit 2:   {tp_levels[1]:.2f}")
                print(f"   - Take Profit 3:   {tp_levels[2]:.2f}")
                print("="*45 + "\n")

        if ts >= freeze_until and position is None:
            l_sig, s_sig = eval_long_sig(i, lp), eval_short_sig(i, sp)
            if proba_filter:
                if l_sig and (p_up is not None) and not (p_up[i] >= min_prob_long): l_sig = False
                if s_sig and (p_dn is not None) and not (p_dn[i] >= min_prob_short): s_sig = False
            if l_sig or s_sig:
                chosen_side = None
                if l_sig and not s_sig: chosen_side = "long"
                elif s_sig and not l_sig: chosen_side = "short"
                elif l_sig and s_sig:
                    if tr_up[i] > tr_dn[i]: chosen_side = "long"
                    elif tr_dn[i] > tr_up[i]: chosen_side = "short"
                    else:
                        up_sum, dn_sum = up30[i] + up60[i] + up90[i] + up180[i], dn30[i] + dn60[i] + dn90[i] + dn180[i]
                        if up_sum > dn_sum: chosen_side = "long"
                        elif dn_sum > up_sum: chosen_side = "short"
                if chosen_side and (j := i + H) < n:
                    future_ret = (close[j] - close[i]) / close[i] * 100.0
                    if (chosen_side == "long" and future_ret >= min_move) or (chosen_side == "short" and future_ret <= -min_move): scheduled_entries[j] = chosen_side
        if position:
            if position == "long":
                if low[i] <= stop_level:
                    exit_price, reason = stop_level, ("TSL@TP2" if stage == 2 else ("TSL@TP1" if stage == 1 else "SL"))
                    pnl = (exit_price - entry_price) / entry_price * 100.0
                    trades.append((entry_time_s, ts, "long", entry_price, exit_price, pnl, reason, active_regime)); position, freeze_until = None, ts + freeze_long_s; continue
                if stage == 0 and high[i] >= tp_levels[0]: stage, stop_level = 1, tp_levels[0]
                if stage == 1 and high[i] >= tp_levels[1]: stage, stop_level = 2, tp_levels[1]
                if stage == 2 and high[i] >= tp_levels[2]:
                    exit_price, pnl = tp_levels[2], (tp_levels[2] - entry_price) / entry_price * 100.0
                    trades.append((entry_time_s, ts, "long", entry_price, exit_price, pnl, "TP3", active_regime)); position, freeze_until = None, ts + freeze_long_s; continue
            else:
                if high[i] >= stop_level:
                    exit_price, reason = stop_level, ("TSL@TP2" if stage == 2 else ("TSL@TP1" if stage == 1 else "SL"))
                    pnl = (entry_price - exit_price) / entry_price * 100.0
                    trades.append((entry_time_s, ts, "short", entry_price, exit_price, pnl, reason, active_regime)); position, freeze_until = None, ts + freeze_short_s; continue
                if stage == 0 and low[i] <= tp_levels[0]: stage, stop_level = 1, tp_levels[0]
                if stage == 1 and low[i] <= tp_levels[1]: stage, stop_level = 2, tp_levels[1]
                if stage == 2 and low[i] <= tp_levels[2]:
                    exit_price, pnl = tp_levels[2], (entry_price - tp_levels[2]) / entry_price * 100.0
                    trades.append((entry_time_s, ts, "short", entry_price, exit_price, pnl, "TP3", active_regime)); position, freeze_until = None, ts + freeze_short_s; continue

    open_trade_state = None
    if position is not None:
        open_trade_state = {
            "side": position, "entry_price": entry_price, "stop_loss": stop_level,
            "take_profits": tp_levels.tolist(), "stage": stage,
        }

    if not trades: return ({"total_trades": 0, "win_rate": 0.0, "total_pnl": 0.0}, pd.DataFrame(columns=trade_cols), open_trade_state)
    trades_df = pd.DataFrame(trades, columns=trade_cols); pnls = trades_df["pnl_pct"].astype(float)
    summary = {"total_trades": len(trades), "long_trades": (trades_df["side"] == "long").sum(), "short_trades": (trades_df["side"] == "short").sum(),
               "win_rate": (pnls > 0).mean() * 100.0 if len(pnls) > 0 else 0.0, "total_pnl": pnls.sum(), "avg_pnl": pnls.mean() if len(pnls) > 0 else 0.0}
    return summary, trades_df, open_trade_state

def create_param_dicts(params: dict):
    long_p = {k.replace('long_', ''): v for k, v in params.items() if k.startswith('long_')}
    short_p = {k.replace('short_', ''): v for k, v in params.items() if k.startswith('short_')}
    confirm = {"horizon_bars": params["confirm_horizon"], "min_confirm_move_pct": params["confirm_min_move_pct"]}
    for p_dict, prefix in [(long_p, "long_"), (short_p, "short_")]:
        p_dict['sl_pct'] = params[f'{prefix}sl']; p_dict['tp1_pct'] = params[f'{prefix}tp1']
        p_dict['tp2_pct'] = params[f'{prefix}tp2']; p_dict['tp3_pct'] = params[f'{prefix}tp3']
        p_dict['freeze_period_minutes'] = params[f'{prefix}freeze']
        for h in [30, 60, 90, 180]: p_dict[f'strength_h{h}'] = params[f'{prefix}str_h{h}']
    return long_p, short_p, confirm

async def run_trading_logic(print_details: bool = False):
    """منطق معاملاتی را بر اساس آخرین پیش‌بینی‌ها اجرا می‌کند."""
    global MONITORED_TRADE
    print(f"[{datetime.now():%H:%M:%S}] 📈 Running trading logic...")
    if not os.path.exists(PREDICTION_RESULTS_CSV):
        print("   - Prediction file not found. Skipping.")
        return

    df_full = pd.read_csv(PREDICTION_RESULTS_CSV)
    df_full = compute_trends(df_full)

    print("   - Loading trading artifacts...")
    try:
        with open(os.path.join(ARTIFACTS_PATH, "global_params.json"), "r") as f: best_global_params = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: 'global_params.json' not found in {ARTIFACTS_PATH}"); return

    gmm_obj, gmm_scaler, ml_models_tuple, regime_overrides = None, None, None, None
    if SKLEARN_OK:
        try: gmm_obj = joblib.load(os.path.join(ARTIFACTS_PATH, "gmm_model.joblib")); gmm_scaler = joblib.load(os.path.join(ARTIFACTS_PATH, "gmm_scaler.joblib"))
        except FileNotFoundError: pass
        try: up_model = joblib.load(os.path.join(ARTIFACTS_PATH, "ml_up_model.joblib")); dn_model = joblib.load(os.path.join(ARTIFACTS_PATH, "ml_dn_model.joblib")); scaler_ml = joblib.load(os.path.join(ARTIFACTS_PATH, "ml_scaler.joblib")); ml_models_tuple = (up_model, dn_model, scaler_ml)
        except FileNotFoundError: pass
    try:
        with open(os.path.join(ARTIFACTS_PATH, "regime_params.json"), "r") as f: regime_overrides = {k: v for k, v in json.load(f).items()}
    except FileNotFoundError: pass

    idx_spikes = detect_volume_spikes(df_full, best_global_params["spike_window"], best_global_params["spike_z"])
    df_used = df_full.loc[idx_spikes].reset_index(drop=True)
    if df_used.empty:
        print("   - No volume spikes detected. No trades."); return

    if gmm_obj and gmm_scaler:
        X_regime = build_regime_features(df_used); Xs_regime = gmm_scaler.transform(X_regime.values)
        df_used["regime"] = gmm_obj.predict(Xs_regime)
    else: df_used["regime"] = 0

    proba_filter = None
    if ml_models_tuple:
        up_model, dn_model, scaler_ml = ml_models_tuple
        X_ml = make_ml_features(df_used); Xs_ml = scaler_ml.transform(X_ml.fillna(0.0).values)
        p_up, p_dn = up_model.predict_proba(Xs_ml)[:, 1], dn_model.predict_proba(Xs_ml)[:, 1]
        proba_filter = {'p_up': p_up, 'p_dn': p_dn, 'min_prob_long': 0.55, 'min_prob_short': 0.55}
    
    long_p, short_p, confirm = create_param_dicts(best_global_params)
    summary, trades_df, open_trade_at_end = backtest_fast_numpy(
        df=df_used, long_params=long_p, short_params=short_p,
        confirm_params=confirm, regime_params=regime_overrides,
        proba_filter=proba_filter, print_details=print_details)
    
    print("="*50); print("🏁 LIVE SIMULATION TICK COMPLETE 🏁")
    
    if summary['total_trades'] > 0:
        print("   - 📊 Performance Summary (based on available history):")
        print(f"     - Total Trades: {summary['total_trades']}")
        print(f"     - Win Rate: {summary['win_rate']:.2f}%")
        print(f"     - Total PnL: {summary['total_pnl']:.2f}%")
        print(f"     - Longs: {summary['long_trades']} | Shorts: {summary['short_trades']}")
        message = (
            f"Trade Summary:\n"
            f"Total Trades: {summary['total_trades']}\n"
            f"Win Rate: {summary['win_rate']:.2f}%\n"
            f"Total PnL: {summary['total_pnl']:.2f}%\n"
            f"Longs: {summary['long_trades']} | Shorts: {summary['short_trades']}"
        )
        # await send_to_allowed(message) 
    else: print("   - 📊 No trades found in the current historical data slice.")

    if not trades_df.empty:
        pnls = trades_df["pnl_pct"]; wins = pnls[pnls > 0]; losses = pnls[pnls <= 0]
        gross_profit = wins.sum(); gross_loss = abs(losses.sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        print("\n   - ⚖️ Risk & Reward Analysis:")
        print(f"     - Profit Factor: {profit_factor:.2f}")
        print(f"     - Wins: {len(wins)} | Losses: {len(losses)}")
        print(f"     - Avg Win PnL: {wins.mean():.2f}% | Avg Loss PnL: {losses.mean():.2f}%")
        print(f"     - Largest Win: {wins.max():.2f}% | Largest Loss: {losses.min():.2f}%")

    if print_details and not trades_df.empty:
        print("\n   - 👉 Individual Completed Trades Found:")
        for index, trade in trades_df.iterrows():
            entry_time_str = pd.to_datetime(trade['entry_time_s'], unit='s', utc=True).strftime('%Y-%m-%d %H:%M')
            exit_time_str = pd.to_datetime(trade['exit_time_s'], unit='s', utc=True).strftime('%Y-%m-%d %H:%M')
            pnl_val = trade['pnl_pct']
            GREEN, RED, RESET = '\033[92m', '\033[91m', '\033[0m'
            pnl_str = f"{pnl_val:.2f}%"
            if pnl_val > 0: pnl_str = f"{GREEN}{pnl_str}{RESET}"
            elif pnl_val < 0: pnl_str = f"{RED}{pnl_str}{RESET}"
            print(f"     - [{index+1:<2}] "
                  f"Time: {entry_time_str} -> {exit_time_str} | "
                  f"Side: {trade['side']:<5} | "
                  f"Entry: {trade['entry']:>8.2f} | "
                  f"Exit: {trade['exit']:>8.2f} | "
                  f"PnL: {pnl_str} | "
                  f"Reason: {trade['reason']}")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = os.path.join(TRADES_OUTPUT_DIR, f"trades_{timestamp}.csv")
        trades_df.to_csv(output_filename, index=False)
        print(f"\n💾 Saved detailed historical trades analysis to: {output_filename}")
    
    if not MONITORED_TRADE and open_trade_at_end:
        MONITORED_TRADE = open_trade_at_end
        print("\n" + "*"*55)
        print("🚀 NEW TRADE DETECTED! NOW MONITORING LIVE STATUS... 🚀")
        print(f"  - Side: {MONITORED_TRADE['side'].upper()}, Entry: {MONITORED_TRADE['entry_price']:.2f}")
        print("*"*55)
        message = (
            f"🚀 NEW TRADE DETECTED! {INSTRUMENT} 🚀\n"
            f"   - Side: {MONITORED_TRADE['side'].upper()} {'🟢' if MONITORED_TRADE['side'] == 'long' else '🔴'}\n" 
            f"   - Entry: {MONITORED_TRADE['entry_price']:.2f}\n"
            f"   - Stop loss: {MONITORED_TRADE['stop_loss']:.2f}\n"
            f"   - Take profit 1: {MONITORED_TRADE['take_profits'][0]:.2f}\n"
            f"   - Take profit 2: {MONITORED_TRADE['take_profits'][1]:.2f}\n"
            f"   - Take profit 3: {MONITORED_TRADE['take_profits'][2]:.2f}\n"
            f"   - Stage: {MONITORED_TRADE['stage']}"
        )
        await send_to_allowed(message)
    print("="*50 + "\n")
# endregion


# ======================================================================
# bölüm 3: حلقه اصلی برنامه (Main Loop) - با قابلیت مانیتورینگ زنده
# ======================================================================

MONITORED_TRADE = {}

async def check_and_display_open_trade_status(latest_ohlc: dict):
    """Checks the status of the single monitored trade against the latest candle."""
    global MONITORED_TRADE
    if not MONITORED_TRADE: return

    side = MONITORED_TRADE['side']; entry_price = MONITORED_TRADE['entry_price']
    stop_level = MONITORED_TRADE['stop_loss']; tp_levels = MONITORED_TRADE['take_profits']
    stage = MONITORED_TRADE['stage']
    
    latest_close = latest_ohlc['Close']
    pnl_pct = ((latest_close - entry_price) / entry_price * 100) if side == 'long' else ((entry_price - latest_close) / entry_price * 100)
    GREEN, RED, RESET = '\033[92m', '\033[91m', '\033[0m'
    pnl_color = GREEN if pnl_pct >= 0 else RED
    
    print("\n" + "="*50); print("💼 LIVE STATUS OF MONITORED TRADE 💼")
    print(f"   - Side: {side.upper()} | Entry: {entry_price:.2f} | Latest Price: {latest_close:.2f}")
    print(f"   - Current PnL: {pnl_color}{pnl_pct:.2f}%{RESET}")
    print(f"   - SL: {stop_level:.2f} | TP1: {tp_levels[0]:.2f} | TP2: {tp_levels[1]:.2f} | TP3: {tp_levels[2]:.2f}")
    print(f"   - Stage: {stage}"); print("="*50)
    # message = (
    # "💼 LIVE STATUS OF MONITORED TRADE 💼"
    # f"   - Side: {side.upper()} | Entry: {entry_price:.2f} | Latest Price: {latest_close:.2f}"
    # f"   - Current PnL: {pnl_color}{pnl_pct:.2f}%{RESET}"
    # f"   - SL: {stop_level:.2f} | TP1: {tp_levels[0]:.2f} | TP2: {tp_levels[1]:.2f} | TP3: {tp_levels[2]:.2f}"
    # f"   - Stage: {stage}"
    # )
    # await send_to_allowed(message)

    is_closed, exit_reason = False, ""
    if side == 'long':
        if latest_ohlc['Low'] <= stop_level:
            is_closed, exit_reason = True, "Stop Loss Hit" 
            await send_to_allowed("Stop Loss Hit ⭕️")
        elif stage == 2 and latest_ohlc['High'] >= tp_levels[2]: 
            is_closed, exit_reason = True, "Final TP Hit"
            await send_to_allowed("Final TP Hit ✅")
        elif stage == 1 and latest_ohlc['High'] >= tp_levels[1]: 
            MONITORED_TRADE['stage'] = 2; MONITORED_TRADE['stop_loss'] = tp_levels[1]
            await send_to_allowed(f"Final TP2 Hit now sl is: {tp_levels[1]} ✅")
        elif stage == 0 and latest_ohlc['High'] >= tp_levels[0]: 
            MONITORED_TRADE['stage'] = 1; MONITORED_TRADE['stop_loss'] = tp_levels[0]
            await send_to_allowed(f"Final TP1 Hit now sl is: {tp_levels[0]} ✅")  

    elif side == 'short':
        if latest_ohlc['High'] >= stop_level: 
            is_closed, exit_reason = True, "Stop Loss Hit"
            await send_to_allowed("Stop Loss Hit ⭕️")
        elif stage == 2 and latest_ohlc['Low'] <= tp_levels[2]: 
            is_closed, exit_reason = True, "Final TP Hit"
            await send_to_allowed("Final TP Hit ✅")
        elif stage == 1 and latest_ohlc['Low'] <= tp_levels[1]: 
            MONITORED_TRADE['stage'] = 2; MONITORED_TRADE['stop_loss'] = tp_levels[1]
            await send_to_allowed(f"Final TP2 Hit now sl is: {tp_levels[1]} ✅")
        elif stage == 0 and latest_ohlc['Low'] <= tp_levels[0]: 
            MONITORED_TRADE['stage'] = 1; MONITORED_TRADE['stop_loss'] = tp_levels[0]
            await send_to_allowed(f"Final TP1 Hit now sl is: {tp_levels[0]} ✅") 
            
    if is_closed:
        print("\n" + "🎉"*25)
        print(f"🎉 MONITORED TRADE CLOSED! Reason: {exit_reason} 🎉")
        print("🎉"*25 + "\n")
        MONITORED_TRADE = {}

async def main():
    """حلقه اصلی که فرآیند را به صورت مداوم اجرا می‌کند."""
    is_first_run = True
    await tg_start()
    while True:
        success = fetch_latest_candles()
        
        if success:
            try:
                df_latest = pd.read_csv(HISTORY_CSV)
                if not df_latest.empty:
                    latest_ohlc = df_latest.iloc[-1].to_dict()
                    await check_and_display_open_trade_status(latest_ohlc)
            except Exception as e:
                print(f"Could not read history for live status check: {e}")

            success_pred = run_prediction_pipeline()

            if success_pred:
                await run_trading_logic(print_details=is_first_run)

        if is_first_run == True:
            is_first_run = False

        now = datetime.now(timezone.utc)
        wait_seconds = 60 - now.second
        print(f"⏳ Waiting {wait_seconds} seconds for the next 1-minute candle...")
        await asyncio.sleep(wait_seconds)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Exiting live trading loop.")