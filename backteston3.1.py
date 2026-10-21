# -*- coding: utf-8 -*-
# Inference: Runs Quantile model for forecasts (Percentages ONLY),
# saving to a SINGLE CSV with OHLCV included.

import os, warnings, joblib, numpy as np, pandas as pd, matplotlib.pyplot as plt, json
from joblib import Parallel, delayed
import torch, torch.nn as nn
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm
warnings.filterwarnings("ignore")

# ───────── paths & knobs ─────────
NEW_PATH     = r"D:\project\Odimm_V2\split_output_data(WTICO_USD)\WTICO_USD_M1_60m_test_5pct.csv"
                
# --- Quantile Model Artifacts ---
QUANT_META_PATH  = r"D:\project\Odimm_V1o\XAUUSD_M1\mhq_ann(XAU_USD)\ann_meta.json"

# --- Output ---
OUT_CSV      = "quantile_pct_results(US30)_pt1.csv"

# --- General Config ---
BATCH_SIZE   = 1024
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
USE_PARALLEL_STATS = True

# ───────── Feature Engineering ─────────
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
    print("🚀 Creating feature set for Quantile model...", flush=True)
    df=df.copy()
    eps = 1e-8

    df["oc_ret"] = np.log(df["Close"] / (df["Open"] + eps))
    df["hl_range"] = (df["High"] - df["Low"]) / (df["Low"] + eps)

    close=df["Close"].values.astype(float)
    logc=np.log(np.maximum(close,eps)); vol=df["Volume"].values.astype(float)
    R=[15,30,60,90,120]; VW=[5,15,30,60,90,180]

    tasks=[(df[c].values.astype(float),w,c) for c in("oc_ret","hl_range") for w in R]
    if USE_PARALLEL_STATS:
        res=list(tqdm(Parallel(n_jobs=max(1,os.cpu_count()//2))(delayed(_rolling_stats_np)(a,w) for a,w,_ in tasks),
                      total=len(tasks), desc="FE Parallel Rolling Stats", dynamic_ncols=True))
    else:
        res=[_rolling_stats_np(a,w) for a,w,_ in tqdm(tasks, desc="FE Rolling Stats", leave=False)]

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
    for w in tqdm(wins, desc="FE Price/Volume Windows", dynamic_ncols=True):
        vm = df["Volume"].rolling(w, min_periods=1).mean()
        df[f"volume_mean_{w}"] = vm
        df[f"volume_change_{w}"] = df["Volume"].diff(w)
        df[f"volume_ratio_{w}"] = df["Volume"] / (vm + eps)

        pm = df["Close"].rolling(w, min_periods=1).mean()
        df[f"price_mean_{w}"] = pm
        df[f"price_change_past_{w}"] = df["Close"].diff(w)
        df[f"price_ratio_{w}"] = df["Close"] / (pm + eps)

        if w > 1:
            df[f"volume_std_{w}"] = df["Volume"].rolling(w, min_periods=2).std()
            df[f"price_std_{w}"] = df["Close"].rolling(w, min_periods=2).std()
        else:
            df[f"volume_std_{w}"] = 0.0
            df[f"price_std_{w}"] = 0.0

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df

# ───────── Model Definitions ─────────
class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden=512, depth=3, dropout=0.1):
        super().__init__()
        layers = []; d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout)]; d = hidden
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

# ───────── Quantile Prediction Logic ─────────
def run_quantile_prediction(df_fe: pd.DataFrame):
    print("\n" + "─"*50)
    print("🚀 Running Quantile Forecast Model on ALL data…")
    with open(QUANT_META_PATH, "r") as f: meta = json.load(f)
    out_dir = os.path.dirname(QUANT_META_PATH)
    scaler_quant = joblib.load(os.path.join(out_dir, meta["artifacts"]["scaler"]))
    feats_quant  = joblib.load(os.path.join(out_dir, meta["artifacts"]["features"]))
    model_blob   = torch.load(os.path.join(out_dir, meta["artifacts"]["model"]), map_location=DEVICE, weights_only=False)
    model_quant = MLP(in_dim=model_blob['in_dim'], out_dim=model_blob['out_dim'], hidden=model_blob['hidden'],
                      depth=model_blob['depth'], dropout=model_blob['dropout']).to(DEVICE)
    model_quant.load_state_dict(model_blob['state_dict']); model_quant.eval()

    missing_feats = set(feats_quant) - set(df_fe.columns)
    if missing_feats: raise ValueError(f"Quantile model missing features: {list(missing_feats)}")

    X_quant = df_fe[feats_quant].values.astype(np.float32)
    X_quant_scaled = scaler_quant.transform(X_quant)

    print("⚙️  Quantile model forward pass (running on all data)...", flush=True)
    all_preds = []
    for i in tqdm(range(0, len(X_quant_scaled), BATCH_SIZE), desc="Quantile Prediction Batches", dynamic_ncols=True):
        batch = torch.from_numpy(X_quant_scaled[i:i+BATCH_SIZE]).to(DEVICE)
        with torch.no_grad():
            preds_batch = model_quant(batch).cpu().numpy()
        all_preds.append(preds_batch)

    yhat_full = np.concatenate(all_preds, axis=0)

    return yhat_full, model_blob['spec']

# ───────── Main Run Function ─────────
def run_inference():
    print("─"*50)
    print("📦 Reading data from CSV…", flush=True)
    df=(pd.read_csv(NEW_PATH,parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True))

    df_fe = create_all_features(df)

    if df_fe.empty:
        raise ValueError("❌ Feature set is empty after engineering. Check data quality/length.")

    # --- Quantile Prediction ---
    quantile_preds_matrix, quantile_spec = run_quantile_prediction(df_fe)

    # NEW: Save OHLCV + quantile forecasts
    out = pd.DataFrame({
        "timestamp": df_fe["timestamp"].values,
        "Open": df_fe["Open"].values,
        "High": df_fe["High"].values,
        "Low": df_fe["Low"].values,
        "Close": df_fe["Close"].values,
        "Volume": df_fe["Volume"].values,
    })

    print("\n📝 Adding quantile prediction results (Percentages Only)...")
    eps = 1e-8
    for i, (side, h, q) in enumerate(tqdm(quantile_spec, desc="Merging Predictions", dynamic_ncols=True)):
        q_str = str(q).replace('.', 'p')
        col_name_pct = f"pred_{side}_h{h}_q{q_str}_pct"
        pred_values = quantile_preds_matrix[:, i]
        out[col_name_pct] = (pred_values / (out['Close'] + eps)) * 100

    # --- Save Final Results ---
    out.to_csv(OUT_CSV, index=False, float_format='%.4f')
    print(f"\n✅ Saved Quantile results (with OHLCV) to: {OUT_CSV}")


if __name__=="__main__":
    run_inference()
