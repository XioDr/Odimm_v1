# -*- coding: utf-8 -*-
"""
Event-Driven Trading System (Anomaly-Centric) — FULL, FIXED
- No argparse (use TRAIN/EXECUTE flags below)
- Pandas deprecation fixed (ffill -> s.ffill().to_numpy())
- LightGBM/XGBoost/LogReg supported; trains with named columns; inference uses DataFrame
- Robust predict_meta(): always returns 3-class probs [skip,long,short] (pads missing classes with zeros)
- Training pads missing classes with synthetic rows to avoid 2-class model
- Safe guards for empty events and empty datasets
"""

import os, joblib, warnings
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from joblib import Parallel, delayed

# ============================ FLAGS ============================
TRAIN   = True     # آموزش متا-لیبلر و کوانتیل هد
EXECUTE = True     # تولید تصمیم‌ها با قانون MVP

# ============================ PATHS ============================
NEW_PATH    = "history.csv"

FEAT_AE     = "iforest_feature_columns.pkl"   # AE feature list
SCALER_AE   = "ae_scaler.pkl"
MODEL_AE    = "btc_autoenc_model.pt"

FEAT_QT     = "feature_list_h90.pkl"          # quantile/ceil-floor feature list
SCALER_QT   = "feature_scaler_h90.pkl"
MODEL_CEIL  = "ceil_model_ann_h90.pth"
MODEL_FLO   = "floor_model_ann_h90.pth"

OUT_DIR     = "evt_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# ============================ HPARAMS ============================
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE  = 1024
THR_Q       = 0.99          # anomaly threshold quantile
DEBOUNCE_K  = 10            # min bars between events
CTX_L       = 60            # context window (past-only)
HORIZON     = 90            # triple-barrier horizon (bars)

# Executor (MVP)
TAU_ENTER   = 0.60          # min prob to enter
R_MIN       = 0.002         # min |q50|
MAX_SIZE    = 1.0
SIZE_K      = 1.0

# Training
META_RANDOM_STATE = 42
Q_EPOCHS = 5
Q_LR     = 1e-3
Q_BATCH  = 256

warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")

# ============================ MODELS ============================
class AutoEncoder(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64))
        self.decoder = nn.Sequential(
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, d))
    def forward(self, x): return self.decoder(self.encoder(x))

class MLP1(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1))
    def forward(self, x): return self.net(x)

class QuantileMLP(nn.Module):
    def __init__(self, d, qs=(0.05, 0.5, 0.95)):
        super().__init__()
        self.qs = torch.tensor(qs, dtype=torch.float32)
        self.feat = nn.Sequential(
            nn.Linear(d, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU())
        self.head = nn.Linear(128, len(qs))
    def forward(self, x):
        h = self.feat(x)
        return self.head(h)

def pinball_loss(pred, target, qs=(0.05,0.5,0.95)):
    qs_t = torch.tensor(qs, dtype=torch.float32, device=pred.device).view(1,-1)
    diff = target.view(-1,1) - pred
    return torch.mean(torch.maximum(qs_t*diff, (qs_t-1)*diff))

# ============================ FEATURE ENG ============================
def _slope(arr: np.ndarray):
    n = arr.size
    if n == 0: return np.nan
    x = np.arange(n); x_mean = (n - 1) / 2.; y_mean = np.nanmean(arr)
    num = np.nansum((x - x_mean)*(arr - y_mean))
    den = np.nansum((x - x_mean)**2)
    return num / den if den else 0.

def _roll_stats(col, s, w):
    return {
        f'{col}_rmean_{w}':  s.rolling(w).mean(),
        f'{col}_rstd_{w}':   s.rolling(w).std(),
        f'{col}_rslope_{w}': s.rolling(w).apply(lambda a: _slope(np.array(a)), raw=False)
    }

def feature_engineering_fast(df, N_JOBS=os.cpu_count() or 1):
    df = df.copy()
    df['oc_ret']   = np.log(df['Close'] / df['Open'])
    df['hl_range'] = (df['High'] - df['Low']) / df['Low']
    logc           = np.log(df['Close'])
    RWS = [15, 30, 60, 90, 120]

    stats = Parallel(N_JOBS)(delayed(_roll_stats)(c, df[c], w) for c in ['oc_ret','hl_range'] for w in RWS)
    for dct in stats:
        for k,v in dct.items(): df[k] = v

    for w in RWS:
        df[f'momentum_{w}'] = df['Close'].pct_change(w)

    for w in [5,15,30,60,90,180]:
        df[f'ret_{w}']   = logc.diff(w)
        df[f'vol_z_{w}'] = (df['Volume']-df['Volume'].rolling(w).mean())/df['Volume'].rolling(w).std()

    delta = df['Close'].diff(); up,down = delta.clip(lower=0), -delta.clip(upper=0)
    rs = up.ewm(alpha=1/14, adjust=False).mean()/down.ewm(alpha=1/14, adjust=False).mean()
    df['rsi_14'] = 100 - 100/(1+rs)

    ma20, s20 = df['Close'].rolling(20).mean(), df['Close'].rolling(20).std()
    df['bb_pos'] = (df['Close']-ma20)/(2*s20)

    df.dropna(inplace=True)
    return preprocess_features_fast(df)

def preprocess_features_fast(df, wins=[1,3,5,15,30,45,60,90]):
    df = df.copy()
    for w in wins:
        mean_v = df['Volume'].rolling(w,1).mean()
        df[f'volume_mean_{w}'] = mean_v
        df[f'volume_std_{w}']  = df['Volume'].rolling(w,1).std()
        df[f'volume_change_{w}'] = df['Volume'].diff(w)
        df[f'volume_ratio_{w}']  = df['Volume'] / (mean_v + 1e-6)

        mean_p = df['Close'].rolling(w,1).mean()
        df[f'price_mean_{w}'] = mean_p
        df[f'price_std_{w}']  = df['Close'].rolling(w,1).std()
        df[f'price_change_past_{w}'] = df['Close'].diff(w)
        df[f'price_ratio_{w}'] = df['Close'] / (mean_p + 1e-6)
    return df

# ============================ ANOMALY ENGINE ============================
def anomaly_engine(df, feat_ae, scaler_ae, ae_model, thr_q=THR_Q, debounce_k=DEBOUNCE_K, rank_window=500):
    X = scaler_ae.transform(df[feat_ae].values.astype(np.float32))
    errs = []
    with torch.no_grad():
        for i in range(0, len(X), BATCH_SIZE):
            xb = torch.from_numpy(X[i:i+BATCH_SIZE]).to(DEVICE)
            recon = ae_model(xb).cpu().numpy()
            errs.append(((X[i:i+BATCH_SIZE]-recon)**2).mean(axis=1))
    err = np.concatenate(errs)

    ser = pd.Series(err, index=df.index)
    ranks = ser.rolling(rank_window, min_periods=50).apply(
        lambda a: pd.Series(a).rank(pct=True).iloc[-1], raw=False)
    ranks = ranks.fillna(ser.rank(pct=True))
    thr   = np.quantile(err[~np.isnan(err)], thr_q)

    out = df.copy()
    out['recon_err'] = err
    out['err_rank']  = ranks.values
    out['anomaly']   = np.where(err > thr, -1, 1)

    is_event = np.zeros(len(out), dtype=bool)
    last_idx = -10**9
    for i in range(1, len(out)):
        if (out['anomaly'].iloc[i] == -1) and (out['anomaly'].iloc[i-1] == 1) and (i - last_idx >= debounce_k):
            is_event[i] = True
            last_idx = i
    out['is_event'] = is_event
    return out, thr

# ============================ CEIL/FLOOR ============================
class MLPWrapper(MLP1): pass

def load_mlp_state(path, d):
    m = MLPWrapper(d).to(DEVICE)
    m.load_state_dict(torch.load(path, map_location=DEVICE))
    m.eval()
    return m

def infer_ceil_floor(df, feat_qt, scaler_qt, ceil_net, floor_net):
    Xq = scaler_qt.transform(df[feat_qt].values.astype(np.float32))
    up, lo = [], []
    with torch.no_grad():
        for i in range(0, len(Xq), BATCH_SIZE):
            xb = torch.from_numpy(Xq[i:i+BATCH_SIZE]).to(DEVICE)
            up.append(ceil_net(xb).cpu().numpy().ravel())
            lo.append(floor_net(xb).cpu().numpy().ravel())
    df = df.copy()
    df['ceil_pred']  = np.concatenate(up)
    df['floor_pred'] = np.concatenate(lo)
    return df

# ============================ EVENTS ============================
def _slope_series(s: pd.Series):
    return _slope(s.ffill().to_numpy())

def build_events(df, ctx_l=CTX_L, feat_cols=None):
    if 'is_event' not in df.columns or df['is_event'].sum() == 0:
        return pd.DataFrame(columns=['event_id','t0','idx'])
    if feat_cols is None:
        skip = {'timestamp','Open','High','Low','Close','Volume','is_event','anomaly'}
        feat_cols = [c for c in df.columns if c not in skip and np.issubdtype(df[c].dtype, np.number)]
    rows = []
    idxs = np.where(df['is_event'].values)[0]
    for i in idxs:
        t0 = df['timestamp'].iloc[i]
        l  = max(0, i - ctx_l); r = i
        dfg = df.iloc[l:r+1]
        feats = {}
        for c in feat_cols:
            s = dfg[c].astype(float)
            feats[f'{c}_last']  = float(s.iloc[-1])
            feats[f'{c}_mean']  = float(s.mean())
            feats[f'{c}_std']   = float(s.std(ddof=0))
            feats[f'{c}_slope'] = float(_slope_series(s))
        rows.append({'event_id': int(i), 't0': pd.to_datetime(t0), 'idx': int(i), **feats})
    return pd.DataFrame(rows).sort_values('t0')

# ============================ LABELING ============================
def triple_barrier_labels(df, evt_df, horizon=HORIZON, alpha_tp=0.75, alpha_sl=1.25):
    if evt_df.empty:
        return evt_df.assign(label_cls=[], tt_hit=[], ret_ev=[])
    out = evt_df.copy()
    labels, tt_hit, ret_fut = [], [], []
    for _, row in out.iterrows():
        i = int(row['idx'])
        if i >= len(df):
            labels.append(0); tt_hit.append(np.nan); ret_fut.append(0.0); continue
        close0 = float(df['Close'].iloc[i])
        up     = float(df['ceil_pred'].iloc[i]) * alpha_tp
        lo     = float(df['floor_pred'].iloc[i]) * alpha_sl
        tp_price = close0 + up
        sl_price = close0 - lo
        end = min(len(df)-1, i + horizon)
        label, tth, ret_ev = 0, np.nan, 0.0
        for j in range(i+1, end+1):
            hi = float(df['High'].iloc[j]) if 'High' in df.columns else float(df['Close'].iloc[j])
            lo_p = float(df['Low'].iloc[j]) if 'Low'  in df.columns else float(df['Close'].iloc[j])
            hit_tp = hi >= tp_price
            hit_sl = lo_p <= sl_price
            if hit_tp and hit_sl:
                label = -1; tth = j - i; ret_ev = (sl_price-close0)/close0; break
            elif hit_tp:
                label = +1; tth = j - i; ret_ev = (tp_price-close0)/close0; break
            elif hit_sl:
                label = -1; tth = j - i; ret_ev = (sl_price-close0)/close0; break
        labels.append(label); tt_hit.append(tth); ret_fut.append(ret_ev)
    out['label_cls'] = labels
    out['tt_hit']    = tt_hit
    out['ret_ev']    = ret_fut
    return out

# ============================ META LABELER ============================
def get_meta_model():
    try:
        import lightgbm as lgb
        return ('lgb', lgb.LGBMClassifier(
            objective='multiclass', num_class=3,
            n_estimators=400, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=META_RANDOM_STATE))
    except Exception:
        try:
            import xgboost as xgb
            return ('xgb', xgb.XGBClassifier(
                objective='multi:softprob', num_class=3,
                n_estimators=400, learning_rate=0.05, max_depth=6,
                subsample=0.8, colsample_bytree=0.8,
                random_state=META_RANDOM_STATE))
        except Exception:
            from sklearn.linear_model import LogisticRegression
            return ('logreg', LogisticRegression(max_iter=300, multi_class='multinomial',
                                                 random_state=META_RANDOM_STATE))

def _pad_missing_classes(X_df: pd.DataFrame, y: np.ndarray):
    """Ensure classes {0,1,2} exist by duplicating first row with new labels (benign trick)."""
    need = {0,1,2}
    have = set(np.unique(y))
    miss = list(need - have)
    if len(miss) == 0:
        return X_df, y
    if X_df.empty:
        return X_df, y  # nothing we can do
    pad_rows = [X_df.iloc[[0]].copy() for _ in miss]
    X_pad = pd.concat([X_df] + pad_rows, ignore_index=True)
    y_pad = np.concatenate([y] + [np.array([m], dtype=int) for m in miss])
    return X_pad, y_pad

def train_meta_labeler(X_df: pd.DataFrame, y: np.ndarray):
    name, model = get_meta_model()
    X_tr, y_tr = _pad_missing_classes(X_df, y)
    model.fit(X_tr, y_tr)
    return name, model

def _to_three_class_proba(model, proba):
    # Some models return list-of-arrays for binary—stack them
    if isinstance(proba, list):
        proba = np.vstack([1-proba[0], proba[0]]).T
    classes = getattr(model, "classes_", np.array([0,1,2]))
    out = np.zeros((proba.shape[0], 3), dtype=float)
    for i, cls in enumerate(classes):
        cls_int = int(cls)
        if 0 <= cls_int <= 2:
            out[:, cls_int] = proba[:, i]
    s = out.sum(axis=1, keepdims=True)
    out = np.where(s > 0, out / s, out)
    return out

def predict_meta(model, X_df: pd.DataFrame):
    try:
        proba = model.predict_proba(X_df)  # keep DataFrame with names
    except Exception:
        df = model.decision_function(X_df)
        if isinstance(df, list):  # rare
            df = np.vstack(df).T
        if df.ndim == 1:
            df = np.vstack([0*df, df, -df]).T
        df = df - df.min(axis=1, keepdims=True)
        proba = df / (df.sum(axis=1, keepdims=True) + 1e-9)
    return _to_three_class_proba(model, proba)

# ============================ QUANTILE ============================
class ArrayDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

def train_quantile_head(X, y, qs=(0.05,0.5,0.95), epochs=Q_EPOCHS, lr=Q_LR, batch=Q_BATCH):
    if len(X) == 0:
        raise ValueError("No samples to train quantile head.")
    model = QuantileMLP(X.shape[1], qs=qs).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ds = ArrayDataset(X, y); dl = DataLoader(ds, batch_size=batch, shuffle=True)
    model.train()
    for ep in range(epochs):
        tot = 0.0
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred = model(xb)
            loss = pinball_loss(pred, yb, qs=qs)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()*len(xb)
        print(f"[Quantile] epoch {ep+1}/{epochs} loss={tot/len(ds):.6f}")
    model.eval()
    return model

def predict_quantiles(model, X):
    if len(X) == 0:
        return np.zeros((0,3))
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        out = model(X_t).cpu().numpy()
    return out

# ============================ EXECUTOR ============================
def execute_trades(evt_df, proba, q_preds, tau=TAU_ENTER, rmin=R_MIN, max_size=MAX_SIZE, k=SIZE_K):
    out = evt_df.copy()
    if len(out) == 0:
        return out.assign(action=[], size=[], p_enter=[], q05=[], q50=[], q95=[])
    q05, q50, q95 = q_preds[:,0], q_preds[:,1], q_preds[:,2]
    out['q05'], out['q50'], out['q95'] = q05, q50, q95

    acts, sizes, p_enters = [], [], []
    for i in range(len(out)):
        p = proba[i]                         # shape (3,)
        p_skip, p_long, p_short = (p[0], p[1], p[2]) if p.shape[0] == 3 else (p[0], p[1] if len(p)>1 else 0.0, 0.0)
        p_enter = 1 - p_skip
        dir_idx = 1 if p_long >= p_short else 2
        direction = ['skip','long','short'][dir_idx]
        cond = (p_enter >= tau) and (abs(q50[i]) >= rmin)
        if not cond:
            acts.append('skip'); sizes.append(0.0); p_enters.append(p_enter); continue
        denom = max(abs(q05[i]), 1e-6)
        size = min(max_size, max(0.0, k * p_enter * (abs(q50[i]) / denom)))
        acts.append(direction); sizes.append(size); p_enters.append(p_enter)

    out['action']  = acts
    out['size']    = sizes
    out['p_enter'] = p_enters
    return out

# ============================ UTILS ============================
def ensure_indexed_timestamp(df):
    if 'timestamp' not in df.columns:
        raise ValueError("Missing 'timestamp' column.")
    return df.sort_values('timestamp').reset_index(drop=True)

def map_labels_cls_to_meta(y_cls):
    y = np.zeros(len(y_cls), dtype=int)  # default skip=0
    y[y_cls ==  1] = 1                   # long
    y[y_cls == -1] = 2                   # short
    return y

# ============================ MAIN ============================
def main():
    # Load data
    df_raw = pd.read_csv(NEW_PATH, parse_dates=['timestamp'])
    df_raw = ensure_indexed_timestamp(df_raw)

    print("🧱 Feature engineering…")
    df = feature_engineering_fast(df_raw)

    print("📦 Loading AE/Scalers…")
    feat_ae   = joblib.load(FEAT_AE);  scaler_ae = joblib.load(SCALER_AE)
    ae = AutoEncoder(len(feat_ae)).to(DEVICE)
    ae.load_state_dict(torch.load(MODEL_AE, map_location=DEVICE)); ae.eval()

    print("⚡ Running anomaly engine…")
    df_a, thr = anomaly_engine(df, feat_ae, scaler_ae, ae, thr_q=THR_Q, debounce_k=DEBOUNCE_K)

    print("📐 Loading ceil/floor models…")
    feat_qt   = joblib.load(FEAT_QT);  scaler_qt = joblib.load(SCALER_QT)
    ceil_net  = load_mlp_state(MODEL_CEIL, len(feat_qt))
    floor_net = load_mlp_state(MODEL_FLO, len(feat_qt))
    df_q = infer_ceil_floor(df_a, feat_qt, scaler_qt, ceil_net, floor_net)

    print("🧩 Building events dataset…")
    evt_df = build_events(df_q, ctx_l=CTX_L)
    if evt_df.empty:
        print("ℹ️ No events created. Exiting gracefully.")
        evt_df.to_csv(os.path.join(OUT_DIR, "events_dataset.csv"), index=False)
        return

    # attach refs
    idxs = evt_df['idx'].values
    evt_df['close0'] = df_q['Close'].iloc[idxs].values
    evt_df['ceil0']  = df_q['ceil_pred'].iloc[idxs].values
    evt_df['floor0'] = df_q['floor_pred'].iloc[idxs].values

    print("🏷️  Labeling (triple-barrier)…")
    evt_lab = triple_barrier_labels(df_q, evt_df, horizon=HORIZON)

    events_csv = os.path.join(OUT_DIR, "events_dataset.csv")
    evt_lab.to_csv(events_csv, index=False)
    print(f"💾 Saved events dataset: {events_csv}  (rows={len(evt_lab)})")

    # Build feature matrix for models
    feat_cols = [c for c in evt_lab.columns if c.endswith(('_last','_mean','_std','_slope'))]
    X_df = evt_lab[feat_cols].fillna(0)               # DataFrame (keeps names)
    y_meta = map_labels_cls_to_meta(evt_lab['label_cls'].values)
    y_q = evt_lab['ret_ev'].values.astype(np.float32)

    if TRAIN:
        print("🎯 Training Meta-Labeler…")
        name, meta_model = train_meta_labeler(X_df, y_meta)
        joblib.dump((name, meta_model, feat_cols), os.path.join(OUT_DIR, "meta_labeler.joblib"))
        print(f"💾 Saved meta model: {name}")

        print("📈 Training Quantile Head (q05,q50,q95) on realized event returns…")
        q_model = train_quantile_head(X_df.values, y_q)
        torch.save(q_model.state_dict(), os.path.join(OUT_DIR, "quantile_head.pt"))
        print("💾 Saved quantile head.")

    if EXECUTE:
        print("🛠️ Executing decisions (MVP)…")
        meta_path = os.path.join(OUT_DIR, "meta_labeler.joblib")
        q_path    = os.path.join(OUT_DIR, "quantile_head.pt")
        if not (os.path.exists(meta_path) and os.path.exists(q_path)):
            print("⚠️ Missing trained models; set TRAIN=True first or provide models in OUT_DIR.")
        else:
            name, meta_model, feat_cols_loaded = joblib.load(meta_path)
            X_inf = evt_lab[feat_cols_loaded].fillna(0)  # DataFrame with same columns
            proba = predict_meta(meta_model, X_inf)      # -> (N,3)

            q_model = QuantileMLP(len(feat_cols_loaded)).to(DEVICE)
            q_model.load_state_dict(torch.load(q_path, map_location=DEVICE)); q_model.eval()
            q_preds = predict_quantiles(q_model, X_inf.values)  # (N,3)

            decisions = execute_trades(evt_lab.copy(), proba, q_preds,
                                       tau=TAU_ENTER, rmin=R_MIN, max_size=MAX_SIZE, k=SIZE_K)
            dec_path = os.path.join(OUT_DIR, "decisions.csv")
            decisions.to_csv(dec_path, index=False)
            print(f"💾 Saved decisions: {dec_path}")

    print("✅ Done.")

if __name__ == "__main__":
    main()
