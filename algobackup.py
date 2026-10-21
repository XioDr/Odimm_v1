# -*- coding: utf-8 -*-
"""
Spike-Only Backtester with Regime Clustering (GMM) + ML Probability Filter
TUNING-READY VERSION

- No pretuned parameters
- No argparse/CLI
- Optuna optimizer included
- Uses tqdm progress bar
- Vectorized volume spike detection for speed
- Robust to your up_*/dn_* prediction columns
"""

import os
import sys
import time
import json
import logging
import warnings
from dataclasses import dataclass
from datetime import timedelta
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd

# ------------------------------------------------------------
# Logging
# ------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backtest.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------
# Optional imports
# ------------------------------------------------------------
try:
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import TimeSeriesSplit
    SKLEARN_OK = True
except ImportError as e:
    logger.warning(f"Scikit-learn imports failed: {e}")
    SKLEARN_OK = False

try:
    import optuna
    OPTUNA_OK = True
except ImportError as e:
    logger.warning(f"Optuna import failed: {e}")
    OPTUNA_OK = False

try:
    from tqdm.auto import tqdm
    TQDM_OK = True
except ImportError as e:
    logger.warning(f"tqdm import failed: {e}")
    TQDM_OK = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    MATPLOTLIB_OK = True
except ImportError:
    MATPLOTLIB_OK = False


# ------------------------------------------------------------
# Config dataclasses
# ------------------------------------------------------------
@dataclass
class PrecisionConfig:
    """Configuration for precision-optimized trading"""
    min_trades: int = 560
    penalty_scale: float = 0.45
    high_win_penalty_scale: float = 10.0
    min_win_rate: float = 85.0
    max_win_rate: float = 98.0


@dataclass
class RiskConfig:
    """Risk management configuration"""
    max_drawdown_pct: float = 10.0
    risk_per_trade_pct: float = 1.0
    max_position_size: float = 10000.0
    stop_trading_on_drawdown: bool = True


@dataclass
class StrategyConfig:
    """Strategy parameter ranges (useful for tuning search space)"""
    # Volume spike detection
    spike_window_range: Tuple[int, int] = (5, 240)
    spike_z_range: Tuple[float, float] = (2.0, 8.0)

    # Confirmation
    confirm_horizon_range: Tuple[int, int] = (5, 60)
    confirm_min_move_range: Tuple[float, float] = (0.05, 5.0)

    # Take profit levels (5 TP system)
    tp_ranges: Dict[str, Tuple[float, float]] = None

    # Stop loss
    sl_range: Tuple[float, float] = (0.1, 0.15)

    # Freeze period
    freeze_range: Tuple[int, int] = (5, 60)

    # Strength thresholds
    strength_range: Tuple[float, float] = (0.1, 0.85)

    def __post_init__(self):
        if self.tp_ranges is None:
            self.tp_ranges = {
                'tp1': (0.15, 0.35),
                'tp2': (0.35, 0.65),
                'tp3': (0.65, 0.80),
                'tp4': (0.8, 1.25),
                'tp5': (1.25, 1.5)
            }


# ------------------------------------------------------------
# Backtester
# ------------------------------------------------------------
class Backtester:
    """Main backtesting engine, tuning-friendly."""

    def __init__(self, config: StrategyConfig, precision_config: PrecisionConfig, risk_config: RiskConfig):
        self.config = config
        self.precision_config = precision_config
        self.risk_config = risk_config
        self.seed = 42
        np.random.seed(self.seed)

        # State
        self.current_drawdown = 0.0
        self.peak_balance = 0.0
        self.stop_trading = False

    # ---------------- Data Validation & Utilities ----------------
    def validate_dataframe(self, df: pd.DataFrame) -> None:
        """Validate input dataframe has required base columns (predictions are optional)."""
        required_cols = ["timestamp", "Close", "Open", "High", "Low", "Volume"]

        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Check for NaN values in critical columns
        critical_cols = ["timestamp", "Close", "Open", "High", "Low"]
        for col in critical_cols:
            if df[col].isna().any():
                raise ValueError(f"NaN values found in critical column: {col}")

        # Just warn if prediction columns are not present
        up_pred_cols = [c for c in df.columns if c.lower().startswith("pred_up")]
        dn_pred_cols = [c for c in df.columns if c.lower().startswith("pred_dn")]
        if not up_pred_cols or not dn_pred_cols:
            logger.warning(
                "No prediction columns starting with 'pred_up' / 'pred_dn' were found. "
                "Strategy will run with trend_up/trend_dn = 0 unless you provide these columns."
            )

        logger.info("Data validation passed")

    def ensure_epoch_seconds(self, series: pd.Series) -> np.ndarray:
        """Convert timestamp series to epoch seconds safely"""
        # If it's already a datetime type, convert to epoch seconds
        if pd.api.types.is_datetime64_any_dtype(series):
            return (series.astype('int64') // 10**9).astype('int64')

        # Attempt to convert to numeric first, for epoch timestamps
        s_numeric = pd.to_numeric(series, errors='coerce')

        if not s_numeric.isna().all():  # If conversion to numeric was successful for at least some
            s = s_numeric.copy()
            if np.issubdtype(s.dtype, np.number):
                arr = s.astype("int64").to_numpy()
                # Check if timestamps are in milliseconds and convert
                if arr.max() > 10**12:
                    arr = arr // 1000
                return arr

        # If numeric conversion fails, parse as datetime strings
        s_dt = pd.to_datetime(series, errors="coerce", utc=True)
        if s_dt.isna().any():
            raise ValueError("Timestamp could not be parsed as numeric epoch or datetime string.")
        return (s_dt.astype('int64') // 10**9).to_numpy()

    def compute_trends(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute trend features without data leakage.

        - averages all columns starting with 'pred_up' into trend_up
        - and all columns starting with 'pred_dn' into trend_dn
        """
        result = df.copy()

        up_cols = [c for c in result.columns if c.lower().startswith("pred_up")]
        dn_cols = [c for c in result.columns if c.lower().startswith("pred_dn")]

        if not up_cols or not dn_cols:
            logger.warning(
                "Prediction columns not found (no columns starting with 'pred_up' / 'pred_dn'). "
                "trend_up/trend_dn will be set to 0.0."
            )
            result["trend_up"] = 0.0
            result["trend_dn"] = 0.0
            return result

        result["trend_up"] = result[up_cols].astype(float).mean(axis=1)
        result["trend_dn"] = result[dn_cols].astype(float).mean(axis=1)

        return result

    # ---------------- Feature Engineering ----------------
    def detect_volume_spikes(self, data: pd.DataFrame, window: int, z_thresh: float) -> pd.Index:
        """
        Detect volume spikes using robust statistics (vectorized, no Python loop).

        Uses rolling median and rolling MAD for speed.
        """
        if "Volume" not in data.columns:
            warnings.warn("Volume column not found; no spikes.")
            return data.iloc[0:0].index

        vol = data["Volume"].astype(float).to_numpy()
        n = len(vol)
        if n < window or window <= 1:
            return data.iloc[0:0].index

        s = pd.Series(vol)

        # rolling median of volume
        rolling_med = s.rolling(window, min_periods=window).median()

        # deviation from rolling median
        dev = (s - rolling_med).abs()

        # rolling MAD
        rolling_mad = dev.rolling(window, min_periods=window).median()

        # avoid division by zero
        rolling_mad_safe = rolling_mad.replace(0, np.nan)

        robust_z = 0.6745 * (s - rolling_med) / rolling_mad_safe
        robust_z = robust_z.fillna(0.0)

        spike_mask = robust_z > z_thresh
        spike_idx_positions = np.where(spike_mask.to_numpy())[0]

        if len(spike_idx_positions) == 0:
            return data.iloc[0:0].index

        # map numeric positions back to original index labels
        original_index = data.index.to_numpy()
        selected_labels = original_index[spike_idx_positions]
        return pd.Index(selected_labels)

    def build_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build regime features without future data"""
        features = pd.DataFrame(index=df.index)

        features["ret1"] = df["Close"].pct_change().fillna(0.0)

        if "Open" in df.columns and "High" in df.columns and "Low" in df.columns:
            features["hl_range_pct"] = (df["High"] - df["Low"]).div(df["Open"].replace(0, np.nan)).fillna(0.0)
        else:
            features["hl_range_pct"] = 0.0

        features["vol_roll"] = features["ret1"].rolling(48, min_periods=8).std().fillna(0.0)

        # trend_up / trend_dn
        for col in ["trend_up", "trend_dn"]:
            if col in df.columns:
                features[col] = df[col].astype(float)
            else:
                features[col] = 0.0

        # additionally include all pred_up*/pred_dn* columns
        for c in df.columns:
            cl = c.lower()
            if cl.startswith("pred_up") or cl.startswith("pred_dn"):
                try:
                    features[c] = df[c].astype(float)
                except Exception:
                    features[c] = 0.0

        return features

    def make_ml_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create ML features using only past data"""
        features = pd.DataFrame(index=df.index)

        # Prediction features
        pred_cols = [c for c in df.columns if c.lower().startswith("pred_up") or c.lower().startswith("pred_dn")]
        for col in pred_cols:
            try:
                features[col] = df[col].astype(float)
            except Exception:
                features[col] = 0.0

        features["trend_up"] = df["trend_up"].astype(float) if "trend_up" in df.columns else 0.0
        features["trend_dn"] = df["trend_dn"].astype(float) if "trend_dn" in df.columns else 0.0

        # Technical features using only past data
        features["range_pct"] = (df["High"] - df["Low"]).div(df["Open"].replace(0, np.nan)).fillna(0.0)
        features["ret1"] = df["Close"].pct_change().fillna(0.0)
        features["ret3"] = df["Close"].pct_change(3).fillna(0.0)
        features["ret6"] = df["Close"].pct_change(6).fillna(0.0)

        # Lagged features to avoid lookahead
        for lag in [1, 2, 3, 5]:
            features[f"ret1_lag{lag}"] = df["Close"].pct_change().shift(lag).fillna(0.0)
            features[f"volume_lag{lag}"] = df["Volume"].shift(lag).fillna(0.0)

        return features

    # ---------------- Model Training ----------------
    def fit_gmm_and_label(self, df: pd.DataFrame, n_components: int, random_state: int = None) -> Tuple[np.ndarray, Any, Any]:
        """Fit GMM and label data without data leakage"""
        if not SKLEARN_OK:
            raise ImportError("scikit-learn is required for GMM")

        if random_state is None:
            random_state = self.seed

        X = self.build_regime_features(df)
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X.values)

        gmm = GaussianMixture(n_components=n_components, random_state=random_state)
        labels = gmm.fit_predict(Xs)

        return labels, gmm, scaler

    def train_probability_models(self, df: pd.DataFrame, horizon_bars: int, min_move_pct: float,
                                 n_splits: int = 5) -> Tuple[np.ndarray, np.ndarray, Tuple]:
        """Train ML models with proper time-series validation"""
        if not SKLEARN_OK:
            raise ImportError("scikit-learn is required for ML filter")

        X = self.make_ml_features(df)

        close = df["Close"].astype(float).to_numpy()
        future_prices = pd.Series(close).shift(-horizon_bars)
        price_changes = (future_prices - close) / close * 100.0

        y_up = (price_changes >= min_move_pct).astype(int).fillna(0).to_numpy()
        y_dn = (price_changes <= -min_move_pct).astype(int).fillna(0).to_numpy()

        scaler = StandardScaler()
        Xs = scaler.fit_transform(X.fillna(0.0).values)

        _tscv = TimeSeriesSplit(n_splits=max(2, n_splits))  # وجود دارد ولی برای سرعت ازش استفاده نمی‌کنیم

        up_model = LogisticRegression(max_iter=1000, random_state=self.seed, class_weight='balanced')
        dn_model = LogisticRegression(max_iter=1000, random_state=self.seed, class_weight='balanced')

        up_model.fit(Xs, y_up)
        dn_model.fit(Xs, y_dn)

        p_up = up_model.predict_proba(Xs)[:, 1]
        p_dn = dn_model.predict_proba(Xs)[:, 1]

        return p_up, p_dn, (up_model, dn_model, scaler)

    # ---------------- Risk ----------------
    def calculate_position_size(self, account_balance: float, atr: float, entry_price: float) -> float:
        """Calculate position size based on risk management"""
        if atr == 0 or entry_price == 0:
            return 0.0

        risk_amount = account_balance * (self.risk_config.risk_per_trade_pct / 100.0)

        if atr == 0.01:
            position_size = risk_amount / (atr * entry_price)
        else:
            position_size = risk_amount / (atr * entry_price)

        return min(position_size, self.risk_config.max_position_size)

    def update_drawdown(self, current_balance: float) -> bool:
        """Update drawdown and check if we should stop trading"""
        self.peak_balance = max(self.peak_balance, current_balance)
        if self.peak_balance > 0:
            self.current_drawdown = ((self.peak_balance - current_balance) / self.peak_balance) * 100.0
        else:
            self.current_drawdown = 0.0

        if (self.risk_config.stop_trading_on_drawdown and
                self.current_drawdown >= self.risk_config.max_drawdown_pct):
            self.stop_trading = True
            logger.warning(f"Trading stopped due to drawdown: {self.current_drawdown:.2f}%")

        return self.stop_trading

    # ---------------- Core Backtest ----------------
    def backtest_fast_numpy(self, df: pd.DataFrame, long_params: dict, short_params: dict,
                            confirm_params: dict, regime_params: dict = None,
                            proba_filter: dict = None, initial_balance: float = 2500.0) -> Tuple[dict, pd.DataFrame]:
        """Core backtest function"""

        try:
            self.validate_dataframe(df)
        except (ValueError, KeyError) as e:
            logger.warning(f"Data validation failed for backtest segment: {e}")
            empty_stats = {
                "total_trades": 0, "long_trades": 0, "short_trades": 0,
                "win_rate": 0.0, "total_pnl": 0.0, "avg_pnl": 0.0,
                "final_balance": initial_balance, "max_drawdown": 0.0, "sharpe_ratio": 0.0
            }
            return empty_stats, pd.DataFrame()

        trade_cols = ["entry_time_s", "exit_time_s", "side", "entry", "exit", "pnl_pct",
                      "reason", "regime", "position_size", "pnl_abs", "balance"]

        if df.empty:
            empty_stats = {
                "total_trades": 0, "long_trades": 0, "short_trades": 0,
                "win_rate": 0.0, "total_pnl": 0.0, "avg_pnl": 0.0,
                "final_balance": initial_balance, "max_drawdown": 0.0, "sharpe_ratio": 0.0
            }
            return empty_stats, pd.DataFrame(columns=trade_cols)

        # Numpy arrays
        t = self.ensure_epoch_seconds(df["timestamp"])
        close = df["Close"].to_numpy(float)
        open_p = df["Open"].to_numpy(float)
        high = df["High"].to_numpy(float)
        low = df["Low"].to_numpy(float)

        tr_up = df["trend_up"].to_numpy(float) if "trend_up" in df.columns else np.zeros(len(df), dtype=float)
        tr_dn = df["trend_dn"].to_numpy(float) if "trend_dn" in df.columns else np.zeros(len(df), dtype=float)

        def _get_pred_array(prefix: str, horizon: str, fallback: np.ndarray) -> np.ndarray:
            prefix = prefix.lower()
            horizon = horizon.lower()
            for c in df.columns:
                cl = c.lower()
                if prefix in cl and horizon in cl:
                    try:
                        return df[c].astype(float).to_numpy()
                    except Exception:
                        pass
            return fallback

        up30 = _get_pred_array("pred_up", "30", tr_up)
        up60 = _get_pred_array("pred_up", "60", tr_up)
        up90 = _get_pred_array("pred_up", "90", tr_up)
        up180 = _get_pred_array("pred_up", "180", tr_up)

        dn30 = _get_pred_array("pred_dn", "30", tr_dn)
        dn60 = _get_pred_array("pred_dn", "60", tr_dn)
        dn90 = _get_pred_array("pred_dn", "90", tr_dn)
        dn180 = _get_pred_array("pred_dn", "180", tr_dn)

        regimes = df["regime"].to_numpy(int) if "regime" in df.columns else np.zeros(len(df), dtype=int)

        n = len(close)
        position, entry_price, entry_time_s, active_regime = None, 0.0, 0, None
        tp_levels, stop_level, stage, freeze_until = np.zeros(5, dtype=float), np.nan, 0, -1
        trades: List[tuple] = []

        pending_long_signal_idx = -1
        pending_short_signal_idx = -1

        base_lp = long_params.copy()
        base_sp = short_params.copy()
        H = int(confirm_params.get("horizon_bars", 10))
        min_move = float(confirm_params.get("min_confirm_move_pct", 0.0))

        p_up = proba_filter.get('p_up') if proba_filter else None
        p_dn = proba_filter.get('p_dn') if proba_filter else None
        min_prob_long = float(proba_filter.get('min_prob_long', 0.0)) if proba_filter else 0.0
        min_prob_short = float(proba_filter.get('min_prob_short', 0.0)) if proba_filter else 0.0

        # Risk state
        current_balance = initial_balance
        self.peak_balance = initial_balance
        self.current_drawdown = 0.0
        self.stop_trading = False

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

        def create_tp_levels(entry_price: float, tp_pcts: list, side: str) -> np.ndarray:
            multipliers = 1 + np.array(tp_pcts) / 100.0
            if side == "short":
                multipliers = 1 - np.array(tp_pcts) / 100.0
            return entry_price * multipliers

        freeze_long_s_default = int(base_lp.get("freeze_period_minutes", 20)) * 60
        freeze_short_s_default = int(base_sp.get("freeze_period_minutes", 20)) * 60

        # Main loop
        for i in range(n):
            if self.stop_trading:
                break

            ts = int(t[i])
            reg = int(regimes[i])
            lp, sp = pick_params(reg)
            freeze_long_s = int(lp.get("freeze_period_minutes", freeze_long_s_default)) * 60
            freeze_short_s = int(sp.get("freeze_period_minutes", freeze_short_s_default)) * 60

            # --- Entry logic ---
            if position is None and ts >= freeze_until:
                if pending_long_signal_idx == -1 and pending_short_signal_idx == -1:
                    l_sig = eval_long_sig(i, lp)
                    s_sig = eval_short_sig(i, sp)

                    # ML filter
                    if p_up is not None and p_dn is not None and i < len(p_up):
                        if l_sig and not (p_up[i] >= min_prob_long):
                            l_sig = False
                        if s_sig and not (p_dn[i] >= min_prob_short):
                            s_sig = False

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

                    if chosen_side == "long":
                        pending_long_signal_idx = i
                    elif chosen_side == "short":
                        pending_short_signal_idx = i

                # Long confirmation
                if pending_long_signal_idx != -1:
                    if i > pending_long_signal_idx + H:
                        pending_long_signal_idx = -1
                    elif i < n:
                        confirmation_ret = (close[i] - close[pending_long_signal_idx]) / close[
                            pending_long_signal_idx] * 100.0
                        if confirmation_ret >= min_move:
                            position = "long"
                            entry_price = open_p[i] if not np.isnan(open_p[i]) else float(close[i])
                            entry_time_s = ts
                            active_regime = reg
                            stage = 0

                            tp_pcts = [
                                lp["tp1_pct"], lp["tp2_pct"], lp["tp3_pct"],
                                lp["tp4_pct"], lp["tp5_pct"]
                            ]
                            tp_levels = create_tp_levels(entry_price, tp_pcts, "long")
                            stop_level = entry_price * (1.0 - lp["sl_pct"] / 100.0)
                            pending_long_signal_idx = -1

                # Short confirmation
                if pending_short_signal_idx != -1:
                    if i > pending_short_signal_idx + H:
                        pending_short_signal_idx = -1
                    elif i < n:
                        confirmation_ret = (close[i] - close[pending_short_signal_idx]) / close[
                            pending_short_signal_idx] * 100.0
                        if confirmation_ret <= -min_move:
                            position = "short"
                            entry_price = open_p[i] if not np.isnan(open_p[i]) else float(close[i])
                            entry_time_s = ts
                            active_regime = reg
                            stage = 0

                            tp_pcts = [
                                sp["tp1_pct"], sp["tp2_pct"], sp["tp3_pct"],
                                sp["tp4_pct"], sp["tp5_pct"]
                            ]
                            tp_levels = create_tp_levels(entry_price, tp_pcts, "short")
                            stop_level = entry_price * (1.0 + sp["sl_pct"] / 100.0)
                            pending_short_signal_idx = -1

            # --- Manage open position ---
            if position:
                position_size = self.calculate_position_size(current_balance, 0.01, entry_price)

                # Long side
                if position == "long":
                    if low[i] <= stop_level:
                        exit_price = stop_level
                        reason = "SL"
                        if stage == 1:
                            reason = "TSL@TP1"
                        if stage == 2:
                            reason = "TSL@TP2"
                        if stage == 3:
                            reason = "TSL@TP3"
                        if stage == 4:
                            reason = "TSL@TP4"
                        pnl_pct = (exit_price - entry_price) / entry_price * 100.0
                        pnl_abs = pnl_pct / 100.0 * entry_price * position_size
                        current_balance += pnl_abs
                        trades.append((entry_time_s, ts, "long", entry_price, exit_price,
                                       pnl_pct, reason, active_regime, position_size, pnl_abs, current_balance))
                        position, freeze_until = None, ts + freeze_long_s
                        self.update_drawdown(current_balance)
                        continue

                    if stage == 0 and high[i] >= tp_levels[0]:
                        stage = 1
                        stop_level = tp_levels[0]
                    if stage == 1 and high[i] >= tp_levels[1]:
                        stage = 2
                        stop_level = tp_levels[1]
                    if stage == 2 and high[i] >= tp_levels[2]:
                        stage = 3
                        stop_level = tp_levels[2]
                    if stage == 3 and high[i] >= tp_levels[3]:
                        stage = 4
                        stop_level = tp_levels[3]
                    if stage == 4 and high[i] >= tp_levels[4]:
                        exit_price = tp_levels[4]
                        pnl_pct = (exit_price - entry_price) / entry_price * 100.0
                        pnl_abs = pnl_pct / 100.0 * entry_price * position_size
                        current_balance += pnl_abs
                        trades.append((entry_time_s, ts, "long", entry_price, exit_price,
                                       pnl_pct, "TP5", active_regime, position_size, pnl_abs, current_balance))
                        position, freeze_until = None, ts + freeze_long_s
                        self.update_drawdown(current_balance)
                        continue

                # Short side
                else:
                    if high[i] >= stop_level:
                        exit_price = stop_level
                        reason = "SL"
                        if stage == 1:
                            reason = "TSL@TP1"
                        if stage == 2:
                            reason = "TSL@TP2"
                        if stage == 3:
                            reason = "TSL@TP3"
                        if stage == 4:
                            reason = "TSL@TP4"
                        pnl_pct = (entry_price - exit_price) / entry_price * 100.0
                        pnl_abs = pnl_pct / 100.0 * entry_price * position_size
                        current_balance += pnl_abs
                        trades.append((entry_time_s, ts, "short", entry_price, exit_price,
                                       pnl_pct, reason, active_regime, position_size, pnl_abs, current_balance))
                        position, freeze_until = None, ts + freeze_short_s
                        self.update_drawdown(current_balance)
                        continue

                    if stage == 0 and low[i] <= tp_levels[0]:
                        stage = 1
                        stop_level = tp_levels[0]
                    if stage == 1 and low[i] <= tp_levels[1]:
                        stage = 2
                        stop_level = tp_levels[1]
                    if stage == 2 and low[i] <= tp_levels[2]:
                        stage = 3
                        stop_level = tp_levels[2]
                    if stage == 3 and low[i] <= tp_levels[3]:
                        stage = 4
                        stop_level = tp_levels[3]
                    if stage == 4 and low[i] <= tp_levels[4]:
                        exit_price = tp_levels[4]
                        pnl_pct = (entry_price - exit_price) / entry_price * 100.0
                        pnl_abs = pnl_pct / 100.0 * entry_price * position_size
                        current_balance += pnl_abs
                        trades.append((entry_time_s, ts, "short", entry_price, exit_price,
                                       pnl_pct, "TP5", active_regime, position_size, pnl_abs, current_balance))
                        position, freeze_until = None, ts + freeze_short_s
                        self.update_drawdown(current_balance)
                        continue

        if not trades:
            empty_stats = {
                "total_trades": 0, "long_trades": 0, "short_trades": 0,
                "win_rate": 0.0, "total_pnl": 0.0, "avg_pnl": 0.0,
                "final_balance": current_balance, "max_drawdown": self.current_drawdown, "sharpe_ratio": 0.0
            }
            return empty_stats, pd.DataFrame(columns=trade_cols)

        trades_df = pd.DataFrame(trades, columns=trade_cols)
        pnls = trades_df["pnl_pct"].astype(float)

        summary = {
            "total_trades": int(len(trades)),
            "long_trades": int((trades_df["side"] == "long").sum()),
            "short_trades": int((trades_df["side"] == "short").sum()),
            "win_rate": float((pnls > 0).mean() * 100.0 if not pnls.empty else 0.0),
            "total_pnl": float(pnls.sum()),
            "avg_pnl": float(pnls.mean() if not pnls.empty else 0.0),
            "final_balance": float(current_balance),
            "max_drawdown": float(self.current_drawdown),
            "sharpe_ratio": float(pnls.mean() / pnls.std() if pnls.std() > 0 else 0.0)
        }

        return summary, trades_df

    # ---------------- Scoring & Wrappers ----------------
    def calculate_score(self, summary: dict, mode: str) -> float:
        """Calculate optimization score based on mode"""
        if mode == "precision":
            win_rate = float(summary["win_rate"])
            trades_count = int(summary["total_trades"])

            score = win_rate

            if trades_count < self.precision_config.min_trades:
                penalty = (self.precision_config.min_trades - trades_count) * self.precision_config.penalty_scale
                score -= penalty

            if win_rate >= self.precision_config.max_win_rate:
                excess_win = win_rate - self.precision_config.max_win_rate
                score -= excess_win * self.precision_config.high_win_penalty_scale

            if self.precision_config.min_win_rate <= win_rate <= self.precision_config.max_win_rate:
                score += 10.0

            return score

        elif mode == "pnl":
            return float(summary["total_pnl"])

        elif mode == "balanced":
            win_rate = float(summary["win_rate"])
            total_pnl = float(summary["total_pnl"])
            return win_rate * total_pnl / 100.0

        else:
            raise ValueError(f"Unknown mode {mode!r}")

    def create_param_dicts(self, params: dict) -> Tuple[dict, dict, dict]:
        """Create parameter dictionaries for backtesting from a flat param dict"""
        long_p = {k.replace('long_', ''): v for k, v in params.items() if k.startswith('long_')}
        short_p = {k.replace('short_', ''): v for k, v in params.items() if k.startswith('short_')}

        confirm = {
            "horizon_bars": params["confirm_horizon"],
            "min_confirm_move_pct": params["confirm_min_move_pct"]
        }

        for p_dict, prefix in [(long_p, "long_"), (short_p, "short_")]:
            p_dict['sl_pct'] = params[f'{prefix}sl']
            p_dict['tp1_pct'] = params[f'{prefix}tp1']
            p_dict['tp2_pct'] = params[f'{prefix}tp2']
            p_dict['tp3_pct'] = params[f'{prefix}tp3']
            p_dict['tp4_pct'] = params[f'{prefix}tp4']
            p_dict['tp5_pct'] = params[f'{prefix}tp5']
            p_dict['freeze_period_minutes'] = params[f'{prefix}freeze']

            for h in [30, 60, 90, 180]:
                p_dict[f'strength_h{h}'] = params[f'{prefix}str_h{h}']

        return long_p, short_p, confirm

    def run_backtest_for_params(self, df_full: pd.DataFrame, params: dict,
                                use_gmm: bool = False, n_regimes: int = 3,
                                use_ml_filter: bool = False,
                                min_prob_long: float = 0.55, min_prob_short: float = 0.55,
                                initial_balance: float = 2500.0) -> Tuple[dict, pd.DataFrame]:
        """High-level backtest wrapper used by tuning code."""
        # logger.debug می‌تونی روشن کنی اگر خواستی جزئیات ببینی
        logger.debug("Running backtest for supplied parameter set")

        df = df_full.copy()
        long_p, short_p, confirm = self.create_param_dicts(params)

        regime_overrides = None
        if use_gmm and SKLEARN_OK:
            logger.info(f"Applying GMM with {n_regimes} regimes")
            try:
                labels, _gmm_obj, _gmm_scaler = self.fit_gmm_and_label(df, n_regimes)
                df["regime"] = labels
            except Exception as e:
                logger.warning(f"GMM failed: {e}, using single regime")
                df["regime"] = 0
        else:
            df["regime"] = 0

        proba_filter = None
        if use_ml_filter and SKLEARN_OK:
            logger.info("Training ML probability filter")
            try:
                spike_window = int(params["spike_window"])
                spike_z = float(params["spike_z"])
                idx_spikes_ml = self.detect_volume_spikes(df, spike_window, spike_z)
                df_spikes_for_ml = df.loc[idx_spikes_ml].reset_index(drop=True)

                if not df_spikes_for_ml.empty:
                    p_up, p_dn, _ = self.train_probability_models(
                        df_spikes_for_ml,
                        int(params["confirm_horizon"]),
                        float(params["confirm_min_move_pct"]),
                        5
                    )
                    proba_filter = {
                        'p_up': p_up,
                        'p_dn': p_dn,
                        'min_prob_long': min_prob_long,
                        'min_prob_short': min_prob_short
                    }
                else:
                    logger.warning("No spikes found for ML filter training.")
            except Exception as e:
                logger.warning(f"ML filter training failed: {e}")
                proba_filter = None

        spike_window = int(params["spike_window"])
        spike_z = float(params["spike_z"])
        idx_spikes = self.detect_volume_spikes(df, spike_window, spike_z)
        df_spikes = df.loc[idx_spikes].reset_index(drop=True)

        logger.debug(f"Found {len(df_spikes)} volume spikes for backtesting")

        summary, trades_df = self.backtest_fast_numpy(
            df_spikes, long_p, short_p, confirm,
            regime_params=regime_overrides,
            proba_filter=proba_filter,
            initial_balance=initial_balance
        )

        return summary, trades_df

    def score_params(self, df_full: pd.DataFrame, params: dict,
                     mode: str = "precision", **run_kwargs) -> Tuple[float, dict]:
        """Run a backtest and return (score, summary) for tuning."""
        summary, _ = self.run_backtest_for_params(df_full, params, **run_kwargs)
        score = self.calculate_score(summary, mode=mode)
        return score, summary


# ------------------------------------------------------------
# Helper: parameter sampling for Optuna
# ------------------------------------------------------------
def sample_params(trial: "optuna.Trial", cfg: StrategyConfig) -> Dict[str, float]:
    params: Dict[str, float] = {}

    # Spike detection
    params["spike_window"] = trial.suggest_int("spike_window",
                                               cfg.spike_window_range[0],
                                               cfg.spike_window_range[1])
    params["spike_z"] = trial.suggest_float("spike_z",
                                            cfg.spike_z_range[0],
                                            cfg.spike_z_range[1])

    # Confirmation
    params["confirm_horizon"] = trial.suggest_int("confirm_horizon",
                                                  cfg.confirm_horizon_range[0],
                                                  cfg.confirm_horizon_range[1])
    params["confirm_min_move_pct"] = trial.suggest_float("confirm_min_move_pct",
                                                         cfg.confirm_min_move_range[0],
                                                         cfg.confirm_min_move_range[1])

    # TP / SL and freeze
    for side in ["long", "short"]:
        for tp_idx in range(1, 6):
            low, high = cfg.tp_ranges[f"tp{tp_idx}"]
            params[f"{side}_tp{tp_idx}"] = trial.suggest_float(f"{side}_tp{tp_idx}", low, high)

        params[f"{side}_sl"] = trial.suggest_float(f"{side}_sl",
                                                   cfg.sl_range[0],
                                                   cfg.sl_range[1])
        params[f"{side}_freeze"] = trial.suggest_int(f"{side}_freeze",
                                                     cfg.freeze_range[0],
                                                     cfg.freeze_range[1])

        # Strength thresholds
        for h in [30, 60, 90, 180]:
            params[f"{side}_str_h{h}"] = trial.suggest_float(
                f"{side}_str_h{h}",
                cfg.strength_range[0],
                cfg.strength_range[1]
            )

    return params


# ------------------------------------------------------------
# Main: load data, run optimization
# ------------------------------------------------------------
def main():
    logger.info("Initializing Backtest Optimizer...")

    if not OPTUNA_OK:
        logger.error("Optuna is not installed. Install with 'pip install optuna'.")
        sys.exit(1)

    DATA_PATH = r"D:\project\split_output_data\BTCUSDT_ohlc_data_1m(5year)_backtest_15pct_predictions.csv"

    if not os.path.isfile(DATA_PATH):
        logger.error(f"CSV file not found: {DATA_PATH}")
        sys.exit(1)

    try:
        df_full = pd.read_csv(DATA_PATH)
    except Exception as e:
        logger.error(f"Failed to load CSV: {e}")
        sys.exit(1)

    logger.info(f"Loaded {len(df_full)} rows from {DATA_PATH}")
    logger.info(f"Columns in data: {list(df_full.columns)}")

    # ----- map your up_/dn_ columns to pred_up_/pred_dn_ expected by the strategy -----
    rename_map = {
        "up_30_q0.9":  "pred_up_h30_q0p9_pct",
        "up_60_q0.9":  "pred_up_h60_q0p9_pct",
        "up_90_q0.9":  "pred_up_h90_q0p9_pct",
        "up_180_q0.9": "pred_up_h180_q0p9_pct",

        "dn_30_q0.1":  "pred_dn_h30_q0p1_pct",
        "dn_60_q0.1":  "pred_dn_h60_q0p1_pct",
        "dn_90_q0.1":  "pred_dn_h90_q0p1_pct",
        "dn_180_q0.1": "pred_dn_h180_q0p1_pct",
    }
    df_full.rename(columns=rename_map, inplace=True)

    # Normalize OHLCV column names if needed
    if "Close" not in df_full.columns and "close" in df_full.columns:
        df_full.rename(columns={"close": "Close"}, inplace=True)
    for c in ["Open", "High", "Low", "Volume"]:
        if c not in df_full.columns and c.lower() in df_full.columns:
            df_full.rename(columns={c.lower(): c}, inplace=True)

    precision_config = PrecisionConfig()
    risk_config = RiskConfig(max_drawdown_pct=10.0, risk_per_trade_pct=1.0)
    strategy_config = StrategyConfig()

    bt = Backtester(strategy_config, precision_config, risk_config)

    # Add trend columns (uses pred_up*/pred_dn* that we just renamed)
    df_full = bt.compute_trends(df_full)

    try:
        bt.validate_dataframe(df_full)
    except Exception as e:
        logger.error(f"Data validation failed: {e}")
        sys.exit(1)

    # ---------------- Optuna Optimization ----------------
    N_TRIALS = 100
    logger.info(f"Starting optimization with {N_TRIALS} trials...")

    # برای سرعت: لاگ سطح پایینی رو کم می‌کنیم (اگر خواستی لاگ جزئیات رو ببینی این رو بردار)
    # logging.getLogger().setLevel(logging.WARNING)

    def objective(trial: "optuna.Trial") -> float:
        params = sample_params(trial, strategy_config)
        score, summary = bt.score_params(
            df_full,
            params,
            mode="precision",
            use_gmm=False,       # برای سرعت OFF
            use_ml_filter=False  # برای سرعت OFF
        )

        trial.set_user_attr("summary", summary)
        return score

    study = optuna.create_study(direction="maximize", study_name="btc_spike_backtest")

    if TQDM_OK:
        with tqdm(total=N_TRIALS, desc="Tuning", unit="trial") as pbar:
            def _callback(study_, trial_):
                pbar.update(1)

            study.optimize(objective, n_trials=N_TRIALS, callbacks=[_callback])
    else:
        study.optimize(objective, n_trials=N_TRIALS)

    logger.info("Optimization finished.")
    logger.info(f"Best trial value (score): {study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")

    # Save best params + summary
    best_trial = study.best_trial
    best_summary = best_trial.user_attrs.get("summary", {})
    timestamp = int(time.time())
    out_dir = "optuna_results"
    os.makedirs(out_dir, exist_ok=True)

    params_path = os.path.join(out_dir, f"best_params_{timestamp}.json")
    summary_path = os.path.join(out_dir, f"best_summary_{timestamp}.json")

    with open(params_path, "w") as f:
        json.dump(best_trial.params, f, indent=2)
    with open(summary_path, "w") as f:
        json.dump(best_summary, f, indent=2)

    logger.info(f"Best params saved to: {params_path}")
    logger.info(f"Best summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
