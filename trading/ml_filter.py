"""learner.py — ML win-probability filter that learns from the bot's own trades.

How it works:
  1. Every entry logs its signal features via trade_store.
  2. After ML_MIN_TRADES closed trades, a gradient-boosting classifier is
     trained: features at entry -> did the trade win?
  3. New signals are scored. If the model has proven predictive power
     (cross-validated AUC >= ML_MIN_AUC) it BLOCKS entries scoring below
     ML_WIN_PROB_THRESHOLD. If not, it stays ADVISORY — scores are logged
     so you can judge it, but it can't veto trades it hasn't earned the
     right to veto. This guards against a noise-fit model on small data.
  4. Retrains automatically every ML_RETRAIN_EVERY new closed trades.
"""
import datetime as dt
import logging
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

import config
import data.trade_store as trade_store
from utils import now_et

log = logging.getLogger("trading.ml_filter")

NUMERIC = trade_store.FEATURE_NUMERIC
CATEGORICAL = ["ticker", "type", "strategy"]
FEATURES = trade_store.FEATURE_COLS


def extract_features(signal, pick) -> dict:
    """Snapshot of everything the bot knew at entry time. Logged with the
    trade so the model can learn which setups actually win."""
    t = signal.time
    market_open = t.replace(hour=9, minute=30, second=0, microsecond=0)
    return {
        "momentum_pct": round(signal.momentum_pct, 3),
        "day_change_pct": round(signal.day_change_pct, 3),
        "rsi": round(signal.rsi, 2),
        "rel_volume": round(signal.rel_volume, 2),
        "vwap_dist_pct": round(signal.vwap_dist_pct, 3),
        "iv": round(pick.iv, 4) if pick.iv is not None else "",
        "spread": round(pick.spread, 3),
        "dte": (pick.expiry - t.date()).days,
        "minutes_since_open": round((t - market_open).total_seconds() / 60),
        "ticker": signal.symbol,
        "type": pick.otype,
        "strategy": signal.strategy,
    }


def _build_pipeline() -> Pipeline:
    # Shallow trees + slow learning rate — small-data overfitting protection.
    return Pipeline([
        ("prep", ColumnTransformer([
            ("num", SimpleImputer(strategy="median"), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ])),
        ("clf", GradientBoostingClassifier(
            n_estimators=100, max_depth=2, learning_rate=0.05,
            subsample=0.8, random_state=42,
        )),
    ])


class Learner:
    def __init__(self):
        self.pipeline: Pipeline | None = None
        self.auc: float = float("nan")
        self.gating: bool = False          # may the model veto entries?
        self.trained_on: int = 0           # trade count at last training
        if config.ML_ENABLED:
            self._load()
            self.maybe_retrain()

    # ---------------- persistence ----------------

    def _load(self):
        if not os.path.exists(config.MODEL_FILE):
            return
        try:
            saved = joblib.load(config.MODEL_FILE)
            self.pipeline = saved["pipeline"]
            self.auc = saved["auc"]
            self.gating = saved["gating"]
            self.trained_on = saved["trained_on"]
            log.info("Loaded model (trained on %d trades, AUC %.2f, %s)",
                     self.trained_on, self.auc,
                     "gating" if self.gating else "advisory")
        except Exception:
            log.exception("model.pkl unreadable — will retrain from trades.csv")

    def _save(self):
        joblib.dump({"pipeline": self.pipeline, "auc": self.auc,
                     "gating": self.gating, "trained_on": self.trained_on},
                    config.MODEL_FILE)

    # ---------------- training ----------------

    def _training_data(self) -> tuple[pd.DataFrame, pd.Series] | None:
        return trade_store.training_data()

    def maybe_retrain(self, force: bool = False):
        """Retrain when enough new trades have accumulated."""
        if not config.ML_ENABLED:
            return
        data = self._training_data()
        if data is None:
            return
        X, y = data
        n = len(X)
        if n < config.ML_MIN_TRADES:
            log.info("Learner collecting data: %d/%d trades before first training",
                     n, config.ML_MIN_TRADES)
            return
        if not force and n - self.trained_on < config.ML_RETRAIN_EVERY and self.pipeline:
            return
        if y.nunique() < 2:
            log.info("Learner: all %d trades have the same outcome — cannot train yet", n)
            return

        pipe = _build_pipeline()
        # Honest skill estimate before granting veto power.
        folds = min(5, int(y.value_counts().min()))
        if folds >= 2:
            cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
            try:
                self.auc = float(np.mean(
                    cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")))
            except Exception:
                log.exception("Cross-validation failed")
                self.auc = float("nan")
        else:
            self.auc = float("nan")

        pipe.fit(X, y)
        self.pipeline = pipe
        self.trained_on = n
        self.gating = bool(self.auc >= config.ML_MIN_AUC)
        self._save()
        log.info("[learner] retrained on %d trades, CV AUC %.2f — %s",
                 n, self.auc,
                 "GATING (can block entries)" if self.gating
                 else f"advisory only (needs AUC >= {config.ML_MIN_AUC})")
        self._log_top_factors(X, y)

    def _log_top_factors(self, X: pd.DataFrame, y: pd.Series):
        """Surface what the model thinks drives wins/losses."""
        try:
            prep = self.pipeline.named_steps["prep"]
            clf = self.pipeline.named_steps["clf"]
            names = prep.get_feature_names_out()
            imp = sorted(zip(names, clf.feature_importances_),
                         key=lambda x: -x[1])[:5]
            pretty = ", ".join(f"{n.split('__')[-1]} ({v:.2f})" for n, v in imp)
            log.info("[learner] top factors: %s", pretty)
        except Exception:
            pass  # diagnostics only

    # ---------------- scoring ----------------

    def score(self, features: dict) -> float | None:
        """P(win) for a prospective entry, or None if no model yet."""
        if not config.ML_ENABLED or self.pipeline is None:
            return None
        # Build single-row DataFrame with columns pre-ordered to match training
        ordered = {col: features.get(col) for col in FEATURES}
        row = pd.DataFrame([ordered])
        for col in NUMERIC:
            row[col] = pd.to_numeric(row[col], errors="coerce")
        try:
            return float(self.pipeline.predict_proba(row)[0, 1])
        except Exception:
            log.exception("Scoring failed — allowing entry")
            return None

    def allows(self, features: dict) -> tuple[bool, float | None]:
        """(entry_allowed, win_prob). Only blocks when the model is gating."""
        prob = self.score(features)
        if prob is None:
            return True, None
        if self.gating and prob < config.ML_WIN_PROB_THRESHOLD:
            return False, prob
        return True, prob


if __name__ == "__main__":
    # Manual check: python learner.py — retrains from trades.csv and reports.
    learner = Learner()
    learner.maybe_retrain(force=True)
    if learner.pipeline is None:
        print("Not enough trade data yet "
              f"(need {config.ML_MIN_TRADES} closed trades with features).")
    else:
        print(f"Model trained on {learner.trained_on} trades | "
              f"CV AUC {learner.auc:.2f} | "
              f"{'GATING' if learner.gating else 'advisory'}")
