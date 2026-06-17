"""meta_labeler.py — the equity meta-labeling model + a time-aware CV splitter.

Deliberately separate from the options `Learner` (which validates with shuffled
k-fold — invalid for time series). Here:

  - PurgedWalkForwardCV: forward-only expanding folds with an embargo gap between
    train and test, so no future or adjacent-leakage sample informs the test fold.
  - build_pipeline / select_features: a small, regularized classifier (logistic
    regression) over the captured feature columns — kept simple for small data.
  - MetaModel: loads the trained artifact at runtime and scores P(win) for a live
    signal (used in shadow mode, and for gating once activated).

Training/evaluation/gating lives in train_meta.py; this module is the shared,
import-safe core so backtest-time training and live-time scoring never diverge.
"""
import logging

import numpy as np
import pandas as pd

import config

log = logging.getLogger("meta_labeler")

# Columns that are identifiers / outcomes / audit — never model inputs.
NON_FEATURES = {
    "signal_id", "ts", "symbol", "src_max_ts",
    "label", "r_multiple", "pnl", "won", "exit_reason", "entry_time", "exit_time",
}
CATEGORICAL = ["strategy", "direction"]


def select_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Pick model-input columns from a captured-signals frame. Returns (X, numeric,
    categorical). Numeric = everything that isn't an identifier/outcome or a known
    categorical; coerced to numeric so stray strings don't poison the matrix."""
    cats = [c for c in CATEGORICAL if c in df.columns]
    numeric = [c for c in df.columns
               if c not in NON_FEATURES and c not in cats]
    X = df[numeric + cats].copy()
    X[numeric] = X[numeric].apply(pd.to_numeric, errors="coerce")
    for c in cats:
        X[c] = X[c].astype("object").where(X[c].notna(), "NA").astype(str)
    return X, numeric, cats


def build_pipeline(numeric: list[str], categorical: list[str]):
    """Small, regularized, reproducible. Logistic regression keeps variance low on
    tiny samples and gives calibrated-ish probabilities."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    num_pipe = Pipeline([("impute", SimpleImputer(strategy="median")),
                         ("scale", StandardScaler())])
    cat_pipe = Pipeline([("oh", OneHotEncoder(handle_unknown="ignore"))])
    pre = ColumnTransformer([("num", num_pipe, numeric),
                             ("cat", cat_pipe, categorical)])
    clf = LogisticRegression(C=0.5, max_iter=1000, class_weight="balanced",
                             random_state=config.RANDOM_SEED)
    return Pipeline([("prep", pre), ("clf", clf)])


class PurgedWalkForwardCV:
    """Forward-only walk-forward splitter with an embargo. For samples ordered by
    time, each fold trains on an expanding prefix and tests on the next contiguous
    block, with `embargo` samples dropped between them to purge adjacency leakage.
    No shuffling, no look-ahead."""

    def __init__(self, n_splits: int = 4, embargo: int = 2):
        self.n_splits = n_splits
        self.embargo = embargo

    def split(self, n: int):
        fold = n // (self.n_splits + 1)
        if fold < 1:
            return
        for i in range(1, self.n_splits + 1):
            train_end = i * fold
            test_start = train_end + self.embargo
            test_end = (test_start + fold) if i < self.n_splits else n
            if test_start >= n:
                break
            train_idx = np.arange(0, train_end)
            test_idx = np.arange(test_start, min(test_end, n))
            if len(test_idx) == 0 or len(train_idx) == 0:
                continue
            yield train_idx, test_idx


class MetaModel:
    """Runtime wrapper: load the trained artifact and score live signals.
    Absent/*unusable* artifact -> `ready=False`, and callers treat every signal as
    a take (the model can never silently block when it hasn't earned the right)."""

    def __init__(self, path: str | None = None):
        self.path = path or config.META_MODEL_FILE
        self.pipeline = None
        self.numeric: list[str] = []
        self.categorical: list[str] = []
        self.threshold = config.EQ_META_THRESHOLD
        self.metrics: dict = {}
        self._load()

    @property
    def ready(self) -> bool:
        return self.pipeline is not None

    def _load(self):
        import os
        if not os.path.exists(self.path):
            return
        try:
            import joblib
            saved = joblib.load(self.path)
            self.pipeline = saved["pipeline"]
            self.numeric = saved["numeric"]
            self.categorical = saved["categorical"]
            self.threshold = saved.get("threshold", self.threshold)
            self.metrics = saved.get("metrics", {})
            log.info("MetaModel loaded (OOS AUC %.3f, threshold %.2f)",
                     self.metrics.get("auc", float("nan")), self.threshold)
        except Exception:
            log.exception("MetaModel artifact unreadable — staying inactive")
            self.pipeline = None

    def score(self, features: dict) -> float | None:
        """P(win) for one captured feature dict, or None if no usable model."""
        if not self.ready:
            return None
        row = pd.DataFrame([features])
        for c in self.numeric:
            row[c] = pd.to_numeric(row.get(c), errors="coerce")
        for c in self.categorical:
            row[c] = str(row.get(c, pd.Series(["NA"])).iloc[0]) if c in row else "NA"
        try:
            return float(self.pipeline.predict_proba(row[self.numeric + self.categorical])[0, 1])
        except Exception:
            log.exception("MetaModel scoring failed — treat as take")
            return None

    def decide(self, prob: float | None) -> bool:
        """Would we TAKE this trade? True if no model (can't block) or prob>=threshold."""
        if prob is None:
            return True
        return prob >= self.threshold
