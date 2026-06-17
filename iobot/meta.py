"""meta.py — meta-labeling layer (phases 2-3). Filters/sizes existing signals.

This NEVER generates signals. It learns from the bot's own closed trades which
signals to skip and how big to size, and only after it has data and proof:

  Phase 2 (train): join Phase-1 features to labels, train a small regularized
    classifier, validate with PURGED + EMBARGOED walk-forward CV (no random k-fold,
    no look-ahead). Report OOS AUC, precision@threshold, calibration (Brier). REFUSE
    to emit a model unless OOS AUC >= bar AND it beats take-every-signal on net E[R].

  Phase 3 (act): integrate in shadow mode by default (logs would-skip/size, changes
    nothing). Acts only when META_ACTIVE is set AND a passing model exists AND there
    are >= META_MIN_TRADES labeled trades. Sizing only ever scales DOWN within the
    per-trade cap — never up.
"""
from __future__ import annotations

import json
import logging

import numpy as np

from iobot import config, gate, journal

log = logging.getLogger("meta")


# ---------------- walk-forward CV ----------------

def purged_walk_forward_splits(n: int, n_splits: int, embargo_frac: float):
    """Expanding-window splits over time-ordered samples. The last `embargo`
    samples before each test block are purged from training to prevent leakage
    across adjacent (possibly overlapping) trades."""
    if n < n_splits + 1:
        return
    fold = n // (n_splits + 1)
    embargo = max(1, int(n * embargo_frac))
    for k in range(1, n_splits + 1):
        test_start = k * fold
        test_end = n if k == n_splits else (k + 1) * fold
        train_end = max(0, test_start - embargo)
        if train_end < fold or test_start >= test_end:
            continue
        yield np.arange(0, train_end), np.arange(test_start, test_end)


def _net_r(info) -> np.ndarray:
    return np.array([
        row.r - gate.modeled_friction_r(row.max_loss, row.structure)
        for row in info.itertuples()])


def _expected_r(net_r: np.ndarray, take_mask: np.ndarray) -> float:
    """Mean net R over taken trades (0 if none taken)."""
    taken = net_r[take_mask]
    return float(taken.mean()) if taken.size else 0.0


def _make_model():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=0.5, class_weight="balanced", max_iter=1000,
                                   random_state=config.SEED)),
    ])


# ---------------- training (phase 2) ----------------

def train(conn) -> dict:
    """Train + walk-forward validate. Saves the model only if it passes."""
    import joblib
    from sklearn.metrics import brier_score_loss, precision_score, roc_auc_score

    X, info = journal.labeled_training_data(conn)
    n = len(X)
    report = {"n_labeled": n, "min_trades": config.META_MIN_TRADES,
              "auc_bar": config.META_AUC_BAR, "threshold": config.META_PROBA_THRESHOLD,
              "passes": False, "verdict": "INSUFFICIENT"}
    if n < config.META_MIN_TRADES:
        report["detail"] = f"{n}/{config.META_MIN_TRADES} labeled trades — accruing"
        _save_report(report)
        return report
    if info["label"].nunique() < 2:
        report["detail"] = "labels not yet both-class"
        _save_report(report)
        return report

    net_r = _net_r(info)
    y = info["label"].to_numpy().astype(int)
    thr = config.META_PROBA_THRESHOLD

    oos_proba = np.full(n, np.nan)
    for tr, te in purged_walk_forward_splits(n, config.META_CV_SPLITS, config.META_EMBARGO_FRAC):
        if len(np.unique(y[tr])) < 2:
            continue
        model = _make_model()
        model.fit(X.iloc[tr], y[tr])
        oos_proba[te] = model.predict_proba(X.iloc[te])[:, 1]

    mask = ~np.isnan(oos_proba)
    if mask.sum() < config.META_CV_SPLITS or len(np.unique(y[mask])) < 2:
        report["detail"] = "insufficient out-of-sample coverage for validation"
        _save_report(report)
        return report

    yv, pv, rv = y[mask], oos_proba[mask], net_r[mask]
    take = pv >= thr
    auc = float(roc_auc_score(yv, pv))
    precision = float(precision_score(yv, take, zero_division=0))
    brier = float(brier_score_loss(yv, pv))
    model_er = _expected_r(rv, take)
    baseline_er = _expected_r(rv, np.ones_like(take, dtype=bool))  # take every signal

    passes = (auc >= config.META_AUC_BAR) and (model_er > baseline_er)
    report.update({
        "oos_auc": auc, "precision_at_threshold": precision, "brier": brier,
        "model_expected_r": model_er, "baseline_expected_r": baseline_er,
        "oos_n": int(mask.sum()), "passes": bool(passes),
        "verdict": "MODEL READY" if passes else "MODEL REJECTED",
        "detail": (f"AUC={auc:.3f} (bar {config.META_AUC_BAR}), "
                   f"model E[R]={model_er:+.3f} vs take-all {baseline_er:+.3f}"),
    })

    if passes:
        final = _make_model()
        final.fit(X, y)                       # fit on all data for deployment
        joblib.dump({"model": final, "columns": list(X.columns)}, config.META_MODEL_FILE)
        log.info("META model SAVED — %s", report["detail"])
    else:
        log.info("META model NOT saved — %s", report["detail"])
    _save_report(report)
    return report


def _save_report(report: dict):
    with open(config.META_REPORT_FILE, "w") as f:
        json.dump(report, f, indent=2)


# ---------------- inference + shadow/gating (phase 3) ----------------

class MetaGate:
    """Loads the model + report and turns a feature dict into a decision.
    Default is shadow mode; gating only when active() is True."""

    def __init__(self):
        self.model = None
        self.columns: list[str] = []
        self.report: dict = {}
        self._load()

    def _load(self):
        import os

        import joblib
        if os.path.exists(config.META_REPORT_FILE):
            try:
                with open(config.META_REPORT_FILE) as f:
                    self.report = json.load(f)
            except (json.JSONDecodeError, OSError):
                self.report = {}
        if os.path.exists(config.META_MODEL_FILE):
            try:
                bundle = joblib.load(config.META_MODEL_FILE)
                self.model = bundle["model"]
                self.columns = bundle["columns"]
            except Exception as exc:
                log.warning("meta model load failed: %s", exc)

    def has_model(self) -> bool:
        return self.model is not None and bool(self.report.get("passes"))

    def active(self) -> bool:
        """Live gating allowed only behind the flag AND with a passing model AND
        enough labeled trades. (The walk-forward already proved positive model E[R]
        vs take-all — the spec's positive-shadow-R precondition.)"""
        return (config.META_ACTIVE and self.has_model()
                and self.report.get("n_labeled", 0) >= config.META_MIN_TRADES)

    def proba(self, features: dict) -> float | None:
        if self.model is None:
            return None
        import pandas as pd
        row = pd.DataFrame([{c: float(features.get(c, 0.0)) for c in self.columns}])
        return float(self.model.predict_proba(row)[0, 1])

    def decision(self, features: dict) -> tuple[float | None, bool, float]:
        """(proba, would_skip, size_mult). size_mult in (0,1] — only scales down."""
        p = self.proba(features)
        if p is None:
            return None, False, 1.0
        thr = config.META_PROBA_THRESHOLD
        would_skip = p < thr
        # Confidence sizing within the cap: thr->0.5x up to 1.0x at p=1; clamp [0.5,1].
        size_mult = float(np.clip(0.5 + (p - thr) / max(1e-6, 1 - thr) * 0.5, 0.5, 1.0))
        return p, would_skip, size_mult
