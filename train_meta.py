"""train_meta.py — offline trainer + hard gate for the equity meta-model.

One command:  python train_meta.py

Joins the captured pre-entry features (SQLite, keyed by signal_id) to the closed
equity trades (trades_equity.csv) on signal_id, labels each (TP-before-SL = 1),
and validates with PURGED, EMBARGOED WALK-FORWARD cross-validation — no shuffling,
no look-ahead. Reports out-of-sample AUC, precision at the operating threshold, and
a calibration curve.

It REFUSES to ship a model unless out-of-sample performance clears the bar:
    OOS AUC > EQ_META_AUC_BAR  AND  model's expected R beats take-everything.
On pass it writes meta_model.pkl + meta_metrics.json. Otherwise it prints why,
writes NO artifact, and exits non-zero. Reproducible (fixed RANDOM_SEED).
"""
import json
import sys

import numpy as np
import pandas as pd

import config
import feature_store
from meta_labeler import PurgedWalkForwardCV, build_pipeline, select_features


def _load_labeled() -> pd.DataFrame | None:
    """Join captured signals -> closed-trade outcomes on signal_id."""
    sig = feature_store.read_signals()
    if sig.empty:
        print("No captured signals yet — nothing to train on.")
        return None
    import os
    if not os.path.exists(config.TRADES_CSV):
        print(f"No closed-trade log at {config.TRADES_CSV} yet.")
        return None
    trades = pd.read_csv(config.TRADES_CSV)
    if "signal_id" not in trades.columns:
        print("Trade log has no signal_id column yet — no labeled trades to join.")
        return None
    trades = trades[trades["signal_id"].notna() & (trades["signal_id"] != "")]
    if trades.empty:
        print("No closed trades carry a signal_id yet.")
        return None

    def label_row(r) -> int:
        reason = str(r.get("exit_reason", "")).lower()
        if "take profit" in reason:
            return 1
        if "stop loss" in reason:
            return 0
        return int(pd.to_numeric(r.get("pnl"), errors="coerce") > 0)   # trailing/time exits

    trades["label"] = trades.apply(label_row, axis=1)
    trades["r_multiple"] = pd.to_numeric(trades.get("r_multiple"), errors="coerce")
    cols = ["signal_id", "label", "r_multiple", "pnl", "exit_reason"]
    df = sig.merge(trades[cols], on="signal_id", how="inner")
    return df.sort_values("ts").reset_index(drop=True)


def _calibration(y: np.ndarray, p: np.ndarray, bins: int = 5) -> list[dict]:
    out = []
    edges = np.linspace(0, 1, bins + 1)
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= edges[i + 1])
        if m.sum() == 0:
            continue
        out.append({"bin": f"{edges[i]:.1f}-{edges[i+1]:.1f}",
                    "n": int(m.sum()),
                    "pred": round(float(p[m].mean()), 3),
                    "actual": round(float(y[m].mean()), 3)})
    return out


def main() -> int:
    np.random.seed(config.RANDOM_SEED)
    df = _load_labeled()
    print("=" * 64)
    print("EQUITY META-MODEL — purged walk-forward training + gate")
    print("=" * 64)
    if df is None or df.empty:
        print(f"DECISION → INSUFFICIENT: 0 labeled trades (need {config.EQ_META_MIN_TRADES}).")
        return 2

    n = len(df)
    print(f"Labeled trades: {n}  |  wins {int(df['label'].sum())}  "
          f"losses {int((df['label'] == 0).sum())}")
    if n < config.EQ_META_MIN_TRADES:
        print(f"DECISION → INSUFFICIENT: {n} labeled trades, need "
              f"{config.EQ_META_MIN_TRADES}. Keep the paper run going.")
        return 2
    if df["label"].nunique() < 2:
        print("DECISION → INSUFFICIENT: all trades share one outcome — cannot train.")
        return 2

    X, numeric, categorical = select_features(df)
    y = df["label"].to_numpy(int)
    r = df["r_multiple"].fillna(0.0).to_numpy(float)

    # ---- purged, embargoed walk-forward: pooled out-of-sample predictions ----
    from sklearn.metrics import precision_score, roc_auc_score
    cv = PurgedWalkForwardCV(n_splits=4, embargo=config.EQ_META_EMBARGO)
    oos_idx, oos_p = [], []
    for tr, te in cv.split(n):
        if len(np.unique(y[tr])) < 2:
            continue
        pipe = build_pipeline(numeric, categorical)
        pipe.fit(X.iloc[tr], y[tr])
        oos_p.append(pipe.predict_proba(X.iloc[te])[:, 1])
        oos_idx.append(te)
    if not oos_idx:
        print("DECISION → INSUFFICIENT: no valid walk-forward fold (class imbalance).")
        return 2
    idx = np.concatenate(oos_idx)
    p = np.concatenate(oos_p)
    y_oos = y[idx]
    r_oos = r[idx]

    if len(np.unique(y_oos)) < 2:
        print("DECISION → INSUFFICIENT: out-of-sample folds are single-class.")
        return 2

    auc = float(roc_auc_score(y_oos, p))
    thr = config.EQ_META_THRESHOLD
    take = p >= thr
    precision = float(precision_score(y_oos, take, zero_division=0))
    calib = _calibration(y_oos, p)

    # expected R: model-selected takes vs take-everything baseline
    base_R = float(r_oos.mean())
    model_R = float(r_oos[take].mean()) if take.any() else float("nan")

    print(f"\nWalk-forward OOS (n={len(y_oos)}):")
    print(f"  AUC:                {auc:.3f}   (bar > {config.EQ_META_AUC_BAR})")
    print(f"  Precision @ {thr:.2f}:   {precision:.3f}")
    print(f"  Expected R/trade:   model {model_R:+.3f}  vs  take-all {base_R:+.3f}")
    print("  Calibration (pred → actual win rate):")
    for b in calib:
        print(f"    {b['bin']}  n={b['n']:>3}  pred {b['pred']:.2f}  actual {b['actual']:.2f}")

    auc_ok = auc > config.EQ_META_AUC_BAR
    r_ok = np.isfinite(model_R) and model_R > base_R
    passed = bool(auc_ok and r_ok)
    print("\n" + "-" * 64)
    if not passed:
        why = []
        if not auc_ok:
            why.append(f"AUC {auc:.3f} ≤ bar {config.EQ_META_AUC_BAR}")
        if not r_ok:
            why.append(f"expected R {model_R:+.3f} does not beat take-all {base_R:+.3f}")
        print(f"DECISION → FAIL: {'; '.join(why)}.")
        print("NO model shipped. The model has not earned the right to gate trades.")
        return 1

    # ---- gate passed: fit final model on all data, ship artifact ----
    import joblib
    final = build_pipeline(numeric, categorical)
    final.fit(X, y)
    metrics = {"auc": round(auc, 4), "precision": round(precision, 4),
               "threshold": thr, "n_trades": n, "n_oos": int(len(y_oos)),
               "model_expected_R": round(model_R, 4),
               "baseline_expected_R": round(base_R, 4),
               "calibration": calib, "gate_passed": True}
    joblib.dump({"pipeline": final, "numeric": numeric, "categorical": categorical,
                 "threshold": thr, "metrics": metrics}, config.META_MODEL_FILE)
    with open(config.META_METRICS_FILE, "w") as fh:
        json.dump(metrics, fh, indent=2, default=str)
    print(f"DECISION → PASS. Shipped {config.META_MODEL_FILE} + {config.META_METRICS_FILE}.")
    print("Model is eligible to activate (still requires EQ_META_ACTIVE=True + the "
          "shadow-mode check).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
