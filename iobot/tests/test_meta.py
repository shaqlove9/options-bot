"""Meta layer: walk-forward splits are leak-free, training gates on data + proof."""
from __future__ import annotations

import numpy as np

from iobot import config, meta


def test_walk_forward_splits_are_ordered_and_embargoed():
    n, splits, embargo = 100, 5, 0.05
    seen_test = set()
    for tr, te in meta.purged_walk_forward_splits(n, splits, embargo):
        assert tr.max() < te.min()                       # train strictly before test
        gap = te.min() - tr.max()
        assert gap >= int(n * embargo)                   # embargo respected
        assert not (set(tr) & set(te))                   # no overlap
        seen_test.update(te.tolist())
    assert len(seen_test) > 0


def test_train_insufficient_without_data(conn, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "META_MIN_TRADES", 40)
    monkeypatch.setattr(config, "META_MODEL_FILE", str(tmp_path / "m.joblib"))
    monkeypatch.setattr(config, "META_REPORT_FILE", str(tmp_path / "r.json"))
    rep = meta.train(conn)
    assert rep["verdict"] == "INSUFFICIENT" and not rep["passes"]


def test_expected_r_take_mask():
    r = np.array([1.0, -1.0, 2.0, -0.5])
    take = np.array([True, False, True, False])
    assert meta._expected_r(r, take) == (1.0 + 2.0) / 2
    assert meta._expected_r(r, np.zeros(4, dtype=bool)) == 0.0


def test_train_end_to_end_on_separable_data(conn, tmp_path, monkeypatch):
    """With a predictive feature, training runs the walk-forward and emits a
    passing model + report (no API, all synthetic)."""
    import json

    from iobot import config
    monkeypatch.setattr(config, "META_MIN_TRADES", 30)
    monkeypatch.setattr(config, "META_CV_SPLITS", 4)
    monkeypatch.setattr(config, "META_MODEL_FILE", str(tmp_path / "m.joblib"))
    monkeypatch.setattr(config, "META_REPORT_FILE", str(tmp_path / "r.json"))

    rng = np.random.default_rng(0)
    for i in range(60):
        win = i % 2                                  # alternate, fully separable feature
        sid = f"s{i}"
        feats = {"rsi": 80.0 if win else 20.0, "minute_of_day": 600.0,
                 "rel_volume": 1.5, "vix": 15.0}
        conn.execute("INSERT INTO features(signal_id, captured_at_ts, max_asof_ts, "
                     "features_json) VALUES (?,?,?,?)",
                     (sid, 1000.0 + i, 1000.0 + i, json.dumps(feats)))
        r = 1.0 if win else -1.0
        conn.execute(
            "INSERT INTO trades(signal_id, structure, symbol, direction, entry_time, "
            "exit_time, max_loss, pnl, r_multiple, exit_reason, label) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (sid, "single", "SPY", "call", f"2026-06-17T10:{i:02d}",
             f"2026-06-17T11:{i:02d}", 400.0, r * 400, r,
             "profit target" if win else "stop", win))
    conn.commit()

    rep = meta.train(conn)
    assert "oos_auc" in rep and rep["oos_n"] > 0
    assert rep["verdict"] in ("MODEL READY", "MODEL REJECTED")
    assert rep["passes"]                              # separable -> should pass
    assert (tmp_path / "m.joblib").exists()
