"""Smoke test for learner.py — synthetic trades with a planted pattern.

Generates ~80 fake closed trades where late-day low-volume entries lose and
morning high-volume entries win, then verifies the learner:
  1. trains and persists a model
  2. earns gating (CV AUC over the planted pattern)
  3. scores a "good" setup above a "bad" one and blocks the bad one

Run:  .venv\\Scripts\\python test_learner.py
(Uses a temp directory — never touches your real trades.csv/model.pkl.)
"""
import csv
import os
import random
import sys
import tempfile

import config

# Redirect data files to a sandbox BEFORE importing the learner.
_tmp = tempfile.mkdtemp(prefix="learner_test_")
config.TRADES_CSV = os.path.join(_tmp, "trades.csv")
config.MODEL_FILE = os.path.join(_tmp, "model.pkl")

from executor import CSV_FIELDS          # noqa: E402
from learner import Learner               # noqa: E402

random.seed(7)


def make_trade(i: int) -> dict:
    good = i % 2 == 0
    minutes = random.randint(15, 120) if good else random.randint(240, 350)
    rel_vol = round(random.uniform(2.0, 3.5), 2) if good else round(random.uniform(1.5, 1.8), 2)
    ticker = "SPY" if good else "TSLA"
    # Planted edge with noise: good setups win ~75%, bad ones ~25%
    win = random.random() < (0.75 if good else 0.25)
    pnl = round(random.uniform(8, 20), 2) if win else round(-random.uniform(6, 15), 2)
    return {
        "entry_time": f"2026-06-0{(i % 5) + 1}T10:00:00",
        "exit_time": f"2026-06-0{(i % 5) + 1}T11:00:00",
        "ticker": ticker, "option_symbol": f"{ticker}260612C00500000",
        "type": "call" if good else "put", "strike": 500, "expiry": "2026-06-12",
        "qty": 1, "entry_price": 0.40, "exit_price": 0.50, "pnl": pnl,
        "pnl_pct": round(pnl / 40 * 100, 1),
        "entry_reason": "test", "exit_reason": "test",
        "strategy": "scalp" if good else "runner",
        "momentum_pct": round(random.uniform(0.5, 1.2), 3),
        "day_change_pct": round(random.uniform(-1, 3), 3),
        "rsi": round(random.uniform(66, 80), 2),
        "rel_volume": rel_vol,
        "vwap_dist_pct": round(random.uniform(0.2, 1.0) if good
                               else random.uniform(-0.3, 0.2), 3),
        "iv": round(random.uniform(0.2, 0.5), 4),
        "spread": round(random.uniform(0.02, 0.09), 3),
        "dte": random.randint(1, 7),
        "minutes_since_open": minutes,
        "win_prob": "",
    }


with open(config.TRADES_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
    writer.writeheader()
    for i in range(80):
        writer.writerow(make_trade(i))

learner = Learner()

assert learner.pipeline is not None, "FAIL: model did not train on 80 trades"
assert os.path.exists(config.MODEL_FILE), "FAIL: model.pkl not persisted"
print(f"trained on {learner.trained_on} trades, CV AUC {learner.auc:.2f}, "
      f"gating={learner.gating}")
assert learner.gating, f"FAIL: AUC {learner.auc:.2f} should clear {config.ML_MIN_AUC} on planted pattern"

good = {"momentum_pct": 0.8, "day_change_pct": 1.5, "rsi": 70, "rel_volume": 2.8,
        "vwap_dist_pct": 0.6, "iv": 0.3, "spread": 0.05, "dte": 3,
        "minutes_since_open": 45, "ticker": "SPY", "type": "call",
        "strategy": "scalp"}
bad = {**good, "rel_volume": 1.6, "minutes_since_open": 320,
       "vwap_dist_pct": -0.1, "ticker": "TSLA", "type": "put",
       "strategy": "runner"}

p_good, p_bad = learner.score(good), learner.score(bad)
print(f"P(win) good setup = {p_good:.2f} | bad setup = {p_bad:.2f}")
assert p_good > p_bad, "FAIL: model didn't separate planted good/bad setups"

allowed_good, _ = learner.allows(good)
allowed_bad, prob = learner.allows(bad)
assert allowed_good, "FAIL: good setup was blocked"
assert not allowed_bad, f"FAIL: bad setup (P={prob:.2f}) should be blocked below {config.ML_WIN_PROB_THRESHOLD}"
print(f"good setup ALLOWED, bad setup BLOCKED (P={prob:.2f} < {config.ML_WIN_PROB_THRESHOLD})")

# Reload from disk — simulates bot restart.
learner2 = Learner()
assert learner2.pipeline is not None and learner2.gating, "FAIL: model didn't reload"
print("model reloads across restart")

print("\nALL LEARNER TESTS PASSED")
