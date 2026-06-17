"""test_feature_capture.py — guards the two correctness-critical pieces of the
equity meta-labeling layer:

  1. NO LOOK-AHEAD LEAK: equity_features.compute_features must ignore any bar
     timestamped after the signal instant. We feed it bars that include future
     bars and assert the output is byte-identical to feeding only causal bars,
     and that the audited src_max_ts never exceeds the signal time.
  2. RISK GOVERNOR: daily-loss + trailing-drawdown kill switches fire, the entry
     gate blocks when halted, and the account-id tag prevents a stale peak from
     another account false-tripping the breaker.

Run:  .venv/bin/python test_feature_capture.py   (plain asserts; no pytest needed)
"""
import datetime as dt
import json
import os
import tempfile

import pandas as pd

import config
from utils import ET


def _bars(start: dt.datetime, n: int, price: float = 100.0) -> pd.DataFrame:
    """n consecutive 1-min bars from `start`, gently drifting up."""
    idx = pd.date_range(start, periods=n, freq="1min", tz=ET)
    rows = []
    for i in range(n):
        c = price + i * 0.05
        rows.append({"open": c - 0.02, "high": c + 0.03, "low": c - 0.03,
                     "close": c, "volume": 1000 + i, "vwap": c})
    return pd.DataFrame(rows, index=idx)


def test_no_lookahead_leak():
    import equity_features as ef
    sig_time = dt.datetime(2026, 6, 16, 11, 0, tzinfo=ET)
    open_ = dt.datetime(2026, 6, 16, 9, 30, tzinfo=ET)
    causal = _bars(open_, 91)                       # 09:30..11:00 inclusive
    future = pd.concat([causal, _bars(
        dt.datetime(2026, 6, 16, 11, 1, tzinfo=ET), 15, price=200.0)])  # spike AFTER signal
    spot = float(causal["close"].iloc[-1])

    f_causal = ef.compute_features(sig_time, spot, causal, None, None, {}, None)
    f_future = ef.compute_features(sig_time, spot, future, None, None, {}, None)

    assert f_causal == f_future, "future bars changed the snapshot — LEAK!"
    assert f_future["src_max_ts"] is not None
    assert dt.datetime.fromisoformat(f_future["src_max_ts"]) <= sig_time, \
        "a source bar later than the signal time was used"
    print("PASS  no-lookahead leak guard")


def _gov(tmp, account="ACCT_A"):
    from risk_governor import EquityRiskGovernor
    return EquityRiskGovernor(account_id=account, state_file=tmp)


def test_daily_loss_kill():
    with tempfile.TemporaryDirectory() as d:
        tmp = os.path.join(d, "gov.json")
        g = _gov(tmp)
        assert g.on_equity(10000.0) == []              # start of day
        ev = g.on_equity(9700.0)                        # -3% > -2% kill
        assert "halted_day" in ev and g.halted
        ok, _ = g.can_enter(0)
        assert not ok, "entry not blocked after daily kill"
    print("PASS  daily-loss kill switch")


def test_trailing_dd_kill():
    with tempfile.TemporaryDirectory() as d:
        tmp = os.path.join(d, "gov.json")
        g = _gov(tmp)
        g.on_equity(10000.0)
        g.on_equity(12000.0)                            # new peak
        ev = g.on_equity(10100.0)                       # -15.8% from peak (daily ~+1%)
        assert "halted_dd" in ev and g.halted
    print("PASS  trailing-drawdown kill switch")


def test_account_tag_guard():
    with tempfile.TemporaryDirectory() as d:
        tmp = os.path.join(d, "gov.json")
        with open(tmp, "w") as f:
            json.dump({"account": "ACCT_A", "peak_equity": 99999.0,
                       "halted_dd": True, "halt_reason": "stale"}, f)
        g = _gov(tmp, account="ACCT_B")                 # different account
        assert g.state["peak_equity"] == 0.0, "stale peak carried across accounts"
        assert not g.halted, "stale halt carried across accounts"
    print("PASS  account-tag guard resets stale peak/halt")


def test_sizing_gated_off():
    with tempfile.TemporaryDirectory() as d:
        tmp = os.path.join(d, "gov.json")
        g = _gov(tmp)
        live, shadow = g.target_notional(10000.0, config.EQ_STOP_LOSS_PCT / 100)
        assert config.EQ_GOV_SIZING_ACTIVE is False
        assert live == config.EQ_NOTIONAL_PER_TRADE, "flat notional must stay live until gate"
        assert shadow > 0, "shadow vol-target size should still be computed"
    print("PASS  vol-target sizing gated off (flat notional live)")


if __name__ == "__main__":
    test_no_lookahead_leak()
    test_daily_loss_kill()
    test_trailing_dd_kill()
    test_account_tag_guard()
    test_sizing_gated_off()
    print("\nAll feature-capture + governor tests passed.")
