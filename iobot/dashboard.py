"""dashboard.py — read-only monitor (Streamlit, port 8501).

Run: streamlit run iobot/dashboard.py --server.port 8501 --server.address 127.0.0.1
Shows open positions w/ defined max loss + live P/L, the closed-trade log with
R-multiples, the rejected-signal log, risk-governor state (P/L, distance to each
kill switch, sizing cap), and Phase-3 shadow-vs-actual decisions. It never trades.
"""
from __future__ import annotations

import json
import os

import pandas as pd
import streamlit as st

from iobot import config, store

st.set_page_config(page_title="iobot — intraday options", layout="wide")


def _status() -> dict:
    if os.path.exists(config.STATUS_FILE):
        try:
            with open(config.STATUS_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


status = _status()
conn = store.connect()

st.title("iobot — intraday options (PAPER)")
if not status:
    st.warning("No status yet — the engine hasn't written a heartbeat. Start the service.")
st.caption(f"Updated {status.get('updated', 'n/a')} · "
           f"market {'OPEN' if status.get('market_open') else 'closed'} · "
           f"signal {status.get('signal')} · "
           f"structure {status.get('structure')} · "
           f"spread {'on' if status.get('spread_enabled') else 'off'} · "
           f"go_live={status.get('go_live')} live_exec={status.get('live_execution')}")

# ---------------- governor / kill switches ----------------
gov = status.get("governor", {})
gt = status.get("gate", {})
if gov:
    st.subheader("Risk governor")
    c = st.columns(5)
    c[0].metric("Equity", f"${gov.get('equity', 0):,.0f}")
    c[1].metric("Day P/L", f"${gov.get('day_realized_pnl', 0):,.0f}",
                f"{-gov.get('day_drawdown_pct', 0):.2f}%")
    c[2].metric("→ daily kill", f"{gov.get('dist_to_daily_kill_pct', 0):.2f}%",
                help=f"halts new entries at -{gov.get('daily_kill_at_pct')}% on the day")
    c[3].metric("→ trailing kill", f"{gov.get('dist_to_trailing_kill_pct', 0):.2f}%",
                help=f"halts sleeve at -{gov.get('trailing_kill_at_pct')}% off peak")
    c[4].metric("Per-trade cap", f"${gov.get('per_trade_risk_cap', 0):,.0f}",
                f"{gov.get('per_trade_risk_pct')}% of equity")
    c = st.columns(4)
    c[0].metric("Trades today", f"{gov.get('trades_today')}/{gov.get('max_trades_day')}")
    c[1].metric("Round-trips", f"{gov.get('round_trips_today')}/{gov.get('max_round_trips')}")
    c[2].metric("Cash-acct mode", "ON" if gov.get("cash_account_mode") else "off")
    c[3].metric("Sleeve", "HALTED" if gov.get("sleeve_halted") else "active")
    if gov.get("sleeve_halted"):
        st.error(f"SLEEVE HALTED — {gov.get('halt_reason')}")

# ---------------- validation gate ----------------
if gt:
    st.subheader("Validation gate (paper → live)")
    cols = st.columns(5)
    cols[0].metric("Closed trades", f"{gt.get('n_trades')}/{gt.get('min_trades')}")
    cols[1].metric("Win rate", f"{(gt.get('win_rate') or 0) * 100:.0f}%")
    cols[2].metric("Net E[R]", f"{gt.get('net_expected_r') or 0:+.3f}")
    cols[3].metric("Recent half E[R]", f"{gt.get('recent_half_net_r') or 0:+.3f}")
    verdict = gt.get("verdict", "INSUFFICIENT")
    cols[4].metric("Verdict", verdict)
    (st.success if gt.get("passes") else st.info)(gt.get("detail", ""))

# ---------------- open positions ----------------
st.subheader("Open positions")
pos = status.get("open_positions", [])
if pos:
    st.dataframe(pd.DataFrame(pos), use_container_width=True, hide_index=True)
else:
    st.write("Flat.")

# ---------------- meta layer ----------------
st.subheader("Meta layer")
if os.path.exists(config.META_REPORT_FILE):
    with open(config.META_REPORT_FILE) as f:
        rep = json.load(f)
    st.json(rep, expanded=False)
    st.caption(f"acting={status.get('meta_active')} · model_ready={status.get('meta_has_model')}")
else:
    st.write("No meta report yet — accruing labeled trades.")
shadow = pd.read_sql_query("SELECT * FROM shadow ORDER BY id DESC LIMIT 50", conn)
if not shadow.empty:
    st.caption("Phase-3 shadow vs actual (most recent 50)")
    st.dataframe(shadow, use_container_width=True, hide_index=True)

# ---------------- closed trades ----------------
st.subheader("Closed trades")
trades = pd.read_sql_query("SELECT * FROM trades ORDER BY id DESC LIMIT 200", conn)
if not trades.empty:
    st.dataframe(trades, use_container_width=True, hide_index=True)
else:
    st.write("No closed trades yet.")

# ---------------- rejected signals ----------------
st.subheader("Rejected signals")
rej = pd.read_sql_query("SELECT * FROM rejects ORDER BY id DESC LIMIT 200", conn)
if not rej.empty:
    st.dataframe(rej, use_container_width=True, hide_index=True)
else:
    st.write("None logged.")
