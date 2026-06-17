"""equity_features.py — leak-safe feature snapshot for the equity meta-labeling layer.

Captures everything the bot knew AT THE MOMENT THE SIGNAL FIRED, before entry, so a
secondary model can later learn which setups win. The cardinal rule is NO LOOK-AHEAD:
`compute_features` slices every bar series to `<= sig_time` on its first line, and
returns `src_max_ts` (the latest source bar actually used) so a test can prove no
future-stamped data influenced the row.

Split into:
  - compute_features(...)  — pure; takes already-fetched data. Unit-testable, leak-safe.
  - market_context(client) — per-cycle regime context (SPY/QQQ MAs, VIX via VIXY proxy).
  - snapshot(...)          — thin wrapper: fetch bars/quote for one symbol, then compute.
"""
import datetime as dt
import logging
import math

import numpy as np
import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

import config
from utils import ET, now_et

log = logging.getLogger("equity_features")

_OPEN = (9, 30)
_CLOSE = (16, 0)


def _session_open(t: dt.datetime) -> dt.datetime:
    return t.replace(hour=_OPEN[0], minute=_OPEN[1], second=0, microsecond=0)


def _session_close(t: dt.datetime) -> dt.datetime:
    return t.replace(hour=_CLOSE[0], minute=_CLOSE[1], second=0, microsecond=0)


def _rv(returns: pd.Series, n: int) -> float | None:
    """Realized vol (%) over the last n one-minute returns."""
    tail = returns.tail(n).dropna()
    if len(tail) < 2:
        return None
    return float(tail.std() * 100)


def _atr(bars: pd.DataFrame, n: int = 14) -> float | None:
    if len(bars) < 2:
        return None
    h, l, c = bars["high"], bars["low"], bars["close"]
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    tail = tr.tail(n).dropna()
    return float(tail.mean()) if len(tail) else None


def compute_features(sig_time: dt.datetime, spot: float,
                     intraday_1m: pd.DataFrame, daily: pd.DataFrame | None,
                     quote: tuple | None, ctx: dict | None,
                     bot_state: dict | None, signal: object | None = None) -> dict:
    """Pure, leak-safe feature builder. `intraday_1m` is 1-min bars (tz-aware ET);
    everything after `sig_time` is dropped immediately so no future data can leak."""
    # ---- LEAK GUARD: nothing after the signal instant may be used ----
    bars = intraday_1m[intraday_1m.index <= sig_time] if intraday_1m is not None \
        else pd.DataFrame()
    src_ts = []

    f: dict = {}

    # --- calendar / time-of-day ---
    open_, close = _session_open(sig_time), _session_close(sig_time)
    mins_since_open = max((sig_time - open_).total_seconds() / 60, 0.0)
    mins_to_close = max((close - sig_time).total_seconds() / 60, 0.0)
    f["minutes_since_open"] = round(mins_since_open, 1)
    f["minutes_to_close"] = round(mins_to_close, 1)
    f["tod_bucket"] = int(min(mins_since_open // 30, 12))   # 30-min buckets, 0..12
    f["day_of_week"] = sig_time.weekday()                   # 0=Mon

    # --- per-symbol intraday structure (today only, causal) ---
    today = bars[bars.index.date == sig_time.date()] if not bars.empty else pd.DataFrame()
    closes = bars["close"] if not bars.empty else pd.Series(dtype=float)
    rets = closes.pct_change()
    f["rv_5m"] = _rv(rets, 5)
    f["rv_15m"] = _rv(rets, 15)
    f["rv_30m"] = _rv(rets, 30)
    f["atr_14"] = _atr(bars, 14)

    ma_fast = float(closes.tail(9).mean()) if len(closes) >= 1 else None
    ma_slow = float(closes.tail(21).mean()) if len(closes) >= 1 else None
    f["px_vs_ma_fast_pct"] = round((spot / ma_fast - 1) * 100, 4) if ma_fast else None
    f["px_vs_ma_slow_pct"] = round((spot / ma_slow - 1) * 100, 4) if ma_slow else None

    # session VWAP (causal): typical price * volume cumulated from the open
    if not today.empty and today["volume"].sum() > 0:
        tp = today["vwap"] if "vwap" in today else \
            today[["high", "low", "close"]].mean(axis=1)
        vwap = float((tp * today["volume"]).sum() / today["volume"].sum())
        f["vwap_dist_pct"] = round((spot / vwap - 1) * 100, 4) if vwap else None
        f["rel_volume_intraday"] = round(
            float(today["volume"].sum()) / max(len(today), 1), 2)
    else:
        f["vwap_dist_pct"] = getattr(signal, "vwap_dist_pct", None)
        f["rel_volume_intraday"] = None

    # gap from prior close (uses only daily bars strictly before today)
    f["gap_pct"] = None
    if daily is not None and not daily.empty and not today.empty:
        prior = daily[daily.index.date < sig_time.date()]
        if not prior.empty:
            prior_close = float(prior["close"].iloc[-1])
            today_open = float(today["open"].iloc[0])
            if prior_close > 0:
                f["gap_pct"] = round((today_open / prior_close - 1) * 100, 4)

    # --- signal's own primary-model readouts (already causal) ---
    if signal is not None:
        f["sig_momentum_pct"] = round(getattr(signal, "momentum_pct", float("nan")), 4)
        f["sig_day_change_pct"] = round(getattr(signal, "day_change_pct", float("nan")), 4)
        f["sig_rsi"] = round(getattr(signal, "rsi", float("nan")), 3)
        f["sig_rel_volume"] = round(getattr(signal, "rel_volume", float("nan")), 3)
        f["strategy"] = getattr(signal, "strategy", None)
        f["direction"] = getattr(signal, "direction", None)

    # --- live quote micro-structure (point-in-time at signal, pre-entry) ---
    if quote is not None:
        bid, ask, bid_sz, ask_sz = quote
        mid = (bid + ask) / 2 if bid and ask else (bid or ask or 0)
        f["spread"] = round(ask - bid, 4) if (bid and ask) else None
        f["spread_pct"] = round((ask - bid) / mid * 100, 4) if mid else None
        f["bid_size"] = float(bid_sz) if bid_sz is not None else None
        f["ask_size"] = float(ask_sz) if ask_sz is not None else None

    # --- market regime context (per-cycle; already causal) ---
    if ctx:
        for k, v in ctx.items():
            f[f"regime_{k}"] = v

    # --- the bot's own recent state (causal: only closed trades so far) ---
    if bot_state:
        f["state_consec_wins"] = bot_state.get("consec_wins")
        f["state_consec_losses"] = bot_state.get("consec_losses")
        f["state_trades_today"] = bot_state.get("trades_today")
        f["state_pnl_today"] = bot_state.get("pnl_today")

    # --- audit: latest source bar actually used (proves causality) ---
    if not bars.empty:
        src_ts.append(bars.index.max())
    f["src_max_ts"] = max(src_ts).isoformat() if src_ts else None

    # NaN -> None so JSON/SQLite stay clean
    return {k: (None if isinstance(v, float) and math.isnan(v) else v)
            for k, v in f.items()}


# ---------------- data fetch (impure) ----------------

def _bars(client, symbol: str, timeframe, start: dt.datetime) -> pd.DataFrame:
    try:
        df = client.get_stock_bars(StockBarsRequest(
            symbol_or_symbols=symbol, timeframe=timeframe, start=start)).df
    except Exception as exc:
        log.warning("%s: bars fetch failed: %s", symbol, exc)
        return pd.DataFrame()
    if df.empty:
        return df
    df = df.droplevel("symbol") if "symbol" in df.index.names else df
    df.index = df.index.tz_convert(ET)
    return df


def market_context(client: StockHistoricalDataClient) -> dict:
    """Per-cycle regime snapshot. Best-effort: any failure yields None fields so a
    capture is never blocked. VIX has no native Alpaca feed — we proxy with VIXY."""
    ctx: dict = {}
    start = now_et() - dt.timedelta(days=120)
    for sym in ("SPY", "QQQ"):
        daily = _bars(client, sym, TimeFrame.Day, start)
        if daily.empty or len(daily) < 50:
            ctx[f"{sym.lower()}_above_ma20"] = None
            ctx[f"{sym.lower()}_above_ma50"] = None
            continue
        close = float(daily["close"].iloc[-1])
        ctx[f"{sym.lower()}_above_ma20"] = int(close > daily["close"].tail(20).mean())
        ctx[f"{sym.lower()}_above_ma50"] = int(close > daily["close"].tail(50).mean())
    # VIX proxy via VIXY ETF: level + 1y percentile.
    vixy = _bars(client, "VIXY", TimeFrame.Day, now_et() - dt.timedelta(days=365))
    if not vixy.empty:
        lvl = float(vixy["close"].iloc[-1])
        ctx["vix_proxy"] = round(lvl, 3)
        ctx["vix_proxy_pctile"] = round(float((vixy["close"] <= lvl).mean()) * 100, 1)
    else:
        ctx["vix_proxy"] = None
        ctx["vix_proxy_pctile"] = None
    return ctx


def snapshot(client: StockHistoricalDataClient, signal, ctx: dict | None,
             bot_state: dict | None) -> dict:
    """Fetch this symbol's causal data and build the feature row."""
    sig_time = signal.time
    intraday = _bars(client, signal.symbol, TimeFrame(1, TimeFrameUnit.Minute),
                     sig_time - dt.timedelta(days=2))
    if not intraday.empty:
        intraday = intraday.between_time("09:30", "16:00")
    daily = _bars(client, signal.symbol, TimeFrame.Day,
                  sig_time - dt.timedelta(days=10))
    quote = None
    try:
        q = client.get_stock_latest_quote(
            StockLatestQuoteRequest(symbol_or_symbols=signal.symbol))[signal.symbol]
        quote = (float(q.bid_price or 0), float(q.ask_price or 0),
                 q.bid_size, q.ask_size)
    except Exception as exc:
        log.warning("%s: quote fetch failed: %s", signal.symbol, exc)
    return compute_features(sig_time, float(signal.spot), intraday, daily,
                            quote, ctx, bot_state, signal)
