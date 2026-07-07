"""earnings.py — blocks single-name entries when earnings fall inside the
option's DTE window.

Why: a 1-7 DTE option on NVDA bought the day before earnings carries inflated
premium that collapses after the report (IV crush) — the stock can move your
way and the option still loses. ETFs (SPY/QQQ) are exempt.

Earnings dates come from Yahoo Finance via yfinance (free, no key) and are
cached for the day. Lookups FAIL OPEN: if Yahoo is unreachable the trade is
allowed with a warning, so a data outage can't silently disable the whole bot.
"""
import datetime as dt
import logging

import config
from utils import now_et

log = logging.getLogger("trading.earnings_guard")

# symbol -> (date the lookup was made, next earnings date or None)
_cache: dict[str, tuple[dt.date, dt.date | None]] = {}


def next_earnings(symbol: str) -> dt.date | None:
    today = now_et().date()
    cached = _cache.get(symbol)
    if cached and cached[0] == today:
        return cached[1]

    nxt: dt.date | None = None
    try:
        import yfinance as yf
        df = yf.Ticker(symbol).get_earnings_dates(limit=12)
        if df is not None and not df.empty:
            future = [ts.date() for ts in df.index if ts.date() >= today]
            if future:
                nxt = min(future)
    except Exception as exc:
        log.warning("%s: earnings lookup failed (%s) — allowing trade", symbol, exc)
        return None   # fail open, and don't cache the failure

    _cache[symbol] = (today, nxt)
    return nxt


def blocks(symbol: str) -> bool:
    """True if this symbol should be skipped because earnings land within
    the bot's MAX_DTE window."""
    if not config.EARNINGS_BLOCK or symbol in config.ETF_SYMBOLS:
        return False
    nxt = next_earnings(symbol)
    if nxt is None:
        return False
    days_away = (nxt - now_et().date()).days
    if days_away <= config.MAX_DTE:
        log.info("%s: earnings on %s (%dd away, inside %dd DTE window) — skipping",
                 symbol, nxt, days_away, config.MAX_DTE)
        return True
    return False


if __name__ == "__main__":
    # Manual check: python earnings.py
    for sym in config.UNIVERSE:
        print(f"{sym:6s} next earnings: {next_earnings(sym)}  blocked: {blocks(sym)}")
