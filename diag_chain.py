"""diag_chain.py — read-only: shows why contracts pass/fail the bot's filters.

Run:  .venv\\Scripts\\python diag_chain.py
"""
import logging

logging.basicConfig(level=logging.CRITICAL)  # keep bot module logs quiet

from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import OptionSnapshotRequest, StockLatestTradeRequest
from alpaca.trading.client import TradingClient

import config
from options_chain import ChainFetcher
from scanner import Signal
from utils import now_et

trading = TradingClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY, paper=True)
opt_data = OptionHistoricalDataClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY)
stk_data = StockHistoricalDataClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY)
fetcher = ChainFetcher(trading, opt_data)

print(f"Filters: spread <= max(${config.MAX_SPREAD:.2f}, {config.MAX_SPREAD_PCT:.0f}% of mid), "
      f"OI > {config.MIN_OPEN_INTEREST}, cost <= ${config.MAX_TRADE_COST:.0f}, "
      f"{config.MIN_DTE}-{config.MAX_DTE} DTE, "
      f"{config.OTM_MIN_PCT:.1f}-{config.OTM_MAX_PCT:.1f}% OTM\n")

for symbol in config.UNIVERSE:
    trade = stk_data.get_stock_latest_trade(StockLatestTradeRequest(symbol_or_symbols=symbol))
    spot = float(trade[symbol].price)
    sig = Signal(symbol=symbol, direction="put", strategy="scalp", momentum_pct=0,
                 day_change_pct=0, rsi=30, rel_volume=1.5, spot=spot,
                 vwap_dist_pct=0, time=now_et())
    try:
        contracts = fetcher._contracts(sig)
    except Exception as exc:
        print(f"{symbol} (spot {spot:.2f}): chain fetch failed: {exc}")
        continue
    targets = fetcher._otm_strikes(contracts, sig)
    cands = {c.symbol: c for c in contracts
             if (c.expiration_date, float(c.strike_price)) in targets}
    print(f"{symbol} (spot {spot:.2f}) — {len(cands)} candidate puts in OTM band:")
    if not cands:
        continue
    snaps = opt_data.get_option_snapshot(OptionSnapshotRequest(symbol_or_symbols=list(cands)))
    for sym, c in sorted(cands.items(), key=lambda kv: (kv[1].expiration_date,
                                                        -float(kv[1].strike_price))):
        oi = int(c.open_interest or 0)
        snap = snaps.get(sym)
        if not snap or not snap.latest_quote:
            print(f"  {sym:<24} OI {oi:>6}  NO QUOTE")
            continue
        bid = float(snap.latest_quote.bid_price or 0)
        ask = float(snap.latest_quote.ask_price or 0)
        spread = ask - bid
        mid = (bid + ask) / 2
        cost = ask * 100 * config.MAX_CONTRACTS
        fails = []
        if oi <= config.MIN_OPEN_INTEREST:
            fails.append("OI")
        if bid <= 0 or ask <= 0:
            fails.append("no-quote")
        elif spread > max(config.MAX_SPREAD, mid * config.MAX_SPREAD_PCT / 100):
            fails.append(f"SPREAD ${spread:.2f}")
        if cost > config.MAX_TRADE_COST:
            fails.append(f"COST ${cost:.0f}")
        verdict = "PASS" if not fails else "fail: " + ", ".join(fails)
        print(f"  {sym:<24} OI {oi:>6}  bid/ask {bid:>6.2f}/{ask:<6.2f} "
              f"spread {spread:>5.2f}  cost ${cost:>6.0f}  {verdict}")
    pick = fetcher.find_contract(sig)
    if pick:
        print(f"  -> BOT WOULD PICK: {pick.option_symbol} strike {pick.strike:.1f} "
              f"exp {pick.expiry} ask {pick.ask:.2f} (cost ${pick.cost:.0f})")
    else:
        print("  -> bot finds NO tradable contract")
    print()
