"""Ad-hoc diagnostic: show live scanner state per symbol. Safe/read-only."""
import config
from alpaca.data.historical import StockHistoricalDataClient
from data.rest import RestStockBarProvider
from trading.scanner import Scanner, rsi_last
from utils import now_et, session_elapsed_fraction

client = StockHistoricalDataClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY)
sc = Scanner(RestStockBarProvider(client))

print(f"now_et={now_et()}  session_elapsed={session_elapsed_fraction(now_et()):.3f}")
print(f"MOMENTUM_PCT={config.MOMENTUM_PCT} RSI_CALL_MIN={config.RSI_CALL_MIN} "
      f"RSI_PUT_MAX={config.RSI_PUT_MAX} REL_VOLUME_MIN={config.REL_VOLUME_MIN} "
      f"VWAP_FILTER={config.VWAP_FILTER}")
print("-" * 90)

intraday_batch = sc._fetch_intraday_batch(list(config.UNIVERSE))

for sym in config.UNIVERSE:
    intraday = intraday_batch.get(sym)
    if intraday is None:
        print(f"{sym}: intraday bars = None (fetch failed or empty)")
        continue
    today = now_et().date()
    today_bars = intraday[intraday.index.date == today]
    n_today = len(today_bars)
    last = intraday.iloc[-1]
    last_ts = intraday.index[-1]
    mom = (last["close"] - last["open"]) / last["open"] * 100 if last["open"] else 0
    rsi = rsi_last(intraday["close"], config.RSI_PERIOD)
    rel = sc._relative_volume(sym, intraday)
    vwap = sc._session_vwap(intraday)
    spot = float(last["close"])
    vdist = (spot - vwap) / vwap * 100 if vwap else float("nan")
    print(f"{sym}: bars={len(intraday)} today_bars={n_today} last_bar={last_ts}")
    print(f"    spot={spot:.2f} last15m_mom={mom:+.2f}% rsi5={rsi:.1f} "
          f"rel_vol={rel:.2f}x vwap={vwap} vwap_dist={vdist:+.2f}%")
