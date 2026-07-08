"""universe_builder.py — discovers active/moving tickers and merges them
into config.UNIVERSE.

Queries Alpaca ScreenerClient for most-actives and market movers every
DYNAMIC_UNIVERSE_REFRESH_MIN minutes.  Optionally queries tradingview-screener
for relative-volume leaders (fail-open if not installed).

Always preserves DYNAMIC_UNIVERSE_CORE symbols.  Falls back to the static
UNIVERSE on any screener failure.
"""
import logging
import threading
import time
from dataclasses import dataclass, field

import config
from utils import now_et

log = logging.getLogger("trading.universe_builder")


@dataclass
class RefreshResult:
    """Snapshot of the last universe refresh for status.json reporting."""
    enabled: bool = False
    symbols: list[str] = field(default_factory=list)
    discovered: list[str] = field(default_factory=list)
    last_refresh: str = ""
    source: str = ""


class UniverseBuilder:
    def __init__(self):
        self._lock = threading.Lock()
        self._next_refresh: float = 0.0
        self._static_universe: list[str] = list(config.UNIVERSE)
        self._screener = None
        self.last_result = RefreshResult()

    # ---------------- screener clients ----------------

    def _get_screener(self):
        """Lazy-init Alpaca ScreenerClient."""
        if self._screener is None:
            from alpaca.data.historical.screener import ScreenerClient
            self._screener = ScreenerClient(
                config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY)
        return self._screener

    def _fetch_alpaca_actives(self) -> list[str]:
        """Most active stocks by volume."""
        from alpaca.data.requests import MostActivesRequest
        screener = self._get_screener()
        result = screener.get_most_actives(
            MostActivesRequest(top=config.DYNAMIC_UNIVERSE_MAX_SYMBOLS))
        return [s.symbol for s in result.most_actives]

    def _fetch_alpaca_movers(self) -> list[str]:
        """Top gainers and losers."""
        from alpaca.data.requests import MarketMoversRequest
        screener = self._get_screener()
        result = screener.get_market_movers(MarketMoversRequest(top=10))
        symbols: list[str] = []
        for mover in list(result.gainers or []) + list(result.losers or []):
            if mover.symbol not in symbols:
                symbols.append(mover.symbol)
        return symbols

    def _fetch_tradingview_rvol(self) -> list[str]:
        """High relative-volume stocks via tradingview-screener (optional)."""
        try:
            from tradingview_screener import Query, col
        except ImportError:
            return []
        try:
            query = (Query()
                     .select("name", "relative_volume_10d_calc")
                     .where(col("relative_volume_10d_calc") > 2.0,
                            col("market_cap_basic") > 2e9)
                     .order_by("relative_volume_10d_calc", ascending=False)
                     .limit(20))
            _count, df = query.get_scanner_data()
            if df is not None and not df.empty:
                return df["name"].tolist()
        except Exception as exc:
            log.debug("tradingview-screener query failed (optional): %s", exc)
        return []

    # ---------------- merge logic ----------------

    def _merge_and_cap(self, discovered: list[str]) -> list[str]:
        """Merge core + static + discovered, deduplicate, cap at max size."""
        core = list(config.DYNAMIC_UNIVERSE_CORE)
        merged = list(dict.fromkeys(core + self._static_universe + discovered))
        return merged[:config.DYNAMIC_UNIVERSE_MAX_SYMBOLS]

    # ---------------- main entry point ----------------

    def maybe_refresh(self) -> bool:
        """Called each main-loop cycle. Returns True if a refresh happened."""
        if not config.DYNAMIC_UNIVERSE:
            if self.last_result.enabled:
                self._apply(self._static_universe, [], "static (disabled)")
                self.last_result.enabled = False
            return False

        if time.monotonic() < self._next_refresh:
            return False

        self._next_refresh = (time.monotonic()
                              + config.DYNAMIC_UNIVERSE_REFRESH_MIN * 60)

        discovered: list[str] = []
        sources: list[str] = []

        try:
            actives = self._fetch_alpaca_actives()
            movers = self._fetch_alpaca_movers()
            discovered = list(dict.fromkeys(actives + movers))
            sources.append("alpaca")
        except Exception as exc:
            log.warning("Alpaca screener failed — keeping current universe: %s",
                        exc)

        try:
            tv_syms = self._fetch_tradingview_rvol()
            if tv_syms:
                discovered = list(dict.fromkeys(discovered + tv_syms))
                sources.append("tradingview")
        except Exception as exc:
            log.debug("tradingview-screener failed (optional): %s", exc)

        if not discovered and not sources:
            self._apply(self._static_universe, [], "static (fallback)")
            return True

        new_universe = self._merge_and_cap(discovered)
        non_core = [s for s in new_universe if s not in self._static_universe]
        source_str = "+".join(sources) if sources else "static (fallback)"
        self._apply(new_universe, non_core, source_str)
        return True

    def _apply(self, symbols: list[str], discovered: list[str], source: str):
        """Thread-safe update of config.UNIVERSE and last_result."""
        with self._lock:
            config.UNIVERSE = symbols
        self.last_result = RefreshResult(
            enabled=config.DYNAMIC_UNIVERSE,
            symbols=list(symbols),
            discovered=discovered,
            last_refresh=now_et().isoformat(timespec="seconds"),
            source=source,
        )
        log.info("Universe refreshed (%s): %s (%d discovered)",
                 source, " ".join(symbols), len(discovered))

    def status_dict(self) -> dict:
        """For inclusion in status.json."""
        r = self.last_result
        return {
            "enabled": r.enabled,
            "symbols": r.symbols,
            "discovered": r.discovered,
            "last_refresh": r.last_refresh,
            "source": r.source,
        }
