"""engine.py — the main loop. Wires signals -> features -> governor -> meta ->
selection -> execution -> journal, every SCAN_INTERVAL while the market is open.

Order of protection for each signal (capital first):
  1. capture Phase-1 features (always, the instant the signal fires);
  2. governor count/kill-switch gate (can_open);
  3. meta layer (shadow by default; gates only when active);
  4. contract selection + liquidity filter;
  5. governor per-trade risk cap + buying-power check;
  6. submit. Any failure SKIPS (logged as a reject) — never resizes to force a fill.
"""
from __future__ import annotations

import json
import logging
import queue
import time

from iobot import (broker, chain, config, executor, features, gate, governor,
                   journal, meta, spread, store, strategies, webhook)
from iobot.clock import in_entry_window, now_et, past_force_close
from iobot.signals import StrategySignal

log = logging.getLogger("engine")


class Engine:
    def __init__(self):
        self.clients = broker.build_clients()
        self.conn = store.connect()
        self.gov = governor.RiskGovernor()
        signal_fn = strategies.REGISTRY.get(config.SIGNAL, strategies.REGISTRY["momentum"])
        self.signals = StrategySignal(self.clients.stock_data, signal_fn, config.SIGNAL)
        self.featbuilder = features.FeatureBuilder(self.clients.stock_data)
        self.exec = executor.Executor(self.clients.trading, self.clients.stock_data,
                                      self.clients.option_data, self.conn)
        self.meta = meta.MetaGate()
        self.exec.reconcile()
        # Optional TradingView webhook receiver: a daemon thread that only
        # ENQUEUES signals; tick() drains them into the same _handle_signal path.
        self.webhook_queue: queue.Queue = queue.Queue()
        webhook.start_webhook_server(self.webhook_queue, self.signals)
        log.info("engine ready — universe %s, signal=%s, structure=%s, meta_active=%s",
                 config.UNIVERSE, config.SIGNAL, config.STRUCTURE, self.meta.active())

    # ---------------- one cycle ----------------

    def tick(self):
        # Drain webhook alerts every tick so none go stale; they only TRADE when
        # the market is open and inside the entry window (same guard as scanner
        # signals), otherwise they're dropped here.
        queued = self._drain_webhook()
        sleeve_eq = self._sleeve_equity()

        if not broker.market_is_open(self.clients.trading):
            for sig in queued:
                self._drop_webhook(sig, "market closed")
            self._write_status(market_open=False, account=None, sleeve_equity=sleeve_eq)
            return
        acct = broker.account_snapshot(self.clients.trading)
        self.gov.begin_day(sleeve_eq)
        self.gov.update_equity(sleeve_eq)

        # Manage / exit first. Overnight-aware: only force EOD-flat the contracts the
        # executor deems unsafe to carry (near expiry / past the hold horizon).
        for pnl in self.exec.manage(force_close=past_force_close(now_et())):
            self.gov.record_exit(pnl)
        sleeve_eq = self._sleeve_equity()        # reflect just-closed trades
        self.gov.update_equity(sleeve_eq)

        if in_entry_window(now_et()):
            self._consider_entries(acct, sleeve_eq, queued)
        else:
            for sig in queued:
                self._drop_webhook(sig, "outside entry window")

        self._write_status(market_open=True, account=acct, sleeve_equity=sleeve_eq)

    def _sleeve_equity(self) -> float:
        """The $1,000 sleeve's equity = seed capital + its own realized P&L. This,
        not the (paper) account equity, is the governor's risk base."""
        return config.SLEEVE_CAPITAL + journal.realized_pnl_total(self.conn)

    def _drain_webhook(self) -> list:
        """Pop all queued webhook signals (FIFO). Empty list if disabled."""
        out: list = []
        while True:
            try:
                out.append(self.webhook_queue.get_nowait())
            except queue.Empty:
                break
        return out

    def _drop_webhook(self, sig, why: str):
        log.info("webhook %s %s dropped — %s", sig.symbol, sig.direction.upper(), why)
        journal.log_reject(self.conn, sig.symbol, sig.direction, "window", why)

    def _consider_entries(self, acct, sleeve_eq, queued=()):
        # Webhook-sourced signals first, then scanner signals — one shared path.
        signals = list(queued) + self.signals.scan()
        if not signals:
            return
        win_streak, loss_streak = journal.recent_streak(self.conn)
        ctx = self.featbuilder.context(win_streak, loss_streak, self.gov.state.trades_today)
        for sig in signals:
            try:
                self._handle_signal(sig, acct, sleeve_eq, ctx)
            except Exception:
                log.exception("%s: signal handling error", sig.symbol)
            if self.exec.open_count() >= config.MAX_CONCURRENT:
                break

    def _handle_signal(self, sig, acct, sleeve_eq, ctx):
        # 1. Phase-1 capture (always, pre-entry, leakage-guarded).
        try:
            row = features.build(sig, ctx)
            features.capture(self.conn, sig, row)
            feats = row.features
        except features.LeakageError as exc:
            log.error("%s: feature capture rejected: %s", sig.symbol, exc)
            journal.log_reject(self.conn, sig.symbol, sig.direction, "features", str(exc))
            return

        # 2. Governor count / kill-switch gate (on SLEEVE equity, not the account).
        ok, why = self.gov.can_open(sleeve_eq, self.exec.open_count())
        if not ok:
            journal.log_reject(self.conn, sig.symbol, sig.direction, "governor", why)
            return
        if self.exec.has_position_in(sig.symbol):
            journal.log_reject(self.conn, sig.symbol, sig.direction, "governor",
                               "already holding this underlying")
            return

        # 2b. Post-close re-entry cooldown on the same symbol+direction. Blocks the
        #     churn where a just-closed name is immediately re-traded (the second fire
        #     reliably underperformed the first). 0 disables.
        if config.SYMBOL_COOLDOWN_MIN > 0:
            mins = journal.minutes_since_last_exit(self.conn, sig.symbol, sig.direction)
            if mins is not None and 0 <= mins < config.SYMBOL_COOLDOWN_MIN:
                journal.log_reject(self.conn, sig.symbol, sig.direction, "cooldown",
                                   f"{mins:.0f}m since last {sig.direction} close "
                                   f"< {config.SYMBOL_COOLDOWN_MIN}m cooldown")
                return

        # 3. Meta layer (shadow by default; gates only when active). A webhook
        #    signal flagged ADVISORY (live data was unavailable at enrichment)
        #    keeps its shadow log but is never vetoed — its features are unreliable.
        advisory = getattr(sig, "advisory", False)
        proba, would_skip, size_mult = self.meta.decision(feats)
        if self.meta.active() and would_skip and not advisory:
            journal.log_shadow(self.conn, sig.signal_id, proba, True, size_mult, "skipped(meta)")
            journal.log_reject(self.conn, sig.symbol, sig.direction, "meta",
                               f"P(win)={proba:.2f} < {config.META_PROBA_THRESHOLD}")
            return

        # 4. Contract selection + liquidity filter.
        want_spread = config.SPREAD_ENABLED and acct.equity >= config.SPREAD_MIN_EQUITY
        if want_spread:
            pick, why = spread.build_spread(sig, self.clients.trading,
                                            self.clients.option_data, acct.equity)
            required_bp = (pick.net_debit * 100 * config.QTY + config.SPREAD_MARGIN_BUFFER
                           if pick else 0.0)
        else:
            pick, why = chain.build_contract(sig, self.clients.trading,
                                             self.clients.option_data)
            required_bp = pick.max_loss if pick else 0.0
        if pick is None:
            journal.log_reject(self.conn, sig.symbol, sig.direction, "selection", why)
            return

        # 5. Per-trade risk cap (% of sleeve) + buying-power check (real account).
        ok, why = self.gov.approve_risk(pick.max_loss, required_bp, acct,
                                        cap_equity=sleeve_eq)
        if not ok:
            journal.log_reject(self.conn, sig.symbol, sig.direction, "risk", why)
            return

        # 6. Submit.
        journal.log_shadow(self.conn, sig.signal_id, proba, bool(would_skip),
                           size_mult, "taken")
        pos = (self.exec.open_spread(pick, sig, feats) if want_spread
               else self.exec.open_single(pick, sig, feats))
        if pos is not None:
            self.gov.record_entry()

    # ---------------- status ----------------

    def _write_status(self, market_open: bool, account, sleeve_equity: float):
        try:
            status = {
                "updated": now_et().isoformat(timespec="seconds"),
                "market_open": market_open,
                "signal": config.SIGNAL,
                "structure": config.STRUCTURE,
                "spread_enabled": config.SPREAD_ENABLED,
                "sleeve_capital": config.SLEEVE_CAPITAL,
                "sleeve_equity": sleeve_equity,
                "account_equity": account.equity if account else None,
                "meta_active": self.meta.active(),
                "meta_has_model": self.meta.has_model(),
                "open_positions": self.exec.positions_snapshot(),
                # governor risk is measured on the sleeve, not the (paper) account.
                "governor": self.gov.snapshot(sleeve_equity),
                # The validation "gate" is INFORMATIONAL only — it never blocks paper
                # entries. Capital protection comes solely from the governor above.
                "gate": gate.evaluate(self.conn),
                "go_live": config.GO_LIVE,
                "live_execution": config.ALLOW_LIVE_EXECUTION,
            }
            with open(config.STATUS_FILE, "w") as f:
                json.dump(status, f, indent=1, default=str)
        except Exception:
            log.exception("status write failed")

    # ---------------- run ----------------

    def run(self):
        log.info("iobot starting — PAPER ONLY (go_live=%s, live_exec=%s)",
                 config.GO_LIVE, config.ALLOW_LIVE_EXECUTION)
        while True:
            try:
                self.tick()
            except broker.LiveExecutionDisabled:
                raise
            except Exception:
                log.exception("tick error — continuing")
            time.sleep(config.SCAN_INTERVAL_SEC)


def main():
    Engine().run()


if __name__ == "__main__":
    main()
