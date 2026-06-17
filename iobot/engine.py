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
import time

from iobot import (broker, chain, config, executor, features, gate, governor,
                   journal, meta, spread, store)
from iobot.clock import in_entry_window, now_et, past_force_close
from iobot.signals import MomentumSignal

log = logging.getLogger("engine")


class Engine:
    def __init__(self):
        self.clients = broker.build_clients()
        self.conn = store.connect()
        self.gov = governor.RiskGovernor()
        self.signals = MomentumSignal(self.clients.stock_data)
        self.featbuilder = features.FeatureBuilder(self.clients.stock_data)
        self.exec = executor.Executor(self.clients.trading, self.clients.stock_data,
                                      self.clients.option_data, self.conn)
        self.meta = meta.MetaGate()
        self.exec.reconcile()
        log.info("engine ready — universe %s, structure=%s, meta_active=%s",
                 config.UNIVERSE, config.STRUCTURE, self.meta.active())

    # ---------------- one cycle ----------------

    def tick(self):
        if not broker.market_is_open(self.clients.trading):
            self._write_status(market_open=False, account=None)
            return
        acct = broker.account_snapshot(self.clients.trading)
        self.gov.begin_day(acct.equity)
        self.gov.update_equity(acct.equity)

        # Manage / exit first (also forces EOD flat).
        for pnl in self.exec.manage(force_close=past_force_close(now_et())):
            self.gov.record_exit(pnl)
        self.gov.update_equity(broker.account_snapshot(self.clients.trading).equity)

        if in_entry_window(now_et()):
            self._consider_entries(acct)

        self._write_status(market_open=True, account=acct)

    def _consider_entries(self, acct):
        signals = self.signals.scan()
        if not signals:
            return
        win_streak, loss_streak = journal.recent_streak(self.conn)
        ctx = self.featbuilder.context(win_streak, loss_streak, self.gov.state.trades_today)
        for sig in signals:
            try:
                self._handle_signal(sig, acct, ctx)
            except Exception:
                log.exception("%s: signal handling error", sig.symbol)
            if self.exec.open_count() >= config.MAX_CONCURRENT:
                break

    def _handle_signal(self, sig, acct, ctx):
        # 1. Phase-1 capture (always, pre-entry, leakage-guarded).
        try:
            row = features.build(sig, ctx)
            features.capture(self.conn, sig, row)
            feats = row.features
        except features.LeakageError as exc:
            log.error("%s: feature capture rejected: %s", sig.symbol, exc)
            journal.log_reject(self.conn, sig.symbol, sig.direction, "features", str(exc))
            return

        # 2. Governor count / kill-switch gate.
        ok, why = self.gov.can_open(acct.equity, self.exec.open_count())
        if not ok:
            journal.log_reject(self.conn, sig.symbol, sig.direction, "governor", why)
            return
        if self.exec.has_position_in(sig.symbol):
            journal.log_reject(self.conn, sig.symbol, sig.direction, "governor",
                               "already holding this underlying")
            return

        # 3. Meta layer (shadow by default; gates only when active).
        proba, would_skip, size_mult = self.meta.decision(feats)
        if self.meta.active() and would_skip:
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

        # 5. Per-trade risk cap + buying-power check.
        ok, why = self.gov.approve_risk(pick.max_loss, required_bp, acct)
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

    def _write_status(self, market_open: bool, account):
        try:
            eq = account.equity if account else self.gov.state.peak_equity
            status = {
                "updated": now_et().isoformat(timespec="seconds"),
                "market_open": market_open,
                "structure": config.STRUCTURE,
                "spread_enabled": config.SPREAD_ENABLED,
                "meta_active": self.meta.active(),
                "meta_has_model": self.meta.has_model(),
                "open_positions": self.exec.positions_snapshot(),
                "governor": self.gov.snapshot(eq),
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
