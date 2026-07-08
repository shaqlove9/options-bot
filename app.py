"""app.py — Streamlit dashboard for the options bot.

Run:  .venv\\Scripts\\python -m streamlit run app.py
      (or double-click "Launch Options Bot.bat")

The dashboard never trades on its own. It launches/stops main.py as a separate
process, reads its status.json heartbeat, and edits settings.json. Stopping
the bot is graceful: it flattens all open positions before exiting.
"""
import datetime as dt
import json
import os
import subprocess
import time

import pandas as pd
import streamlit as st

import ai_analyst
import config
import data.trade_store as trade_store

BOT_DIR = os.path.dirname(os.path.abspath(__file__))
# venv layout differs by OS: Scripts/python.exe on Windows, bin/python on POSIX.
if os.name == "nt":
    PYTHON = os.path.join(BOT_DIR, ".venv", "Scripts", "python.exe")
else:
    PYTHON = os.path.join(BOT_DIR, ".venv", "bin", "python")
PID_FILE = os.path.join(BOT_DIR, "bot.pid")
ALL_SYMBOLS = sorted({"SPY", "QQQ", "IWM", "NVDA", "TSLA", "AAPL", "AMZN",
                      "META", "MSFT", "AMD", "GOOGL", *config.UNIVERSE})

st.set_page_config(page_title="Options Bot", page_icon="📈", layout="wide")


# ---------------- helpers ----------------

def read_status() -> dict | None:
    try:
        with open(config.STATUS_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def heartbeat_age(status: dict) -> float | None:
    try:
        updated = dt.datetime.fromisoformat(status["updated"])
        return (dt.datetime.now(tz=updated.tzinfo) - updated).total_seconds()
    except (KeyError, ValueError):
        return None


def _pid_alive(pid: int) -> bool:
    """Check if a process PID is still running (cross-platform)."""
    if os.name == "nt":
        try:
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                capture_output=True, text=True, timeout=3,
            )
            return str(pid) in result.stdout
        except Exception:
            return False
    # POSIX: signal 0 probes the process without affecting it.
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True   # exists but owned by another user
    except Exception:
        return False


def bot_running(status: dict | None = None) -> bool:
    """Prefer PID check (instantaneous); fall back to heartbeat."""
    if os.path.exists(PID_FILE):
        try:
            with open(PID_FILE) as f:
                pid = int(f.read().strip())
            if _pid_alive(pid):
                return True
            os.remove(PID_FILE)   # stale PID file
        except Exception:
            pass
    if not status or not status.get("running", True):
        return False
    age = heartbeat_age(status)
    return age is not None and age < 120


def start_bot():
    if os.path.exists(config.STOP_FLAG_FILE):
        os.remove(config.STOP_FLAG_FILE)
    # CREATE_NO_WINDOW only exists on Windows; on POSIX detach via start_new_session.
    kwargs = ({"creationflags": subprocess.CREATE_NO_WINDOW}
              if os.name == "nt" else {"start_new_session": True})
    console = open(config.CONSOLE_LOG_FILE, "a")
    try:
        proc = subprocess.Popen(
            [PYTHON, os.path.join(BOT_DIR, "main.py")],
            cwd=BOT_DIR, stdout=console, stderr=subprocess.STDOUT,
            **kwargs,
        )
    finally:
        console.close()
    with open(PID_FILE, "w") as f:
        f.write(str(proc.pid))
    st.session_state["started_at"] = time.time()


def stop_bot():
    open(config.STOP_FLAG_FILE, "w").close()


def restart_bot():
    stop_bot()
    st.session_state["restarting"] = True


def _finish_restart_if_needed():
    """Called each render cycle: once the bot actually stops, re-launch it."""
    if not st.session_state.get("restarting"):
        return
    status = read_status()
    if not bot_running(status):
        st.session_state.pop("restarting")
        start_bot()


def load_trades() -> pd.DataFrame:
    return trade_store.all_trades()


def tail(path: str, lines: int = 50) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return "".join(f.readlines()[-lines:]) or "(empty)"
    except FileNotFoundError:
        return "(no log yet)"


def save_settings(universe, max_cost, max_pos, max_loss, tp, sl, mom,
                   ml_on, ml_thr, dyn_uni):
    tmp = config.SETTINGS_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump({
            "UNIVERSE": universe,
            "MAX_TRADE_COST": max_cost,
            "MAX_OPEN_POSITIONS": int(max_pos),
            "MAX_DAILY_LOSS": max_loss,
            "TAKE_PROFIT_PCT": tp,
            "STOP_LOSS_PCT": sl,
            "MOMENTUM_PCT": mom,
            "ML_ENABLED": ml_on,
            "ML_WIN_PROB_THRESHOLD": ml_thr,
            "DYNAMIC_UNIVERSE": dyn_uni,
        }, f, indent=1)
    os.replace(tmp, config.SETTINGS_FILE)


# ---------------- sidebar: control + settings ----------------

_finish_restart_if_needed()

with st.sidebar:
    st.title("📈 Options Bot")
    if config.LIVE_MODE:
        st.error("🔴 **LIVE MODE — real money**")
    else:
        st.success("🟢 Paper mode")

    status = read_status()
    running = bot_running(status)
    restarting = st.session_state.get("restarting", False)
    stopping = running and os.path.exists(config.STOP_FLAG_FILE)
    starting_up = (not running
                   and time.time() - st.session_state.get("started_at", 0) < 20)

    if config.MONITOR_ONLY:
        # VM mode: systemd owns the bot. Showing Start/Stop here would let the
        # dashboard launch/kill a second main.py and fight systemd (double
        # orders), so controls are hidden — this dashboard is monitoring only.
        if running:
            st.info("🖥️ Monitoring only — bot is managed by **systemd** on this "
                    "host. Control it with `systemctl` over SSH.")
        else:
            st.error("🖥️ Monitoring only — bot **not running**. Start it on the "
                     "host with `sudo systemctl start optionsbot`.")
    else:
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            if st.button("▶ Start",
                         disabled=running or starting_up or restarting,
                         width="stretch", type="primary"):
                start_bot()
                st.toast("Bot starting…")
                st.rerun()
        with col_b:
            if st.button("⏹ Stop",
                         disabled=not running or stopping or restarting,
                         width="stretch"):
                stop_bot()
                st.toast("Stop sent — flattening positions…")
                st.rerun()
        with col_c:
            if st.button("↺ Restart",
                         disabled=(stopping or restarting or starting_up),
                         width="stretch"):
                restart_bot()
                st.toast("Restarting…")
                st.rerun()

        if restarting:
            st.warning("⏳ Restarting — waiting for clean shutdown…")
        elif stopping:
            st.warning("⏳ Stopping — closing positions, a few seconds…")
        elif starting_up:
            st.info("⏳ Starting up…")

        st.caption("Stop is graceful: open positions are closed before shutdown. "
                   "Restart applies saved settings.")

    st.divider()
    st.subheader("⚙️ Settings")

    with st.form("settings"):
        universe = st.multiselect("Universe", ALL_SYMBOLS, default=list(config.UNIVERSE))
        max_cost = st.number_input("Max $ per trade", 10.0, 500.0,
                                   float(config.MAX_TRADE_COST), 5.0)
        max_pos = st.number_input("Max open positions", 1, 10,
                                  int(config.MAX_OPEN_POSITIONS))
        max_loss = st.number_input("Daily loss halt ($)", 25.0, 500.0,
                                   float(config.MAX_DAILY_LOSS), 5.0)
        tp = st.number_input("Take profit (%)", 10.0, 300.0,
                             float(config.TAKE_PROFIT_PCT), 5.0)
        sl = st.number_input("Stop loss (%)", 10.0, 90.0,
                             float(config.STOP_LOSS_PCT), 5.0)
        mom = st.number_input("Momentum trigger (% / 15min)", 0.1, 3.0,
                              float(config.MOMENTUM_PCT), 0.1)
        ml_on = st.toggle("ML win-probability filter", value=bool(config.ML_ENABLED))
        ml_thr = st.slider("ML block threshold P(win)", 0.20, 0.80,
                           float(config.ML_WIN_PROB_THRESHOLD), 0.05)
        dyn_uni = st.toggle("Dynamic universe (screener)",
                            value=bool(config.DYNAMIC_UNIVERSE))

        saved = st.form_submit_button("💾 Save settings", width="stretch")

    if saved:
        if not universe:
            st.error("Universe can't be empty.")
        else:
            save_settings(universe, max_cost, max_pos, max_loss,
                          tp, sl, mom, ml_on, ml_thr, dyn_uni)
            if config.MONITOR_ONLY:
                st.success("Saved to settings.json — apply on the host with "
                           "`sudo systemctl restart optionsbot`.")
            elif running:
                if st.button("↺ Restart now to apply", type="primary",
                             width="stretch", key="restart_after_save"):
                    restart_bot()
                    st.toast("Restarting to apply new settings…")
                    st.rerun()
                else:
                    st.success("Saved — press ↺ Restart to apply, or it takes "
                               "effect on the next start.")
            else:
                st.success("Saved — settings will apply on next start.")


# ---------------- main panel (auto-refreshes every 5s) ----------------

@st.fragment(run_every="5s")
def dashboard():
    status = read_status()
    running = bot_running(status)
    restarting = st.session_state.get("restarting", False)
    s = status or {}

    # Status banner
    if restarting:
        st.warning("⏳ Restarting — waiting for clean shutdown before relaunch…")
    elif status is None:
        st.info("👋 The bot has never run. Press **▶ Start** in the sidebar.")
    elif running and os.path.exists(config.STOP_FLAG_FILE):
        st.warning("⏳ Stop requested — flattening positions before shutdown.")
    elif running:
        if s.get("halted"):
            st.error("🛑 **HALTED** — daily loss limit hit. No more trades today.")
        elif s.get("paused_until"):
            try:
                until = dt.datetime.fromisoformat(s["paused_until"])
                if dt.datetime.now(tz=until.tzinfo) < until:
                    st.warning(f"⏸️ Paused after losses — resumes "
                               f"{until.strftime('%I:%M %p')} ET")
                else:
                    st.success("✅ Bot running")
            except ValueError:
                st.success("✅ Bot running")
        elif not s.get("market_open"):
            st.info("🌙 Bot running — market closed, waiting for the open.")
        else:
            age = heartbeat_age(status)
            age_str = f"{age:.0f}s ago" if age is not None else "unknown"
            st.success(f"✅ Bot running — last heartbeat {age_str}")
    else:
        starting_up = time.time() - st.session_state.get("started_at", 0) < 20
        if starting_up:
            st.info("⏳ Starting up… (if this persists, check the Logs tab)")
        else:
            st.warning("⏹ Bot is stopped.")

    # Metric row
    pnl = float(s.get("daily_pnl", 0) or 0)
    open_pos = s.get("open_positions", [])
    learner = s.get("learner") or {}
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Daily P&L", f"${pnl:+,.2f}",
              delta=f"{pnl / config.CAPITAL * 100:+.1f}% of account",
              delta_color="normal" if pnl else "off")
    c2.metric("Open positions", f"{len(open_pos)} / {config.MAX_OPEN_POSITIONS}")
    c3.metric("Trades today", f"{s.get('trades', 0)}",
              delta=f"{s.get('wins', 0)}W · {s.get('losses', 0)}L", delta_color="off")
    c4.metric("Win rate today", f"{float(s.get('win_rate', 0) or 0):.0f}%")
    if not learner.get("trained_on"):
        model_txt, model_sub = "learning", "collecting trade data"
    elif learner.get("gating"):
        model_txt, model_sub = "gating", f"AUC {learner.get('auc')}"
    else:
        model_txt, model_sub = "advisory", f"AUC {learner.get('auc')}"
    c5.metric("ML model", model_txt, delta=model_sub, delta_color="off")

    # Dynamic universe status
    dyn = s.get("dynamic_universe")
    if dyn and dyn.get("enabled"):
        with st.expander(f"🔍 Dynamic universe ({len(dyn.get('symbols', []))} symbols, "
                         f"via {dyn.get('source', '?')})"):
            st.caption(f"Last refresh: {dyn.get('last_refresh', 'never')}")
            if dyn.get("discovered"):
                st.write("**Discovered:** " + ", ".join(dyn["discovered"]))
            st.write("**Active universe:** " + ", ".join(dyn.get("symbols", [])))

    tab_pos, tab_hist, tab_ai, tab_logs = st.tabs(
        ["📌 Open positions", "📜 Trade history", "🤖 AI Analyst", "🧾 Logs"])

    with tab_pos:
        if open_pos:
            rows = []
            for p in open_pos:
                pnl_d = ((p["last_bid"] - p["entry_price"]) * 100 * p["qty"]
                         if p.get("last_bid") else None)
                rows.append({
                    "Underlying": p["underlying"],
                    "Type": p["type"].upper(),
                    "Strike": p["strike"],
                    "Expiry": p["expiry"],
                    "Qty": p["qty"],
                    "Entry": f"${p['entry_price']:.2f}",
                    "Last bid": f"${p['last_bid']:.2f}" if p.get("last_bid") else "—",
                    "P&L %": f"{p['pnl_pct']:+.0f}%" if p.get("pnl_pct") is not None else "—",
                    "P&L $": f"${pnl_d:+.2f}" if pnl_d is not None else "—",
                })
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
            st.caption(f"Targets: +{config.TAKE_PROFIT_PCT:.0f}% take profit · "
                       f"−{config.STOP_LOSS_PCT:.0f}% stop loss · all flat by 3:45 PM ET")
        else:
            st.caption("No open positions.")

    with tab_hist:
        trades = load_trades()
        if trades.empty:
            st.caption("No closed trades yet — history appears here after the first exit.")
        else:
            total = trades["pnl"].sum()
            wins = (trades["pnl"] > 0).sum()
            h1, h2, h3 = st.columns(3)
            h1.metric("All-time P&L", f"${total:+,.2f}")
            h2.metric("Total trades", len(trades))
            h3.metric("Win rate", f"{wins / len(trades) * 100:.0f}%")

            curve = trades.sort_values("exit_time").set_index("exit_time")["pnl"]
            equity = (config.CAPITAL + curve.cumsum()).rename("Account equity ($)")
            st.line_chart(equity, height=260)

            show = trades.sort_values("exit_time", ascending=False).head(200)
            cols = ["exit_time", "ticker", "type", "strike", "expiry",
                    "entry_price", "exit_price", "pnl", "pnl_pct",
                    "exit_reason", "win_prob"]
            st.dataframe(show[[c for c in cols if c in show.columns]],
                         width="stretch", hide_index=True)
            st.download_button("⬇ Download full trades.csv",
                               trades.to_csv(index=False), "trades.csv", "text/csv")

    with tab_ai:
        if not config.ANTHROPIC_API_KEY:
            st.info("To turn on the AI analyst, add this line to your `.env` "
                    "file and restart the dashboard:\n\n"
                    "`ANTHROPIC_API_KEY=sk-ant-...`\n\n"
                    "Get a key at https://platform.claude.com — reports cost "
                    "a few cents each.")
        else:
            a1, a2 = st.columns(2)
            with a1:
                if st.button("🌅 Generate morning briefing",
                             width="stretch"):
                    with st.spinner("Reading overnight news… (~30-60s)"):
                        out = ai_analyst.morning_briefing()
                    if out is None:
                        st.warning("Nothing generated — check the Logs tab.")
            with a2:
                if st.button("🧠 Generate end-of-day report",
                             width="stretch"):
                    with st.spinner("Analyzing today's trades…"):
                        out = ai_analyst.daily_report()
                    if out is None:
                        st.warning("No closed trades today yet (or an error — "
                                   "see the Logs tab).")
            st.caption("These run automatically: briefing ~9:00 AM ET, "
                       "report after the 3:45 PM close. Advisory only.")
        for path, label in ((config.AI_BRIEFING_FILE, "morning briefing"),
                            (config.AI_REPORT_FILE, "end-of-day report")):
            st.divider()
            if os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    st.markdown(f.read())
            else:
                st.caption(f"No {label} yet.")

    with tab_logs:
        lcol, _ = st.columns([1, 4])
        with lcol:
            if st.button("🔄 Refresh logs", width="stretch"):
                st.rerun()
        st.code(tail(config.BOT_LOG_FILE, 80), language="log")
        with st.expander("Process output (startup errors land here)"):
            st.code(tail(config.CONSOLE_LOG_FILE, 30), language="log")


dashboard()
