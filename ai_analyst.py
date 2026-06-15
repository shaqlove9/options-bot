"""ai_analyst.py — Claude-powered plain-English analysis (advisory only).

Two jobs, both read-only on the trading side — this module can never place,
modify, or block a trade:

  morning_briefing()  pre-market: web-searches overnight news on the bot's
                      universe and posts a heads-up to Discord.
  daily_report()      end-of-day: explains today's trades in plain English —
                      what happened, what worked, what to watch.

Both post to Discord and save a markdown file the dashboard displays.
Needs ANTHROPIC_API_KEY in .env; silently skips if missing (like alerts.py).
Cost: roughly $0.05-0.15 per run on claude-fable-5.

Manual run:  python ai_analyst.py briefing
             python ai_analyst.py report
"""
import logging
import sys

import pandas as pd

import alerts
import config
from utils import now_et

log = logging.getLogger("ai_analyst")

_SYSTEM = (
    "You are the friendly analyst for a small automated options-scalping bot. "
    "The user is a beginner trader — explain everything in plain English, and "
    "define any jargon in a few words the first time you use it. Be honest "
    "about losses and risk; never hype. You are informational only: do NOT "
    "give buy/sell recommendations or predictions presented as certainty. "
    "Keep it under 400 words, with short markdown sections and bullet points."
)


def _enabled() -> bool:
    if not config.AI_ENABLED:
        return False
    if not config.ANTHROPIC_API_KEY:
        log.debug("No ANTHROPIC_API_KEY set — AI analyst disabled")
        return False
    return True


def _ask(prompt: str, tools: list | None = None) -> str:
    """One Claude call; follows pause_turn continuations during web search."""
    import anthropic  # lazy import — the bot still runs without the package

    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    kwargs = dict(
        model=config.AI_MODEL,
        max_tokens=16000,
        thinking={"type": "adaptive"},
        system=_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    if tools:
        kwargs["tools"] = tools
    response = client.messages.create(**kwargs)
    # Server-side web search can pause mid-turn; re-send to let it resume.
    for _ in range(5):
        if response.stop_reason != "pause_turn":
            break
        kwargs["messages"] = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response.content},
        ]
        response = client.messages.create(**kwargs)
    return "\n".join(b.text for b in response.content if b.type == "text").strip()


def _deliver(title: str, text: str, path: str):
    """Save the report for the dashboard and post it to Discord."""
    stamp = now_et().strftime("%A, %B %d %Y — %I:%M %p ET")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"## {title}\n*{stamp}*\n\n{text}\n")
    alerts.ai_report(title, text)


def morning_briefing() -> str | None:
    """Pre-market news scan for the universe. Returns the text, or None."""
    if not _enabled():
        return None
    tickers = ", ".join(config.UNIVERSE)
    today = now_et().strftime("%A, %B %d %Y")
    prompt = (
        f"Today is {today}, before the US market open. The bot trades "
        f"short-dated options on: {tickers}.\n\n"
        "Search the web for overnight and pre-market news, then write a briefing:\n"
        "1. **Market mood** — futures, and any big economic data or Fed events "
        "scheduled today (include the release times in ET).\n"
        "2. **Per ticker** — one or two sentences each, only where there is real "
        "news (earnings, analyst moves, product news, unusual pre-market moves). "
        "Say 'quiet' if nothing notable.\n"
        "3. **Heads up** — anything that could cause sharp moves or whipsaws "
        "today.\n"
        "Information only — no trade recommendations."
    )
    try:
        text = _ask(prompt, tools=[{
            "type": "web_search_20260209", "name": "web_search", "max_uses": 8,
        }])
    except Exception:
        log.exception("Morning briefing failed")  # never crash the trading loop
        return None
    if text:
        _deliver("🌅 Morning briefing", text, config.AI_BRIEFING_FILE)
    return text or None


def daily_report() -> str | None:
    """Plain-English explanation of today's closed trades. None if no trades."""
    if not _enabled():
        return None
    today = now_et().date()
    try:
        df = pd.read_csv(config.TRADES_CSV)
        df["exit_time"] = pd.to_datetime(df["exit_time"], errors="coerce")
        df = df[df["exit_time"].dt.date == today]
    except (FileNotFoundError, KeyError, ValueError):
        df = pd.DataFrame()
    if df.empty:
        log.info("AI daily report skipped — no closed trades today")
        return None

    cols = [c for c in ("ticker", "type", "strike", "expiry", "entry_time",
                        "exit_time", "entry_price", "exit_price", "pnl",
                        "pnl_pct", "entry_reason", "exit_reason", "strategy",
                        "win_prob") if c in df.columns]
    pnl = pd.to_numeric(df["pnl"], errors="coerce").sum()
    mode = "LIVE (real money)" if config.LIVE_MODE else "PAPER (practice money)"
    prompt = (
        f"The bot just finished the trading day in {mode} mode. "
        f"Total P&L today: ${pnl:+.2f} on a ${config.CAPITAL:.0f} account.\n\n"
        f"Today's closed trades (CSV):\n{df[cols].to_csv(index=False)}\n\n"
        "Write the end-of-day report:\n"
        "1. **What happened** — quick recap of the day.\n"
        "2. **What worked / what didn't** — look at the entry and exit reasons.\n"
        "3. **Patterns worth watching** — only patterns this data actually "
        "supports; with just a handful of trades, say clearly that it's too "
        "early to conclude much.\n"
        "4. **One question to think about** — something the user could check or "
        "tweak in settings (a thought to explore, not advice)."
    )
    try:
        text = _ask(prompt)
    except Exception:
        log.exception("Daily report failed")
        return None
    if text:
        _deliver("🧠 AI end-of-day report", text, config.AI_REPORT_FILE)
    return text or None


if __name__ == "__main__":
    logging.basicConfig(level="INFO", format="%(levelname)s %(name)s: %(message)s")
    which = sys.argv[1] if len(sys.argv) > 1 else "report"
    result = morning_briefing() if which == "briefing" else daily_report()
    print(result or "(nothing generated — check ANTHROPIC_API_KEY and data)")
