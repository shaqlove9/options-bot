# TradingView ORB strategy → iobot webhook

`iobot_orb_vwap.pine` is an **Opening Range Breakout + VWAP** strategy that fires the
bot's webhook on each entry. ORB is the most evidence-backed intraday strategy we found
(Zarattini, Barbon & Aziz, SSRN 2023/2024 — ~2.4 Sharpe, beta≈0; outperformed buy-&-hold
2016–2023). It was also marginally positive in iobot's own backtest.

## Install (per symbol)
1. TradingView → chart **SPY**, **5-minute** timeframe.
2. Pine Editor → paste `iobot_orb_vwap.pine` → **Add to chart**.
3. In the strategy settings, paste your bot secret into **Webhook secret**
   (it's `IOBOT_WEBHOOK_SECRET` in `~/options-bot/.env`).
4. Check the **Strategy Tester** tab — that's the backtest on your data (see caveats).
5. Repeat on a **QQQ** 5-min chart (the bot's universe is SPY + QQQ).

## Wire the alert (per chart)
1. Click the strategy's **⋯ → Add alert** (or the clock icon).
2. **Condition:** the strategy, **"alert() function calls only"**.
3. **Notifications → Webhook URL:** your public tunnel URL, e.g.
   `https://webhook.yourdomain.com/tv-webhook` (set up via
   `deploy/setup-cloudflare-tunnel.sh`). Leave the message box as-is — the JSON comes
   from the script.
4. Expiration: **Open-ended**. Create. (Repeat for QQQ.)
   A *paid TradingView plan is required for webhook alerts.*

## What actually happens
The alert POSTs `{"secret","id","symbol","direction":"call"|"put","price"}` to the bot.
The bot authenticates it, then runs it through the **same pipeline as its own signals**:
governor caps, liquidity/contract selection, stops, and EOD flat. The webhook is the
*entry signal* only.

## Honest caveats (read before trusting any number)
- **The TV backtest ≠ what the bot will do.** The paper's edge comes from letting winners
  run to **10R**; the bot manages its *own* exit (`IOBOT_TARGET_R`, default 1.5R) and a
  0.40% underlying stop — which **truncates the fat-tail runners** the ORB edge depends on.
  So you're testing "ORB *entries* under the bot's exit engine," not the paper verbatim.
  To test ORB more faithfully, raise `IOBOT_TARGET_R` (changes the bot globally).
- **Options drag.** The paper traded shares/TQQQ; the bot buys SPY/QQQ ITM options. Every
  edge we've tested thinned or died under options costs — expect the same risk here.
- **OR length vs entry window.** Default `orMin=15` makes the breakout land at 09:45, inside
  the bot's entry window. The paper's `orMin=5` breaks out ~09:35, which the bot would
  **drop** unless you set `IOBOT_ENTRY_START=09:35`.
- This is a **paper test.** Let it run for weeks, compare its paper trades to the bot's
  current scanner, and only then judge. Don't pay for TradingView or risk real money until
  the paper sample says something.
