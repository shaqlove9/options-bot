# Spread sleeve — deployment (paper-only)

Prerequisites you must provide (Claude can't):
1. A **separate Alpaca PAPER account with Level 3 options** approval (multi-leg
   debit spreads require L3).
2. Its API key/secret.

## Steps
```bash
cd ~/options-bot

# 1. Wire the isolated L3 paper account
cp .env.spread.example .env.spread
#    edit .env.spread, paste the L3 paper account's ALPACA_API_KEY / SECRET

# 2. Sanity-check selection against the live chain (read-only, no orders)
./run_spread.sh --dry-run

# 3. Install + start the service
sudo cp deploy/optionsbot-spread.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now optionsbot-spread.service
systemctl status optionsbot-spread.service        # confirm active
tail -f spread.log                                 # watch it scan

# 4. Validation scoreboard (run anytime; cron optional)
./run_spread.sh score
```

## Safety / isolation
- Paper-only: `spread_bot.py` refuses to run under `LIVE_MODE`. No live-money path exists.
- Own account (.env.spread), own state/logs (`status_spread.json`, `spread_governor_state.json`,
  `spread_trades.csv`, `spread_rejects.csv`, `spread.log`, `spread_meta.db`). No collision
  with the equity or trend sleeves.
- Every loss is defined at entry (max loss = net debit). Kill switches: −5% day, −20%
  trailing. Caps: 1 concurrent spread, 6 trades/day (configurable in config.py).
- Stays paper until the gate passes: `./run_spread.sh score` must read PASS
  (≥40 trades, positive expected R net of costs). Going live is a separate, deliberate
  change — `SPREAD_ACTIVE` is a placeholder flag; live execution is NOT implemented here.
