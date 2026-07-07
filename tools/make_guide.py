"""Generates 'Options Bot - Beginners Guide.pdf' — run once, keep the PDF."""
from fpdf import FPDF

FONTS = r"C:\Windows\Fonts"
NAVY = (28, 40, 65)
BLUE = (41, 98, 255)
RED = (190, 30, 45)
GRAY = (110, 110, 110)
LIGHT = (240, 243, 250)


class Guide(FPDF):
    def __init__(self):
        super().__init__("P", "mm", "A4")
        self.add_font("Arial", "", rf"{FONTS}\arial.ttf")
        self.add_font("Arial", "B", rf"{FONTS}\arialbd.ttf")
        self.add_font("Arial", "I", rf"{FONTS}\ariali.ttf")
        self.set_auto_page_break(True, margin=18)
        self.set_margins(18, 16, 18)

    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Arial", "I", 8)
        self.set_text_color(*GRAY)
        self.cell(0, 5, "Options Trading Bot - Beginner's Guide",
                  align="R", new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def footer(self):
        self.set_y(-13)
        self.set_font("Arial", "I", 8)
        self.set_text_color(*GRAY)
        self.cell(0, 6, f"Page {self.page_no()}", align="C")

    # ---- building blocks ----
    def h1(self, text):
        self.ln(3)
        self.set_font("Arial", "B", 16)
        self.set_text_color(*NAVY)
        self.cell(0, 9, text, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*BLUE)
        self.set_line_width(0.6)
        self.line(self.l_margin, self.get_y(), self.l_margin + 60, self.get_y())
        self.ln(4)

    def h2(self, text):
        self.ln(2)
        self.set_font("Arial", "B", 12)
        self.set_text_color(*BLUE)
        self.cell(0, 7, text, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def p(self, text):
        self.set_font("Arial", "", 10.5)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 5.6, text, new_x="LMARGIN", new_y="NEXT")
        self.ln(1.5)

    def bullet(self, text, bold_lead=None):
        self.set_text_color(30, 30, 30)
        self.set_font("Arial", "B", 10.5)
        self.cell(5, 5.6, "•")
        x = self.get_x()
        if bold_lead:
            self.set_font("Arial", "B", 10.5)
            w = self.get_string_width(bold_lead + " ")
            self.cell(w, 5.6, bold_lead + " ")
            self.set_font("Arial", "", 10.5)
            self.multi_cell(0, 5.6, text, new_x="LMARGIN", new_y="NEXT")
        else:
            self.set_font("Arial", "", 10.5)
            self.multi_cell(0, 5.6, text, new_x="LMARGIN", new_y="NEXT")
        self.ln(0.8)

    def warn(self, title, text):
        self.ln(1)
        self.set_fill_color(253, 235, 235)
        self.set_draw_color(*RED)
        self.set_line_width(0.4)
        y0 = self.get_y()
        self.set_font("Arial", "B", 11)
        self.set_text_color(*RED)
        self.multi_cell(0, 6, title, fill=True, border="LTR", new_x="LMARGIN", new_y="NEXT")
        self.set_font("Arial", "", 10)
        self.set_text_color(60, 20, 20)
        self.multi_cell(0, 5.4, text, fill=True, border="LBR", new_x="LMARGIN", new_y="NEXT")
        self.ln(3)

    def rule_table(self, rows):
        self.set_font("Arial", "", 9.8)
        self.set_text_color(30, 30, 30)
        self.set_draw_color(200, 205, 215)
        self.set_line_width(0.2)
        with self.table(col_widths=(52, 122), line_height=5.4,
                        borders_layout="HORIZONTAL_LINES",
                        cell_fill_color=LIGHT, cell_fill_mode="ROWS",
                        first_row_as_headings=True) as table:
            head = table.row()
            head.cell("Safety rule")
            head.cell("What it means for you")
            for k, v in rows:
                r = table.row()
                r.cell(k)
                r.cell(v)
        self.ln(3)


pdf = Guide()

# ============ PAGE 1 — title ============
pdf.add_page()
pdf.ln(28)
pdf.set_font("Arial", "B", 26)
pdf.set_text_color(*NAVY)
pdf.multi_cell(0, 12, "Your Options Trading Bot", align="C",
               new_x="LMARGIN", new_y="NEXT")
pdf.set_font("Arial", "", 14)
pdf.set_text_color(*GRAY)
pdf.multi_cell(0, 8, "A Beginner's Guide - no trading experience required",
               align="C", new_x="LMARGIN", new_y="NEXT")
pdf.ln(10)

pdf.warn("READ THIS BEFORE ANYTHING ELSE",
         "This bot trades options, one of the riskiest things you can do in a "
         "stock market. With a $500 account, a bad day can cost $75 (15% of "
         "your money) before the bot stops itself. The bot starts in PAPER "
         "MODE - fake money, real prices. Keep it there for at least a few "
         "weeks. Nothing in this guide is financial advice. Never trade money "
         "you cannot afford to lose.")

pdf.h2("Start in 3 steps")
pdf.bullet("Double-click \"Launch Options Bot.bat\" in this folder. A dashboard "
           "opens in your web browser.", "1.")
pdf.bullet("Press the blue \"Start\" button in the left sidebar. The bot begins "
           "watching the market (it only trades 9:45 AM - 3:30 PM Eastern, "
           "Monday-Friday).", "2.")
pdf.bullet("Watch. The dashboard shows every trade, your profit/loss for the "
           "day, and the bot's reasoning in the Logs tab.", "3.")
pdf.ln(2)
pdf.p("Everything below explains what the bot is actually doing and why - in "
      "plain English.")

# ============ Options 101 ============
pdf.add_page()
pdf.h1("Options in plain English")
pdf.p("A stock option is a contract that bets on where a stock's price is "
      "going, without buying the stock itself. There are only two kinds, and "
      "this bot uses both:")
pdf.bullet("is a bet that the stock will go UP. If you buy a call and the "
           "stock rises, your option becomes more valuable.", "A CALL")
pdf.bullet("is a bet that the stock will go DOWN. If you buy a put and the "
           "stock falls, your option becomes more valuable.", "A PUT")
pdf.p("Words you'll see on the dashboard:")
pdf.bullet("The price you pay for the option. One contract covers 100 shares, "
           "so an option priced at $0.40 costs $40. This is the MOST you can "
           "lose on that trade.", "Premium:")
pdf.bullet("The stock price your bet is measured against. The bot buys "
           "strikes slightly beyond the current price (called 'out of the "
           "money') because they are cheaper.", "Strike:")
pdf.bullet("The date the contract dies. This bot only buys options expiring "
           "in 1-7 days, and it NEVER holds one overnight - everything is "
           "sold by 3:45 PM the same day.", "Expiry:")
pdf.p("Why options move so fast: they are leveraged (small stock moves cause "
      "big option moves, in both directions) and they lose value every hour "
      "just from time passing ('time decay'). That is why this bot gets in "
      "and out within minutes to hours - holding is the enemy.")

# ============ How it finds trades ============
pdf.h1("How the bot decides to buy")
pdf.p("Every 30 seconds the bot checks six heavily-traded names: SPY and QQQ "
      "(funds that track the whole market) plus NVDA, TSLA, AAPL and AMZN. It "
      "is looking for a stock that just made a strong, confirmed move:")
pdf.bullet("The stock moved more than 0.5% in the last 15-minute candle. "
           "Something is happening.", "Momentum:")
pdf.bullet("An indicator called RSI confirms the move is strong (above 65 for "
           "up-moves, below 35 for down-moves). This filters out weak "
           "wiggles.", "Confirmation:")
pdf.bullet("Today's trading volume must be at least 1.5x normal. Big volume "
           "means real buyers/sellers are behind the move, not noise.",
           "Crowd check:")
pdf.p("Only when ALL THREE agree does it shop for an option: up-move = buy a "
      "call, down-move = buy a put. Then the option itself must pass quality "
      "checks - it must be cheap enough (under $50), actively traded (so the "
      "bot can sell it back instantly), fairly priced (not inflated premium), "
      "and have a tight gap between buying and selling price.")
pdf.p("Most scans find nothing. That is normal and good - the bot is picky on "
      "purpose. Zero trades on a quiet day beats forced bad trades.")

# ============ Protection ============
pdf.add_page()
pdf.h1("How it protects your money")
pdf.p("Every rule below runs automatically. You do not have to do anything.")
pdf.rule_table([
    ("Max $50 per trade", "One contract, never more than $50 of premium. One bad trade cannot hurt you badly."),
    ("Max 3 open trades", "At most $150 at risk at any moment."),
    ("Take profit +40%", "The option is sold automatically once it gains 40%."),
    ("Stop loss -30%", "The option is sold automatically once it loses 30%. Losses are cut fast."),
    ("3:45 PM flatten", "Everything is sold before the close, every day, no exceptions. You never wake up to an overnight surprise."),
    ("Daily halt at -$75", "If the day's losses reach $75, the bot sells everything, stops for the day, and sends you a Discord alert."),
    ("2-loss cool-down", "Two losing trades in a row = 30 minute timeout. Stops 'revenge trading' loops."),
    ("Entry window only", "No trades in the chaotic first 15 minutes or last 30 minutes of the day."),
])
pdf.p("Every completed trade is written to trades.csv - what it bought, why, "
      "what it paid, what it sold for, and the result. That file is your "
      "permanent record and it is also what the AI learns from.")

# ============ AI ============
pdf.h1("The AI that learns from mistakes")
pdf.p("Each trade is saved with a snapshot of the conditions at the moment of "
      "entry: how strong the momentum was, the RSI value, the volume, the "
      "time of day, which stock, and more. After 50 completed trades, the bot "
      "trains a small machine-learning model on that history to answer one "
      "question: 'trades that looked like THIS one - did they usually win or "
      "lose?'")
pdf.bullet("the model studies your trades silently. You will see 'learning' "
           "on the dashboard.", "First 50 trades:")
pdf.bullet("the model scores every new opportunity but cannot block anything "
           "yet. It must first prove on past data that its predictions "
           "actually beat a coin flip.", "Advisory mode:")
pdf.bullet("once proven, it vetoes any trade it scores below a 45% chance of "
           "winning. The dashboard shows 'gating' with its accuracy score "
           "(AUC - above 0.55 means it has real predictive power).",
           "Gating mode:")
pdf.p("In short: the strategy finds trades, the risk rules cap the damage, "
      "and the AI slowly learns which of the strategy's ideas are its worst "
      "ones - and starts skipping them.")

# ============ Dashboard tour ============
pdf.add_page()
pdf.h1("The dashboard, piece by piece")
pdf.h2("Left sidebar")
pdf.bullet("Green 'Paper mode' badge = fake money. A red badge means LIVE "
           "real-money trading.", "Mode badge:")
pdf.bullet("Start launches the bot; Stop closes all open trades first, then "
           "shuts down. Closing the browser tab does NOT stop the bot - only "
           "the Stop button does.", "Start / Stop:")
pdf.bullet("Change which stocks it watches, dollar limits, profit/loss "
           "targets and the AI threshold. Press Save, then Stop and Start "
           "the bot for changes to take effect.", "Settings:")
pdf.h2("Main screen (refreshes every 10 seconds)")
pdf.bullet("Today's profit or loss in dollars and as a % of your $500.",
           "Daily P&L:")
pdf.bullet("What the bot is holding right now and how each position is "
           "doing.", "Open positions tab:")
pdf.bullet("Every past trade, your total results, and a chart of your "
           "account value over time.", "Trade history tab:")
pdf.bullet("The bot's diary - every decision and the reason for it. If "
           "anything looks wrong, look here first.", "Logs tab:")
pdf.h2("Discord alerts (optional)")
pdf.p("If you add a Discord webhook address to the .env file, the bot "
      "messages you on every entry, every exit, any halt, and a daily "
      "summary - so you can leave it running and stay informed from your "
      "phone.")

# ============ Files ============
pdf.h1("What the files in this folder are")
pdf.bullet("the app you double-click.", "Launch Options Bot.bat -")
pdf.bullet("your trade record. Open it in Excel anytime.", "trades.csv -")
pdf.bullet("your private keys and settings. NEVER share this file or its "
           "contents with anyone.", ".env -")
pdf.bullet("the AI's saved brain and its training settings.", "model.pkl -")
pdf.bullet("the bot's code: scanner (finds moves), options_chain (picks "
           "contracts), executor (buys/sells), risk_manager (the safety "
           "rules), learner (the AI), alerts (Discord), app (the dashboard), "
           "main (ties it all together).", "The .py files -")
pdf.bullet("settings saved from the dashboard, the bot's status heartbeat, "
           "and log files. Safe to ignore.",
           "settings.json / status.json / bot.log -")

# ============ Going live ============
pdf.add_page()
pdf.h1("Before you even think about real money")
pdf.bullet("Run paper mode for AT LEAST 3-4 weeks. This also gives the IV "
           "filter and the AI time to warm up - both need history to work.")
pdf.bullet("Check trades.csv weekly. Is it actually profitable AFTER a real "
           "stretch of days? A lucky week proves nothing.")
pdf.bullet("Understand that paper results are optimistic: real orders fill "
           "slightly worse than paper ones.")
pdf.bullet("Going live requires TWO deliberate changes in .env: setting "
           "LIVE_MODE=true AND swapping in your live Alpaca keys. That "
           "friction is intentional.")
pdf.bullet("If live trading starts badly, stop. The bot halting at -$75 daily "
           "is a circuit breaker, not a strategy.")
pdf.ln(2)
pdf.warn("THE HONEST TRUTH",
         "Most short-term options traders lose money. The edge this bot hunts "
         "for is small and may not exist in every market. Its real strengths "
         "are discipline - it never panics, never doubles down, always takes "
         "the stop loss - and a paper trail you can study. Treat the first "
         "months as tuition: the goal is to learn whether the strategy works, "
         "not to get rich.")

pdf.h1("Mini glossary")
for term, defn in [
    ("Alpaca", "The brokerage whose service the bot trades through."),
    ("Paper trading", "Simulated trading: real market prices, fake money."),
    ("Candle", "A bar summarizing the price action of one time slice (here, 15 minutes)."),
    ("RSI", "Relative Strength Index - a 0-100 score of how strong a recent move is."),
    ("Bid / Ask", "The price buyers will pay / sellers want. The bot needs these close together."),
    ("Open interest", "How many contracts exist. High = easy to buy and sell quickly."),
    ("IV (implied volatility)", "How expensive an option is relative to expected movement. The bot avoids overpriced ones."),
    ("P&L", "Profit and loss."),
    ("Flatten", "Sell everything, hold nothing."),
]:
    pdf.bullet(defn, f"{term}:")

pdf.output(r"C:\Users\Shaq\Desktop\options-bot\Options Bot - Beginners Guide.pdf")
print("PDF written")

