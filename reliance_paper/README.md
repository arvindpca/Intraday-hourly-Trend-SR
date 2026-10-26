# Reliance Paper Book (live, GitHub Actions)

Rs 1 lakh capital, Reliance 5-min, **ek waqt pe sirf 1 position** (no duplicate entry), Telegram alerts.

| Strategy | Side | Rules (short) |
|---|---|---|
| 20-Min Momentum (MOM) | BUY/SELL | (Close − Close 4 candle pehle) ÷ ATR14 > +1.5 BUY / < −1.5 SELL, agli candle open pe entry, SL 2×ATR trailing, koi target nahi, din me 1. **BUY sirf:** RSI14 ≤ 65, Close > EMA20, ATR% ≥ 0.17. **SELL sirf:** RSI14 ≥ 35, Close < EMA20, aur price VWAP se 0–1 ATR neeche na ho |
| VWAP Pullback Short (PBS) | SELL | Close < VWAP **aur < EMA20** + 5-min swing high (2 left/2 right) + (Nifty 15-min > +0.1% **ya** ADX14 > 30) → confirm ke baad agli candle open pe SELL, SL = swing high + 1×ATR, target 2R, din me 2 |

Entry 9:25/9:30 – 14:30, sab position 15:15 pe band. Dono ka signal ek saath → MOM pehle.

**Version: FINAL** (27 Oct 2026 se) — loss analysis filters (RSI, EMA20, ATR%, VWAP gap).
Backtest (Oct 2025 – Oct 2026, ek position, charges 0.06%): 283 trades, 47% win, PF 1.84, +Rs 40,169, max DD −Rs 3,338.
26 Oct ka 1 trade purane version (bina filter) ka hai.

- `paper/paper_trades.csv` — har band trade (trade ID se duplicate nahi)
- `paper/STATUS.md` — total + strategy-wise result
- `paper/state.json` — aaj kaunse alert ja chuke
- Workflow `reliance-paper.yml` — subah ek job shuru hota hai aur 15:45 tak har 5 min khud check karta hai (backup start bhi)

Research / backtest code: private repo `Indian-Stock-Intraday-15Minuts-Chart/reliance_strategy`.
