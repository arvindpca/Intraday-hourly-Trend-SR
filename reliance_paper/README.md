# Reliance Paper Book (live, GitHub Actions)

Rs 1 lakh capital, Reliance 5-min, **ek waqt pe sirf 1 position** (no duplicate entry), Telegram alerts.

| Strategy | Side | Rules (short) |
|---|---|---|
| 20-Min Momentum (MOM) | BUY/SELL | (Close − Close 4 candle pehle) ÷ ATR14 > +1.5 BUY / < −1.5 SELL, agli candle open pe entry, SL 2×ATR trailing, koi target nahi, din me 1 |
| VWAP Pullback Short (PBS) | SELL | Close < VWAP + 5-min swing high (2 left/2 right) + (Nifty 15-min > +0.1% **ya** ADX14 > 30) → confirm ke baad agli candle open pe SELL, SL = swing high + 1×ATR, target 2R, din me 2 |

Entry 9:25/9:30 – 14:30, sab position 15:15 pe band. Dono ka signal ek saath → MOM pehle.

Backtest (Oct 2025 – Oct 2026, ek position, charges 0.06%): 336 trades, PF 1.32, +Rs 23,400, max DD −Rs 9,500.

- `paper/paper_trades.csv` — har band trade (trade ID se duplicate nahi)
- `paper/STATUS.md` — total + strategy-wise result
- `paper/state.json` — aaj kaunse alert ja chuke
- Workflow `reliance-paper.yml` — subah ek job shuru hota hai aur 15:45 tak har 5 min khud check karta hai (backup start bhi)

Research / backtest code: private repo `Indian-Stock-Intraday-15Minuts-Chart/reliance_strategy`.
