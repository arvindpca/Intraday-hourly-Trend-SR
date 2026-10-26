# Paper V2.1 "Shant" - paper trading bot

V1 (paper_swing/) se **bilkul alag** strategy aur alag paper capital (Rs 1 lakh). Koi asli order nahi.

| | |
|---|---|
| Trend | EMA20 > EMA50 > EMA200 |
| Breakout | Close > pichhle 20 din ka sabse ooncha High |
| Candle | Body > ATR(14), body > 60% range, upar ki wick < 25% range |
| Volume | > 1.5x SMA(Volume, 50) |
| Zyada bhaga nahi | Close < EMA20 x 1.08 |
| Liquidity | 20 din avg turnover > Rs 5 crore |
| V2.1 filters | 52-week high se 6% ke andar, RSI(14) > 65, 20 din me 5%+ upar |
| Market | Nifty 50 close > 50 EMA |
| Entry | Usi din close (3 PM alert se 3:20-3:29) |
| Size | Rs 20,000 x 5, ek stock me ek trade, zyada signal -> sabse zyada volume (V/SMA50) pehle |
| SL | Entry - 2 x ATR(14) (gap down -> open pe) |
| Exit | 10 trading din ke close pe |

Backtest (Aug 2023 - Oct 2026, Rs 1 lakh): 216 trades, 58% win, ~Rs 49,800 (0.2% + DP Rs 24), PF ~1.74, max girawat ~8%.

Telegram: 3 PM "V2.1 (shant) Alert" + shaam "Paper V2.1 (shant)" update. Status: `state/STATUS.md`
