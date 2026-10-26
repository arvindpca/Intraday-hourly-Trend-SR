# Paper Swing Trader

Daily Swing strategy (MACD crossover + 2x volume, Nifty > 50 EMA filter) ka **paper trading bot**.
Koi asli order nahi lagta.

- Roz Mon-Fri **4:37 PM IST** chalta hai (6:37 PM pe retry), GitHub Actions: `Paper Swing Trader`
- Har din Telegram pe: Nifty filter, naye BUY, SELL (SL / 20 din), khule trades aur kul P&L
- **3 PM alert** (Mon-Fri ~3:00 PM IST, `Paper Swing 3PM Alert`): Angel One live data se aaj ke BUY signals + aaj bechne wale trades, taaki 3:20-3:29 me order laga sako
- Live status: [`state/STATUS.md`](state/STATUS.md)
- Saare band trades: `state/closed_trades.csv` | Roz ka equity: `state/daily_log.csv` | Signals: `state/signals_log.csv`

## Rules
| | |
|---|---|
| Stocks | Nifty 500, Close 200-3000, turnover > 5 cr |
| Market | Nifty 50 close > 50 EMA, warna naya trade nahi |
| Signal | MACD aaj cross up, RSI 50-70, AccDist up, green candle, Vol > 2x SMA20 aur > 10 din max, Close > VWAP, EMA8 > EMA21 |
| Entry | Signal din ke close pe, Rs 20,000/trade (kul Rs 1 lakh), max 5 khule, no duplicate, zyada signal -> sabse zyada volume pehle |
| SL | Entry se 5% neeche (gap down -> open pe) |
| Exit | 20 trading din ke close pe, koi target nahi |
| Cost | 0.2% round trip |

Reset karna ho to `state/` folder ki files delete karke commit karo.
