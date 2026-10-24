# NSE MACD + Volume Breakout: SL/TP Backtest

Chartink screener wali strategy ka backtest. Nifty 500 stocks pe pichhle 180 trading din me
98 SL/TP combinations test karke best combo nikalta hai.

## Entry conditions (daily candle)

| # | Condition |
|---|-----------|
| 1 | Close 200 se 3000 ke beech |
| 2 | Close × Volume > ₹5 crore (liquidity) |
| 3 | RSI(14) 50 se 70 ke beech |
| 4 | MACD Line > Signal (aaj) aur MACD Line ≤ Signal (kal): fresh bullish crossover |
| 5 | AccDist aaj > kal |
| 6 | Close > Open (green candle) |
| 7 | Volume > SMA(Volume, 20) × 2 |
| 8 | Volume > pichhle 10 din ka max volume |
| 9 | Close > VWAP (typical price) |
| 10 | EMA(8) > EMA(21) |

Entry: signal ke agle din ka open. Cost: 0.2% round trip.

## Kya test hota hai

- **SL:** signal candle low, 3% / 5% / 7%, 1.5x / 2x / 3x ATR
- **TP:** 1R, 1.5R, 2R, 3R, 5%, 10%, ya EMA21 ke neeche close pe exit
- **Max hold:** 10 / 20 din

Same candle me SL aur TP dono touch hon to SL maana jata hai (conservative).

## GitHub pe chalana (kuch install nahi karna)

1. Repo ke **Actions** tab pe jao
2. **Run Backtest** select karo, phir **Run workflow** dabao
3. 10–15 minute baad result:
   - `results/LATEST.txt`: top 10 combos aur best combo
   - `results/<date>/sl_tp_results.csv`: saare combinations
   - `results/<date>/best_combo_trades.csv`: best combo ke saare trades

## Apne computer pe chalana

```
pip install -r requirements.txt
python backtest_macd_breakout.py
```

Options: `--days 180`, `--cost 0.2`, `--min-trades 15`, `--symbols-file ind_nifty500list.csv`

## Limitations

- Aaj ki Nifty 500 list use hoti hai (survivorship bias), isliye result thoda optimistic ho sakta hai.
- 180 din chhota period hai. Sirf #1 combo nahi, top 3–5 me common pattern dekho.
- Data Yahoo Finance se aata hai (split-adjusted).
