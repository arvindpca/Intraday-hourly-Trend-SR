# 🔒 Daily Swing Strategy - LOCKED v1 (26 Oct 2026)

Ye rules **lock** hain. Paper trading ke dauraan (kam se kam 1-2 mahine) inme koi badlav nahi.
Badlav karna ho to naya version (v2) banega, aur v1 ka paper result alag rakha jayega.

## Rules
| Step | Rule |
|---|---|
| Stocks | Nifty 500, Close Rs 200-3000, Close x Volume > Rs 5 crore |
| Market filter | Nifty 50 close > Nifty 50 EMA(50). Neeche ho to naya trade NAHI |
| Signal (daily candle) | MACD(12,26,9) aaj upar cross (kal neeche/barabar) · RSI(14) 50-70 · AccDist aaj > kal · Close > Open · Volume > 2x SMA(Vol,20) · Volume > pichhle 10 din ka max · Close > VWAP · EMA8 > EMA21 |
| Entry | Usi din 3:20-3:29 (3 PM alert se). Paper bot: close price |
| Size | Rs 20,000 / trade, max 5 khule (capital Rs 1,00,000). Ek stock me ek hi trade |
| Zyada signal | Sabse zyada volume uchhal (Volume / SMA20) wale pehle |
| Stop-loss | Entry se 5% neeche. Gap down SL ke neeche -> open pe exit |
| Exit | Koi target nahi. 20 trading din ke close pe becho |
| Cost | 0.2% round trip |

## Baseline - backtest (Nifty 500, Dec 2023 - Oct 2026, Yahoo daily data)
| | Usi din close entry | Agle din open entry |
|---|---|---|
| Trades | 159 | 189 |
| Win rate | 45.3% | 34.9% |
| Net profit (Rs 1 lakh) | Rs 94,907 | Rs 40,848 |
| Saalana (approx) | ~34% | ~15% |
| Profit factor | 2.13 | 1.33 |
| Avg win / loss | +Rs 2,484 / -Rs 965 | +Rs 2,486 / -Rs 1,002 |
| Max drawdown | -Rs 8,990 (9%) | -Rs 20,317 (20%) |
| Saal-wise | 2024 +51,468 · 2025 +38,873 · 2026 (Jan-Sep) +5,604 | |

## Paper result ko kaise jaanchein (1-2 mahine baad)
- Win rate 35-50% ke beech, avg win avg loss ka ~2.5 guna -> strategy theek chal rahi
- Profit factor 1.3 se upar -> theek; 1 se neeche -> rok ke dobara dekho
- Drawdown Rs 10-12k se zyada -> backtest se bura, review karo
- Dhyan: Nifty 50 EMA ke neeche ho to hafton tak trade nahi aayega - ye normal hai

## Files
- Rules code: `paper_trader.py` + `strategy_core.py` | 3 PM alert: `preclose_alert.py`
- Live hisaab: `state/STATUS.md`, `state/closed_trades.csv`, `state/daily_log.csv`
- Locked: 26 Oct 2026 (pehle Indian-Stock-Intraday-15Minuts-Chart commit `fea80be`, 26 Oct ko is public repo me shift). Rules files me badlav nahi.
