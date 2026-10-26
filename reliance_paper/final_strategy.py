"""
RELIANCE "20-Minute Momentum + 2-ATR Trailing Stop" -- final intraday strategy
=============================================================================
Reverse engineering se mila: Reliance me jab 20 minute me tez move aata hai, wo aksar
thoda aur aage jaata hai (1 ghante ke scale pe momentum). Profit tab aata hai jab
chhota SL na rakh ke, trailing stop se winner ko bhaagne diya jaye.

RULES (5-minute chart):
  1. ATR(14) nikalo (5-min candles pe).
  2. Move = (abhi ka close - 4 candle pehle ka close) / ATR
       Move >  +1.5  -> BUY signal
       Move <  -1.5  -> SELL signal
  3. Signal candle close hone ke baad, AGLI candle ke open pe entry.
  4. Entry sirf 9:30 se 14:30 ke beech. Din me sirf 1 trade.
  5. Stop loss = entry se 2 x ATR. Har candle close pe trailing:
       BUY:  SL = max(purana SL, close - 2 x ATR)
       SELL: SL = min(purana SL, close + 2 x ATR)
     Koi fixed target nahi.
  6. 15:15 pe jo bhi position ho, band.

Usage:
  python reliance_strategy/final_strategy.py                 # data/RELIANCE_5m.csv pe backtest
  python reliance_strategy/final_strategy.py --signal-today  # aaj ka live signal check (data file ke last candle tak)
"""
import argparse
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
LOOKBACK, THRESH, TRAIL_ATR = 4, 1.5, 2.0
START, LAST_ENTRY, EXIT = 570, 870, 915    # minutes from midnight: 9:30, 14:30, 15:15
COST = 0.0006                              # 0.06% round trip (brokerage+STT+slippage approx)
SPLIT = pd.Timestamp("2026-06-22")         # rules chune gaye is date se pehle ke data pe


def atr(d, n=14):
    pc = d.close.shift()
    tr = pd.concat([d.high - d.low, (d.high - pc).abs(), (d.low - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def prepare(path):
    d = pd.read_csv(path, parse_dates=["datetime"]).set_index("datetime")
    return prepare_df(d)


def prepare_df(d):
    d = d.between_time("09:15", "15:29").copy()
    d["day"] = d.index.normalize()
    d["tod"] = d.index.hour * 60 + d.index.minute
    d["atr"] = atr(d)
    move = (d.close - d.close.shift(LOOKBACK)) / d.atr
    d["move"] = move
    d["signal"] = np.where(move > THRESH, 1, np.where(move < -THRESH, -1, 0))
    return d


def backtest(d):
    o, h, l, c, a = d.open.values, d.high.values, d.low.values, d.close.values, d.atr.values
    tod, day, sig, idx = d.tod.values, d.day.values, d.signal.values, d.index
    trades, i, n, done_day = [], 0, len(d), None
    while i < n - 1:
        s = sig[i]
        if s and day[i] != done_day and START <= tod[i] <= LAST_ENTRY and day[i + 1] == day[i]:
            e, R = o[i + 1], a[i]
            stop, j, ex, why = e - s * TRAIL_ATR * R, i + 1, None, ""
            while True:
                if (s == 1 and l[j] <= stop) or (s == -1 and h[j] >= stop):
                    ex, why = stop, "trail SL"
                else:
                    stop = max(stop, c[j] - TRAIL_ATR * R) if s == 1 else min(stop, c[j] + TRAIL_ATR * R)
                    if tod[j] >= EXIT or j + 1 >= n or day[j + 1] != day[j]:
                        ex, why = c[j], "15:15 exit"
                if ex is not None:
                    break
                j += 1
            trades.append(dict(entry_time=idx[i + 1], side="BUY" if s == 1 else "SELL", entry=round(e, 2),
                               exit_time=idx[j], exit=round(ex, 2), reason=why,
                               ret_pct=round((s * (ex - e) / e - COST) * 100, 3)))
            done_day, i = day[i], j + 1
            continue
        i += 1
    return pd.DataFrame(trades)


def stats(t):
    r = t.ret_pct / 100
    eq = r.cumsum()
    m = t.groupby(pd.to_datetime(t.entry_time).dt.to_period("M")).ret_pct.sum()
    return {
        "Trades": len(t), "Win %": round((r > 0).mean() * 100, 1),
        "Avg win %": round(r[r > 0].mean() * 100, 2), "Avg loss %": round(r[r < 0].mean() * 100, 2),
        "Profit Factor": round(r[r > 0].sum() / -r[r < 0].sum(), 2),
        "Total %": round(r.sum() * 100, 1), "Max DD %": round((eq - eq.cummax()).min() * 100, 1),
        "Profitable months": f"{(m > 0).sum()}/{len(m)}",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "data", "RELIANCE_5m.csv"))
    ap.add_argument("--signal-today", action="store_true")
    args = ap.parse_args()
    d = prepare(args.data)

    if args.signal_today:
        last = d.iloc[-1]
        side = {1: "BUY", -1: "SELL", 0: "No signal"}[int(last.signal)]
        if last.signal and not (START <= last.tod <= LAST_ENTRY):
            side += " (lekin entry time 9:30-14:30 ke bahar hai -> trade nahi)"
        print(f"{d.index[-1]}  close={last.close:.2f}  move={last.move:+.2f} ATR  -> {side}")
        if last.signal:
            print(f"Initial SL distance = {TRAIL_ATR * last.atr:.2f} rupees (2 x ATR)")
        return

    t = backtest(d)
    t["period"] = np.where(pd.to_datetime(t.entry_time) < SPLIT, "train (rules chune)", "test (unseen)")
    rows = {"FULL": stats(t)}
    for p, g in t.groupby("period"):
        rows[p] = stats(g)
    for s, g in t.groupby("side"):
        rows[f"only {s}"] = stats(g)
    res = pd.DataFrame(rows).T
    pd.set_option("display.width", 200)
    print(res.to_string())
    out = os.path.join(HERE, "results")
    os.makedirs(out, exist_ok=True)
    t.to_csv(os.path.join(out, "final_strategy_trades.csv"), index=False)
    monthly = t.groupby(pd.to_datetime(t.entry_time).dt.to_period("M")).ret_pct.agg(["count", "sum"]).round(2)
    with open(os.path.join(out, "final_strategy_summary.md"), "w") as f:
        f.write(f"## Reliance final strategy backtest ({d.index[0].date()} to {d.index[-1].date()})\n\n")
        f.write(res.to_markdown() + "\n\n### Monthly\n\n" + monthly.to_markdown() + "\n")
    print("\nSaved:", out)


if __name__ == "__main__":
    main()
