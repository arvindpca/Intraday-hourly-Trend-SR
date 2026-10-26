"""
RELIANCE paper book -- 2 strategies, ek hi position, Rs 1 lakh capital
======================================================================
Strategy A  "MOM"  20-Minute Momentum + 2xATR Trailing SL   (BUY + SELL, din me 1)
    Signal: (Close - Close 4 candle pehle) / ATR14  > +1.5 BUY,  < -1.5 SELL
    FINAL FILTERS (loss analysis se):
      BUY  sirf jab: RSI14 <= 65, Close > EMA20 (5m), ATR% (ATR/price) >= 0.17
      SELL sirf jab: RSI14 >= 35, Close < EMA20 (5m), aur price VWAP se 0..1 ATR neeche NA ho
Strategy B  "PBS"  VWAP Pullback Short                       (sirf SELL, din me 2)
    1. Reliance VWAP ke neeche (close < VWAP)
    2. 5-min swing high: candle ka high, 2 candle pehle aur 2 candle baad se zyada
    3. Us swing candle pe:  Nifty 15-min return > +0.1%   YA   ADX(14) > 30
    3b. Swing candle ka close < EMA20 (5m)  [final filter]
    4. Swing confirm (2 candle baad) -> agli candle ke OPEN pe SELL
    5. SL = swing high + 1xATR, Target = 2 x risk, 15:15 pe exit
    Entry 9:25 - 14:30.

NO DUPLICATE ENTRY:
  * Ek waqt pe Reliance me sirf EK position (kisi bhi strategy ki).
  * Dono ka signal ek hi candle pe aaye to MOM pehle (purani strategy).
  * Jis candle pe position band hui, us candle pe nayi entry nahi.
Position size: floor(CAPITAL / entry price) shares (bina leverage).

Backtest:  python reliance_strategy/book.py
"""
import math
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from final_strategy import (COST, EXIT, LAST_ENTRY, START, TRAIL_ATR,  # noqa: E402
                            prepare_df)

CAPITAL = float(os.getenv("PAPER_CAPITAL", "100000"))
PB_START, PB_MAX_PER_DAY, PB_SL_ATR, PB_RR = 565, 2, 1.0, 2.0
PB_NIFTY_RET, PB_ADX = 0.10, 30.0
MOM_MAX_PER_DAY = 1
RSI_BUY_MAX, RSI_SELL_MIN, EMA_TREND, MIN_ATRP_BUY, SELL_VWAP_GAP = 65.0, 35.0, 20, 0.17, 1.0
NAMES = {"MOM": "20-Min Momentum", "PBS": "VWAP Pullback Short"}


def _rma(s, n):
    return s.ewm(alpha=1 / n, adjust=False).mean()


def _rsi(s, n=14):
    x = s.diff()
    return 100 - 100 / (1 + _rma(x.clip(lower=0), n) / _rma(-x.clip(upper=0), n))


def prepare_all(rel, nif):
    """rel, nif: 5-min OHLCV DataFrames (DatetimeIndex). Return: saare columns ke saath Reliance df."""
    d = prepare_df(rel)                                     # day, tod, atr, move, signal (MOM)
    h, l, c, v = d.high, d.low, d.close, d.volume
    tp = (h + l + c) / 3
    d["vwap"] = (tp * v).groupby(d.day).cumsum() / v.groupby(d.day).cumsum().replace(0, np.nan)
    d["vwap"] = d["vwap"].ffill()
    up, dn = h.diff(), -l.diff()
    pdm = up.where((up > dn) & (up > 0), 0.0)
    ndm = dn.where((dn > up) & (dn > 0), 0.0)
    pdi, ndi = 100 * _rma(pdm, 14) / d.atr, 100 * _rma(ndm, 14) / d.atr
    d["adx"] = _rma(100 * (pdi - ndi).abs() / (pdi + ndi), 14)
    n = nif.between_time("09:15", "15:29")
    d["nret3"] = (n.close.pct_change(3) * 100).reindex(d.index)
    same = (d.day.shift(-2) == d.day) & (d.day.shift(2) == d.day)
    d["fhigh"] = (h == h.rolling(5, center=True).max()) & same
    d["rsi14"] = _rsi(c)
    d["ema20"] = c.ewm(span=EMA_TREND, adjust=False).mean()
    d["vwd"] = (c - d.vwap) / d.atr
    d["atrp"] = d.atr / c * 100
    raw = d.signal.copy()
    d["mom_raw"] = raw
    buy_ok = (d.rsi14 <= RSI_BUY_MAX) & (c > d.ema20) & (d.atrp >= MIN_ATRP_BUY)
    sell_ok = (d.rsi14 >= RSI_SELL_MIN) & (c < d.ema20) & ~((d.vwd > -SELL_VWAP_GAP) & (d.vwd < 0))
    d["signal"] = np.where((raw == 1) & buy_ok, 1, np.where((raw == -1) & sell_ok, -1, 0))
    d["pb_filter"] = (c < d.vwap) & ((d.nret3 > PB_NIFTY_RET) | (d.adx > PB_ADX)) & (c < d.ema20)
    return d


def simulate_day(t, capital=CAPITAL):
    """Ek din ke candles (t) pe dono strategy chalao, ek position ke rule ke saath.
    Return: list of trade dicts. status = closed / open."""
    o, h, l, c, a = (t[k].values for k in ("open", "high", "low", "close", "atr"))
    tod, sig, fh, pf = t.tod.values, t.signal.values, t.fhigh.values, t.pb_filter.values
    idx, n = t.index, len(t)
    trades, pos, last_exit = [], None, -1
    count = {"MOM": 0, "PBS": 0}

    def check_exit(p, j):
        s = p["side_n"]
        if p["strategy"] == "MOM":
            if (s == 1 and l[j] <= p["sl"]) or (s == -1 and h[j] >= p["sl"]):
                return p["sl"], "Trailing SL hit"
            p["sl"] = max(p["sl"], c[j] - TRAIL_ATR * p["atr"]) if s == 1 else min(p["sl"], c[j] + TRAIL_ATR * p["atr"])
        else:
            if h[j] >= p["sl"]:
                return p["sl"], "Stop loss hit"
            if l[j] <= p["tp"]:
                return p["tp"], "Target hit (2R)"
        if tod[j] >= EXIT:
            return c[j], "15:15 time exit"
        return None, None

    def close(p, j, px, why):
        q = p["qty"]
        p.update(status="closed", exit=round(float(px), 2), exit_time=str(idx[j]), reason=why)
        gross = p["side_n"] * (p["exit"] - p["entry"]) * q
        p["pnl_rs"] = round(gross - COST * p["entry"] * q, 2)
        p["ret_pct"] = round(p["pnl_rs"] / (p["entry"] * q) * 100, 3)

    for j in range(n):
        if pos is not None:
            px, why = check_exit(pos, j)
            if px is not None:
                close(pos, j, px, why)
                pos, last_exit = None, j
                continue
        if pos is not None or j <= last_exit or j == 0:
            continue
        new = None
        i = j - 1   # MOM: signal candle = pichli candle
        if count["MOM"] < MOM_MAX_PER_DAY and sig[i] != 0 and START <= tod[i] <= LAST_ENTRY:
            s = int(sig[i])
            new = dict(strategy="MOM", side_n=s, signal_time=str(idx[i]), atr=float(a[i]),
                       sl=float(o[j] - s * TRAIL_ATR * a[i]), tp=None)
        i = j - 3   # PBS: swing candle (2 candle baad confirm, phir agli candle entry)
        if new is None and i >= 0 and count["PBS"] < PB_MAX_PER_DAY and fh[i] and pf[i] \
                and PB_START <= tod[j] <= LAST_ENTRY:
            sl = h[i] + PB_SL_ATR * a[i]
            risk = sl - o[j]
            if 0.3 * a[i] < risk <= 5 * a[i]:
                new = dict(strategy="PBS", side_n=-1, signal_time=str(idx[i]), atr=float(a[i]),
                           sl=float(sl), tp=float(o[j] - PB_RR * risk))
        if new is None:
            continue
        e = float(o[j])
        new.update(entry=round(e, 2), entry_time=str(idx[j]), init_sl=round(new["sl"], 2),
                   qty=max(1, math.floor(capital / e)), status="open",
                   side="BUY" if new["side_n"] == 1 else "SELL",
                   id=f"{new['strategy']}-{idx[j]:%Y%m%d-%H%M}")
        count[new["strategy"]] += 1
        pos = new
        trades.append(pos)
        px, why = check_exit(pos, j)          # entry wali candle me hi SL/target
        if px is not None:
            close(pos, j, px, why)
            pos, last_exit = None, j
    for p in trades:
        p["sl"] = round(p["sl"], 2)
        if p["tp"] is not None:
            p["tp"] = round(p["tp"], 2)
    return trades


def pending_signal(t):
    """Last candle pe signal aaya hai jiski entry agli candle pe hogi (sirf info)."""
    j = len(t)
    if j < 4:
        return None
    if t.signal.values[-1] != 0 and START <= t.tod.values[-1] <= LAST_ENTRY:
        return "MOM", int(t.signal.values[-1])
    if t.fhigh.values[j - 3] and t.pb_filter.values[j - 3]:
        return "PBS", -1
    return None


def backtest(d):
    rows = []
    for day, t in d.groupby("day"):
        rows += [p for p in simulate_day(t) if p["status"] == "closed"]
    return pd.DataFrame(rows)


def stats(t):
    if t.empty:
        return {"Trades": 0}
    p = t.pnl_rs
    eq = p.cumsum()
    m = t.groupby(pd.to_datetime(t.entry_time).dt.to_period("M")).pnl_rs.sum()
    return {"Trades": len(t), "Win %": round((p > 0).mean() * 100, 1),
            "Profit Factor": round(p[p > 0].sum() / -p[p < 0].sum(), 2),
            "Net P&L Rs": round(p.sum()), "Return on 1L %": round(p.sum() / CAPITAL * 100, 1),
            "Max DD Rs": round((eq - eq.cummax()).min()), "Profit months": f"{(m > 0).sum()}/{len(m)}"}


def main():
    rel = pd.read_csv(os.path.join(HERE, "data", "RELIANCE_5m.csv"), parse_dates=["datetime"]).set_index("datetime")
    nif = pd.read_csv(os.path.join(HERE, "data", "NIFTY_5m.csv"), parse_dates=["datetime"]).set_index("datetime")
    d = prepare_all(rel, nif)
    t = backtest(d)
    split = pd.Timestamp("2026-06-22")
    rows = {"COMBINED (1 position)": stats(t)}
    for s, g in t.groupby("strategy"):
        rows[f"  {NAMES[s]}"] = stats(g)
    rows["COMBINED pehle 8 mahine"] = stats(t[pd.to_datetime(t.entry_time) < split])
    rows["COMBINED aakhri 4 mahine"] = stats(t[pd.to_datetime(t.entry_time) >= split])
    res = pd.DataFrame(rows).T
    pd.set_option("display.width", 200)
    print(f"Capital Rs {CAPITAL:,.0f} | {d.index[0].date()} se {d.index[-1].date()}\n")
    print(res.to_string())
    out = os.path.join(HERE, "results")
    os.makedirs(out, exist_ok=True)
    t.to_csv(os.path.join(out, "book_trades.csv"), index=False)
    m = t.groupby([pd.to_datetime(t.entry_time).dt.to_period("M"), "strategy"]).pnl_rs.sum().unstack().fillna(0).round(0)
    m["TOTAL"] = m.sum(axis=1)
    print("\nMonthly P&L (Rs):\n" + m.to_string())
    with open(os.path.join(out, "book_summary.md"), "w") as f:
        f.write(f"## Reliance book backtest (Rs {CAPITAL:,.0f}, ek position)\n\n" + res.to_markdown()
                + "\n\n### Monthly P&L (Rs)\n\n" + m.to_markdown() + "\n")


if __name__ == "__main__":
    main()
