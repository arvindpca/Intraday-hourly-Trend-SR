"""
V2 - TREND + 20 DIN HIGH BREAKOUT (Chartink screener se) - Backtest
==================================================================
v1 (locked) se bilkul alag. Nifty 500, ~3 saal daily data (Yahoo), Rs 20,000 x 5, 0.2% cost.

Entry conditions (daily):
  EMA20 > EMA50 > EMA200
  Close > pichhle 20 din ka sabse ooncha High
  Volume > SMA(Volume,50) x 1.5
  |Close - Open| > ATR(14)
  (Close - Open) > 0.60 x (High - Low)        -> badi hari candle
  (High - Close) < 0.25 x (High - Low)        -> high ke paas close
  Close < EMA20 x 1.08                         -> zyada bhaga hua nahi
  Liquidity (FIXED): SMA(Close x Volume, 20) > Rs 5 crore
     (original Chartink: SMA(Close,20) > 5000 -> "ORIGINAL" variant me alag se dikhaya)
  Fundamentals (profit, D/E, ROE, EPS growth): backtest me NAHI (purana data bharosemand nahi).

Test grid:
  Nifty filter : OFF / ON (Nifty close > Nifty EMA50)
  Entry        : same-day close (3 PM jaisa) / next-day open
  SL           : breakout candle low / 2x ATR / 5%
  Exit         : 10 din / 20 din / 2R target (warna 20 din) / close EMA20 ke neeche (max 60 din)
Portfolio: max 5 khule, ek stock me ek trade, zyada signal -> sabse zyada volume uchhal pehle.
"""
import itertools
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import backtest_macd_breakout as base  # download + load_symbols  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)
PER_TRADE, MAX_POS, COST = 20_000, 5, 0.2
CAPITAL = PER_TRADE * MAX_POS
WARMUP = 220

SL_MODES = ["Candle low", "2x ATR", "5%"]
EXIT_MODES = ["10 din", "20 din", "2R target / 20 din", "EMA20 close exit"]
ENTRY_MODES = ["same-day close", "next-day open"]
FILTERS = ["OFF", "ON"]

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_rows", 300)
LOG = []


def out(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def indicators(d):
    d = d.copy()
    c, o, h, l, v = d["Close"], d["Open"], d["High"], d["Low"], d["Volume"]
    d["ema20"] = c.ewm(span=20, adjust=False).mean()
    d["ema50"] = c.ewm(span=50, adjust=False).mean()
    d["ema200"] = c.ewm(span=200, adjust=False).mean()
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    d["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    d["vsma50"] = v.rolling(50).mean()
    rng = (h - l)
    tech = (
        (d["ema20"] > d["ema50"]) & (d["ema50"] > d["ema200"])
        & (c > h.shift(1).rolling(20).max())
        & (v > d["vsma50"] * 1.5)
        & ((c - o).abs() > d["atr"])
        & ((c - o) > rng * 0.60)
        & ((h - c) < rng * 0.25)
        & (c < d["ema20"] * 1.08)
    )
    d["sig_fixed"] = (tech & ((c * v).rolling(20).mean() > 5e7)).fillna(False)
    d["sig_original"] = (tech & (c.rolling(20).mean() > 5000)).fillna(False)
    d["vr"] = v / d["vsma50"]
    return d


def trades_for(sym, d, start, mkt_ok, sigcol, use_filter, entry_mode, sl_mode, exit_mode):
    O, H, L, C = (d[k].to_numpy() for k in ["Open", "High", "Low", "Close"])
    atr, ema20, sig, vr = d["atr"].to_numpy(), d["ema20"].to_numpy(), d[sigcol].to_numpy(), d["vr"].to_numpy()
    dates, n = d.index, len(d)
    rows, last = [], -1
    for i in range(start, n - 1):
        if not sig[i] or (use_filter and not mkt_ok[i]):
            continue
        e = i if entry_mode == "same-day close" else i + 1
        if e <= last:
            continue
        entry = C[i] if e == i else O[e]
        sl = L[i] if sl_mode == "Candle low" else (entry - 2 * atr[i] if sl_mode == "2x ATR" else entry * 0.95)
        if not np.isfinite(sl) or sl >= entry:
            continue
        risk = entry - sl
        tp = entry + 2 * risk if exit_mode.startswith("2R") else np.inf
        hold = 10 if exit_mode == "10 din" else (60 if exit_mode.startswith("EMA20") else 20)
        j = i + 1
        while True:
            if j > e and O[j] <= sl:
                px, why = O[j], "SL (gap)"
            elif L[j] <= sl:
                px, why = sl, "SL"
            elif j > e and O[j] >= tp:
                px, why = O[j], "Target (gap)"
            elif H[j] >= tp:
                px, why = tp, "Target 2R"
            elif exit_mode.startswith("EMA20") and C[j] < ema20[j]:
                px, why = C[j], "EMA20 ke neeche"
            elif j - e >= hold:
                px, why = C[j], f"{hold} din pure"
            elif j == n - 1:
                px, why = C[j], "Abhi khula"
            else:
                j += 1
                continue
            break
        rows.append((sym.replace(".NS", ""), dates[i], dates[e], dates[j], entry, sl, px, why, vr[i], j - e))
        last = j
    return rows


COLS = ["symbol", "signal_date", "entry_time", "exit_time", "entry", "sl", "exit", "reason", "vr", "days"]


def portfolio(c):
    if c.empty:
        return pd.DataFrame()
    c = c.sort_values(["entry_time", "vr"], ascending=[True, False])
    open_pos, taken = [], []
    for r in c.itertuples(index=False):
        open_pos = [p for p in open_pos if p[0] > r.entry_time]
        if len(open_pos) >= MAX_POS or any(p[1] == r.symbol for p in open_pos):
            continue
        qty = math.floor(PER_TRADE / r.entry)
        if qty < 1:
            continue
        pnl = qty * (r.exit - r.entry) - qty * r.entry * COST / 100
        taken.append(dict(symbol=r.symbol, signal_date=r.signal_date.date(), entry_time=r.entry_time,
                          exit_time=r.exit_time, days=r.days, qty=qty, entry=round(r.entry, 2), sl=round(r.sl, 2),
                          exit=round(r.exit, 2), ret_pct=round((r.exit / r.entry - 1) * 100, 2),
                          pnl=round(pnl, 2), reason=r.reason))
        open_pos.append((r.exit_time, r.symbol))
    return pd.DataFrame(taken)


def stats(t, mid):
    if t.empty:
        return dict(trades=0, net_pnl=0)
    t = t.sort_values("exit_time")
    p = t["pnl"]
    eq = p.cumsum()
    w, l_ = p[p > 0], p[p <= 0]
    return dict(trades=len(p), win_pct=round(len(w) / len(p) * 100, 1), net_pnl=round(p.sum()),
                return_pct=round(p.sum() / CAPITAL * 100, 1),
                pf=round(w.sum() / abs(l_.sum()), 2) if l_.sum() else np.inf,
                avg_win=round(w.mean()) if len(w) else 0, avg_loss=round(l_.mean()) if len(l_) else 0,
                max_dd=round((eq - eq.cummax()).min()), avg_days=round(t["days"].mean(), 1),
                first_half=round(p[t["entry_time"] < mid].sum()), second_half=round(p[t["entry_time"] >= mid].sum()))


def main():
    syms = base.load_symbols(None)
    out(f"{len(syms)} symbols | Rs {PER_TRADE:,} x {MAX_POS} | capital Rs {CAPITAL:,} | cost {COST}%")
    data = base.download(syms, days=1500)
    nraw = yf.download("^NSEI", period="6y", interval="1d", auto_adjust=True, progress=False)
    nc = nraw["Close"]
    nc = (nc.iloc[:, 0] if isinstance(nc, pd.DataFrame) else nc).dropna()
    nifty_ok = nc > nc.ewm(span=50, adjust=False).mean()

    prepared = {}
    for s, d in data.items():
        if len(d) < WARMUP + 50:
            continue
        d = indicators(d)
        ok = nifty_ok.reindex(d.index).ffill().fillna(False).astype(bool).to_numpy()
        prepared[s] = (d, ok)
    all_dates = sorted(set().union(*[d.index for d, _ in prepared.values()]))
    start_date = all_dates[WARMUP]
    mid = all_dates[WARMUP + (len(all_dates) - WARMUP) // 2]
    out(f"Stocks {len(prepared)} | test {start_date.date()} se {all_dates[-1].date()}")

    nsig = sum(int(d["sig_fixed"].loc[start_date:].sum()) for d, _ in prepared.values())
    nsig_f = sum(int((d["sig_fixed"] & ok).loc[start_date:].sum()) for d, ok in
                 [(d, pd.Series(ok, index=d.index)) for d, ok in prepared.values()])
    nsig_o = sum(int(d["sig_original"].loc[start_date:].sum()) for d, _ in prepared.values())
    months = max(1, (all_dates[-1] - start_date).days / 30.4)
    out(f"\nSignals (test period): FIXED liquidity {nsig} (~{nsig / months:.1f}/mahina) | "
        f"FIXED + Nifty filter {nsig_f} (~{nsig_f / months:.1f}/mahina) | "
        f"ORIGINAL (SMA close > 5000) {nsig_o} (~{nsig_o / months:.1f}/mahina)")

    rows, store = [], {}
    for filt, em, slm, xm in itertools.product(FILTERS, ENTRY_MODES, SL_MODES, EXIT_MODES):
        c = []
        for s, (d, ok) in prepared.items():
            st = int(np.searchsorted(d.index, start_date))
            c += trades_for(s, d, st, ok, "sig_fixed", filt == "ON", em, slm, xm)
        t = portfolio(pd.DataFrame(c, columns=COLS))
        key = (filt, em, slm, xm)
        store[key] = t
        rows.append(dict(nifty_filter=filt, entry=em, SL=slm, exit=xm, **stats(t, mid)))
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "v2_grid.csv", index=False)

    # original (bug) variant - ek default setup pe
    c = []
    for s, (d, ok) in prepared.items():
        st = int(np.searchsorted(d.index, start_date))
        c += trades_for(s, d, st, ok, "sig_original", False, "same-day close", "Candle low", "20 din")
    orig = stats(portfolio(pd.DataFrame(c, columns=COLS)), mid)
    out(f"\nORIGINAL Chartink (SMA close>5000, no filter, close entry, candle-low SL, 20 din): {orig}")

    ok = res[res["trades"] >= 25]
    if ok.empty:
        ok = res[res["trades"] > 0]
    if ok.empty:
        out("Koi trade nahi bana.")
        (OUT / "V2_REPORT.txt").write_text("\n".join(LOG), encoding="utf-8")
        return
    out("\n================ SAARE COMBOS (25+ trades), profit ke hisaab se ================")
    out(ok.sort_values("net_pnl", ascending=False).to_string(index=False))
    out(f"\nProfit wale combos: {(ok['net_pnl'] > 0).sum()} / {len(ok)} | "
        f"dono aadhe hisson me profit: {((ok['first_half'] > 0) & (ok['second_half'] > 0)).sum()}")

    for col in ["nifty_filter", "entry", "SL", "exit"]:
        g = ok.groupby(col).agg(combos=("net_pnl", "size"), profit=("net_pnl", lambda x: int((x > 0).sum())),
                                median_pnl=("net_pnl", "median"), median_pf=("pf", "median")).round(2)
        out(f"\n{col} ke hisaab se:\n{g.to_string()}")

    robust = ok[(ok["first_half"] > 0) & (ok["second_half"] > 0)]
    best = (robust if not robust.empty else ok).sort_values(["pf", "net_pnl"], ascending=False).iloc[0]
    key = (best["nifty_filter"], best["entry"], best["SL"], best["exit"])
    t = store[key].copy()
    t.to_csv(OUT / "v2_best_trades.csv", index=False)
    out(f"\n================ BEST (consistent, PF se): filter {key[0]} | {key[1]} | SL {key[2]} | exit {key[3]} ================")
    out(best.to_string())
    t["year"] = t["exit_time"].dt.year
    t["month"] = t["exit_time"].dt.to_period("M")
    out("\nSaal-wise:\n" + t.groupby("year")["pnl"].agg(trades="count", pnl="sum",
                                                       win=lambda x: (x > 0).sum()).round(0).to_string())
    m = t.groupby("month")["pnl"].agg(trades="count", pnl="sum").round(0)
    m["cum"] = m["pnl"].cumsum()
    out("\nMonth-wise:\n" + m.to_string())
    out("\nExit reason:\n" + t.groupby("reason")["pnl"].agg(trades="count", avg="mean", total="sum").round(0).to_string())
    out("\nTop 8 trades:\n" + t.nlargest(8, "pnl")[["symbol", "entry_time", "exit_time", "days", "ret_pct", "pnl"]].to_string(index=False))
    (OUT / "V2_REPORT.txt").write_text("\n".join(LOG), encoding="utf-8")


if __name__ == "__main__":
    main()
