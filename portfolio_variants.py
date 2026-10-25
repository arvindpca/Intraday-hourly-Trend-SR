"""
Daily Swing - position size variants backtest (3 saal, Nifty 500)
Rules fixed: Nifty > 50 EMA, SL 5%, koi TP nahi, 20 din hold, 0.2% cost, no duplicate,
zyada signal -> sabse zyada volume uchhal pehle.

Variants: (per-trade, max khule) x (entry: signal din close / agle din open)
"""
import math
import sys

import numpy as np
import pandas as pd
import yfinance as yf

import backtest_macd_breakout as core

VARIANTS = [(20_000, 5), (50_000, 5), (10_000, 10)]
ENTRY_MODES = ["close", "next_open"]
SL_PCT, MAX_HOLD, COST = 0.05, 20, 0.2

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_rows", 200)
LOG = []


def out(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def candidates(sym, d, start, mkt_ok, mode):
    O, H, L, C = (d[k].to_numpy() for k in ["Open", "High", "Low", "Close"])
    sig = d["signal"].to_numpy()
    vr = (d["Volume"] / d["vol_sma20"]).to_numpy()
    dates, n = d.index, len(d)
    rows, last = [], -1
    for i in range(start, n - 1):
        if not sig[i] or not mkt_ok[i]:
            continue
        if mode == "close":
            e, entry, first = i, C[i], i + 1
        else:
            e, entry, first = i + 1, O[i + 1], i + 1
        if e <= last:
            continue
        sl = entry * (1 - SL_PCT)
        j = first
        while True:
            if j > e and O[j] <= sl:
                px, why = O[j], "SL (gap)"
            elif L[j] <= sl:
                px, why = sl, "SL"
            elif j - e >= MAX_HOLD:
                px, why = C[j], "Time exit"
            elif j == n - 1:
                px, why = C[j], "Abhi khula"
            else:
                j += 1
                continue
            break
        rows.append((sym.replace(".NS", ""), dates[i], dates[e], dates[j], entry, px, why, vr[i]))
        last = j
    return pd.DataFrame(rows, columns=["symbol", "signal_time", "entry_time", "exit_time", "entry", "exit", "reason", "vr"])


def portfolio(c, per_trade, max_pos):
    c = c.sort_values(["entry_time", "vr"], ascending=[True, False])
    open_pos, taken, skipped_full = [], [], 0
    for r in c.itertuples(index=False):
        open_pos = [p for p in open_pos if p[0] > r.entry_time]
        if any(p[1] == r.symbol for p in open_pos):
            continue
        if len(open_pos) >= max_pos:
            skipped_full += 1
            continue
        qty = math.floor(per_trade / r.entry)
        if qty < 1:
            continue
        pnl = qty * (r.exit - r.entry) - qty * r.entry * COST / 100
        taken.append(dict(symbol=r.symbol, entry_time=r.entry_time, exit_time=r.exit_time, qty=qty,
                          invested=round(qty * r.entry), entry=round(r.entry, 2), exit=round(r.exit, 2),
                          ret_pct=round((r.exit / r.entry - 1) * 100, 2), pnl=round(pnl, 2), reason=r.reason))
        open_pos.append((r.exit_time, r.symbol))
    return pd.DataFrame(taken), skipped_full


def summary(t, capital, skipped, label):
    t = t.sort_values("exit_time")
    p = t["pnl"]
    eq = p.cumsum()
    w, l_ = p[p > 0], p[p <= 0]
    days = pd.bdate_range(t["entry_time"].min(), t["exit_time"].max())
    occ = np.array([((t["entry_time"] <= d) & (t["exit_time"] > d)).sum() for d in days])
    years = (t["exit_time"].max() - t["entry_time"].min()).days / 365.25
    return dict(variant=label, capital=capital, trades=len(p), win_rate=round(len(w) / len(p) * 100, 1),
                net_pnl=round(p.sum()), return_pct=round(p.sum() / capital * 100, 1),
                per_year_pct=round(p.sum() / capital * 100 / years, 1),
                profit_factor=round(w.sum() / abs(l_.sum()), 2), avg_win=round(w.mean()), avg_loss=round(l_.mean()),
                max_dd=round((eq - eq.cummax()).min()), max_dd_pct=round((eq - eq.cummax()).min() / capital * 100, 1),
                avg_open=round(occ.mean(), 1), max_open=int(occ.max()), signals_missed_full=skipped)


def main():
    syms = core.load_symbols(None)
    out(f"{len(syms)} symbols")
    data = core.download(syms, days=1150)
    nraw = yf.download("^NSEI", period="5y", interval="1d", auto_adjust=True, progress=False)
    nc = nraw["Close"]
    nc = (nc.iloc[:, 0] if isinstance(nc, pd.DataFrame) else nc).dropna()
    nifty_ok = nc > nc.ewm(span=50, adjust=False).mean()

    prepared = {}
    for s, d in data.items():
        d = core.add_indicators(d)
        ok = nifty_ok.reindex(d.index).ffill().fillna(False).astype(bool).to_numpy()
        prepared[s] = (d, ok)
    all_dates = sorted(set().union(*[d.index for d, _ in prepared.values()]))
    start_date = all_dates[60]
    out(f"Stocks {len(prepared)} | test {start_date.date()} se {all_dates[-1].date()}")

    rows, store = [], {}
    for mode in ENTRY_MODES:
        c = pd.concat([candidates(s, d, int(np.searchsorted(d.index, start_date)), ok, mode)
                       for s, (d, ok) in prepared.items()])
        for per, mx in VARIANTS:
            t, skipped = portfolio(c, per, mx)
            label = f"Rs {per // 1000}k x {mx} | entry {'same-day close' if mode == 'close' else 'next-day open'}"
            store[label] = t
            rows.append(summary(t, per * mx, skipped, label))
            t.to_csv(f"variant_{per // 1000}k_x{mx}_{mode}.csv", index=False)

    res = pd.DataFrame(rows)
    out("\n================ COMPARISON ================")
    out(res.to_string(index=False))

    for label, t in store.items():
        if not label.startswith("Rs 20k"):
            continue
        out(f"\n================ {label} ================")
        t = t.copy()
        t["year"] = t["exit_time"].dt.year
        t["month"] = t["exit_time"].dt.to_period("M")
        out("Saal-wise:\n" + t.groupby("year")["pnl"].agg(trades="count", pnl="sum",
                                                          win=lambda x: (x > 0).sum()).round(0).to_string())
        m = t.groupby("month")["pnl"].agg(trades="count", pnl="sum").round(0)
        m["cum"] = m["pnl"].cumsum()
        out("Month-wise:\n" + m.to_string())
        out("Exit reason:\n" + t.groupby("reason")["pnl"].agg(trades="count", avg="mean", total="sum").round(0).to_string())

    with open("variants_output.txt", "w") as f:
        f.write("\n".join(LOG))


if __name__ == "__main__":
    main()
