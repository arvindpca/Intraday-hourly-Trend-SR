"""
V2.1 vs V2.2 - portfolio backtest (V2.2 = V2.1 + EMA50 10 din me 1%+ upar)
(adapted from backtest_v21.py) (Rs 20,000 x 5, Nifty 500, ~3 saal)
V2.1 = V2 + loss analysis ke 3 filter (round numbers):
  A: Close 52-week high se 6% ke andar
  B: RSI(14) > 65
  C: Pichhle 20 din me Close 5%+ upar
Nifty filter ON (dono me). Grid: entry (close / next open) x SL (candle low / 2x ATR / 5%)
x exit (10 din / 20 din / 2R / EMA20 exit).
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import backtest_macd_breakout as base  # noqa: E402
import backtest_v2 as v2  # noqa: E402

OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
LOG = []
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_rows", 300)


def out(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def rsi(c, n=14):
    d = c.diff()
    ag = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    al = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + ag / al.replace(0, np.nan))


def main():
    syms = base.load_symbols(None)
    data = base.download(syms, days=1500)
    nraw = yf.download("^NSEI", period="6y", interval="1d", auto_adjust=True, progress=False)
    nc = nraw["Close"]
    nc = (nc.iloc[:, 0] if isinstance(nc, pd.DataFrame) else nc).dropna()
    nifty_ok = nc > nc.ewm(span=50, adjust=False).mean()

    prepared = {}
    for s, d in data.items():
        if len(d) < v2.WARMUP + 50:
            continue
        d = v2.indicators(d)
        c, h = d["Close"], d["High"]
        extra = ((c / h.rolling(252, min_periods=120).max() - 1) * 100 > -6) & (rsi(c) > 65) & \
                ((c / c.shift(20) - 1) * 100 > 5)
        d["sig_v21"] = (d["sig_fixed"] & extra.fillna(False)).astype(bool)
        d["sig_v22"] = (d["sig_v21"] & (d["ema50"] / d["ema50"].shift(10) > 1.01).fillna(False)).astype(bool)
        ok = nifty_ok.reindex(d.index).ffill().fillna(False).astype(bool).to_numpy()
        prepared[s] = (d, ok)
    all_dates = sorted(set().union(*[d.index for d, _ in prepared.values()]))
    start = all_dates[v2.WARMUP]
    mid = all_dates[v2.WARMUP + (len(all_dates) - v2.WARMUP) // 2]
    months = (all_dates[-1] - start).days / 30.4
    out(f"Stocks {len(prepared)} | test {start.date()} se {all_dates[-1].date()} ({months:.0f} mahine) | "
        f"Rs {v2.PER_TRADE:,} x {v2.MAX_POS}")
    for col in ["sig_v21", "sig_v22"]:
        n = sum(int((d[col] & pd.Series(ok, index=d.index)).loc[start:].sum()) for d, ok in prepared.values())
        out(f"  {'V2.1' if col == 'sig_v21' else 'V2.2'} signals (Nifty ON): {n} (~{n / months:.1f}/mahina)")

    rows, store = [], {}
    for ver, col in [("V2.1", "sig_v21"), ("V2.2", "sig_v22")]:
        for em, slm, xm in itertools.product(v2.ENTRY_MODES, v2.SL_MODES, v2.EXIT_MODES):
            c = []
            for s, (d, ok) in prepared.items():
                st = int(np.searchsorted(d.index, start))
                c += v2.trades_for(s, d, st, ok, col, True, em, slm, xm)
            t = v2.portfolio(pd.DataFrame(c, columns=v2.COLS))
            store[(ver, em, slm, xm)] = t
            rows.append(dict(version=ver, entry=em, SL=slm, exit=xm, **v2.stats(t, mid)))
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "v22_grid.csv", index=False)

    out("\n================ V2.1 vs V2.2 - har setting pe ================")
    piv = res.pivot_table(index=["entry", "SL", "exit"], columns="version",
                          values=["net_pnl", "pf", "max_dd", "trades"]).round(2)
    piv.columns = [f"{a}_{b}" for a, b in piv.columns]
    piv["V2.2_better"] = piv["net_pnl_V2.2"] > piv["net_pnl_V2.1"]
    out(piv[["trades_V2.1", "trades_V2.2", "net_pnl_V2.1", "net_pnl_V2.2", "pf_V2.1", "pf_V2.2", "max_dd_V2.1",
             "max_dd_V2.2", "V2.2_better"]].sort_values("net_pnl_V2.2", ascending=False).to_string())
    for ver in ["V2.1", "V2.2"]:
        r = res[res.version == ver]
        out(f"\n{ver}: profit combos {(r.net_pnl > 0).sum()}/{len(r)} | dono hisson me profit "
            f"{((r.first_half > 0) & (r.second_half > 0)).sum()} | median profit Rs {r.net_pnl.median():,.0f} | "
            f"median PF {r.pf.median():.2f} | median max DD Rs {r.max_dd.median():,.0f}")
    out(f"V2.2 better in {int(piv['V2.2_better'].sum())}/{len(piv)} settings")

    # V2.1 ke best setting (close, 2x ATR, 10 din) pe dono ka seedha mukabla + DP Rs 24
    from size_test import portfolio as pf_fixed
    for ver, col in [("V2.1", "sig_v21"), ("V2.2", "sig_v22")]:
        c = []
        for s_, (d, ok) in prepared.items():
            c += v2.trades_for(s_, d, int(np.searchsorted(d.index, start)), ok, col, True, "same-day close", "2x ATR", "10 din")
        tt, _, _ = pf_fixed(pd.DataFrame(c, columns=v2.COLS), 20_000, 5, 0.2, 24)
        p = tt["pnl"]
        out(f"{ver} | close, 2x ATR, 10 din | 0.2% + DP Rs24: trades {len(p)} | profit Rs {p.sum():,.0f} | "
            f"PF {p[p > 0].sum() / -p[p <= 0].sum():.2f} | yearly {tt.groupby(tt.exit_time.dt.year)['pnl'].sum().round(0).to_dict()}")
    r21 = res[(res.version == "V2.2") & (res.trades >= 25)]
    if r21.empty:
        r21 = res[(res.version == "V2.2") & (res.trades > 0)]
    robust = r21[(r21.first_half > 0) & (r21.second_half > 0)]
    best = (robust if not robust.empty else r21).sort_values(["net_pnl"], ascending=False).iloc[0]
    key = ("V2.2", best["entry"], best["SL"], best["exit"])
    t = store[key].copy()
    t.to_csv(OUT / "v22_best_trades.csv", index=False)
    out(f"\n================ V2.2 BEST: {best['entry']} | SL {best['SL']} | exit {best['exit']} ================")
    out(best.to_string())
    same_v2 = res[(res.version == "V2.1") & (res.entry == best["entry"]) & (res.SL == best["SL"]) &
                  (res.exit == best["exit"])].iloc[0]
    out(f"(Same setting pe V2.1: profit Rs {same_v2['net_pnl']:,} | PF {same_v2['pf']} | DD Rs {same_v2['max_dd']:,})")
    t["year"] = t["exit_time"].dt.year
    t["month"] = t["exit_time"].dt.to_period("M")
    out("\nSaal-wise:\n" + t.groupby("year")["pnl"].agg(trades="count", pnl="sum",
                                                       win=lambda x: (x > 0).sum()).round(0).to_string())
    m = t.groupby("month")["pnl"].agg(trades="count", pnl="sum").round(0)
    m["cum"] = m["pnl"].cumsum()
    out("\nMonth-wise:\n" + m.to_string())
    out(f"Profit mahine {(m.pnl > 0).sum()} | loss mahine {(m.pnl < 0).sum()}")
    out("\nExit reason:\n" + t.groupby("reason")["pnl"].agg(trades="count", avg="mean", total="sum").round(0).to_string())
    out("\nTop 8:\n" + t.nlargest(8, "pnl")[["symbol", "entry_time", "exit_time", "days", "ret_pct", "pnl"]].to_string(index=False))
    out("\nTop 8 loss:\n" + t.nsmallest(8, "pnl")[["symbol", "entry_time", "exit_time", "days", "ret_pct", "pnl", "reason"]].to_string(index=False))
    (OUT / "V22_REPORT.txt").write_text("\n".join(LOG), encoding="utf-8")


if __name__ == "__main__":
    main()
