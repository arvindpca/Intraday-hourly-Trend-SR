"""
V2 / V2.1 - position size test: Rs 5,000 per signal, max 10 / 15 / 20 khule (+ Rs 20k x 5 reference)
Setup: Nifty filter ON, same-day close entry, SL 2x ATR, exit 10 din (V2.1 best).
Cost do tarah: (a) 0.2% round trip  (b) 0.2% + Rs 40 fixed per trade (DP charge + min brokerage - chhote trade me asli)
"""
import math
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
from backtest_v21 import rsi  # noqa: E402

OUT = HERE / "results"
SIZES = [(20_000, 5), (5_000, 10), (5_000, 15), (5_000, 20)]
COSTS = [("0.2%", 0.2, 0), ("0.2% + Rs40", 0.2, 40)]
LOG = []
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)


def out(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def portfolio(c, per, mx, pct, fixed):
    c = c.sort_values(["entry_time", "vr"], ascending=[True, False])
    open_pos, taken, missed, skipped_price = [], [], 0, 0
    for r in c.itertuples(index=False):
        open_pos = [p for p in open_pos if p[0] > r.entry_time]
        if any(p[1] == r.symbol for p in open_pos):
            continue
        if len(open_pos) >= mx:
            missed += 1
            continue
        qty = math.floor(per / r.entry)
        if qty < 1:
            skipped_price += 1
            continue
        inv = qty * r.entry
        pnl = qty * (r.exit - r.entry) - inv * pct / 100 - fixed
        taken.append(dict(symbol=r.symbol, entry_time=r.entry_time, exit_time=r.exit_time, invested=inv,
                          pnl=pnl, ret_pct=(r.exit / r.entry - 1) * 100, days=r.days))
        open_pos.append((r.exit_time, r.symbol))
    return pd.DataFrame(taken), missed, skipped_price


def stats(t, cap, mid, missed, skp):
    t = t.sort_values("exit_time")
    p = t["pnl"]
    eq = p.cumsum()
    w, l_ = p[p > 0], p[p <= 0]
    days = pd.bdate_range(t["entry_time"].min(), t["exit_time"].max())
    used = np.array([t[(t.entry_time <= d) & (t.exit_time > d)]["invested"].sum() for d in days])
    yrs = (t["exit_time"].max() - t["entry_time"].min()).days / 365.25
    yearly = t.groupby(t["exit_time"].dt.year)["pnl"].sum().round(0).to_dict()
    return dict(capital=cap, trades=len(p), win_pct=round(len(w) / len(p) * 100, 1), net_pnl=round(p.sum()),
                return_pct=round(p.sum() / cap * 100, 1), per_year_pct=round(p.sum() / cap * 100 / yrs, 1),
                pf=round(w.sum() / abs(l_.sum()), 2), max_dd=round((eq - eq.cummax()).min()),
                max_dd_pct=round((eq - eq.cummax()).min() / cap * 100, 1),
                avg_invested=round(t["invested"].mean()), avg_capital_used_pct=round(used.mean() / cap * 100),
                signals_missed_full=missed, skipped_price_gt_per_trade=skp,
                first_half=round(p[t.entry_time < mid].sum()), second_half=round(p[t.entry_time >= mid].sum()),
                yearly=yearly)


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
        prepared[s] = (d, nifty_ok.reindex(d.index).ffill().fillna(False).astype(bool).to_numpy())
    all_dates = sorted(set().union(*[d.index for d, _ in prepared.values()]))
    start = all_dates[v2.WARMUP]
    mid = all_dates[v2.WARMUP + (len(all_dates) - v2.WARMUP) // 2]
    out(f"Stocks {len(prepared)} | test {start.date()} se {all_dates[-1].date()} | Nifty ON, close entry, SL 2x ATR, 10 din exit")

    rows = []
    for ver, col in [("V2.1", "sig_v21"), ("V2", "sig_fixed")]:
        cand = []
        for s, (d, ok) in prepared.items():
            st = int(np.searchsorted(d.index, start))
            cand += v2.trades_for(s, d, st, ok, col, True, "same-day close", "2x ATR", "10 din")
        cand = pd.DataFrame(cand, columns=v2.COLS)
        for (per, mx), (cn, pct, fx) in [(sz, c) for sz in SIZES for c in COSTS]:
            t, missed, skp = portfolio(cand, per, mx, pct, fx)
            rows.append(dict(version=ver, setup=f"Rs {per:,} x {mx}", cost=cn,
                             **stats(t, per * mx, mid, missed, skp)))
    res = pd.DataFrame(rows)
    res.drop(columns=["yearly"]).to_csv(OUT / "size_test.csv", index=False)
    show = ["version", "setup", "cost", "capital", "trades", "win_pct", "net_pnl", "return_pct", "per_year_pct", "pf",
            "max_dd", "max_dd_pct", "avg_capital_used_pct", "signals_missed_full", "skipped_price_gt_per_trade",
            "first_half", "second_half"]
    out("\n" + res[show].to_string(index=False))
    out("\nSaal-wise (Rs):")
    for r in res.itertuples():
        out(f"  {r.version} | {r.setup} | {r.cost}: {r.yearly}")
    (OUT / "SIZE_TEST_REPORT.txt").write_text("\n".join(LOG), encoding="utf-8")


if __name__ == "__main__":
    main()
