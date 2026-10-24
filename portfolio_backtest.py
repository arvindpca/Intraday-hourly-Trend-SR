"""
PORTFOLIO BACKTEST - Rupees me result
====================================

Rules (dono strategies pe):
  - Har trade me Rs 50,000 (qty = 50,000 / entry price, poore shares)
  - Ek time pe maximum 5 trades khule (kul capital Rs 2,50,000)
  - Ek stock me ek hi trade, jab tak wo band na ho (no duplicate)
  - 5 se zyada signal ek saath aayen to jisme volume sabse zyada uchhla (volume / average) wo pehle
  - Signal wale din hi entry

PART A - DAILY SWING (MACD + Volume breakout), ~3 saal ka data
  - Entry: signal wale din ke CLOSE pe (3:20-3:25 ke aas paas buy)
  - Filter OFF / ON (Nifty 50 close > Nifty 50 EMA)
  - SL: 2x ATR / 3x ATR / 5% / 7%   TP: 1R / 2R / 3R / TP nahi   Max hold: 10 / 20 din
  - Cost: 0.2% round trip (delivery)

PART B - INTRADAY (VWAP Pullback, ORB, SMC FVG, SMC Sweep), 5m/15m, ~50 din
  - Entry: signal candle ke baad wali candle pe (same day), 3:15 square-off
  - Cost: 0.1% round trip

Chalane ka tarika:
    pip install yfinance pandas numpy
    python portfolio_backtest.py              # dono
    python portfolio_backtest.py --part daily
    python portfolio_backtest.py --part intraday
"""

import argparse
import itertools
import math
import sys

import numpy as np
import pandas as pd
import yfinance as yf

import backtest_macd_breakout as daily_mod
import backtest_intraday_mtf_smc as intra_mod

PER_TRADE = 50_000
MAX_POS = 5
CAPITAL = PER_TRADE * MAX_POS

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)


# --------------------------------------------------------------------------------------
# Portfolio engine (dono parts ke liye common)
# --------------------------------------------------------------------------------------
def run_portfolio(cands, cost_pct):
    """cands: DataFrame [symbol, side(+1/-1 ya Buy/Short), entry_time, exit_time, entry, exit, reason, vr]"""
    if cands.empty:
        return cands
    c = cands.sort_values(["entry_time", "vr"], ascending=[True, False]).reset_index(drop=True)
    open_pos = []  # (exit_time, symbol)
    taken = []
    for r in c.itertuples(index=False):
        open_pos = [p for p in open_pos if p[0] > r.entry_time]
        if len(open_pos) >= MAX_POS or any(p[1] == r.symbol for p in open_pos):
            continue
        qty = math.floor(PER_TRADE / r.entry)
        if qty < 1:
            continue
        side = 1 if r.side in (1, "Buy") else -1
        gross = qty * (r.exit - r.entry) * side
        charges = qty * r.entry * cost_pct / 100
        taken.append(dict(symbol=r.symbol, side="Buy" if side == 1 else "Short", entry_time=r.entry_time,
                          exit_time=r.exit_time, qty=qty, entry=r.entry, exit=r.exit, reason=r.reason,
                          pnl=round(gross - charges, 2)))
        open_pos.append((r.exit_time, r.symbol))
    return pd.DataFrame(taken)


def money_stats(t, mid):
    if t.empty:
        return dict(trades=0)
    t = t.sort_values("exit_time")
    p = t["pnl"]
    eq = p.cumsum()
    w, l_ = p[p > 0], p[p <= 0]
    return dict(
        trades=len(p), win_rate=round(len(w) / len(p) * 100, 1),
        total_pnl=round(p.sum()), return_pct=round(p.sum() / CAPITAL * 100, 1),
        avg_pnl=round(p.mean()), profit_factor=round(w.sum() / abs(l_.sum()), 2) if l_.sum() else np.inf,
        max_dd=round((eq - eq.cummax()).min()),
        first_half=round(p[t["entry_time"] < mid].sum()), second_half=round(p[t["entry_time"] >= mid].sum()),
    )


def report(res, store, title, min_trades, period_col):
    print(f"\n{'=' * 30} {title} {'=' * 30}")
    ok = res[res["trades"] >= min_trades].sort_values("total_pnl", ascending=False)
    if ok.empty:
        print("Kisi combo me kaafi trades nahi.")
        return
    print(f"\nTOP 15 (kam se kam {min_trades} trades, total profit Rs ke hisaab se):")
    print(ok.head(15).to_string(index=False))
    both = ok[(ok["first_half"] > 0) & (ok["second_half"] > 0)]
    print(f"\nDono hisson (pehla aadha + doosra aadha) me profit wale combos: {len(both)} / {len(ok)}")
    if not both.empty:
        print(both.head(10).to_string(index=False))
    b = (both if not both.empty else ok).iloc[0]
    key = tuple(b[k] for k in res.columns[:list(res.columns).index("trades")])
    t = store[key].copy()
    print(f"\nBEST (consistent): {' | '.join(str(x) for x in key)}")
    print(f"Trades {b['trades']} | Win {b['win_rate']}% | Total Rs {b['total_pnl']:,} "
          f"({b['return_pct']}% on Rs {CAPITAL:,}) | Max drawdown Rs {b['max_dd']:,} | PF {b['profit_factor']}")
    t["period"] = pd.to_datetime(t["entry_time"]).dt.to_period(period_col)
    print("\nPeriod ke hisaab se (Rs):")
    print(t.groupby("period")["pnl"].agg(trades="count", pnl="sum").round(0).to_string())
    print("\nExit reason:")
    print(t.groupby("reason")["pnl"].agg(trades="count", avg="mean", total="sum").round(0).to_string())
    return t


# --------------------------------------------------------------------------------------
# PART A - DAILY
# --------------------------------------------------------------------------------------
D_SL = [("2x ATR", "atr", 2.0), ("3x ATR", "atr", 3.0), ("SL 5%", "pct", 0.05), ("SL 7%", "pct", 0.07)]
D_TP = [("1R", 1.0), ("2R", 2.0), ("3R", 3.0), ("No TP", None)]
D_HOLD = [10, 20]


def daily_candidates(sym, d, start, mkt_ok, use_filter, sl_opt, tp_r, max_hold):
    O, H, L, C = (d[k].to_numpy() for k in ["Open", "High", "Low", "Close"])
    atr, sig = d["atr"].to_numpy(), d["signal"].to_numpy()
    vr = (d["Volume"] / d["vol_sma20"]).to_numpy()
    dates, n = d.index, len(d)
    out, last = [], -1
    for i in range(start, n - 1):
        if not sig[i] or i <= last or (use_filter and not mkt_ok[i]):
            continue
        entry = C[i]  # same day close pe entry
        sl = entry - sl_opt[2] * atr[i] if sl_opt[1] == "atr" else entry * (1 - sl_opt[2])
        if not np.isfinite(sl) or sl >= entry:
            continue
        tp = entry + tp_r * (entry - sl) if tp_r else np.inf
        j = i + 1
        while True:
            if O[j] <= sl:
                px, why = O[j], "SL (gap)"
            elif O[j] >= tp:
                px, why = O[j], "TP (gap)"
            elif L[j] <= sl:
                px, why = sl, "SL"
            elif H[j] >= tp:
                px, why = tp, "TP"
            elif j - i >= max_hold:
                px, why = C[j], "Time exit"
            elif j == n - 1:
                px, why = C[j], "Abhi khula (last close)"
            else:
                j += 1
                continue
            break
        out.append((sym.replace(".NS", ""), 1, dates[i], dates[j], entry, px, why, vr[i]))
        last = j
    return out


def part_daily(symbols, min_trades):
    print("\n##### PART A: DAILY SWING (3 saal) #####")
    data = daily_mod.download(symbols, days=1150)
    print(f"   {len(data)} stocks")
    nraw = yf.download("^NSEI", period="5y", interval="1d", auto_adjust=True, progress=False)
    nc = nraw["Close"]
    nc = (nc.iloc[:, 0] if isinstance(nc, pd.DataFrame) else nc).dropna()
    nifty_ok = nc > nc.ewm(span=50, adjust=False).mean()

    prepared = {}
    for s, d in data.items():
        d = daily_mod.add_indicators(d)
        ok = nifty_ok.reindex(d.index).ffill().fillna(False).astype(bool).to_numpy()
        prepared[s] = (d, 60, ok)  # pehle 60 din warmup
    all_dates = sorted(set().union(*[d.index for d, _, _ in prepared.values()]))
    start_date, mid = all_dates[60], all_dates[60 + (len(all_dates) - 60) // 2]
    print(f"   test period: {start_date.date()} se {all_dates[-1].date()}")

    rows, store = [], {}
    for filt, sl_opt, (tpn, tpr), mh in itertools.product(["OFF", "ON"], D_SL, D_TP, D_HOLD):
        c = []
        for s, (d, st, ok) in prepared.items():
            c += daily_candidates(s, d, st, ok, filt == "ON", sl_opt, tpr, mh)
        cands = pd.DataFrame(c, columns=["symbol", "side", "entry_time", "exit_time", "entry", "exit", "reason", "vr"])
        t = run_portfolio(cands, 0.2)
        key = (filt, sl_opt[0], tpn, mh)
        store[key] = t
        rows.append(dict(nifty_filter=filt, SL=sl_opt[0], TP=tpn, max_hold=mh, **money_stats(t, mid)))
    res = pd.DataFrame(rows)
    res.to_csv("portfolio_daily_results.csv", index=False)
    best = report(res, store, "DAILY SWING - Rs 50k/trade, max 5 khule", min_trades, "Q")
    if best is not None:
        best.to_csv("portfolio_daily_best_trades.csv", index=False)


# --------------------------------------------------------------------------------------
# PART B - INTRADAY
# --------------------------------------------------------------------------------------
def part_intraday(symbols, min_trades):
    print("\n##### PART B: INTRADAY 5m/15m (~50 din) #####")
    data = intra_mod.download_5m(symbols)
    print(f"   {len(data)} stocks")
    nifty_ctx, nifty_trend_ctx = intra_mod.nifty_state()
    all_days = sorted(set().union(*[set(d.index.normalize()) for d in data.values()]))
    w = intra_mod.CFG["warmup_days"]
    test_start, mid = all_days[w], all_days[w + (len(all_days) - w) // 2]
    print(f"   test period: {test_start.date()} se {all_days[-1].date()}")

    prepared = {tf: {} for tf in intra_mod.TIMEFRAMES}
    for s, d5 in data.items():
        tctx = intra_mod.trend_1h(d5)
        for tf in intra_mod.TIMEFRAMES:
            prepared[tf][s] = intra_mod.prepare(d5, tf, tctx, nifty_ctx, nifty_trend_ctx)

    rows, store = [], {}
    combos = itertools.product(intra_mod.STRATEGIES, intra_mod.TIMEFRAMES, intra_mod.FILTERS,
                               intra_mod.SL_MODES, intra_mod.TP_MODES)
    for strat, tf, filt, slm, (tpn, tpr) in combos:
        c = []
        for s, (A, sig) in prepared[tf].items():
            if sig.empty:
                continue
            ss = sig[sig["strategy"] == strat]
            if filt == "ON":
                ss = ss[ss["aligned"]]
            elif filt == "STRICT":
                ss = ss[ss["strict"]]
            if not ss.empty:
                c += intra_mod.simulate(s, A, ss.sort_values("entry_idx"), test_start, slm, tpr, 0)
        cands = pd.DataFrame(c, columns=intra_mod.COLS)
        for side in ["Buy", "Short", "Buy+Short"]:
            sub = cands if side == "Buy+Short" else cands[cands["side"] == side]
            t = run_portfolio(sub, 0.1)
            key = (strat, f"{tf}m", filt, side, slm, tpn)
            store[key] = t
            rows.append(dict(strategy=strat, tf=f"{tf}m", filter=filt, side=side, SL=slm, TP=tpn,
                             **money_stats(t, mid)))
    res = pd.DataFrame(rows)
    res.to_csv("portfolio_intraday_results.csv", index=False)
    best = report(res, store, "INTRADAY - Rs 50k/trade, max 5 khule", min_trades, "W")
    if best is not None:
        best.to_csv("portfolio_intraday_best_trades.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["both", "daily", "intraday"], default="both")
    ap.add_argument("--symbols-file")
    ap.add_argument("--min-trades-daily", type=int, default=30)
    ap.add_argument("--min-trades-intraday", type=int, default=40)
    args = ap.parse_args()
    symbols = daily_mod.load_symbols(args.symbols_file)
    print(f"{len(symbols)} symbols | Rs {PER_TRADE:,} per trade | max {MAX_POS} khule | capital Rs {CAPITAL:,}")
    if args.part in ("both", "daily"):
        part_daily(symbols, args.min_trades_daily)
    if args.part in ("both", "intraday"):
        part_intraday(symbols, args.min_trades_intraday)


if __name__ == "__main__":
    main()
