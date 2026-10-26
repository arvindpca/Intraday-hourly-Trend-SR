"""
MACD + Volume Breakout - SL/TP Optimizer Backtest (Nifty 500, last 180 trading days)
=================================================================================

Chalane ka tarika:
    pip install yfinance pandas numpy
    python backtest_macd_breakout.py

Script khud karegi:
  1. Nifty 500 ki list NSE se download (fail ho to --symbols-file se apni CSV do)
  2. Yahoo Finance se daily data download (~400 calendar din, indicators ke warmup ke liye)
  3. Pichhle 180 trading din me Chartink wali entry conditions check
  4. Har SL x TP combination ka backtest
  5. Best combo + saare results CSV me save

Entry  : signal candle ke NEXT DAY open pe
Costs  : default 0.2% round trip (brokerage + STT + slippage) - --cost se badlo
Rule   : ek stock me ek time pe ek hi trade
Rule   : same candle me SL aur TP dono touch ho to SL maana (conservative)
Rule   : gap down SL ke neeche khule to open pe exit (real jaisa nuksaan)
"""

import argparse
import itertools
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    sys.exit("yfinance install nahi hai. Chalao:  pip install yfinance pandas numpy")


# --------------------------------------------------------------------------------------
# CONFIG - yahan se strategy badal sakte ho
# --------------------------------------------------------------------------------------
CFG = dict(
    price_min=200,            # Daily Close >= 200
    price_max=3000,           # Daily Close <= 3000
    min_turnover=5e7,         # Close * Volume > 5 crore (liquidity)
    rsi_min=50, rsi_max=70,   # RSI(14) range
    vol_mult=2.0,             # Volume > SMA(Volume,20) * 2
    vol_max_lookback=10,      # Volume > pichhle 10 din ka max volume (0 = filter off)
    test_days=180,            # kitne trading din pe test
)

# SL options: (naam, type, value)
SL_OPTIONS = [
    ("Signal candle low", "candle_low", None),
    ("SL 3%", "pct", 0.03),
    ("SL 5%", "pct", 0.05),
    ("SL 7%", "pct", 0.07),
    ("SL 1.5x ATR", "atr", 1.5),
    ("SL 2x ATR", "atr", 2.0),
    ("SL 3x ATR", "atr", 3.0),
]

# TP options: (naam, type, value)  R = risk ka multiple (entry - SL)
TP_OPTIONS = [
    ("TP 1R", "r", 1.0),
    ("TP 1.5R", "r", 1.5),
    ("TP 2R", "r", 2.0),
    ("TP 3R", "r", 3.0),
    ("TP 5%", "pct", 0.05),
    ("TP 10%", "pct", 0.10),
    ("No TP, exit on Close < EMA21", "ema_exit", None),
]

MAX_HOLD_OPTIONS = [10, 20]  # itne din baad bhi trade khula ho to close pe exit


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
NIFTY500_URLS = [
    "https://archives.nseindia.com/content/indices/ind_nifty500list.csv",
    "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv",
]


def load_symbols(path=None):
    if path:
        df = pd.read_csv(path)
    else:
        df = None
        for url in NIFTY500_URLS:
            try:
                df = pd.read_csv(url, storage_options={"User-Agent": "Mozilla/5.0"})
                break
            except Exception as e:
                print(f"  list download fail: {url} ({e})")
        if df is None:
            sys.exit(
                "Nifty 500 list download nahi hui. NSE site se 'ind_nifty500list.csv' "
                "download karke chalao:  python backtest_macd_breakout.py --symbols-file ind_nifty500list.csv"
            )
    col = "Symbol" if "Symbol" in df.columns else df.columns[0]
    return sorted({str(s).strip().upper() + ".NS" for s in df[col].dropna()})


def download(symbols, days=420):
    start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    data = {}
    for i in range(0, len(symbols), 50):
        chunk = symbols[i:i + 50]
        print(f"  downloading {i + 1}-{i + len(chunk)} / {len(symbols)} ...")
        raw = yf.download(chunk, start=start, interval="1d", auto_adjust=True,
                          group_by="ticker", progress=False, threads=True)
        for s in chunk:
            try:
                d = raw[s] if len(chunk) > 1 else raw
                if isinstance(d.columns, pd.MultiIndex):
                    d = d.droplevel(0, axis=1) if s in d.columns.get_level_values(0) else d.droplevel(1, axis=1)
                d = d[["Open", "High", "Low", "Close", "Volume"]].dropna()
                if len(d) > 80:
                    data[s] = d
            except Exception:
                pass
    return data


# --------------------------------------------------------------------------------------
# Indicators + entry signal (Chartink conditions)
# --------------------------------------------------------------------------------------
def add_indicators(d):
    d = d.copy()
    c, o, h, l, v = d["Close"], d["Open"], d["High"], d["Low"], d["Volume"]

    delta = c.diff()
    ag = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    al = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    d["rsi"] = 100 - 100 / (1 + ag / al.replace(0, np.nan))

    d["macd"] = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    d["macd_sig"] = d["macd"].ewm(span=9, adjust=False).mean()

    rng = (h - l).replace(0, np.nan)
    d["adl"] = ((((c - l) - (h - c)) / rng).fillna(0) * v).cumsum()

    d["vwap"] = (h + l + c) / 3  # daily candle ka VWAP approx (typical price)
    d["ema8"] = c.ewm(span=8, adjust=False).mean()
    d["ema21"] = c.ewm(span=21, adjust=False).mean()
    d["vol_sma20"] = v.rolling(20).mean()

    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    d["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()

    sig = (
        (c >= CFG["price_min"]) & (c <= CFG["price_max"])
        & (c * v > CFG["min_turnover"])
        & (d["rsi"] > CFG["rsi_min"]) & (d["rsi"] < CFG["rsi_max"])
        & (d["macd"] > d["macd_sig"])
        & (d["macd"].shift(1) <= d["macd_sig"].shift(1))
        & (d["adl"] > d["adl"].shift(1))
        & (c > o)
        & (v > d["vol_sma20"] * CFG["vol_mult"])
        & (c > d["vwap"])
        & (d["ema8"] > d["ema21"])
    )
    if CFG["vol_max_lookback"] > 0:
        sig &= v > v.shift(1).rolling(CFG["vol_max_lookback"]).max()
    d["signal"] = sig.fillna(False)
    return d


# --------------------------------------------------------------------------------------
# Trade simulation
# --------------------------------------------------------------------------------------
def simulate(sym, d, test_start_idx, sl_opt, tp_opt, max_hold, cost):
    O, H, L, C = (d[k].to_numpy() for k in ["Open", "High", "Low", "Close"])
    ema21, atr, sigs = d["ema21"].to_numpy(), d["atr"].to_numpy(), d["signal"].to_numpy()
    dates = d.index
    n = len(d)
    trades = []
    i = test_start_idx
    while i < n - 1:
        if not sigs[i]:
            i += 1
            continue
        e_idx = i + 1
        entry = O[e_idx]

        _, sl_t, sl_v = sl_opt
        if sl_t == "candle_low":
            sl = L[i]
        elif sl_t == "pct":
            sl = entry * (1 - sl_v)
        else:
            sl = entry - sl_v * atr[i]
        if not np.isfinite(sl) or sl >= entry:  # gap down ne SL pehle hi tod diya - skip
            i += 1
            continue
        risk = entry - sl

        _, tp_t, tp_v = tp_opt
        tp = entry + tp_v * risk if tp_t == "r" else entry * (1 + tp_v) if tp_t == "pct" else np.inf

        exit_px, reason, j = None, None, e_idx
        while j < n:
            if j > e_idx and O[j] <= sl:
                exit_px, reason = O[j], "SL (gap)"
            elif j > e_idx and O[j] >= tp:
                exit_px, reason = O[j], "TP (gap)"
            elif L[j] <= sl:
                exit_px, reason = sl, "SL"
            elif H[j] >= tp:
                exit_px, reason = tp, "TP"
            elif tp_t == "ema_exit" and C[j] < ema21[j]:
                exit_px, reason = C[j], "EMA21 exit"
            elif j - e_idx + 1 >= max_hold:
                exit_px, reason = C[j], "Time exit"
            if exit_px is not None:
                break
            j += 1
        if exit_px is None:
            j = n - 1
            exit_px, reason = C[j], "Open (last close)"

        trades.append(dict(
            symbol=sym.replace(".NS", ""), signal_date=dates[i].date(), entry_date=dates[e_idx].date(),
            exit_date=dates[j].date(), entry=round(entry, 2), sl=round(sl, 2),
            tp=round(tp, 2) if np.isfinite(tp) else None, exit=round(exit_px, 2), reason=reason,
            days=j - e_idx + 1, ret_pct=round((exit_px / entry - 1 - cost) * 100, 3),
            r_mult=round((exit_px - entry) / risk, 2),
        ))
        i = j + 1  # ek stock me ek hi trade at a time
    return trades


def stats(trades):
    if not trades:
        return dict(trades=0)
    df = pd.DataFrame(trades).sort_values("exit_date")
    r = df["ret_pct"]
    wins, losses = r[r > 0], r[r <= 0]
    pf = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else np.inf
    eq = r.cumsum()
    dd = (eq - eq.cummax()).min()
    return dict(
        trades=len(df), win_rate=round(len(wins) / len(df) * 100, 1),
        avg_ret=round(r.mean(), 2), avg_win=round(wins.mean(), 2) if len(wins) else 0,
        avg_loss=round(losses.mean(), 2) if len(losses) else 0,
        profit_factor=round(pf, 2), total_ret_sum=round(r.sum(), 1),
        max_dd_sum=round(dd, 1), avg_days=round(df["days"].mean(), 1),
    )


def market_regimes():
    """Nifty 50 trend filters. Signal din ka Nifty close check hota hai."""
    raw = yf.download("^NSEI", period="2y", interval="1d", auto_adjust=True, progress=False)
    c = raw["Close"]
    if isinstance(c, pd.DataFrame):
        c = c.iloc[:, 0]
    c = c.dropna()
    return {
        "No filter": None,
        "Nifty > 50 EMA": c > c.ewm(span=50, adjust=False).mean(),
        "Nifty > 200 SMA": c > c.rolling(200).mean(),
    }


# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols-file", help="Nifty 500 CSV (Symbol column) agar auto download fail ho")
    ap.add_argument("--days", type=int, default=CFG["test_days"], help="test trading days (default 180)")
    ap.add_argument("--cost", type=float, default=0.2, help="round-trip cost %% (default 0.2)")
    ap.add_argument("--min-trades", type=int, default=15, help="ranking ke liye kam se kam trades")
    args = ap.parse_args()
    cost = args.cost / 100

    print("1) Nifty 500 list ...")
    symbols = load_symbols(args.symbols_file)
    print(f"   {len(symbols)} symbols")

    print("2) Data download ...")
    data = download(symbols)
    print(f"   {len(data)} stocks ka data mila")

    print("3) Nifty 50 market filter ...")
    regimes = market_regimes()

    print("4) Indicators + signals ...")
    prepared = {}
    all_dates = sorted(set().union(*[d.index for d in data.values()]))
    test_start_date = all_dates[-args.days] if len(all_dates) >= args.days else all_dates[0]
    for s, d in data.items():
        d = add_indicators(d)
        d["base_signal"] = d["signal"]
        idx = int(np.searchsorted(d.index, test_start_date))
        prepared[s] = (d, idx)
    print(f"   test period: {pd.Timestamp(test_start_date).date()} se {pd.Timestamp(all_dates[-1]).date()}")

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    print("5) Har market filter ke saath SL x TP x MaxHold test ...")
    rows, all_trades = [], {}
    for fname, regime in regimes.items():
        total_signals = 0
        for s, (d, idx) in prepared.items():
            if regime is None:
                d["signal"] = d["base_signal"]
            else:
                ok_mkt = regime.reindex(d.index).ffill().fillna(False).astype(bool)
                d["signal"] = d["base_signal"] & ok_mkt
            total_signals += int(d["signal"].iloc[idx:].sum())
        print(f"   [{fname}] signals: {total_signals}")
        for sl_opt, tp_opt, mh in itertools.product(SL_OPTIONS, TP_OPTIONS, MAX_HOLD_OPTIONS):
            trades = []
            for s, (d, idx) in prepared.items():
                trades += simulate(s, d, idx, sl_opt, tp_opt, mh, cost)
            key = (fname, sl_opt[0], tp_opt[0], mh)
            all_trades[key] = trades
            rows.append(dict(filter=fname, SL=sl_opt[0], TP=tp_opt[0], max_hold=mh, **stats(trades)))

    res = pd.DataFrame(rows)
    res.to_csv("sl_tp_results.csv", index=False)

    summary, bests = [], []
    for fname in regimes:
        sub = res[(res["filter"] == fname) & (res["trades"] >= args.min_trades)]
        if sub.empty:
            sub = res[(res["filter"] == fname) & (res["trades"] > 0)]
        if sub.empty:
            print(f"\n[{fname}] koi trade nahi")
            continue
        sub = sub.sort_values(["avg_ret", "profit_factor"], ascending=False)
        print(f"\n================ FILTER: {fname} - TOP 5 ================")
        print(sub.head(5).drop(columns=["filter"]).to_string(index=False))
        b = sub.iloc[0]
        bests.append(b)
        summary.append(dict(filter=fname, profitable_combos=f"{int((res[res['filter']==fname]['avg_ret']>0).sum())}/{len(res[res['filter']==fname])}",
                            best_SL=b["SL"], best_TP=b["TP"], max_hold=b["max_hold"], trades=b["trades"],
                            win_rate=b["win_rate"], avg_ret=b["avg_ret"], profit_factor=b["profit_factor"],
                            max_dd_sum=b["max_dd_sum"]))

    print("\n================ COMPARISON (har filter ka best combo) ================")
    print(pd.DataFrame(summary).to_string(index=False))

    best = max(bests, key=lambda b: (b["avg_ret"], b["profit_factor"]))
    key = (best["filter"], best["SL"], best["TP"], best["max_hold"])
    pd.DataFrame(all_trades[key]).sort_values("entry_date").to_csv("best_combo_trades.csv", index=False)

    print("\n================ OVERALL BEST ================")
    print(f"Filter: {best['filter']} | SL: {best['SL']} | TP: {best['TP']} | Max hold: {best['max_hold']} din")
    print(f"Trades: {best['trades']} | Win rate: {best['win_rate']}% | Avg/trade: {best['avg_ret']}% "
          f"| Profit factor: {best['profit_factor']} | Max DD (sum %): {best['max_dd_sum']}")
    print("\nFiles: sl_tp_results.csv (saare combos), best_combo_trades.csv (best combo ke trades)")
    print("Note: 180 din chhota period hai - top 3-5 combos dekho, sirf #1 pe bharosa mat karo.")


if __name__ == "__main__":
    main()
