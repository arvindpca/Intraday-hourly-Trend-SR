"""
INTRADAY 1-Hour Backtest - MACD + Volume Breakout (Buy + Short), Nifty 500
=========================================================================

Chalane ka tarika:
    pip install yfinance pandas numpy
    python backtest_intraday_1h.py

Rules:
  - Saari conditions 1 HOUR candle pe check hoti hain
  - Signal candle close hone ke baad, AGLI candle ke open pe entry
  - Entry sirf 10:15 se 2:15 tak (signal candle 9:15 se 1:15 wali)
  - 3:15 pe har trade square-off (us candle ke open pe)
  - Buy aur Short dono test, alag-alag aur saath me
  - Same candle me SL aur TP dono touch ho to SL maana (conservative)
  - Ek stock me ek time pe ek hi trade (exit ke baad usi din dobara signal aa sakta hai)
  - Cost: default 0.1% round trip (intraday brokerage + STT + slippage)

Note: Yahoo free data me 1h candle ~2 saal tak milti hai, isliye 180 din ho jata hai.
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
# CONFIG
# --------------------------------------------------------------------------------------
CFG = dict(
    price_min=200, price_max=3000,
    min_prev_day_turnover=5e7,     # pichhle din ka turnover > 5 crore
    rsi_long=(50, 70),             # Buy: RSI 50-70
    rsi_short=(30, 50),            # Short: RSI 30-50
    vol_mult=2.0,                  # candle volume > (usi time ki pichhle 20 din ki avg volume) x 2
    last_signal_candle="13:15",    # 1:15 wali candle last signal -> 2:15 last entry
    squareoff="15:15",
    test_days=180,
)

SL_OPTIONS = [
    ("Signal candle low/high", "candle", None),
    ("SL 0.5%", "pct", 0.005),
    ("SL 1%", "pct", 0.01),
    ("SL 1.5%", "pct", 0.015),
    ("SL 1x ATR", "atr", 1.0),
    ("SL 1.5x ATR", "atr", 1.5),
]

TP_OPTIONS = [
    ("TP 1R", "r", 1.0),
    ("TP 1.5R", "r", 1.5),
    ("TP 2R", "r", 2.0),
    ("TP 3R", "r", 3.0),
    ("TP 1%", "pct", 0.01),
    ("TP 2%", "pct", 0.02),
    ("No TP, 3:15 square-off", "none", None),
]

DIRECTIONS = ["Long", "Short"]


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
            sys.exit("Nifty 500 list download nahi hui. --symbols-file ind_nifty500list.csv do.")
    col = "Symbol" if "Symbol" in df.columns else df.columns[0]
    return sorted({str(s).strip().upper() + ".NS" for s in df[col].dropna()})


def download(symbols, days=330):
    start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    data = {}
    for i in range(0, len(symbols), 50):
        chunk = symbols[i:i + 50]
        print(f"  downloading {i + 1}-{i + len(chunk)} / {len(symbols)} ...")
        raw = yf.download(chunk, start=start, interval="1h", auto_adjust=True,
                          group_by="ticker", progress=False, threads=True)
        for s in chunk:
            try:
                d = raw[s] if len(chunk) > 1 else raw
                if isinstance(d.columns, pd.MultiIndex):
                    d = d.droplevel(0, axis=1) if s in d.columns.get_level_values(0) else d.droplevel(1, axis=1)
                d = d[["Open", "High", "Low", "Close", "Volume"]].dropna()
                d = d[d["Volume"] > 0]
                idx = d.index
                if idx.tz is not None:
                    idx = idx.tz_convert("Asia/Kolkata").tz_localize(None)
                d.index = idx
                if len(d) > 300:
                    data[s] = d
            except Exception:
                pass
    return data


# --------------------------------------------------------------------------------------
# Indicators + signals (1h)
# --------------------------------------------------------------------------------------
def add_indicators(d):
    d = d.copy()
    c, o, h, l, v = d["Close"], d["Open"], d["High"], d["Low"], d["Volume"]
    d["date"] = d.index.normalize()
    d["mins"] = d.index.hour * 60 + d.index.minute
    d["hhmm"] = d.index.strftime("%H:%M")

    delta = c.diff()
    ag = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    al = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    d["rsi"] = 100 - 100 / (1 + ag / al.replace(0, np.nan))

    d["macd"] = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    d["macd_sig"] = d["macd"].ewm(span=9, adjust=False).mean()

    rng = (h - l).replace(0, np.nan)
    d["adl"] = ((((c - l) - (h - c)) / rng).fillna(0) * v).cumsum()

    tp = (h + l + c) / 3
    d["vwap"] = (tp * v).groupby(d["date"]).cumsum() / v.groupby(d["date"]).cumsum()  # intraday VWAP

    d["ema8"] = c.ewm(span=8, adjust=False).mean()
    d["ema21"] = c.ewm(span=21, adjust=False).mean()

    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    d["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()

    # 9:15 candle ka volume hamesha zyada hota hai, isliye usi time ki pichhle 20 din ki avg se compare
    d["vol_avg"] = v.groupby(d["hhmm"]).transform(lambda x: x.shift(1).rolling(20).mean())

    daily_turnover = (c * v).groupby(d["date"]).sum().shift(1)
    d["prev_turnover"] = d["date"].map(daily_turnover)

    last_sig = int(CFG["last_signal_candle"][:2]) * 60 + int(CFG["last_signal_candle"][3:])
    common = (
        (c >= CFG["price_min"]) & (c <= CFG["price_max"])
        & (d["prev_turnover"] > CFG["min_prev_day_turnover"])
        & (v > d["vol_avg"] * CFG["vol_mult"])
        & (d["mins"] <= last_sig)
    )
    cross_up = (d["macd"] > d["macd_sig"]) & (d["macd"].shift(1) <= d["macd_sig"].shift(1))
    cross_dn = (d["macd"] < d["macd_sig"]) & (d["macd"].shift(1) >= d["macd_sig"].shift(1))

    d["sig_Long"] = (
        common & cross_up
        & (d["rsi"] > CFG["rsi_long"][0]) & (d["rsi"] < CFG["rsi_long"][1])
        & (d["adl"] > d["adl"].shift(1)) & (c > o) & (c > d["vwap"]) & (d["ema8"] > d["ema21"])
    ).fillna(False)
    d["sig_Short"] = (
        common & cross_dn
        & (d["rsi"] > CFG["rsi_short"][0]) & (d["rsi"] < CFG["rsi_short"][1])
        & (d["adl"] < d["adl"].shift(1)) & (c < o) & (c < d["vwap"]) & (d["ema8"] < d["ema21"])
    ).fillna(False)
    return d


# --------------------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------------------
def simulate(sym, d, start_idx, direction, sl_opt, tp_opt, cost):
    O, H, L, C = (d[k].to_numpy() for k in ["Open", "High", "Low", "Close"])
    atr, mins = d["atr"].to_numpy(), d["mins"].to_numpy()
    dates, sigs = d["date"].to_numpy(), d[f"sig_{direction}"].to_numpy()
    ts = d.index
    sq = int(CFG["squareoff"][:2]) * 60 + int(CFG["squareoff"][3:])
    long = direction == "Long"
    n = len(d)
    trades = []
    i = start_idx
    while i < n - 1:
        e = i + 1
        if not sigs[i] or dates[e] != dates[i] or mins[e] >= sq:
            i += 1
            continue
        entry = O[e]

        _, sl_t, sl_v = sl_opt
        if sl_t == "candle":
            sl = L[i] if long else H[i]
        elif sl_t == "pct":
            sl = entry * (1 - sl_v) if long else entry * (1 + sl_v)
        else:
            sl = entry - sl_v * atr[i] if long else entry + sl_v * atr[i]
        if not np.isfinite(sl) or (long and sl >= entry) or (not long and sl <= entry):
            i += 1
            continue
        risk = abs(entry - sl)

        _, tp_t, tp_v = tp_opt
        if tp_t == "r":
            tp = entry + tp_v * risk if long else entry - tp_v * risk
        elif tp_t == "pct":
            tp = entry * (1 + tp_v) if long else entry * (1 - tp_v)
        else:
            tp = np.inf if long else -np.inf

        exit_px, reason, j = None, None, e
        while True:
            if mins[j] >= sq:
                exit_px, reason = O[j], "Square-off 3:15"
                break
            if j > e:
                if (long and O[j] <= sl) or (not long and O[j] >= sl):
                    exit_px, reason = O[j], "SL (gap)"
                    break
                if (long and O[j] >= tp) or (not long and O[j] <= tp):
                    exit_px, reason = O[j], "TP (gap)"
                    break
            if (long and L[j] <= sl) or (not long and H[j] >= sl):
                exit_px, reason = sl, "SL"
                break
            if (long and H[j] >= tp) or (not long and L[j] <= tp):
                exit_px, reason = tp, "TP"
                break
            if j + 1 >= n or dates[j + 1] != dates[i]:
                exit_px, reason = C[j], "Day end"
                break
            j += 1

        sign = 1 if long else -1
        trades.append(dict(
            symbol=sym.replace(".NS", ""), side=direction, signal_time=ts[i], entry_time=ts[e],
            exit_time=ts[j], entry=round(entry, 2), sl=round(sl, 2),
            tp=round(tp, 2) if np.isfinite(tp) else None, exit=round(exit_px, 2), reason=reason,
            ret_pct=round((sign * (exit_px / entry - 1) - cost) * 100, 3),
            r_mult=round(sign * (exit_px - entry) / risk, 2),
        ))
        i = j + 1
    return trades


def stats(trades):
    if not trades:
        return dict(trades=0)
    df = pd.DataFrame(trades).sort_values("exit_time")
    r = df["ret_pct"]
    wins, losses = r[r > 0], r[r <= 0]
    pf = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else np.inf
    eq = r.cumsum()
    return dict(
        trades=len(df), win_rate=round(len(wins) / len(df) * 100, 1),
        avg_ret=round(r.mean(), 3), avg_win=round(wins.mean(), 2) if len(wins) else 0,
        avg_loss=round(losses.mean(), 2) if len(losses) else 0,
        profit_factor=round(pf, 2), total_ret_sum=round(r.sum(), 1),
        max_dd_sum=round((eq - eq.cummax()).min(), 1),
    )


# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols-file")
    ap.add_argument("--days", type=int, default=CFG["test_days"])
    ap.add_argument("--cost", type=float, default=0.1, help="round-trip cost %% (default 0.1)")
    ap.add_argument("--min-trades", type=int, default=30)
    args = ap.parse_args()
    cost = args.cost / 100

    print("1) Nifty 500 list ...")
    symbols = load_symbols(args.symbols_file)
    print(f"   {len(symbols)} symbols")
    print("2) 1-hour data download ...")
    data = download(symbols)
    print(f"   {len(data)} stocks ka data mila")
    if not data:
        sys.exit("Data nahi mila.")

    print("3) Indicators + signals ...")
    all_days = sorted(set().union(*[set(d.index.normalize()) for d in data.values()]))
    test_start = all_days[-args.days] if len(all_days) >= args.days else all_days[0]
    prepared, n_sig = {}, {k: 0 for k in DIRECTIONS}
    for s, d in data.items():
        d = add_indicators(d)
        idx = int(np.searchsorted(d.index, test_start))
        prepared[s] = (d, idx)
        for k in DIRECTIONS:
            n_sig[k] += int(d[f"sig_{k}"].iloc[idx:].sum())
    print(f"   test period: {pd.Timestamp(test_start).date()} se {pd.Timestamp(all_days[-1]).date()}")
    print(f"   signals -> Long: {n_sig['Long']} | Short: {n_sig['Short']}")

    print("4) SL x TP combinations test ...")
    trades_by = {}
    for direction, sl_opt, tp_opt in itertools.product(DIRECTIONS, SL_OPTIONS, TP_OPTIONS):
        t = []
        for s, (d, idx) in prepared.items():
            t += simulate(s, d, idx, direction, sl_opt, tp_opt, cost)
        trades_by[(direction, sl_opt[0], tp_opt[0])] = t

    rows, all_trades = [], {}
    for sl_opt, tp_opt in itertools.product(SL_OPTIONS, TP_OPTIONS):
        lt = trades_by[("Long", sl_opt[0], tp_opt[0])]
        st = trades_by[("Short", sl_opt[0], tp_opt[0])]
        for side, t in [("Long", lt), ("Short", st), ("Long+Short", lt + st)]:
            all_trades[(side, sl_opt[0], tp_opt[0])] = t
            rows.append(dict(side=side, SL=sl_opt[0], TP=tp_opt[0], **stats(t)))

    res = pd.DataFrame(rows)
    res.to_csv("intraday_sl_tp_results.csv", index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    summary, bests = [], []
    for side in ["Long", "Short", "Long+Short"]:
        sub = res[(res["side"] == side) & (res["trades"] >= args.min_trades)]
        if sub.empty:
            sub = res[(res["side"] == side) & (res["trades"] > 0)]
        if sub.empty:
            print(f"\n[{side}] koi trade nahi")
            continue
        sub = sub.sort_values(["avg_ret", "profit_factor"], ascending=False)
        print(f"\n================ {side} - TOP 5 ================")
        print(sub.head(5).drop(columns=["side"]).to_string(index=False))
        b = sub.iloc[0]
        bests.append(b)
        allside = res[res["side"] == side]
        summary.append(dict(side=side, profitable_combos=f"{int((allside['avg_ret'] > 0).sum())}/{len(allside)}",
                            best_SL=b["SL"], best_TP=b["TP"], trades=b["trades"], win_rate=b["win_rate"],
                            avg_ret=b["avg_ret"], profit_factor=b["profit_factor"], max_dd_sum=b["max_dd_sum"]))

    print("\n================ COMPARISON ================")
    print(pd.DataFrame(summary).to_string(index=False))

    best = max(bests, key=lambda b: (b["avg_ret"], b["profit_factor"]))
    bt = pd.DataFrame(all_trades[(best["side"], best["SL"], best["TP"])]).sort_values("entry_time")
    bt.to_csv("intraday_best_trades.csv", index=False)

    print("\n================ OVERALL BEST ================")
    print(f"Side: {best['side']} | SL: {best['SL']} | TP: {best['TP']}")
    print(f"Trades: {best['trades']} | Win rate: {best['win_rate']}% | Avg/trade: {best['avg_ret']}% "
          f"| Profit factor: {best['profit_factor']} | Max DD (sum %): {best['max_dd_sum']}")

    bt["month"] = pd.to_datetime(bt["entry_time"]).dt.to_period("M")
    bt["entry_hour"] = pd.to_datetime(bt["entry_time"]).dt.strftime("%H:%M")
    print("\nMahine ke hisaab se (best combo):")
    print(bt.groupby("month")["ret_pct"].agg(trades="count", avg="mean", total="sum").round(2).to_string())
    print("\nEntry time ke hisaab se (best combo):")
    print(bt.groupby("entry_hour")["ret_pct"].agg(trades="count", avg="mean", total="sum").round(2).to_string())
    print("\nExit reason:")
    print(bt.groupby("reason")["ret_pct"].agg(trades="count", avg="mean").round(2).to_string())

    print("\nFiles: intraday_sl_tp_results.csv (saare combos), intraday_best_trades.csv (best combo ke trades)")


if __name__ == "__main__":
    main()
