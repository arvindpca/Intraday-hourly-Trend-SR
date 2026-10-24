"""
INTRADAY MTF + SMC BACKTEST (Nifty 500, Buy + Short, 5 min aur 15 min entry)
===========================================================================

Chalane ka tarika:
    pip install yfinance pandas numpy
    python backtest_intraday_mtf_smc.py

Data: Yahoo se 5 min data (free me sirf ~60 din milta hai). 15 min aur 1 hour candles
isi 5 min data se banti hain. Pehle ~10 din indicators ke warmup me jate hain.

FILTER LEVELS:  OFF = koi filter nahi | ON = stock 1h trend + Nifty intraday | STRICT = ON + Nifty 1h trend
TREND (bada timeframe, 1 hour):
    Uptrend   = 1h Close > EMA50  aur  EMA20 > EMA50   -> sirf BUY
    Downtrend = 1h Close < EMA50  aur  EMA20 < EMA50   -> sirf SHORT
MARKET (Nifty):
    Buy sirf jab Nifty apne aaj ke average price ke upar ho, Short jab neeche ho
(Har setup "filter ON" aur "filter OFF" dono tarah test hota hai, taaki fark dikhe.)

ENTRY SETUPS (5 min aur 15 min dono pe):
 1. VWAP_PULLBACK : uptrend me price VWAP tak wapas aaye, VWAP chhu ke upar green close -> Buy
                    (Short me ulta). SL = us candle ka low/high
 2. ORB           : 9:15-9:30 ki range ka high, volume ke saath, pehli baar todhe -> Buy
                    (low tode -> Short). SL = range ka beech (mid)
 3. SMC_FVG       : SMC - Break of Structure (pichhla swing high close se toota), phir
                    us move me bane Fair Value Gap (FVG) tak price wapas aaye -> limit Buy.
                    SL = us move ka low. Short me ulta.
 4. SMC_SWEEP     : SMC - Liquidity sweep: price pichhle swing low ke neeche wick maare
                    lekin upar green close kare (stop-loss hunting ke baad reversal) -> Buy.
                    SL = sweep candle ka low. Short me ulta.

EXIT:
    SL  : "Structure" (setup ka apna SL) ya "1x ATR"
    TP  : 1.5R / 2R / 3R / TP nahi (3:15 square-off)
    Entry sirf 2:30 tak, 3:15 pe sab band. Same candle me SL+TP dono -> SL maana.
    Cost: 0.1% round trip. Risk 0.2% se kam ya 3% se zyada wale trades skip.
"""

import argparse
import itertools
import sys

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    sys.exit("pip install yfinance pandas numpy")


CFG = dict(
    price_min=200, price_max=3000,
    min_prev_day_turnover=1e8,      # pichhle din ka turnover > 10 crore (intraday liquidity)
    entry_cutoff="14:30",
    squareoff="15:15",
    warmup_days=10,
    orb_vol_mult=1.5,
    fvg_wait_minutes=90,            # FVG retest ka max wait
    min_risk=0.002, max_risk=0.03,
)

STRATEGIES = ["VWAP_PULLBACK", "ORB", "SMC_FVG", "SMC_SWEEP"]
TIMEFRAMES = [5, 15]
FILTERS = ["OFF", "ON", "STRICT"]
SL_MODES = ["Structure", "1x ATR"]
TP_MODES = [("1.5R", 1.5), ("2R", 2.0), ("3R", 3.0), ("3:15 exit", None)]


def hm(s):
    return int(s[:2]) * 60 + int(s[3:])


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
            sys.exit("Nifty 500 list nahi mili. --symbols-file do.")
    col = "Symbol" if "Symbol" in df.columns else df.columns[0]
    return sorted({str(s).strip().upper() + ".NS" for s in df[col].dropna()})


def clean(d):
    if isinstance(d.columns, pd.MultiIndex):
        d = d.droplevel(1, axis=1) if "Close" in d.columns.get_level_values(0) else d.droplevel(0, axis=1)
    d = d[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Open", "Close"])
    idx = d.index
    if idx.tz is not None:
        idx = idx.tz_convert("Asia/Kolkata").tz_localize(None)
    d = d.copy()
    d.index = idx
    m = d.index.hour * 60 + d.index.minute
    return d[(m >= hm("09:15")) & (m < hm("15:30"))]


def download_5m(symbols):
    data = {}
    for i in range(0, len(symbols), 40):
        chunk = symbols[i:i + 40]
        print(f"  downloading {i + 1}-{i + len(chunk)} / {len(symbols)} ...")
        raw = yf.download(chunk, period="60d", interval="5m", auto_adjust=True,
                          group_by="ticker", progress=False, threads=True)
        for s in chunk:
            try:
                d = clean(raw[s] if len(chunk) > 1 else raw)
                if len(d) > 500:
                    data[s] = d
            except Exception:
                pass
    return data


def resample(d, minutes):
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    if minutes == 5:
        return d
    offset = "15min" if minutes == 60 else None
    r = d.resample(f"{minutes}min", offset=offset, label="left", closed="left").agg(agg)
    return r.dropna(subset=["Open"])


# --------------------------------------------------------------------------------------
# Context: 1h trend + Nifty
# --------------------------------------------------------------------------------------
def trend_1h(d5):
    h = resample(d5, 60)
    c = h["Close"]
    e20, e50 = c.ewm(span=20, adjust=False).mean(), c.ewm(span=50, adjust=False).mean()
    trend = np.where((c > e50) & (e20 > e50), 1, np.where((c < e50) & (e20 < e50), -1, 0))
    ends = (h.index + pd.Timedelta(minutes=60)).values
    # aakhri candle 15:15 ki hai, wo 15:30 pe khatam
    return ends, trend


def nifty_state():
    raw = yf.download("^NSEI", period="60d", interval="5m", auto_adjust=True, progress=False)
    n = clean(raw)
    tp = (n["High"] + n["Low"] + n["Close"]) / 3
    avg = tp.groupby(n.index.normalize()).expanding().mean().droplevel(0)
    state = np.where(n["Close"] > avg, 1, -1)
    ends = (n.index + pd.Timedelta(minutes=5)).values
    return (ends, state), trend_1h(n)


def asof(ends_src, vals_src, ends_tgt):
    pos = np.searchsorted(ends_src, ends_tgt, side="right") - 1
    out = np.zeros(len(ends_tgt), dtype=int)
    ok = pos >= 0
    out[ok] = vals_src[pos[ok]]
    return out


# --------------------------------------------------------------------------------------
# Signals
# --------------------------------------------------------------------------------------
def prepare(d5, tf, trend_ctx, nifty_ctx, nifty_trend_ctx):
    d = resample(d5, tf).copy()
    O, H, L, C, V = (d[k].to_numpy(float) for k in ["Open", "High", "Low", "Close", "Volume"])
    idx = d.index
    dates = idx.normalize().to_numpy()
    mins = (idx.hour * 60 + idx.minute).to_numpy()
    ends = (idx + pd.Timedelta(minutes=tf)).values

    tp = (H + L + C) / 3
    g = pd.Series(dates)
    cum_pv = pd.Series(tp * V).groupby(g).cumsum().to_numpy()
    cum_v = pd.Series(V).groupby(g).cumsum().to_numpy()
    vwap = np.where(cum_v > 0, cum_pv / np.maximum(cum_v, 1e-9), tp)

    prev_c = np.r_[np.nan, C[:-1]]
    tr = np.nanmax(np.vstack([H - L, np.abs(H - prev_c), np.abs(L - prev_c)]), axis=0)
    atr = pd.Series(tr).ewm(alpha=1 / 14, adjust=False).mean().to_numpy()

    hhmm = pd.Series(idx.strftime("%H:%M"))
    vol_avg = pd.Series(V).groupby(hhmm).transform(lambda x: x.shift(1).rolling(10, min_periods=5).mean()).to_numpy()

    turnover = pd.Series(C * V).groupby(g).sum()
    prev_turn = pd.Series(dates).map(turnover.shift(1)).to_numpy()

    trend = asof(trend_ctx[0], trend_ctx[1], ends)
    nifty = asof(nifty_ctx[0], nifty_ctx[1], ends)
    nifty_tr = asof(nifty_trend_ctx[0], nifty_trend_ctx[1], ends)

    base_ok = (C >= CFG["price_min"]) & (C <= CFG["price_max"]) & (np.nan_to_num(prev_turn) > CFG["min_prev_day_turnover"])

    A = dict(O=O, H=H, L=L, C=C, atr=atr, mins=mins, dates=dates, idx=idx, n=len(d))
    sigs = []  # (strategy, side, decision_idx, entry_idx, entry_px, sl_struct)

    same_day_next = np.r_[dates[1:] == dates[:-1], False]
    cutoff = hm(CFG["entry_cutoff"])
    next_mins = np.r_[mins[1:], 99999]
    can_enter = same_day_next & (next_mins <= cutoff)

    # ---- 1. VWAP pullback
    pv = np.r_[np.nan, vwap[:-1]]
    pc = np.r_[np.nan, C[:-1]]
    first_bar = np.r_[True, dates[1:] != dates[:-1]]
    long_pb = ~first_bar & (pc > pv) & (L <= vwap) & (C > vwap) & (C > O)
    short_pb = ~first_bar & (pc < pv) & (H >= vwap) & (C < vwap) & (C < O)
    for i in np.where(long_pb & can_enter & base_ok)[0]:
        sigs.append(("VWAP_PULLBACK", 1, i, i + 1, O[i + 1], L[i]))
    for i in np.where(short_pb & can_enter & base_ok)[0]:
        sigs.append(("VWAP_PULLBACK", -1, i, i + 1, O[i + 1], H[i]))

    # ---- 2. ORB (9:15-9:30)
    in_or = mins < hm("09:30")
    or_h = pd.Series(np.where(in_or, H, np.nan)).groupby(g).transform("max").to_numpy()
    or_l = pd.Series(np.where(in_or, L, np.nan)).groupby(g).transform("min").to_numpy()
    or_mid = (or_h + or_l) / 2
    vol_ok = V > np.nan_to_num(vol_avg, nan=np.inf) * CFG["orb_vol_mult"]
    brk_up = ~in_or & (C > or_h) & (pc <= or_h)
    brk_dn = ~in_or & (C < or_l) & (pc >= or_l)
    first_up = pd.Series(brk_up.astype(int)).groupby(g).cumsum().to_numpy() == 1
    first_dn = pd.Series(brk_dn.astype(int)).groupby(g).cumsum().to_numpy() == 1
    for i in np.where(brk_up & first_up & vol_ok & can_enter & base_ok)[0]:
        sigs.append(("ORB", 1, i, i + 1, O[i + 1], or_mid[i]))
    for i in np.where(brk_dn & first_dn & vol_ok & can_enter & base_ok)[0]:
        sigs.append(("ORB", -1, i, i + 1, O[i + 1], or_mid[i]))

    # ---- 3 & 4. SMC (per day loop)
    wait = max(1, CFG["fvg_wait_minutes"] // tf)
    day_starts = np.r_[0, np.where(dates[1:] != dates[:-1])[0] + 1, len(d)]
    for a, b in zip(day_starts[:-1], day_starts[1:]):
        sh = sl_ = None  # (price, idx)
        sh_broken = sl_broken = True
        pending = []
        for k in range(a, b):
            m = k - 2
            if m - 2 >= a:
                if H[m] > max(H[m - 2], H[m - 1]) and H[m] >= max(H[m + 1], H[m + 2]):
                    sh, sh_broken = (H[m], m), False
                if L[m] < min(L[m - 2], L[m - 1]) and L[m] <= min(L[m + 1], L[m + 2]):
                    sl_, sl_broken = (L[m], m), False

            # pending FVG limit orders
            keep = []
            for p in pending:
                side, top, bot, stop, dec, exp = p
                if k > exp or mins[k] > cutoff:
                    continue
                if side == 1 and L[k] <= top:
                    if O[k] > stop and base_ok[dec]:
                        sigs.append(("SMC_FVG", 1, dec, k, min(O[k], top), stop))
                    continue
                if side == -1 and H[k] >= top:
                    if O[k] < stop and base_ok[dec]:
                        sigs.append(("SMC_FVG", -1, dec, k, max(O[k], top), stop))
                    continue
                keep.append(p)
            pending = keep

            # sweep
            if sl_ and not sl_broken and L[k] < sl_[0]:
                if C[k] > sl_[0] and C[k] > O[k] and can_enter[k] and base_ok[k]:
                    sigs.append(("SMC_SWEEP", 1, k, k + 1, O[k + 1], L[k]))
                sl_broken = True if C[k] > sl_[0] else sl_broken
            if sh and not sh_broken and H[k] > sh[0]:
                if C[k] < sh[0] and C[k] < O[k] and can_enter[k] and base_ok[k]:
                    sigs.append(("SMC_SWEEP", -1, k, k + 1, O[k + 1], H[k]))
                sh_broken = True if C[k] < sh[0] else sh_broken

            # BOS + FVG
            if sh and not sh_broken and C[k] > sh[0]:
                sh_broken = True
                start = sl_[1] if (sl_ and sl_[1] < k) else a
                for q in range(k, max(start, a + 2) - 1, -1):
                    if q - 2 >= a and L[q] > H[q - 2]:
                        stop = L[start:k + 1].min()
                        pending.append((1, L[q], H[q - 2], stop, k, k + wait))
                        break
            if sl_ and not sl_broken and C[k] < sl_[0]:
                sl_broken = True
                start = sh[1] if (sh and sh[1] < k) else a
                for q in range(k, max(start, a + 2) - 1, -1):
                    if q - 2 >= a and H[q] < L[q - 2]:
                        stop = H[start:k + 1].max()
                        pending.append((-1, H[q], L[q - 2], stop, k, k + wait))
                        break

    sig_df = pd.DataFrame(sigs, columns=["strategy", "side", "dec", "entry_idx", "entry_px", "sl_struct"])
    if not sig_df.empty:
        sig_df["aligned"] = (trend[sig_df["dec"]] == sig_df["side"]) & (nifty[sig_df["dec"]] == sig_df["side"])
        sig_df["strict"] = sig_df["aligned"] & (nifty_tr[sig_df["dec"]] == sig_df["side"])
        sig_df["vr"] = V[sig_df["dec"]] / np.nan_to_num(vol_avg[sig_df["dec"]], nan=np.inf)
    return A, sig_df


# --------------------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------------------
def simulate(sym, A, sig, test_start, sl_mode, tp_r, cost):
    O, H, L, C, atr, mins, dates, idx, n = (A[k] for k in ["O", "H", "L", "C", "atr", "mins", "dates", "idx", "n"])
    sq = hm(CFG["squareoff"])
    trades, last_exit = [], -1
    for row in sig.itertuples(index=False):
        e = row.entry_idx
        if e <= last_exit or idx[row.dec] < test_start:
            continue
        side, entry = row.side, row.entry_px
        sl = row.sl_struct if sl_mode == "Structure" else entry - side * atr[row.dec]
        risk = (entry - sl) * side
        if not np.isfinite(risk) or risk <= 0:
            continue
        if not (CFG["min_risk"] <= risk / entry <= CFG["max_risk"]):
            continue
        tp = entry + side * tp_r * risk if tp_r else side * np.inf

        j, day = e, dates[e]
        while True:
            if mins[j] >= sq:
                px, why = O[j], "Square-off"
                break
            if j > e:
                if (O[j] - sl) * side <= 0:
                    px, why = O[j], "SL (gap)"
                    break
                if (O[j] - tp) * side >= 0:
                    px, why = O[j], "TP (gap)"
                    break
            if (L[j] if side == 1 else H[j]) * side <= sl * side:
                px, why = sl, "SL"
                break
            if (H[j] if side == 1 else L[j]) * side >= tp * side:
                px, why = tp, "TP"
                break
            if j + 1 >= n or dates[j + 1] != day:
                px, why = C[j], "Day end"
                break
            j += 1
        last_exit = j
        trades.append((sym, "Buy" if side == 1 else "Short", idx[e], idx[j], round(entry, 2),
                       round(sl, 2), round(px, 2), why, side * (px / entry - 1) * 100 - cost * 100,
                       getattr(row, "vr", 0.0)))
    return trades


COLS = ["symbol", "side", "entry_time", "exit_time", "entry", "sl", "exit", "reason", "ret_pct", "vr"]


def stats(df, mid):
    if df.empty:
        return dict(trades=0)
    df = df.sort_values("exit_time")
    r = df["ret_pct"]
    w, l_ = r[r > 0], r[r <= 0]
    eq = r.cumsum()
    h1 = r[df["entry_time"] < mid]
    h2 = r[df["entry_time"] >= mid]
    return dict(
        trades=len(r), win_rate=round(len(w) / len(r) * 100, 1), avg_ret=round(r.mean(), 3),
        profit_factor=round(w.sum() / abs(l_.sum()), 2) if l_.sum() != 0 else np.inf,
        total_sum=round(r.sum(), 1), max_dd_sum=round((eq - eq.cummax()).min(), 1),
        first_half_avg=round(h1.mean(), 3) if len(h1) else np.nan,
        second_half_avg=round(h2.mean(), 3) if len(h2) else np.nan,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols-file")
    ap.add_argument("--cost", type=float, default=0.1)
    ap.add_argument("--min-trades", type=int, default=40)
    args = ap.parse_args()
    cost = args.cost / 100

    print("1) Nifty 500 list ...")
    symbols = load_symbols(args.symbols_file)
    print("2) 5-min data download (60 din) ...")
    data = download_5m(symbols)
    print(f"   {len(data)} stocks ka data mila")
    nifty_ctx, nifty_trend_ctx = nifty_state()

    all_days = sorted(set().union(*[set(d.index.normalize()) for d in data.values()]))
    test_start = all_days[CFG["warmup_days"]]
    mid = all_days[CFG["warmup_days"] + (len(all_days) - CFG["warmup_days"]) // 2]
    print(f"   test period: {test_start.date()} se {all_days[-1].date()} ({len(all_days) - CFG['warmup_days']} din)")

    print("3) Signals ...")
    prepared = {tf: {} for tf in TIMEFRAMES}
    for s, d5 in data.items():
        tctx = trend_1h(d5)
        for tf in TIMEFRAMES:
            prepared[tf][s] = prepare(d5, tf, tctx, nifty_ctx, nifty_trend_ctx)
    for tf in TIMEFRAMES:
        allsig = pd.concat([x[1] for x in prepared[tf].values() if not x[1].empty])
        print(f"   {tf}m signals:\n" + allsig.groupby(["strategy", "side"])[["aligned", "strict"]].agg(["count", "sum"]).iloc[:, [0, 1, 3]].set_axis(["total", "ON", "STRICT"], axis=1).to_string())

    print("4) Backtest combos ...")
    rows, store = [], {}
    for strat, tf, filt, slm, (tpn, tpr) in itertools.product(STRATEGIES, TIMEFRAMES, FILTERS, SL_MODES, TP_MODES):
        tr = []
        for s, (A, sig) in prepared[tf].items():
            if sig.empty:
                continue
            ss = sig[sig["strategy"] == strat]
            if filt == "ON":
                ss = ss[ss["aligned"]]
            elif filt == "STRICT":
                ss = ss[ss["strict"]]
            if ss.empty:
                continue
            tr += simulate(s, A, ss.sort_values("entry_idx"), test_start, slm, tpr, cost)
        df = pd.DataFrame(tr, columns=COLS)
        for side in ["Buy", "Short", "Buy+Short"]:
            sub = df if side == "Buy+Short" else df[df["side"] == side]
            key = (strat, f"{tf}m", filt, side, slm, tpn)
            store[key] = sub
            rows.append(dict(strategy=strat, tf=f"{tf}m", filter=filt, side=side, SL=slm, TP=tpn, **stats(sub, mid)))

    res = pd.DataFrame(rows)
    res.to_csv("mtf_smc_results.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)

    ok = res[res["trades"] >= args.min_trades].sort_values(["avg_ret", "profit_factor"], ascending=False)
    print(f"\n================ TOP 20 (kam se kam {args.min_trades} trades) ================")
    print(ok.head(20).to_string(index=False))

    print("\n================ HAR SETUP KA BEST (Buy+Short alag) ================")
    best_each = ok.sort_values("avg_ret", ascending=False).groupby(["strategy", "tf", "side"]).head(1)
    print(best_each.sort_values(["strategy", "tf", "side"]).to_string(index=False))

    print("\n================ FILTER ON vs OFF (har setup ka average, saare SL/TP combos) ================")
    print(res[res["trades"] > 0].groupby(["strategy", "tf", "side", "filter"])["avg_ret"].mean().unstack().round(3).to_string())

    if ok.empty:
        print("Kisi combo me itne trades nahi.")
        return
    b = ok.iloc[0]
    key = (b["strategy"], b["tf"], b["filter"], b["side"], b["SL"], b["TP"])
    bt = store[key].sort_values("entry_time").copy()
    bt.to_csv("mtf_smc_best_trades.csv", index=False)
    print("\n================ OVERALL BEST ================")
    print(f"{b['strategy']} | {b['tf']} | Filter {b['filter']} | {b['side']} | SL {b['SL']} | TP {b['TP']}")
    print(f"Trades {b['trades']} | Win {b['win_rate']}% | Avg {b['avg_ret']}% | PF {b['profit_factor']} | "
          f"1st half avg {b['first_half_avg']}% | 2nd half avg {b['second_half_avg']}%")
    bt["hour"] = pd.to_datetime(bt["entry_time"]).dt.hour
    bt["week"] = pd.to_datetime(bt["entry_time"]).dt.to_period("W")
    print("\nEntry ghante ke hisaab se:")
    print(bt.groupby("hour")["ret_pct"].agg(trades="count", avg="mean", total="sum").round(2).to_string())
    print("\nHafte ke hisaab se:")
    print(bt.groupby("week")["ret_pct"].agg(trades="count", avg="mean", total="sum").round(2).to_string())
    print("\nExit reason:")
    print(bt.groupby("reason")["ret_pct"].agg(trades="count", avg="mean").round(2).to_string())
    print("\nFiles: mtf_smc_results.csv, mtf_smc_best_trades.csv")


if __name__ == "__main__":
    main()
