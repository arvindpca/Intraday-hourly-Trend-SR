"""
V2 LOSS ANALYSIS - loss wale trades me kya alag tha?
Setup: Nifty filter ON, same-day close entry, SL = breakout candle low, exit 20 din (V2 best by profit).
Har SIGNAL ko alag se simulate (portfolio limit ke bina) taaki sample bada ho.
Har feature ko 5 barabar hisson (quintile) me baant ke avg return dekhte hain - pehle aadhe aur
doosre aadhe period me ALAG, taaki sirf wahi filter chune jo dono me kaam kare.
"""
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

COST = 0.2
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
LOG = []
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_rows", 400)


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
    n_ema50 = nc.ewm(span=50, adjust=False).mean()
    nifty = pd.DataFrame({"n_ok": nc > n_ema50, "n_ext": (nc / n_ema50 - 1) * 100,
                          "n_ret20": (nc / nc.shift(20) - 1) * 100, "n_ret5": (nc / nc.shift(5) - 1) * 100})

    rows = []
    for s, d in data.items():
        if len(d) < v2.WARMUP + 50:
            continue
        d = v2.indicators(d)
        c, o, h, l, v = d["Close"], d["Open"], d["High"], d["Low"], d["Volume"]
        f = pd.DataFrame(index=d.index)
        f["rsi"] = rsi(c)
        f["ext_ema20"] = (c / d["ema20"] - 1) * 100
        f["breakout_pct"] = (c / h.shift(1).rolling(20).max() - 1) * 100
        f["body_atr"] = (c - o) / d["atr"]
        f["range_atr"] = (h - l) / d["atr"]
        f["risk_pct"] = (c - l) / c * 100
        f["vol_ratio"] = v / d["vsma50"]
        f["ret20_before"] = (c / c.shift(20) - 1) * 100
        f["ret60_before"] = (c / c.shift(60) - 1) * 100
        f["from_52w_high"] = (c / h.rolling(252, min_periods=120).max() - 1) * 100
        f["ema50_slope10"] = (d["ema50"] / d["ema50"].shift(10) - 1) * 100
        f["ema20_over_ema50"] = (d["ema20"] / d["ema50"] - 1) * 100
        f["atr_pct"] = d["atr"] / c * 100
        f["price"] = c
        f["turnover_cr"] = (c * v).rolling(20).mean() / 1e7
        nf = nifty.reindex(d.index).ffill()
        f = f.join(nf)
        O, H, L, C = o.to_numpy(), h.to_numpy(), l.to_numpy(), c.to_numpy()
        sig = d["sig_fixed"].to_numpy() & f["n_ok"].fillna(False).to_numpy()
        idx = d.index
        for i in np.where(sig)[0]:
            if i < v2.WARMUP or i >= len(d) - 1:
                continue
            entry, sl = C[i], L[i]
            if sl >= entry:
                continue
            j, mae = i + 1, 0.0
            while True:
                mae = min(mae, (L[j] / entry - 1) * 100)
                if O[j] <= sl:
                    px, why = O[j], "SL"
                elif L[j] <= sl:
                    px, why = sl, "SL"
                elif j - i >= 20:
                    px, why = C[j], "20 din"
                elif j == len(d) - 1:
                    px, why = C[j], "khula"
                else:
                    j += 1
                    continue
                break
            if why == "khula":
                continue
            r = f.iloc[i].to_dict()
            r.update(symbol=s.replace(".NS", ""), date=idx[i], ret=(px / entry - 1) * 100 - COST, reason=why,
                     days=j - i, mae=mae, mfe=(H[i + 1:j + 1].max() / entry - 1) * 100,
                     next_day_ret=(C[i + 1] / entry - 1) * 100, weekday=idx[i].day_name()[:3])
            rows.append(r)
    t = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    t["same_day_signals"] = t.groupby("date")["symbol"].transform("count")
    mid = t["date"].iloc[len(t) // 2]
    t["half"] = np.where(t["date"] < mid, "H1", "H2")
    t["win"] = t["ret"] > 0
    t.to_csv(OUT / "v2_signal_features.csv", index=False)

    out(f"Signals (Nifty filter ON): {len(t)} | {t['date'].min().date()} se {t['date'].max().date()} | split {mid.date()}")
    out(f"Win {t['win'].mean() * 100:.1f}% | avg {t['ret'].mean():.2f}% | avg win {t[t.win]['ret'].mean():.2f}% | "
        f"avg loss {t[~t.win]['ret'].mean():.2f}%")
    out(f"H1 avg {t[t.half == 'H1']['ret'].mean():.2f}% (n={int((t.half == 'H1').sum())}) | "
        f"H2 avg {t[t.half == 'H2']['ret'].mean():.2f}% (n={int((t.half == 'H2').sum())})")

    # ---- loss kaise hote hain
    L_ = t[~t["win"]]
    out("\n=== LOSS kaise hue ===")
    out(L_.groupby("reason")["ret"].agg(trades="count", avg="mean").round(2).to_string())
    out("SL kitne din me laga:\n" + pd.cut(L_[L_.reason == "SL"]["days"], [0, 1, 3, 5, 10, 20],
                                         labels=["1 din", "2-3", "4-5", "6-10", "11-20"]).value_counts().sort_index().to_string())
    out(f"\nLoss trades me agle din ka avg return: {L_['next_day_ret'].mean():.2f}% | win trades me: {t[t.win]['next_day_ret'].mean():.2f}%")
    out(f"Loss trades me max upar gaya (MFE) median: {L_['mfe'].median():.2f}% | 25% loss trades {L_['mfe'].quantile(.75):.1f}%+ upar gaye the phir gire")

    # ---- feature-wise
    feats = ["rsi", "ext_ema20", "breakout_pct", "body_atr", "range_atr", "risk_pct", "vol_ratio", "ret20_before",
             "ret60_before", "from_52w_high", "ema50_slope10", "ema20_over_ema50", "atr_pct", "price", "turnover_cr",
             "n_ext", "n_ret20", "n_ret5", "same_day_signals"]
    out("\n=== WIN vs LOSS - median values ===")
    cmp = pd.DataFrame({"win_median": t[t.win][feats].median(), "loss_median": t[~t.win][feats].median()}).round(2)
    out(cmp.to_string())

    summary = []
    for ft in feats:
        x = t[[ft, "ret", "half", "win"]].dropna()
        if x[ft].nunique() < 5:
            continue
        x["q"] = pd.qcut(x[ft], 5, labels=False, duplicates="drop")
        g = x.groupby("q").agg(lo=(ft, "min"), hi=(ft, "max"), n=("ret", "size"), win=("win", "mean"),
                               avg=("ret", "mean"))
        for hf in ["H1", "H2"]:
            g[f"avg_{hf}"] = x[x.half == hf].groupby("q")["ret"].mean()
        g["win"] = (g["win"] * 100).round(0)
        out(f"\n--- {ft} (5 hisse, chhote se bade) ---\n{g.round(2).to_string()}")
        worst = g["avg"].idxmin()
        both_bad = (g.loc[worst, "avg_H1"] < t[t.half == "H1"]["ret"].mean()) and \
                   (g.loc[worst, "avg_H2"] < t[t.half == "H2"]["ret"].mean())
        summary.append(dict(feature=ft, worst_bucket=f"{g.loc[worst, 'lo']:.2f} - {g.loc[worst, 'hi']:.2f}",
                            worst_avg=round(g.loc[worst, "avg"], 2), worst_H1=round(g.loc[worst, "avg_H1"], 2),
                            worst_H2=round(g.loc[worst, "avg_H2"], 2), both_halves_bad=both_bad,
                            spread=round(g["avg"].max() - g["avg"].min(), 2)))
    out("\n=== FEATURE SUMMARY (spread = best hissa - worst hissa) ===")
    out(pd.DataFrame(summary).sort_values("spread", ascending=False).to_string(index=False))

    out("\n=== Weekday ===\n" + t.groupby("weekday")["ret"].agg(n="count", avg="mean",
                                                                win=lambda x: (x > 0).mean() * 100).round(2).to_string())
    (OUT / "LOSS_REPORT.txt").write_text("\n".join(LOG), encoding="utf-8")


if __name__ == "__main__":
    main()
