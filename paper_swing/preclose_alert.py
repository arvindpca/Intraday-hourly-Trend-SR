#!/usr/bin/env python3
"""
3 PM PRE-CLOSE ALERT - Daily Swing strategy (koi order nahi lagta)
=================================================================

Market band hone se pehle (~3:00 PM IST) chalta hai:
  - Pichhle dino ka daily data: Yahoo
  - AAJ ki candle (open/high/low/abhi ka price/volume): Angel One live quote (getMarketData FULL)
    Angel fail ho to Yahoo ka aaj ka (delayed) data - message me likha jata hai.
  - Wahi strategy rules lagakar aaj ke BUY signals nikalta hai
  - Paper bot ke khule trades dekh ke batata hai: kitni jagah khaali, kaunsa trade aaj 20va din (becho)

Telegram pe list bhejta hai taaki aap 3:20-3:29 ke beech khud order laga sako.
Final paper entry/exit shaam 4:37 wale run me actual close pe hi hoti hai.
3 baje ke baad price/volume badal sakta hai - kuch signal close tak gayab ho sakte hain.
"""
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import yfinance as yf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import strategy_core as core  # noqa: E402
import paper_trader as pt  # noqa: E402  (telegram, inr, symbols, PER_TRADE, MAX_POS, SL_PCT)

IST = ZoneInfo("Asia/Kolkata")
INSTRUMENT_URL = "https://margincalculator.angelone.in/OpenAPI_File/files/OpenAPIScripMaster.json"
NIFTY_TOKEN = "99926000"
LOG_F = pt.STATE / "preclose_log.csv"


def angel_quotes(symbols):
    """{symbol: (open, high, low, ltp, volume)}, nifty_ltp - Angel One live."""
    from angel_client import get_client
    master = requests.get(INSTRUMENT_URL, timeout=60).json()
    tok = {str(i.get("symbol", "")).upper(): str(i.get("token", ""))
           for i in master if str(i.get("exch_seg", "")).upper() in ("NSE", "NSE_CM")}
    want = {tok[f"{s}-EQ"]: s for s in symbols if f"{s}-EQ" in tok}
    client = get_client()
    quotes, nifty = {}, None
    ids = list(want) + [NIFTY_TOKEN]
    for i in range(0, len(ids), 50):
        chunk = ids[i:i + 50]
        r = client.getMarketData("FULL", {"NSE": chunk})
        if not r or not r.get("status"):
            raise RuntimeError(str((r or {}).get("message", "empty"))[:150])
        for q in (r.get("data") or {}).get("fetched", []):
            t = str(q.get("symbolToken"))
            if t == NIFTY_TOKEN:
                nifty = float(q["ltp"])
                continue
            s = want.get(t)
            if s and q.get("ltp"):
                quotes[s] = (float(q["open"]), float(q["high"]), float(q["low"]),
                             float(q["ltp"]), float(q.get("tradeVolume") or 0))
        time.sleep(1.1)
    if nifty is None or len(quotes) < len(want) * 0.8:
        raise RuntimeError(f"Angel quotes adhoore: {len(quotes)}/{len(want)}, nifty={nifty}")
    return quotes, nifty


def yahoo_today(symbols):
    """Backup: Yahoo ka aaj ka (delayed) daily row."""
    quotes = {}
    for i in range(0, len(symbols), 50):
        chunk = [s + ".NS" for s in symbols[i:i + 50]]
        raw = yf.download(chunk, period="5d", interval="1d", auto_adjust=False, group_by="ticker", progress=False)
        for s in chunk:
            try:
                d = raw[s].dropna()
                r = d.iloc[-1]
                if d.index[-1].date() == datetime.now(IST).date():
                    quotes[s[:-3]] = (float(r["Open"]), float(r["High"]), float(r["Low"]), float(r["Close"]),
                                      float(r["Volume"]))
            except Exception:
                pass
    n = yf.download("^NSEI", period="5d", interval="1d", progress=False)["Close"]
    n = (n.iloc[:, 0] if isinstance(n, pd.DataFrame) else n).dropna()
    return quotes, float(n.iloc[-1])


def main():
    now = datetime.now(IST)
    today = pd.Timestamp(os.getenv("PAPER_AS_OF") or now.date())
    test = os.getenv("PRECLOSE_TEST") == "1"
    meta = json.loads(pt.META_F.read_text()) if pt.META_F.exists() else {}
    if not test and meta.get("preclose_sent") == str(today.date()):
        print("Aaj ka 3 PM alert pehle hi bhej chuke - skip")
        return
    if not test and now.hour * 60 + now.minute > 15 * 60 + 22 and not os.getenv("PAPER_AS_OF"):
        print("3:22 ke baad - alert ka fayda nahi, skip")
        return
    syms = pt.symbols()
    names = [s.replace(".NS", "") for s in syms]

    # 1) aaj ka live data
    try:
        quotes, nifty_ltp = angel_quotes(names)
        src = "Angel One live"
    except Exception as e:
        print(f"::warning title=Angel quotes fail::{str(e)[:180]}")
        quotes, nifty_ltp = yahoo_today(names)
        src = "Yahoo (15-20 min delayed)"
    print(f"::notice title=3PM data::quotes {len(quotes)} | source {src} | nifty {nifty_ltp}")
    if len(quotes) < 100:
        pt.telegram(f"⚠️ <b>3 PM Alert</b>: aaj ka live data nahi mila ({len(quotes)} stocks). "
                    "Aaj market chhutti to nahi? Shaam wala paper run normal chalega.")
        return

    # 2) history (aaj se pehle tak) + aaj ki candle
    data, nifty = pt.download_daily(syms)
    nifty = nifty[nifty.index < today]
    nifty.loc[today] = nifty_ltp
    ema = nifty.ewm(span=50, adjust=False).mean()
    market_ok = bool(nifty.iloc[-1] > ema.iloc[-1])

    cands = []
    for s, d in data.items():
        name = s.replace(".NS", "")
        q = quotes.get(name)
        if q is None:
            continue
        d = d[d.index < today].copy()
        if len(d) < 80:
            continue
        d.loc[today, ["Open", "High", "Low", "Close", "Volume"]] = q
        d = core.add_indicators(d)
        r = d.iloc[-1]
        if bool(r["signal"]):
            cands.append(dict(symbol=name, price=float(r["Close"]), vr=float(r["Volume"] / r["vol_sma20"]),
                              rsi=float(r["rsi"]), chg=(float(r["Close"]) / float(d["Close"].iloc[-2]) - 1) * 100))
    cands.sort(key=lambda x: -x["vr"])

    # 3) paper bot ke khule trades
    positions = json.loads(pt.POS_F.read_text()) if pt.POS_F.exists() else []
    sell_today = [p for p in positions if int(p.get("bars_held", 0)) + 1 >= pt.MAX_HOLD]
    free = pt.MAX_POS - len(positions) + len(sell_today)
    held = {p["symbol"] for p in positions if p not in sell_today}

    lines = [("🧪 <b>TEST RUN</b> (market band hai - sirf connection check, isse trade mat lena)\n" if test else "")
             + f"⏰ <b>3 PM Swing Alert | {today:%d %b %Y}</b>  <i>({src}, {now:%H:%M})</i>",
             f"Nifty {nifty_ltp:,.0f} vs 50 EMA {ema.iloc[-1]:,.0f} → "
             + ("✅ naya trade allowed" if market_ok else "⛔ naya trade NAHI (Nifty 50 EMA ke neeche)")]
    if sell_today:
        lines += ["", "<b>AAJ BECHO (20va din, 3:20–3:29)</b>"]
        for p in sell_today:
            lines.append(f"🔵 {p['symbol']} × {p['qty']} (entry {p['entry']:.2f})")
    log_rows = []
    if market_ok and cands:
        lines += ["", f"<b>BUY SIGNALS</b> (khaali jagah: {max(free, 0)}/{pt.MAX_POS})"]
        taken = 0
        for c in cands:
            if c["symbol"] in held:
                tag = "pehle se khula"
            elif taken >= max(free, 0):
                tag = "jagah nahi"
            else:
                tag = "BUY"
                taken += 1
            qty = math.floor(pt.PER_TRADE / c["price"])
            sl = c["price"] * (1 - pt.SL_PCT)
            if tag == "BUY":
                lines.append(f"🟢 <b>{c['symbol']}</b> ~{c['price']:.2f} × {qty} | SL {sl:.2f} | "
                             f"Vol {c['vr']:.1f}× | RSI {c['rsi']:.0f} | {c['chg']:+.1f}%")
            else:
                lines.append(f"⚪ {c['symbol']} ~{c['price']:.2f} ({tag}) | Vol {c['vr']:.1f}×")
            log_rows.append(dict(date=str(today.date()), time=f"{now:%H:%M}", source=src, symbol=c["symbol"],
                                 price=round(c["price"], 2), vol_ratio=round(c["vr"], 2), rsi=round(c["rsi"], 1),
                                 action=tag))
    elif market_ok:
        lines += ["", "Aaj koi BUY signal nahi."]
    else:
        lines += ["", f"Signals mile: {len(cands)} (filter ki wajah se nahi lene)" if cands else "Koi signal nahi."]
    lines += ["", f"Rule: {pt.inr(pt.PER_TRADE)}/trade, max {pt.MAX_POS}, SL 5%, 20 din hold.",
              "<i>3 baje ka data hai - close tak signal badal sakta hai. Paper bot final entry close pe karega.</i>"]
    pt.telegram("\n".join(lines))
    print(f"::notice title=3PM result::market_ok={market_ok} signals={len(cands)} sell_today={len(sell_today)}")
    if not test:
        pt.append_csv(LOG_F, log_rows)
        meta = json.loads(pt.META_F.read_text()) if pt.META_F.exists() else {}
        meta["preclose_sent"] = str(today.date())
        pt.META_F.write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
