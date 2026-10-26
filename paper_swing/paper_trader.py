#!/usr/bin/env python3
"""
DAILY SWING - LIVE PAPER TRADING BOT (koi asli order nahi lagta)
===============================================================

Roz market band hone ke baad (GitHub Actions) chalta hai, bilkul backtest wale rules se:

  Universe : Nifty 500, Close 200-3000, Close x Volume > 5 crore
  Market   : Nifty 50 close > Nifty 50 EMA(50), warna naya trade nahi
  Signal   : MACD aaj upar cross, RSI 50-70, AccDist badha, green candle,
             Volume > 2x SMA20 aur > pichhle 10 din ka max, Close > VWAP, EMA8 > EMA21
  Entry    : signal wale din ke CLOSE pe (paper), Rs 20,000 / trade, max 5 khule,
             ek stock me ek hi trade, zyada signal -> sabse zyada volume uchhal pehle
  SL       : entry se 5% neeche. Gap down SL ke neeche khule -> open pe exit
  Exit     : koi target nahi; 20 trading din pure -> us din ke close pe
  Cost     : 0.2% round trip

State files (repo me commit hoti hain): paper_swing/state/
  positions.json      khule trades
  closed_trades.csv   band trades
  signals_log.csv     har din ke saare signals (liye / chhode)
  daily_log.csv       roz ka equity
  STATUS.md           readable summary

Env: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID (optional), PAPER_AS_OF=YYYY-MM-DD (testing/backfill)
"""
import json
import math
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yfinance as yf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import strategy_core as core  # noqa: E402

STATE = Path(os.getenv("PAPER_STATE_DIR", str(HERE / "state")))
STATE.mkdir(parents=True, exist_ok=True)
POS_F = STATE / "positions.json"
CLOSED_F = STATE / "closed_trades.csv"
SIG_F = STATE / "signals_log.csv"
DAILY_F = STATE / "daily_log.csv"
META_F = STATE / "meta.json"
SYMS_F = STATE / "nifty500_symbols.csv"
STATUS_F = STATE / "STATUS.md"

IST = ZoneInfo("Asia/Kolkata")
PER_TRADE = 20_000
MAX_POS = 5
CAPITAL = PER_TRADE * MAX_POS
SL_PCT = 0.05
MAX_HOLD = 20
COST = 0.002

def _clean(v):
    """Secret me copy hue space / newline / invisible chars / quotes / 'bot' prefix hatao."""
    import unicodedata
    v = "".join(c for c in (v or "") if unicodedata.category(c) != "Cf").strip().strip("\"'").strip()
    return v[3:] if v.lower().startswith("bot") and ":" in v else v


TG_TOKEN = _clean(os.getenv("TELEGRAM_BOT_TOKEN"))
TG_CHAT = _clean(os.getenv("TELEGRAM_CHAT_ID"))


# ---------------------------------------------------------------- helpers
def inr(x, sign=False):
    neg = x < 0
    n = str(int(round(abs(x))))
    if len(n) > 3:
        h, t = n[:-3], n[-3:]
        parts = []
        while len(h) > 2:
            parts.insert(0, h[-2:])
            h = h[:-2]
        if h:
            parts.insert(0, h)
        n = ",".join(parts) + "," + t
    return ("-" if neg else ("+" if sign and x > 0 else "")) + "Rs " + n


def telegram(text):
    print("\n----- TELEGRAM -----\n" + text + "\n--------------------", flush=True)
    if not (TG_TOKEN and TG_CHAT):
        print("Telegram: configured nahi (sirf log)")
        return
    try:
        r = requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
                          data={"chat_id": TG_CHAT, "text": text, "parse_mode": "HTML",
                                "disable_web_page_preview": True}, timeout=20)
        print("Telegram:", r.status_code)
    except Exception as e:
        print("Telegram fail:", e)


def load_json(p, default):
    return json.loads(p.read_text()) if p.exists() else default


def append_csv(p, rows):
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(p, mode="a", header=not p.exists(), index=False)


def symbols():
    try:
        s = core.load_symbols(None)
        if len(s) > 400:
            pd.DataFrame({"Symbol": [x.replace(".NS", "") for x in s]}).to_csv(SYMS_F, index=False)
            return s
    except SystemExit:
        pass
    except Exception as e:
        print("Nifty 500 list fail:", e)
    if SYMS_F.exists():
        print("Saved Nifty 500 list use kar rahe hain")
        return core.load_symbols(str(SYMS_F))
    sys.exit("Nifty 500 list nahi mili")


def download_daily(syms):
    data = core.download(syms, days=420)
    nraw = yf.download("^NSEI", period="2y", interval="1d", auto_adjust=True, progress=False)
    nc = nraw["Close"]
    nc = (nc.iloc[:, 0] if isinstance(nc, pd.DataFrame) else nc).dropna()
    return data, nc


# ---------------------------------------------------------------- main
def main():
    now = datetime.now(IST)
    as_of = pd.Timestamp(os.getenv("PAPER_AS_OF") or now.date())
    meta = load_json(META_F, {"last_processed": None, "started": None})
    positions = load_json(POS_F, [])

    print(f"PAPER SWING | as_of {as_of.date()} | khule trades {len(positions)}", flush=True)
    syms = symbols()
    data, nifty = download_daily(syms)
    nifty = nifty[nifty.index <= as_of]
    if nifty.empty:
        sys.exit("Nifty data nahi mila")
    D = nifty.index[-1]
    if meta["last_processed"] and pd.Timestamp(meta["last_processed"]) >= D:
        print(f"{D.date()} pehle hi process ho chuka - kuch nahi karna.")
        return
    if (as_of - D).days > 4:
        telegram(f"⚠️ <b>Paper Swing</b>: latest data {D.date()} ka hai (purana). Aaj skip.")
        return
    nifty_ema = nifty.ewm(span=50, adjust=False).mean()
    market_ok = bool(nifty.iloc[-1] > nifty_ema.iloc[-1])

    prepared = {}
    for s, d in data.items():
        d = d[d.index <= D]
        if len(d) > 80:
            prepared[s.replace(".NS", "")] = core.add_indicators(d)

    msgs_exit, msgs_entry, closed_rows = [], [], []

    # ---------- 1) khule trades update (SL / 20 din)
    still_open = []
    for p in positions:
        d = prepared.get(p["symbol"])
        if d is None:
            still_open.append(p)
            continue
        last_chk = pd.Timestamp(p.get("last_checked", p["entry_date"]))
        new_bars = d[d.index > last_chk]
        exit_px = reason = exit_dt = None
        for dt, bar in new_bars.iterrows():
            p["bars_held"] = int(p.get("bars_held", 0)) + 1
            if bar["Open"] <= p["sl"]:
                exit_px, reason = float(bar["Open"]), "SL (gap down, open pe)"
            elif bar["Low"] <= p["sl"]:
                exit_px, reason = float(p["sl"]), "Stop-loss 5%"
            elif p["bars_held"] >= MAX_HOLD:
                exit_px, reason = float(bar["Close"]), "20 din pure (time exit)"
            if exit_px is not None:
                exit_dt = dt
                break
            p["last_checked"] = str(dt.date())
            p["last_close"] = float(bar["Close"])
        if exit_px is None:
            still_open.append(p)
            continue
        pnl = p["qty"] * (exit_px - p["entry"]) - p["qty"] * p["entry"] * COST
        ret = (exit_px / p["entry"] - 1) * 100
        closed_rows.append(dict(symbol=p["symbol"], entry_date=p["entry_date"], exit_date=str(exit_dt.date()),
                                days_held=p["bars_held"], qty=p["qty"], entry=round(p["entry"], 2),
                                sl=round(p["sl"], 2), exit=round(exit_px, 2), ret_pct=round(ret, 2),
                                pnl=round(pnl, 2), reason=reason))
        icon = "✅" if pnl > 0 else "🔴"
        msgs_exit.append(f"{icon} <b>SELL {p['symbol']}</b> @ {exit_px:.2f}\n"
                         f"   {reason} | {p['bars_held']} din | {ret:+.1f}% | <b>{inr(pnl, True)}</b>")
    positions = still_open

    # ---------- 2) naye signals
    sig_rows = []
    held = {p["symbol"] for p in positions}
    cands = []
    for s, d in prepared.items():
        if d.index[-1] != D or not bool(d["signal"].iloc[-1]):
            continue
        row = d.iloc[-1]
        cands.append(dict(symbol=s, close=float(row["Close"]), vr=float(row["Volume"] / row["vol_sma20"]),
                          rsi=float(row["rsi"])))
    cands.sort(key=lambda x: -x["vr"])
    slots = MAX_POS - len(positions)
    for c in cands:
        status = ""
        if not market_ok:
            status = "skip: Nifty 50 EMA ke neeche"
        elif c["symbol"] in held:
            status = "skip: pehle se khula"
        elif slots <= 0:
            status = "skip: 5 trade already khule"
        else:
            qty = math.floor(PER_TRADE / c["close"])
            if qty < 1:
                status = "skip: price zyada"
            else:
                sl = c["close"] * (1 - SL_PCT)
                positions.append(dict(symbol=c["symbol"], entry_date=str(D.date()), entry=c["close"], qty=qty,
                                      sl=sl, bars_held=0, last_checked=str(D.date()), last_close=c["close"],
                                      vol_ratio=round(c["vr"], 2)))
                held.add(c["symbol"])
                slots -= 1
                status = "BOUGHT (paper)"
                msgs_entry.append(f"🟢 <b>BUY {c['symbol']}</b> @ {c['close']:.2f} × {qty}\n"
                                  f"   SL {sl:.2f} (−5%) | exit: 20 din ya SL | Vol {c['vr']:.1f}× | RSI {c['rsi']:.0f}")
        sig_rows.append(dict(date=str(D.date()), symbol=c["symbol"], close=round(c["close"], 2),
                             vol_ratio=round(c["vr"], 2), rsi=round(c["rsi"], 1), action=status))

    # ---------- 3) save + summary
    append_csv(CLOSED_F, closed_rows)
    append_csv(SIG_F, sig_rows)
    closed = pd.read_csv(CLOSED_F) if CLOSED_F.exists() else pd.DataFrame(columns=["pnl"])
    realized = float(closed["pnl"].sum()) if len(closed) else 0.0
    unreal = sum(p["qty"] * (p["last_close"] - p["entry"]) for p in positions)
    append_csv(DAILY_F, [dict(date=str(D.date()), nifty=round(float(nifty.iloc[-1]), 2),
                              nifty_ema50=round(float(nifty_ema.iloc[-1]), 2), market_ok=market_ok,
                              signals=len(cands), entries=len(msgs_entry), exits=len(msgs_exit),
                              open_positions=len(positions), realized=round(realized), unrealized=round(unreal),
                              total=round(realized + unreal))])
    POS_F.write_text(json.dumps(positions, indent=2))
    meta["last_processed"] = str(D.date())
    meta["started"] = meta["started"] or str(D.date())
    META_F.write_text(json.dumps(meta, indent=2))

    wins = int((closed["pnl"] > 0).sum()) if len(closed) else 0
    lines = [f"📊 <b>Paper Swing | {D:%d %b %Y}</b>",
             f"Nifty {nifty.iloc[-1]:,.0f} vs 50 EMA {nifty_ema.iloc[-1]:,.0f} → "
             + ("✅ trade allowed" if market_ok else "⛔ naya trade nahi")]
    if msgs_exit:
        lines += ["", "<b>EXITS</b>"] + msgs_exit
    if msgs_entry:
        lines += ["", "<b>NEW ENTRIES</b> (close price pe)"] + msgs_entry
    if not msgs_exit and not msgs_entry:
        lines += ["", f"Aaj koi entry/exit nahi. Signals: {len(cands)}"]
    if positions:
        lines += ["", f"<b>KHULE TRADES ({len(positions)}/5)</b>"]
        for p in positions:
            u = p["qty"] * (p["last_close"] - p["entry"])
            lines.append(f"• {p['symbol']}: {(p['last_close']/p['entry']-1)*100:+.1f}% ({inr(u, True)}) | "
                         f"din {p['bars_held']}/20 | SL {p['sl']:.0f}")
    lines += ["", f"Realized: <b>{inr(realized, True)}</b> ({len(closed)} trades, {wins} profit)",
              f"Open P&L: {inr(unreal, True)} | Kul: <b>{inr(realized + unreal, True)}</b> on {inr(CAPITAL)}",
              "<i>Paper trade - koi asli order nahi</i>"]
    telegram("\n".join(lines))
    write_status(D, market_ok, nifty, nifty_ema, positions, closed, realized, unreal, meta)


def write_status(D, market_ok, nifty, ema, positions, closed, realized, unreal, meta):
    md = [f"# Paper Swing - Live Status\n",
          f"**Last update:** {D:%d %b %Y} | **Started:** {meta['started']} | Capital {inr(CAPITAL)} ({inr(PER_TRADE)} × {MAX_POS})\n",
          f"**Nifty:** {nifty.iloc[-1]:,.0f} vs 50 EMA {ema.iloc[-1]:,.0f} → "
          f"{'trade allowed' if market_ok else 'naya trade nahi'}\n",
          f"| Realized | Open P&L | Kul | Closed trades | Win |",
          f"|---|---|---|---|---|",
          f"| {inr(realized, True)} | {inr(unreal, True)} | **{inr(realized + unreal, True)}** | {len(closed)} | "
          f"{int((closed['pnl'] > 0).sum()) if len(closed) else 0} |\n",
          f"## Khule trades ({len(positions)}/5)\n"]
    if positions:
        md += ["| Stock | Entry date | Entry | Qty | SL | Last close | P&L % | Din |", "|---|---|---|---|---|---|---|---|"]
        for p in positions:
            md.append(f"| {p['symbol']} | {p['entry_date']} | {p['entry']:.2f} | {p['qty']} | {p['sl']:.2f} | "
                      f"{p['last_close']:.2f} | {(p['last_close']/p['entry']-1)*100:+.1f}% | {p['bars_held']}/20 |")
    else:
        md.append("Koi nahi.")
    md.append("\n## Band trades (latest 20)\n")
    if len(closed):
        md += ["| Stock | Entry | Exit | Din | Return | P&L | Reason |", "|---|---|---|---|---|---|---|"]
        for r in closed.tail(20).iloc[::-1].itertuples():
            md.append(f"| {r.symbol} | {r.entry_date} | {r.exit_date} | {r.days_held} | {r.ret_pct:+.1f}% | "
                      f"{inr(r.pnl, True)} | {r.reason} |")
    else:
        md.append("Abhi koi nahi.")
    md.append("\n_Paper trade - koi asli order nahi lagta. Rules: paper_swing/paper_trader.py_")
    STATUS_F.write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
