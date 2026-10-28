"""
Reliance PAPER TRADE (live) -- 2 strategies, 1 position, Rs 1 lakh
------------------------------------------------------------------
Strategies (book.py me rules):
  MOM = 20-Min Momentum + 2xATR trailing SL (BUY/SELL)
  PBS = VWAP Pullback Short (sirf SELL, fixed SL + 2R target)
Kaun si chalani hai: env PAPER_STRATEGIES="MOM,PBS" (ya sirf "PBS" / "MOM").

Har run:
  1. Angel One se Reliance + Nifty ke 5-min candle (sirf poori band candles).
  2. Aaj ka din shuru se dobara simulate (book.simulate_day) -> run late/miss ho to bhi same result.
  3. Har trade ka ek ID hota hai (strategy + entry time). state.json me likha rehta hai
     ki kaunsa alert ja chuka -> koi alert / entry DUPLICATE nahi hoti.
  4. Band trade paper/paper_trades.csv me (ek ID ek baar).

Modes:
  python reliance_strategy/paper_live.py            # ek baar check
  python reliance_strategy/paper_live.py --loop     # market band hone tak har 5 min (GitHub job)
  python reliance_strategy/paper_live.py --replay 2026-10-15   # purane data pe din replay (dry)
  python reliance_strategy/paper_live.py --hello    # Telegram test
"""
import argparse
import json
import os
import subprocess
import sys
import time
import unicodedata
from datetime import datetime, timedelta

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from book import CAPITAL, NAMES, pending_signal, prepare_all, simulate_day  # noqa: E402
from final_strategy import EXIT, TRAIL_ATR  # noqa: E402

PAPER = os.path.join(HERE, "paper")
STATE = os.path.join(PAPER, "state.json")
TRADES = os.path.join(PAPER, "paper_trades.csv")
STRATS = [s.strip().upper() for s in os.getenv("PAPER_STRATEGIES", "MOM,PBS").split(",") if s.strip()]
SL_ALERT_STEP = 1.0          # MOM ka trailing SL itna (x ATR) khiske tab update alert
TOKENS = {"RELIANCE": "2885", "NIFTY": "99926000"}
COLS = ["id", "date", "strategy", "side", "entry_time", "entry", "exit_time", "exit", "reason",
        "qty", "ret_pct", "pnl_rs"]


def _env(name):
    return "".join(ch for ch in os.getenv(name, "").strip() if unicodedata.category(ch) != "Cf")


def ist_now():
    return datetime.utcnow() + timedelta(hours=5, minutes=30)


# ------------------------------------------------------------------ data
_api = None


def angel():
    global _api
    if _api is None:
        from SmartApi import SmartConnect
        import pyotp
        api = SmartConnect(api_key=_env("ANGEL_API_KEY"))
        login = api.generateSession(_env("ANGEL_CLIENT_ID"), _env("ANGEL_PASSWORD"),
                                    pyotp.TOTP(_env("ANGEL_TOTP_SECRET")).now())
        if not login or not login.get("status"):
            raise RuntimeError(f"Angel One login fail: {login and login.get('message')}")
        _api = api
    return _api


def fetch(sym, now):
    global _api
    params = {"exchange": "NSE", "symboltoken": TOKENS[sym], "interval": "FIVE_MINUTE",
              "fromdate": (now - timedelta(days=7)).strftime("%Y-%m-%d 09:15"),
              "todate": now.strftime("%Y-%m-%d %H:%M")}
    for attempt in range(4):
        try:
            rows = (angel().getCandleData(params) or {}).get("data") or []
            if rows:
                d = pd.DataFrame(rows, columns=["datetime", "open", "high", "low", "close", "volume"])
                d["datetime"] = pd.to_datetime(d["datetime"]).dt.tz_localize(None)
                return d.drop_duplicates("datetime").set_index("datetime").sort_index()
        except Exception as e:
            print(f"{sym} fetch retry {attempt + 1}:", e)
            if attempt == 1:
                _api = None          # dobara login
        time.sleep(3)
    raise RuntimeError(f"{sym}: Angel One se candle nahi mile")


def completed(d, now):
    return d[d.index + pd.Timedelta(minutes=5) <= now]


# ----------------------------------------------------------- telegram
def tg(text, dry=False):
    print("TELEGRAM >>", text.replace("\n", " | "))
    tok, chat = _env("TELEGRAM_BOT_TOKEN").strip("\"'"), _env("TELEGRAM_CHAT_ID").strip("\"'")
    if tok.lower().startswith("bot") and ":" in tok:
        tok = tok[3:]
    if dry or not (tok and chat):
        return
    import requests
    err = ""
    for attempt in range(3):
        try:
            data = {"chat_id": chat, "text": text}
            if attempt < 2:
                data["parse_mode"] = "HTML"      # 3rd try: plain text (HTML galti ho to bhi msg jaye)
            r = requests.post(f"https://api.telegram.org/bot{tok}/sendMessage", data=data, timeout=15)
            if r.status_code == 200:
                print("Telegram sent OK")
                return True
            err = f"HTTP {r.status_code}: {r.text[:200]}"
        except Exception as e:
            err = repr(e)[:200]
        print("Telegram fail:", err)
        time.sleep(3)
    try:  # fail ho to repo me log rahe (git_save ise commit karta hai)
        os.makedirs(PAPER, exist_ok=True)
        with open(os.path.join(PAPER, "telegram_errors.log"), "a") as f:
            f.write(f"{ist_now():%Y-%m-%d %H:%M} {err} | {text[:60]!r}\n")
    except Exception:
        pass
    return False


def hhmm(ts):
    return pd.Timestamp(ts).strftime("%H:%M")


def side_txt(s):
    return "🟢 BUY" if s == "BUY" else "🔴 SELL"


# --------------------------------------------------------------- files
def load_state(today):
    try:
        with open(STATE) as f:
            st = json.load(f)
    except Exception:
        st = {}
    if st.get("date") != str(today) or "trades" not in st:
        st = {"date": str(today), "trades": {}}
    return st


def save_state(st):
    os.makedirs(PAPER, exist_ok=True)
    with open(STATE, "w") as f:
        json.dump(st, f, indent=2)


def load_trades():
    if not os.path.exists(TRADES):
        return pd.DataFrame(columns=COLS)
    df = pd.read_csv(TRADES)
    if "id" not in df.columns:               # purana format (sirf MOM, qty 100) -> naya format
        df["strategy"] = "MOM"
        df["id"] = "MOM-" + pd.to_datetime(df.entry_time).dt.strftime("%Y%m%d-%H%M")
    return df.reindex(columns=COLS)


def record_trade(p, today):
    os.makedirs(PAPER, exist_ok=True)
    df = load_trades()
    if (df["id"] == p["id"]).any():          # pehle se likha hai -> duplicate nahi
        return df
    row = {k: p.get(k) for k in COLS}
    row["date"] = str(today)
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(TRADES, index=False)
    return df


def write_status(df):
    lines = [f"## Reliance paper trade status (capital ₹{CAPITAL:,.0f}, ek position)\n\n",
             f"Strategies: {', '.join(NAMES[s] for s in STRATS)}\n\n"]
    if df.empty:
        lines.append("Abhi tak koi paper trade band nahi hua.\n")
    else:
        def line(name, g, bt="—"):
            p = g.pnl_rs.astype(float)
            w = int((p > 0).sum())
            pf = p[p > 0].sum() / -p[p < 0].sum() if (p < 0).any() else float("inf")
            aw = f"₹{p[p > 0].mean():+,.0f}" if w else "—"
            al = f"₹{p[p < 0].mean():+,.0f}" if (p < 0).any() else "—"
            return (f"| {name} | {len(g)} | {w} / {len(g) - w} | **{w / len(g) * 100:.0f}%** | {bt} | ₹{p.sum():+,.0f} | "
                    f"{aw} / {al} | {pf:.2f} |\n")
        lines.append("| Strategy | Trades | Jeet / Haar | Win % (paper) | Win % (backtest) | P&L | Avg jeet / haar | Profit Factor |\n"
                     "|---|---|---|---|---|---|---|---|\n")
        for s in ["MOM", "PBS"]:
            g = df[df.strategy == s]
            if len(g):
                lines.append(line(NAMES[s], g, BT_WIN[s]))
        lines.append(line("**TOTAL**", df))
        lines.append(f"\n**Lagatar loss abhi:** " + " | ".join(
            f"{NAMES[x]}: {loss_streak(df[df.strategy == x])}" for x in ["MOM", "PBS"]) +
            f" | Total: {loss_streak(df)}  (alert {STREAK_ALERT} pe)\n")
        lines.append("\n_26 Oct ka MOM trade purane version (bina filter) ka tha._\n")
        lines.append("\n### Last 20 trades\n\n" + df.tail(20).drop(columns=["id"]).to_markdown(index=False) + "\n")
    with open(os.path.join(PAPER, "STATUS.md"), "w") as f:
        f.write("".join(lines))


BT_WIN = {"MOM": "44% (1 saal) / 36% (3 saal)", "PBS": "52% (1 saal) / 46% (3 saal)"}   # backtest win rate


STREAK_ALERT = int(os.getenv("STREAK_ALERT", "5"))
BT_STREAK = {"MOM": "1 saal me max 6, 3 saal me max 22", "PBS": "1 saal me max 4, 3 saal me max 10",
             "ALL": "1 saal me max 8, 3 saal me max 10"}


def loss_streak(g):
    """Aakhri se peeche ginti: lagatar kitne trade loss me."""
    n = 0
    for v in reversed(g.pnl_rs.astype(float).tolist()):
        if v < 0:
            n += 1
        else:
            break
    return n


def streak_alerts(df, strat):
    msgs = []
    k = loss_streak(df[df.strategy == strat])
    if k >= STREAK_ALERT:
        g = df[df.strategy == strat].tail(k)
        msgs.append(f"⚠️ <b>RELIANCE PAPER: {NAMES[strat]} — lagatar {k} trade LOSS</b>\n"
                    f"In {k} trade ka loss: ₹{g.pnl_rs.astype(float).sum():,.0f}\n"
                    f"Backtest record: {BT_STREAK[strat]}\n"
                    f"{'Ye backtest ki seema se upar hai — strategy rokne pe sochna chahiye.' if k > 6 and strat == 'MOM' or k > 4 and strat == 'PBS' else 'Ye backtest jaisa hi hai, par dhyan rakho.'}")
    k = loss_streak(df)
    if k >= STREAK_ALERT and k != loss_streak(df[df.strategy == strat]):
        msgs.append(f"⚠️ <b>RELIANCE PAPER: dono strategy milake lagatar {k} trade LOSS</b>\n"
                    f"Loss: ₹{df.tail(k).pnl_rs.astype(float).sum():,.0f}\nBacktest record: {BT_STREAK['ALL']}")
    return msgs


def _one(g):
    p = g.pnl_rs.astype(float)
    w = int((p > 0).sum())
    return f"{len(g)} trades, {w} jeet / {len(g) - w} haar = {w / len(g) * 100:.0f}% win, ₹{p.sum():+,.0f}"


def totals_line(df):
    """Har strategy ka win rate ALAG, phir total."""
    lines = ["<b>Ab tak:</b>"]
    for s in ["MOM", "PBS"]:
        g = df[df.strategy == s]
        if len(g):
            lines.append(f"• {NAMES[s]}: {_one(g)}  (backtest win {BT_WIN[s]})")
    lines.append(f"• Total: {_one(df)} ({df.pnl_rs.astype(float).sum() / CAPITAL * 100:+.2f}%)")
    return "\n".join(lines)


# ------------------------------------------------------------------ core
def run(rel, nif, now, dry=False):
    today = now.date()
    rel, nif = completed(rel, now), completed(nif, now)
    if rel.empty or rel.index[-1].date() != today:
        print(f"{now:%Y-%m-%d %H:%M} -- aaj ka koi candle nahi (holiday / market abhi khula nahi).")
        return False
    d = prepare_all(rel, nif)
    if "MOM" not in STRATS:
        d["signal"] = 0
    if "PBS" not in STRATS:
        d["pb_filter"] = False
    t = d[d.day == pd.Timestamp(today)]
    trades = simulate_day(t)
    st = load_state(today)
    last = t.iloc[-1]
    print(f"{now:%H:%M} IST | last candle {t.index[-1]:%H:%M} close {last.close:.2f} | "
          f"move {last.move:+.2f} ATR | RSI {last.rsi14:.1f} | EMA20 {last.ema20:.2f} | VWAP {last.vwap:.2f} | ADX {last.adx:.1f} | Nifty15m {last.nret3:+.2f}% | "
          f"trades: {[(p['id'], p['status']) for p in trades]} | pending: {pending_signal(t)}")

    changed = False
    for p in trades:
        s = st["trades"].setdefault(p["id"], {})
        if not s.get("entry_sent"):
            risk = abs(p["entry"] - p["init_sl"])
            tgt = (f"Target: ₹{p['tp']} (2R)" if p["strategy"] == "PBS"
                   else f"Target: nahi — trailing SL (2×ATR = ₹{TRAIL_ATR * p['atr']:.2f})")
            tg(f"<b>RELIANCE PAPER {side_txt(p['side'])}</b>  [{NAMES[p['strategy']]}]\n"
               f"Entry: ₹{p['entry']} ({hhmm(p['entry_time'])}) × {p['qty']} shares = ₹{p['entry'] * p['qty']:,.0f}\n"
               f"Stop loss: ₹{p['init_sl']} (risk ₹{risk * p['qty']:,.0f})\n{tgt}\n"
               f"Signal candle: {hhmm(p['signal_time'])}", dry)
            s.update(entry_sent=True, last_sl=p["init_sl"])
            changed = True
        if p["status"] == "open" and p["strategy"] == "MOM" \
                and abs(p["sl"] - s.get("last_sl", p["init_sl"])) >= SL_ALERT_STEP * p["atr"]:
            sgn = 1 if p["side"] == "BUY" else -1
            locked = sgn * (p["sl"] - p["entry"]) * p["qty"]
            tg(f"<b>RELIANCE PAPER SL update</b> {side_txt(p['side'])}  [{NAMES['MOM']}]\n"
               f"Naya SL: ₹{p['sl']} (pehle ₹{s.get('last_sl')})\n"
               f"Abhi price: ₹{last.close:.2f} | Open P&L: ₹{sgn * (last.close - p['entry']) * p['qty']:+,.0f}\n"
               f"{'Profit lock: ₹%+.0f' % locked if locked > 0 else 'Abhi profit lock nahi'}", dry)
            s["last_sl"] = p["sl"]
            changed = True
        if p["status"] == "closed" and not s.get("exit_sent"):
            df = record_trade(p, today)
            tg(f"<b>RELIANCE PAPER EXIT</b> {side_txt(p['side'])}  [{NAMES[p['strategy']]}]\n"
               f"₹{p['entry']} → ₹{p['exit']} ({hhmm(p['exit_time'])}) × {p['qty']}\n"
               f"Wajah: {p['reason']}\n"
               f"Result: ₹{p['pnl_rs']:+,.0f} ({p['ret_pct']:+.2f}%, charges ke baad)\n{totals_line(df)}", dry)
            write_status(df)
            if p["pnl_rs"] < 0:
                for m in streak_alerts(df, p["strategy"]):
                    tg(m, dry)
            s["exit_sent"] = True
            changed = True

    if last.tod >= EXIT and not st.get("day_end_sent"):
        if not trades:
            tg("RELIANCE paper: aaj koi signal nahi aaya (dono strategy).", dry)
        else:
            day_pnl = sum(p.get("pnl_rs") or 0 for p in trades if p["status"] == "closed")
            tg(f"<b>RELIANCE paper — din khatam</b>\nAaj {len(trades)} trade, P&L ₹{day_pnl:+,.0f}\n"
               f"{totals_line(load_trades())}", dry)
        st["day_end_sent"] = True
        changed = True
    if changed or not os.path.exists(STATE):
        save_state(st)
    return True


def git_save():
    rel = os.path.relpath(PAPER, os.path.dirname(HERE))
    cmds = [f'git add -f {rel}',
            'git diff --cached --quiet || git commit -q -m "Reliance paper update $(TZ=Asia/Kolkata date \'+%F %H:%M\') [skip ci]"',
            'for i in 1 2 3; do git pull -q --rebase origin main && git push -q && break; sleep 5; done']
    for c in cmds:
        subprocess.run(c, shell=True, cwd=os.path.dirname(HERE))


def loop():
    started = time.time()
    st_now = ist_now()
    if st_now.hour * 60 + st_now.minute < 8 * 60 + 45:
        print("Market khulne me der hai (8:45 se pehle) -- loop shuru nahi kiya"); return
    max_secs = float(os.getenv("LOOP_MAX_MIN", "345")) * 60      # GitHub job 6h limit se pehle
    while True:
        now = ist_now()
        if now.weekday() >= 5 or now.hour * 60 + now.minute > 15 * 60 + 45:
            print("Market band -- loop khatam"); break
        if time.time() - started > max_secs:
            print("Job time limit -- agla job sambhalega"); break
        if now.hour * 60 + now.minute >= 9 * 60 + 20:
            try:
                run(fetch("RELIANCE", now), fetch("NIFTY", now), now)
                git_save()
            except Exception as e:
                print("RUN ERROR:", repr(e))
        nxt = (now.replace(second=0, microsecond=0) + timedelta(minutes=5 - now.minute % 5, seconds=20))
        time.sleep(max(30, (nxt - ist_now()).total_seconds()))


def main():
    global PAPER, STATE, TRADES
    ap = argparse.ArgumentParser()
    ap.add_argument("--hello", action="store_true")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--check", action="store_true", help="sirf Angel login + data check, koi alert/state nahi")
    ap.add_argument("--replay", help="YYYY-MM-DD: data/ CSV pe ek din har 5 min step (dry run)")
    args = ap.parse_args()
    if args.hello:
        tg("<b>✅ RELIANCE PAPER BOOK ACTIVE</b>\n"
           f"Capital: ₹{CAPITAL:,.0f} | Ek waqt pe sirf 1 position (no duplicate entry)\n"
           f"Strategies: {', '.join(NAMES[s] for s in STRATS)}\n"
           "• 20-Min Momentum: move > 1.5×ATR, trailing SL 2×ATR, din me 1\n"
           "   BUY sirf: RSI≤65, price>EMA20, ATR%≥0.17 | SELL sirf: RSI≥35, price<EMA20, VWAP se 1 ATR+ neeche (ya VWAP ke upar)\n"
           "• VWAP Pullback Short: VWAP+EMA20 ke neeche swing high + (Nifty ↑ ya ADX>30), SL 1×ATR, target 2R, din me 2\n"
           "Version: FINAL (loss-analysis filters)\n"
           "Alerts: Entry, SL update, Exit + P&L, din ka summary")
        return
    if args.replay:
        PAPER = "/tmp/reliance_paper_replay"
        STATE, TRADES = os.path.join(PAPER, "state.json"), os.path.join(PAPER, "paper_trades.csv")
        for f in (STATE, TRADES):
            if os.path.exists(f):
                os.remove(f)
        rd = lambda s: pd.read_csv(os.path.join(HERE, "data", f"{s}_5m.csv"), parse_dates=["datetime"]).set_index("datetime")
        rel, nif = rd("RELIANCE"), rd("NIFTY")
        day = pd.Timestamp(args.replay)
        start = day - pd.Timedelta(days=7)
        rel, nif = rel[rel.index >= start], nif[nif.index >= start]
        for m in range(9 * 60 + 20, 15 * 60 + 41, 5):
            now = day + pd.Timedelta(minutes=m)
            run(rel[rel.index < now], nif[nif.index < now], now, dry=True)
        return
    if args.loop:
        loop()
        return
    if args.check:
        now = ist_now()
        rel, nif = fetch("RELIANCE", now), fetch("NIFTY", now)
        d = prepare_all(completed(rel, now), completed(nif, now))
        last = d.iloc[-1]
        print(f"CHECK OK | Reliance {len(rel)} candles, last {d.index[-1]} close {last.close:.2f} "
              f"ATR {last.atr:.2f} VWAP {last.vwap:.2f} ADX {last.adx:.1f} | Nifty {len(nif)} candles")
        return
    now = ist_now()
    run(fetch("RELIANCE", now), fetch("NIFTY", now), now)


if __name__ == "__main__":
    main()
