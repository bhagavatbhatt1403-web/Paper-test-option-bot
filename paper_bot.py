"""
BTC Paper Trading Bot (separate Telegram bot)
=============================================
Copies every signal from the Signal Bot (signals.json) and trades it with
FAKE money, using LIVE Delta Exchange India prices:

  * Opens a virtual SELL at the signal premium (best bid) minus fee + GST.
  * Every hour checks the live price (best ask = price to buy back):
      - Stop loss : price >= 200% of entry premium  -> close
      - Target    : price <= 25% of entry premium   -> close (75% decay)
      - Expiry    : settles at BTC price at 12:00 UTC (5:30 PM IST)
  * You can also PASTE a signal message into this bot's chat; it is picked up
    on the next hourly run (or a manual run). Send "report" to get the P/L.
  * Sends you: trade opened, trade closed with P/L, and a daily summary.

No real orders are ever placed. Prices are checked once an hour, so a fast
spike between checks can be missed; real results can be worse.
"""
import os
import sys
import re
import json
import html
import logging
from datetime import datetime, timezone, timedelta

import requests

# ============================== SETTINGS =================================
LOTS_PER_TRADE = 10           # 10 lots x 0.001 BTC = 0.01 BTC per trade
START_CAPITAL = 100.0         # virtual starting money in USD
ONE_PER_SIDE = True           # max one open PUT and one open CALL at a time
DAILY_SUMMARY_HOUR_IST = 21   # daily P/L summary on the run after 9 PM IST
AUTO_COPY = False             # True = copy signals automatically from signals.json
READ_PASTED = True            # True = also read signal messages you paste into this bot's chat

FEE_RATE_NOTIONAL = 0.0003    # Delta India option fee: 0.03% of notional (taker)
FEE_CAP_PREMIUM = 0.035       # ...capped at 3.5% of premium
GST = 0.18                    # 18% GST on fees

PAPER_BOT_TOKEN = os.getenv("PAPER_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
FORCE_REPORT = os.getenv("FORCE_SEND", "0") == "1"   # manual run -> always send summary
DRY_RUN = os.getenv("DRY_RUN", "0") == "1"

HERE = os.path.dirname(os.path.abspath(__file__))
SIGNALS_FILE = os.path.join(HERE, "signals.json")
PAPER_FILE = os.path.join(HERE, "paper_state.json")

DELTA_URL = "https://api.india.delta.exchange/v2/tickers"
BINANCE_URL = "https://data-api.binance.vision/api/v3/klines"
IST = timezone(timedelta(hours=5, minutes=30))
HTTP_TIMEOUT = 20

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("paper")
session = requests.Session()
session.headers.update({"User-Agent": "btc-paper-bot/1.0"})


# ============================== HELPERS ==================================
def fnum(x, default=None):
    try:
        v = float(x)
        return v if v == v else default
    except (TypeError, ValueError):
        return default


def load(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default


def save(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=1)


def fee(price, spot):
    """Delta India fee (incl. GST) per 1 BTC to trade an option at this price."""
    if not price or price <= 0:
        return 0.0
    f = FEE_RATE_NOTIONAL * (spot or 0)
    f = min(f, FEE_CAP_PREMIUM * price) if f else FEE_CAP_PREMIUM * price
    return f * (1 + GST)


def ist(ts_iso):
    return datetime.fromisoformat(ts_iso).astimezone(IST).strftime("%d %b %I:%M %p")


def send(text):
    if DRY_RUN or not PAPER_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("----- PAPER BOT (dry run) -----\n" + text)
        return
    url = f"https://api.telegram.org/bot{PAPER_BOT_TOKEN}/sendMessage"
    r = session.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text,
                                "parse_mode": "HTML", "disable_web_page_preview": True},
                     timeout=HTTP_TIMEOUT)
    if r.status_code == 400 and "parse" in r.text:
        import re
        plain = html.unescape(re.sub(r"<[^>]+>", "", text))
        r = session.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": plain},
                         timeout=HTTP_TIMEOUT)
    if r.status_code != 200:
        raise RuntimeError(f"Telegram error {r.status_code}: {r.text[:300]}")


def live_prices():
    """symbol -> {ask, mark, spot} for all live Delta BTC options."""
    r = session.get(DELTA_URL, params={"contract_types": "call_options,put_options",
                                       "underlying_asset_symbols": "BTC"}, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    out = {}
    for t in r.json().get("result", []):
        q = t.get("quotes") or {}
        out[t["symbol"]] = {"ask": fnum(q.get("best_ask")), "bid": fnum(q.get("best_bid")),
                            "mark": fnum(t.get("mark_price")),
                            "spot": fnum(t.get("spot_price"))}
    return out


def btc_at(when):
    """BTC price (Binance 1-minute open) at a given UTC time = settlement estimate."""
    ms = int(when.timestamp() * 1000)
    r = session.get(BINANCE_URL, params={"symbol": "BTCUSDT", "interval": "1m",
                                         "startTime": ms, "limit": 1}, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    k = r.json()
    return float(k[0][1]) if k else None


def usd(x):
    return f"-${abs(x):,.2f}" if x < 0 else f"+${x:,.2f}"


# ============================ PAPER TRADING ==============================
def open_new(ps, signals):
    traded = {p["symbol"] for p in ps["open"]} | {p["symbol"] for p in ps["closed"]}
    seen = set(ps.get("seen_keys", []))
    msgs = []
    for sig in signals:
        if sig["key"] in seen:
            continue
        seen.add(sig["key"])
        for leg in sig["legs"]:
            if leg["symbol"] in traded:
                continue
            if datetime.fromisoformat(leg["expiry"]) <= datetime.now(timezone.utc):
                continue
            if ONE_PER_SIDE and any(p["type"] == leg["type"] for p in ps["open"]):
                log.info("Skip %s: a %s is already open", leg["symbol"], leg["type"])
                continue
            pos = {
                "symbol": leg["symbol"], "type": leg["type"], "strike": leg["strike"],
                "expiry": leg["expiry"], "entry": leg["premium"], "entry_fee": leg["fee"],
                "sl": leg["sl"], "target": leg["target"],
                "cv": leg.get("contract_value", 0.001), "lots": LOTS_PER_TRADE,
                "opened": sig["time"], "btc_at_entry": sig.get("btc"),
            }
            ps["open"].append(pos)
            traded.add(pos["symbol"])
            size = pos["cv"] * pos["lots"]
            msgs.append(
                f"📝 <b>Paper trade OPENED</b>\n"
                f"SELL <code>{html.escape(pos['symbol'])}</code> | {pos['lots']} lots ({size:g} BTC)\n"
                f"Entry premium: ${pos['entry']:,.1f} (you receive ${pos['entry']*size:,.2f})\n"
                f"Stop loss: ${pos['sl']:,.1f} | Target: ${pos['target']:,.1f}\n"
                f"Expiry: {ist(pos['expiry'])} IST")
    ps["seen_keys"] = list(seen)[-500:]
    return msgs


SIG_RE = re.compile(r"SELL\s+(PUT|CALL)\s+([CP]-BTC-(\d+)-(\d{6}))")


def read_pasted(ps, prices):
    """Read signal messages you pasted into the paper bot chat (Telegram keeps them 24h)."""
    if not PAPER_BOT_TOKEN:
        return []
    url = f"https://api.telegram.org/bot{PAPER_BOT_TOKEN}/getUpdates"
    r = session.get(url, params={"offset": ps.get("tg_offset", 0), "timeout": 0}, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    sigs = []
    for u in r.json().get("result", []):
        ps["tg_offset"] = u["update_id"] + 1
        m = u.get("message") or u.get("channel_post") or {}
        if str(m.get("chat", {}).get("id")) != str(TELEGRAM_CHAT_ID):
            continue                      # ignore strangers
        text = m.get("text") or m.get("caption") or ""
        found = SIG_RE.search(text)       # first SELL line = main strike (backups ignored)
        if not found:
            if text.strip().lower() in ("/report", "report", "p/l", "pl"):
                ps["want_report"] = True
            continue
        side, sym, strike, ddmmyy = found.groups()
        q = prices.get(sym, {})
        mp = re.search(r"Premium:\s*\$([\d,.]+)", text[found.end():])
        live_bid = q.get("bid")
        prem = live_bid if live_bid and live_bid > 0 else (float(mp.group(1).replace(",", "")) if mp else None)
        if not prem:
            sigs.append({"error": f"{sym}: no live price, trade skipped"})
            continue
        exp = datetime.strptime(ddmmyy, "%d%m%y").replace(hour=12, tzinfo=timezone.utc)
        sigs.append({"key": f"PASTE|{u['update_id']}", "time": datetime.now(timezone.utc).isoformat(),
                     "btc": q.get("spot"), "legs": [{
                         "label": f"SELL {side}", "symbol": sym, "type": side.lower(),
                         "strike": float(strike), "expiry": exp.isoformat(), "premium": prem,
                         "fee": fee(prem, q.get("spot")), "sl": prem * 2, "target": prem * 0.25,
                         "contract_value": 0.001}]})
    return sigs


def close(ps, pos, exit_price, reason, spot, settled=False):
    size = pos["cv"] * pos["lots"]
    exit_fee = 0.0 if (settled and exit_price == 0) else fee(exit_price, spot)
    pnl = ((pos["entry"] - exit_price) - pos["entry_fee"] - exit_fee) * size
    pos.update(exit=exit_price, exit_fee=exit_fee, reason=reason, pnl=round(pnl, 4),
               closed=datetime.now(timezone.utc).isoformat())
    ps["open"].remove(pos)
    ps["closed"].append(pos)
    ps["capital"] = round(ps["capital"] + pnl, 4)
    icon = "✅" if pnl >= 0 else "❌"
    return (f"{icon} <b>Paper trade CLOSED: {reason}</b>\n"
            f"<code>{html.escape(pos['symbol'])}</code> | sold ${pos['entry']:,.1f} → bought back ${exit_price:,.1f}\n"
            f"P/L (after fees): <b>{usd(pnl)}</b>\n"
            f"Virtual balance: ${ps['capital']:,.2f}")


def check_open(ps, prices):
    msgs, now = [], datetime.now(timezone.utc)
    for pos in list(ps["open"]):
        exp = datetime.fromisoformat(pos["expiry"])
        if now >= exp:
            s = btc_at(exp)
            if s is None:
                continue
            k = pos["strike"]
            intrinsic = max(k - s, 0) if pos["type"] == "put" else max(s - k, 0)
            reason = f"Expired (BTC ${s:,.0f})" + (" worthless 🎉" if intrinsic == 0 else " in the money")
            msgs.append(close(ps, pos, intrinsic, reason, s, settled=True))
            continue
        q = prices.get(pos["symbol"])
        if not q:
            continue
        price = q["ask"] if q["ask"] and q["ask"] > 0 else q["mark"]
        if price is None:
            continue
        pos["last"] = price
        if price >= pos["sl"]:
            msgs.append(close(ps, pos, price, "Stop loss hit 🛑", q["spot"]))
        elif price <= pos["target"]:
            msgs.append(close(ps, pos, price, "Target hit 🎯", q["spot"]))
    return msgs


def summary(ps):
    closed = ps["closed"]
    wins = [p for p in closed if p["pnl"] >= 0]
    realized = sum(p["pnl"] for p in closed)
    open_pnl = sum(((p["entry"] - p.get("last", p["entry"])) - p["entry_fee"]) * p["cv"] * p["lots"]
                   for p in ps["open"])
    today = datetime.now(IST).date()
    today_pnl = sum(p["pnl"] for p in closed
                    if datetime.fromisoformat(p["closed"]).astimezone(IST).date() == today)
    win_rate = (len(wins) / len(closed) * 100) if closed else 0
    worst = min((p["pnl"] for p in closed), default=0)
    msg = (f"📊 <b>Paper Trading Report</b>\n<i>{datetime.now(IST):%d %b %Y, %I:%M %p} IST</i>\n\n"
           f"Today's P/L: <b>{usd(today_pnl)}</b>\n"
           f"Total realized P/L: <b>{usd(realized)}</b>\n"
           f"Open trades (unrealized): {usd(open_pnl)}\n"
           f"Virtual balance: ${ps['capital']:,.2f} (started ${START_CAPITAL:,.0f})\n\n"
           f"Trades closed: {len(closed)} | Wins: {len(wins)} | Win rate: {win_rate:.0f}%\n"
           f"Biggest loss: {usd(worst)}\n")
    if ps["open"]:
        msg += "\n<b>Open now:</b>\n"
        for p in ps["open"]:
            msg += (f" • <code>{html.escape(p['symbol'])}</code> sold ${p['entry']:,.1f}, "
                    f"now ${p.get('last', p['entry']):,.1f}\n")
    msg += "\n<i>Paper trading only. Fake money, live prices, checked hourly.</i>"
    return msg


# ================================ MAIN ===================================
def main():
    ps = load(PAPER_FILE, None) or {"capital": START_CAPITAL, "open": [], "closed": [],
                                    "seen_keys": [], "started": datetime.now(timezone.utc).isoformat()}
    signals = load(SIGNALS_FILE, [])
    first_run = "last_summary" not in ps

    prices = live_prices()
    msgs = []
    if READ_PASTED:
        pasted = read_pasted(ps, prices)
        for e in [x for x in pasted if "error" in x]:
            msgs.append(f"⚠️ {html.escape(e['error'])}")
        signals = (signals if AUTO_COPY else []) + [x for x in pasted if "error" not in x]
    elif not AUTO_COPY:
        signals = []
    msgs += open_new(ps, signals)
    if ps["open"]:
        msgs += check_open(ps, prices)

    now_ist = datetime.now(IST)
    today = now_ist.date().isoformat()
    if (FORCE_REPORT or first_run or ps.pop("want_report", False) or
            (now_ist.hour >= DAILY_SUMMARY_HOUR_IST and ps.get("last_summary") != today)):
        msgs.append(summary(ps))
        ps["last_summary"] = today

    for m in msgs:
        send(m)
    ps["last_run"] = datetime.now(timezone.utc).isoformat()
    save(PAPER_FILE, ps)
    log.info("Done: %d open, %d closed, balance %.2f", len(ps["open"]), len(ps["closed"]), ps["capital"])


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.exception("Paper bot failed")
        try:
            err = str(e).replace(PAPER_BOT_TOKEN or "~~", "***")
            send(f"⚠️ <b>Paper bot error</b>\n<code>{html.escape(err)[:3000]}</code>")
        except Exception:
            pass
        sys.exit(1)
