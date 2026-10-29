"""
BTC Option Selling Signal Bot
-----------------------------
Script for GitHub Actions (cron). Signals only, no order execution.
Remembers the last signal in state.json so the same trade is not re-sent every hour.

Data:
  1. Binance  -> BTCUSDT 1h candles: 20/50 EMA, ADX(14), ATR(14)
  2. Deribit  -> PCR (OI), DVOL + IV Rank, 25-delta skew
  3. Delta Exchange India -> BTC option chain for the nearest live expiry
Output:
  Telegram message (HTML) with trend, PCR, volatility, strikes, premium, SL, target.
"""

import os
import sys
import json
import math
import html
import logging
from datetime import datetime, timezone, timedelta

import requests
import pandas as pd

# ============================== CONFIG ===================================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
DRY_RUN = os.getenv("DRY_RUN", "0") == "1"      # 1 = print only, no Telegram

SYMBOL = "BTCUSDT"
INTERVAL = "1h"
KLINE_LIMIT = 300            # enough history for 50 EMA + Wilder smoothing

EMA_FAST = 20
EMA_SLOW = 50
ADX_PERIOD = 14
ATR_PERIOD = 14
ADX_TREND_LEVEL = 25
ADX_EXIT_LEVEL = 20          # trend ends when ADX falls below this
USE_ICHIMOKU = True          # Ichimoku cloud must agree with the trend
ICHI_TENKAN, ICHI_KIJUN, ICHI_SPAN_B, ICHI_SHIFT = 9, 26, 52, 26
TREND_BUFFER_ATR = 0.3        # close must be this many ATRs beyond the 50 EMA to count as a trend

SD_MULTIPLIER = 2.0          # 2 SD boundaries
MAX_ABS_DELTA = 0.10         # option delta must be below this
USE_IV_CHECK = True          # also compute IV-based 2 SD and use the WIDER boundary
MIN_HOURS_TO_EXPIRY = 1.0    # skip an expiry that settles in less than this
MIN_PREMIUM_USD = 5.0        # ignore options whose NET premium after fees (per 1 BTC) is below this
ALT_STRIKES = 2              # also show this many safer (further away) backup strikes
ALT_MIN_PREMIUM_USD = 2.0    # backups may pay less, but at least this much net (per 1 BTC)

# Delta Exchange India option fees (official page, Oct 2026): % of notional, capped at % of premium, + GST
FEE_RATE_NOTIONAL = 0.0003   # 0.03% taker (worst case; maker/promo is 0.01%)
FEE_CAP_PREMIUM = 0.035      # fee never more than 3.5% of the premium
GST = 0.18                   # 18% GST on the fee
# OTM options that expire worthless pay no settlement fee

# Duplicate control: state.json is saved back to the repo by the workflow
STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "state.json")
# Every signal is also written here so the separate Paper Trading bot can copy it
SIGNALS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "signals.json")
SIGNALS_KEEP = 200
HEARTBEAT_HOURS = 24         # if nothing was sent for this long, send a short "bot alive" note
FORCE_SEND = os.getenv("FORCE_SEND", "0") == "1"   # manual run from Actions tab always sends

# Deribit DVOL (BTC implied-volatility index) and 25-delta skew
DVOL_LOOKBACK_DAYS = 365     # window for IV Rank / IV Percentile
IV_RANK_LOW = 20             # below this, premiums are cheap -> warning
BLOCK_LOW_IV = False         # True = send "No trade" when IV Rank < IV_RANK_LOW
SKEW_MIN_DAYS = 1.0          # measure skew on the nearest expiry with >= this many days left
SKEW_FEAR = 5.0              # put IV - call IV above this (vol points) -> skip selling PUTS
SKEW_GREED = -5.0            # put IV - call IV below this -> skip selling CALLS

SL_MULTIPLIER = 2.0          # stop loss when option price = 200% of entry premium
TARGET_DECAY = 0.75          # book profit when 75% of premium has decayed

IST = timezone(timedelta(hours=5, minutes=30))
HTTP_TIMEOUT = 20

BINANCE_URLS = [
    "https://data-api.binance.vision/api/v3/klines",  # public mirror, works from US (GitHub) servers
    "https://api.binance.com/api/v3/klines",          # blocked from US IPs (HTTP 451)
]
DERIBIT_URL = "https://www.deribit.com/api/v2/public/get_book_summary_by_currency"
DERIBIT_DVOL_URL = "https://www.deribit.com/api/v2/public/get_volatility_index_data"
DELTA_URL = "https://api.india.delta.exchange/v2/tickers"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("bot")
session = requests.Session()
session.headers.update({"User-Agent": "btc-option-signal-bot/1.0"})


# ============================== HELPERS ==================================
def get_json(url, params=None):
    r = session.get(url, params=params, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    return r.json()


def fnum(x, default=None):
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


# ========================= 1. BINANCE: TREND & VOL =======================
def fetch_klines():
    last_err = None
    for url in BINANCE_URLS:
        try:
            data = get_json(url, {"symbol": SYMBOL, "interval": INTERVAL, "limit": KLINE_LIMIT})
            cols = ["open_time", "open", "high", "low", "close", "volume", "close_time",
                    "qav", "trades", "tbb", "tbq", "ignore"]
            df = pd.DataFrame(data, columns=cols)
            for c in ["open", "high", "low", "close", "volume"]:
                df[c] = df[c].astype(float)
            df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
            log.info("Binance klines from %s (%d rows)", url, len(df))
            return df
        except Exception as e:  # try next mirror
            last_err = e
            log.warning("Binance source failed (%s): %s", url, e)
    raise RuntimeError(f"All Binance sources failed: {last_err}")


def wilder(series, period):
    """Wilder's smoothing (RMA), same as TradingView's ATR/ADX."""
    return series.ewm(alpha=1 / period, adjust=False).mean()


def add_indicators(df):
    df = df.copy()
    df["ema_fast"] = df["close"].ewm(span=EMA_FAST, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=EMA_SLOW, adjust=False).mean()

    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = wilder(tr, ATR_PERIOD)

    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = ((up > down) & (up > 0)) * up
    minus_dm = ((down > up) & (down > 0)) * down
    atr_adx = wilder(tr, ADX_PERIOD)
    plus_di = 100 * wilder(plus_dm, ADX_PERIOD) / atr_adx
    minus_di = 100 * wilder(minus_dm, ADX_PERIOD) / atr_adx
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, float("nan"))
    df["adx"] = wilder(dx.fillna(0), ADX_PERIOD)
    df["plus_di"], df["minus_di"] = plus_di, minus_di

    # Ichimoku cloud (standard 9/26/52, cloud plotted 26 candles ahead)
    mid = lambda n: (df["high"].rolling(n).max() + df["low"].rolling(n).min()) / 2
    df["tenkan"], df["kijun"] = mid(ICHI_TENKAN), mid(ICHI_KIJUN)
    span_a = ((df["tenkan"] + df["kijun"]) / 2).shift(ICHI_SHIFT)
    span_b = mid(ICHI_SPAN_B).shift(ICHI_SHIFT)
    df["cloud_top"] = pd.concat([span_a, span_b], axis=1).max(axis=1)
    df["cloud_bot"] = pd.concat([span_a, span_b], axis=1).min(axis=1)
    df["cloud_green"] = span_a > span_b
    return df


def ichi_side(r):
    """+1 above cloud, -1 below cloud, 0 inside (= no clear trend)."""
    if r.close > r.cloud_top:
        return 1
    if r.close < r.cloud_bot:
        return -1
    return 0


def _entry(r, prev):
    buf = TREND_BUFFER_ATR * r.atr
    if r.adx <= ADX_TREND_LEVEL:
        return "Sideways"
    if (r.close > r.ema_slow + buf and prev.close > prev.ema_slow
            and r.ema_fast > r.ema_slow and r.plus_di > r.minus_di
            and (not USE_ICHIMOKU or (ichi_side(r) == 1 and r.tenkan > r.kijun))):
        return "Pure Long"
    if (r.close < r.ema_slow - buf and prev.close < prev.ema_slow
            and r.ema_fast < r.ema_slow and r.minus_di > r.plus_di
            and (not USE_ICHIMOKU or (ichi_side(r) == -1 and r.tenkan < r.kijun))):
        return "Pure Short"
    return "Sideways"


def _exit(trend, r, prev):
    """A trend ends only on clear evidence: 2 closes on the wrong side of the 50 EMA, or ADX < 20."""
    if r.adx < ADX_EXIT_LEVEL:
        return True
    if USE_ICHIMOKU:   # 2 closes back inside / through the cloud ends the trend
        want = 1 if trend == "Pure Long" else -1
        if ichi_side(r) != want and ichi_side(prev) != want:
            return True
    if trend == "Pure Long":
        return r.close < r.ema_slow and prev.close < prev.ema_slow
    return r.close > r.ema_slow and prev.close > prev.ema_slow


def classify_trend(df):
    """
    Walks through CLOSED 1h candles with hysteresis, so the trend can't flip-flop:
      enter Pure Long : 2 closes above 50 EMA (last one by 0.3 x ATR), 20 EMA > 50 EMA, +DI > -DI, ADX > 25
                        + close above the Ichimoku cloud and Tenkan > Kijun
      exit            : 2 closes back below 50 EMA, 2 closes back inside the cloud, or ADX < 20
      (Pure Short is the mirror)
    """
    trend = "Sideways"
    rows = list(df.iloc[:-1].itertuples())          # drop the live (unfinished) candle
    for prev, r in zip(rows[EMA_SLOW:-1], rows[EMA_SLOW + 1:]):
        if trend != "Sideways" and _exit(trend, r, prev):
            trend = "Sideways"
        if trend == "Sideways":
            trend = _entry(r, prev)
    return trend


def analyse_trend(df):
    closed = df.iloc[-2]
    price = float(df.iloc[-1]["close"])            # live price, for display / strikes only
    close = float(closed["close"])
    ema_slow, ema_fast, adx = float(closed["ema_slow"]), float(closed["ema_fast"]), float(closed["adx"])
    pdi, mdi = float(closed["plus_di"]), float(closed["minus_di"])
    trend = classify_trend(df)

    atr = float(closed["atr"])
    atr_avg = float(df["atr"].iloc[-52:-2].mean())
    ratio = atr / atr_avg if atr_avg else 1.0
    if ratio > 1.3:
        vol_state = "High (expanding)"
    elif ratio < 0.8:
        vol_state = "Low (contracting)"
    else:
        vol_state = "Normal"

    return {
        "price": price,
        "ema_fast": ema_fast,
        "close": close,
        "cloud_top": float(closed["cloud_top"]),
        "cloud_bot": float(closed["cloud_bot"]),
        "cloud_green": bool(closed["cloud_green"]),
        "tenkan": float(closed["tenkan"]),
        "kijun": float(closed["kijun"]),
        "plus_di": pdi,
        "minus_di": mdi,
        "ema_slow": ema_slow,
        "adx": adx,
        "atr": atr,
        "atr_ratio": ratio,
        "vol_state": vol_state,
        "trend": trend,
    }


def sd_boundaries(price, atr_1h, hours_to_expiry, atm_iv=None):
    """
    ATR method: 1 SD move until expiry ~= ATR(1h) * sqrt(hours to expiry)
    IV method (optional): 1 SD = price * IV * sqrt(days / 365)
    The wider (safer) of the two is used.
    """
    h = max(hours_to_expiry, 1.0)
    sd_atr = atr_1h * math.sqrt(h)
    sd_iv = price * atm_iv * math.sqrt(h / (24 * 365)) if (USE_IV_CHECK and atm_iv) else 0.0
    one_sd = max(sd_atr, sd_iv)
    move = SD_MULTIPLIER * one_sd
    return {
        "sd_atr": sd_atr,
        "sd_iv": sd_iv,
        "one_sd": one_sd,
        "upper": price + move,
        "lower": price - move,
        "method": "IV" if sd_iv > sd_atr else "ATR",
    }


# ================== 2. DERIBIT: PCR, DVOL, 25-DELTA SKEW =================
def parse_deribit_expiry(name):
    # BTC-31OCT26-87000-C -> expiry 08:00 UTC that date
    d = datetime.strptime(name.split("-")[1], "%d%b%y")
    return d.replace(hour=8, tzinfo=timezone.utc)


def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bs_delta(fwd, strike, iv, years, is_call):
    """Black-76 delta (Deribit prices options off the futures/forward price, r ~ 0)."""
    if not (fwd and strike and iv and years > 0):
        return None
    d1 = (math.log(fwd / strike) + 0.5 * iv * iv * years) / (iv * math.sqrt(years))
    return norm_cdf(d1) if is_call else norm_cdf(d1) - 1


def deribit_options():
    data = get_json(DERIBIT_URL, {"currency": "BTC", "kind": "option"})["result"]
    now = datetime.now(timezone.utc)
    rows = []
    for o in data:
        name = o.get("instrument_name", "")
        try:
            exp = parse_deribit_expiry(name)
            strike = float(name.split("-")[2])
        except Exception:
            continue
        if exp <= now:
            continue
        rows.append({
            "expiry": exp,
            "years": (exp - now).total_seconds() / (365 * 86400),
            "type": name[-1],                       # "C" or "P"
            "strike": strike,
            "oi": fnum(o.get("open_interest"), 0.0),
            "iv": fnum(o.get("mark_iv")),           # in % (e.g. 42.2)
            "fwd": fnum(o.get("underlying_price")),
        })
    if not rows:
        raise RuntimeError("No live Deribit options found")
    return rows


def calc_pcr(rows):
    nearest = min(r["expiry"] for r in rows)
    put_oi = sum(r["oi"] for r in rows if r["expiry"] == nearest and r["type"] == "P")
    call_oi = sum(r["oi"] for r in rows if r["expiry"] == nearest and r["type"] == "C")
    all_put = sum(r["oi"] for r in rows if r["type"] == "P")
    all_call = sum(r["oi"] for r in rows if r["type"] == "C")
    pcr = put_oi / call_oi if call_oi else float("nan")
    pcr_all = all_put / all_call if all_call else float("nan")
    if pcr > 1.2:
        mood = "Bearish/hedged (puts heavy)"
    elif pcr < 0.7:
        mood = "Bullish (calls heavy)"
    else:
        mood = "Neutral"
    return {"pcr": pcr, "pcr_all": pcr_all, "mood": mood}


def calc_skew(rows):
    """25-delta skew = IV(25d put) - IV(25d call), in vol points."""
    eligible = [r for r in rows if r["years"] * 365 >= SKEW_MIN_DAYS and r["iv"]]
    if not eligible:
        return None
    exp = min(r["expiry"] for r in eligible)
    chain = []
    for r in eligible:
        if r["expiry"] != exp:
            continue
        d = bs_delta(r["fwd"], r["strike"], r["iv"] / 100, r["years"], r["type"] == "C")
        if d is not None:
            chain.append((r, d))
    calls = [(r, d) for r, d in chain if r["type"] == "C"]
    puts = [(r, d) for r, d in chain if r["type"] == "P"]
    if not calls or not puts:
        return None
    c, cd = min(calls, key=lambda x: abs(x[1] - 0.25))
    p, pd_ = min(puts, key=lambda x: abs(x[1] + 0.25))
    skew = p["iv"] - c["iv"]
    if skew > SKEW_FEAR:
        state = "Fear (puts expensive)"
    elif skew < SKEW_GREED:
        state = "Greed (calls expensive)"
    else:
        state = "Balanced"
    return {"skew": skew, "state": state, "expiry": exp,
            "put_strike": p["strike"], "put_iv": p["iv"],
            "call_strike": c["strike"], "call_iv": c["iv"]}


def calc_dvol():
    """Current DVOL plus IV Rank and IV Percentile over the lookback window."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=DVOL_LOOKBACK_DAYS)
    res = get_json(DERIBIT_DVOL_URL, {
        "currency": "BTC",
        "start_timestamp": int(start.timestamp() * 1000),
        "end_timestamp": int(end.timestamp() * 1000),
        "resolution": "1D",
    })["result"]["data"]                     # [ts, open, high, low, close], oldest first
    if len(res) < 30:
        raise RuntimeError("Not enough DVOL history")
    closes = [float(c[4]) for c in res]
    current = closes[-1]
    hi, lo = max(float(c[2]) for c in res), min(float(c[3]) for c in res)
    iv_rank = (current - lo) / (hi - lo) * 100 if hi > lo else 50.0
    iv_pct = sum(1 for x in closes[:-1] if x < current) / (len(closes) - 1) * 100
    if iv_rank >= 50:
        state = "Rich (good for selling)"
    elif iv_rank >= IV_RANK_LOW:
        state = "Fair"
    else:
        state = "Cheap (poor premiums)"
    return {"dvol": current, "iv_rank": iv_rank, "iv_pct": iv_pct,
            "high": hi, "low": lo, "state": state}


def deribit_sentiment():
    out = {"pcr": float("nan"), "pcr_all": float("nan"), "mood": "Unavailable",
           "skew": None, "dvol": None}
    try:
        rows = deribit_options()
        out.update(calc_pcr(rows))
        out["skew"] = calc_skew(rows)
    except Exception as e:
        log.warning("Deribit options failed: %s", e)
    try:
        out["dvol"] = calc_dvol()
    except Exception as e:
        log.warning("Deribit DVOL failed: %s", e)
    return out


# ====================== 3. DELTA EXCHANGE: STRIKES =======================
def parse_delta_expiry(symbol):
    # C-BTC-87000-291026 -> 29 Oct 2026, settles 12:00 UTC (5:30 PM IST)
    d = datetime.strptime(symbol.split("-")[-1], "%d%m%y")
    return d.replace(hour=12, tzinfo=timezone.utc)


def fetch_delta_chain():
    data = get_json(DELTA_URL, {
        "contract_types": "call_options,put_options",
        "underlying_asset_symbols": "BTC",
    })
    if not data.get("success"):
        raise RuntimeError(f"Delta Exchange error: {data}")

    now = datetime.now(timezone.utc)
    opts = []
    for t in data["result"]:
        try:
            exp = parse_delta_expiry(t["symbol"])
        except Exception:
            continue
        hours_left = (exp - now).total_seconds() / 3600
        if hours_left < MIN_HOURS_TO_EXPIRY:
            continue
        greeks = t.get("greeks") or {}
        quotes = t.get("quotes") or {}
        opts.append({
            "symbol": t["symbol"],
            "type": "call" if t["contract_type"] == "call_options" else "put",
            "strike": fnum(t.get("strike_price")),
            "expiry": exp,
            "hours_left": hours_left,
            "delta": fnum(greeks.get("delta")),
            "mark": fnum(t.get("mark_price")),
            "bid": fnum(quotes.get("best_bid")),
            "ask": fnum(quotes.get("best_ask")),
            "iv": fnum(quotes.get("mark_iv")) or fnum(t.get("mark_vol")),
            "oi": fnum(t.get("oi_contracts"), 0.0),
            "contract_value": fnum(t.get("contract_value"), 0.001),
            "spot": fnum(t.get("spot_price")),
        })
    if not opts:
        raise RuntimeError("No live Delta Exchange BTC options found")

    nearest = min(o["expiry"] for o in opts)
    chain = [o for o in opts if o["expiry"] == nearest and o["strike"] and o["delta"] is not None]
    return nearest, chain


def atm_iv(chain, price):
    with_iv = [o for o in chain if o["iv"]]
    if not with_iv:
        return None
    atm = min(with_iv, key=lambda o: abs(o["strike"] - price))
    return atm["iv"]


def sell_price(o):
    """What you'd realistically receive when selling: best bid, else mark."""
    return o["bid"] if o["bid"] and o["bid"] > 0 else o["mark"]


def entry_fee(premium, spot):
    """Delta India fee to SELL one option (per 1 BTC), incl. GST."""
    fee = FEE_RATE_NOTIONAL * (spot or 0)
    fee = min(fee, FEE_CAP_PREMIUM * premium) if fee else FEE_CAP_PREMIUM * premium
    return fee * (1 + GST)


def enrich(o, price):
    prem = sell_price(o)
    o = dict(o)
    o["premium"] = prem
    o["fee"] = entry_fee(prem, o["spot"] or price)
    o["net"] = prem - o["fee"]
    o["net_per_lot"] = o["net"] * o["contract_value"]
    o["premium_per_lot"] = prem * o["contract_value"]
    o["sl"] = prem * SL_MULTIPLIER
    o["target"] = prem * (1 - TARGET_DECAY)
    spread = (o["ask"] - o["bid"]) if (o["ask"] and o["bid"]) else None
    o["spread_pct"] = (spread / o["ask"] * 100) if spread is not None and o["ask"] else None
    o["oi_btc"] = o["oi"] * o["contract_value"]
    return o


def pick_strike(chain, side, bounds, price):
    """
    Safest-but-still-paying strike, plus safer backups further away:
      PUT : strike <= lower 2SD bound, |delta| < MAX_ABS_DELTA -> HIGHEST such strike, backups below it
      CALL: strike >= upper 2SD bound, |delta| < MAX_ABS_DELTA -> LOWEST such strike, backups above it
    Premium is judged AFTER fees.
    """
    if side == "put":
        cands = [o for o in chain if o["type"] == "put" and o["strike"] <= bounds["lower"]]
        key = lambda o: -o["strike"]
    else:
        cands = [o for o in chain if o["type"] == "call" and o["strike"] >= bounds["upper"]]
        key = lambda o: o["strike"]

    cands = [enrich(o, price) for o in sorted(cands, key=key)
             if abs(o["delta"]) < MAX_ABS_DELTA and (sell_price(o) or 0) > 0]
    main_leg = next((o for o in cands if o["net"] >= MIN_PREMIUM_USD), None)
    if not main_leg:
        return None
    after = cands[cands.index(main_leg) + 1:]
    main_leg["alts"] = [o for o in after if o["net"] >= ALT_MIN_PREMIUM_USD][:ALT_STRIKES]
    return main_leg


# ============================= 4. TELEGRAM ===============================
def send_telegram(text):
    if DRY_RUN or not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("----- TELEGRAM (dry run) -----\n" + text)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    r = session.post(url, json={
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }, timeout=HTTP_TIMEOUT)
    if r.status_code == 400 and "parse" in r.text:
        # Formatting problem: resend as plain text so the alert is never lost
        log.error("Telegram HTML parse error, resending as plain text: %s", r.text)
        import re as _re
        plain = html.unescape(_re.sub(r"<[^>]+>", "", text))
        r = session.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": plain,
                                    "disable_web_page_preview": True}, timeout=HTTP_TIMEOUT)
    if r.status_code != 200:
        log.error("Telegram error %s: %s", r.status_code, r.text)
        raise RuntimeError(f"Telegram error {r.status_code}: {r.text[:300]}")


def fmt_leg(o, label):
    spread = f"{o['spread_pct']:.0f}%" if o["spread_pct"] is not None else "n/a"
    msg = (
        f"<b>{label}</b> <code>{html.escape(o['symbol'])}</code>\n"
        f"  Strike: <b>{o['strike']:,.0f}</b> | Delta: {o['delta']:+.3f}\n"
        f"  Premium: ${o['premium']:,.1f} | Fee+GST: ${o['fee']:,.2f} | <b>Net: ${o['net']:,.1f}</b>\n"
        f"  Per 0.001 BTC lot: net ${o['net_per_lot']:,.4f} (10 lots ≈ ${o['net_per_lot']*10:,.3f})\n"
        f"  Stop loss: ${o['sl']:,.1f} ({SL_MULTIPLIER*100:.0f}% of premium)\n"
        f"  Target: ${o['target']:,.1f} ({TARGET_DECAY*100:.0f}% decay)\n"
        f"  Bid-ask spread: {spread} | OI: {o['oi']:,.0f} lots (≈ {o['oi_btc']:,.0f} BTC)\n"
    )
    if o.get("alts"):
        msg += "  <i>Safer backups (if strike not available):</i>\n"
        for a in o["alts"]:
            msg += (f"   • <code>{html.escape(a['symbol'])}</code> Δ {a['delta']:+.3f} | "
                    f"net ${a['net']:,.1f} | SL ${a['sl']:,.1f}\n")
    msg += "  <i>Never pick a strike CLOSER to price than the one above.</i>\n"
    return msg


def fmt_ichi(t):
    if not USE_ICHIMOKU:
        return ""
    c = t["close"]
    pos = "above cloud 🟢" if c > t["cloud_top"] else ("below cloud 🔴" if c < t["cloud_bot"] else "inside cloud 🟡")
    tk = "Tenkan > Kijun" if t["tenkan"] > t["kijun"] else "Tenkan < Kijun"
    color = "green" if t["cloud_green"] else "red"
    return (f"<b>Ichimoku:</b> {pos} (${t['cloud_bot']:,.0f}–${t['cloud_top']:,.0f}, {color}) | {html.escape(tk)}\n")


def fmt_dvol(d):
    if not d:
        return "<b>DVOL:</b> unavailable\n"
    return (f"<b>DVOL:</b> {d['dvol']:.1f} | IV Rank {d['iv_rank']:.0f} | "
            f"IV Pctl {d['iv_pct']:.0f} → {d['state']}\n")


def fmt_skew(k):
    if not k:
        return "<b>25Δ Skew:</b> unavailable\n"
    exp = k["expiry"].astimezone(IST).strftime("%d %b")
    return (f"<b>25Δ Skew ({exp}):</b> {k['skew']:+.1f} vol pts "
            f"(put {k['put_iv']:.1f} vs call {k['call_iv']:.1f}) → {k['state']}\n")


def build_message(t, p, bounds, expiry, legs, notes=()):
    now_ist = datetime.now(IST).strftime("%d %b %Y, %I:%M %p IST")
    exp_ist = expiry.astimezone(IST).strftime("%d %b, %I:%M %p IST")
    hours_left = (expiry - datetime.now(timezone.utc)).total_seconds() / 3600
    emoji = {"Pure Long": "🟢", "Pure Short": "🔴", "Sideways": "🟡"}[t["trend"]]

    msg = (
        f"<b>📊 BTC Option Selling Signal</b>\n<i>{now_ist}</i>\n\n"
        f"<b>BTC Price:</b> ${t['price']:,.0f}\n"
        f"<b>Trend:</b> {emoji} {t['trend']} (ADX {t['adx']:.1f}, +DI {t['plus_di']:.0f} / -DI {t['minus_di']:.0f})\n"
        f"<b>EMA:</b> 20 = ${t['ema_fast']:,.0f} | 50 = ${t['ema_slow']:,.0f} | last 1h close ${t['close']:,.0f}\n"
        f"{fmt_ichi(t)}"
        f"<b>Volatility:</b> {t['vol_state']} (ATR 1h ${t['atr']:,.0f}, {t['atr_ratio']:.2f}× avg)\n"
        f"<b>Deribit PCR (OI):</b> {p['pcr']:.2f} nearest expiry, {p['pcr_all']:.2f} all expiries → {p['mood']}\n"
        f"{fmt_dvol(p['dvol'])}{fmt_skew(p['skew'])}\n"
        f"<b>Expiry:</b> {exp_ist} ({hours_left:.1f}h left)\n"
        f"<b>2 SD range:</b> ${bounds['lower']:,.0f} – ${bounds['upper']:,.0f} "
        f"(1 SD ${bounds['one_sd']:,.0f}, {bounds['method']} based)\n\n"
    )
    for n in notes:
        msg += f"⚠️ {html.escape(n)}\n"
    if notes:
        msg += "\n"
    if not legs:
        msg += "⚪ <b>No trade:</b> no strike passed the filters."
        return msg

    sides = {lbl for lbl, _ in legs}
    strategy = "Short Strangle" if len(sides) == 2 else ("Sell PUT" if "SELL PUT" in sides else "Sell CALL")
    msg += f"<b>🎯 Strategy: {strategy}</b>\n"
    for label, o in legs:
        msg += "\n" + fmt_leg(o, label)
    if len(legs) == 2:
        total = sum(o["net"] for _, o in legs)
        msg += f"\n<b>Combined net premium:</b> ${total:,.1f} | Breakevens: " \
               f"${legs[0][1]['strike'] - total:,.0f} / ${legs[1][1]['strike'] + total:,.0f}\n"
    msg += "\n<i>Signal only. Not financial advice. Always place your stop loss.</i>"
    return msg


# ============================ 5. STATE (no repeats) ======================
def load_state():
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def save_state(st):
    with open(STATE_FILE, "w") as f:
        json.dump(st, f, indent=2)


def signal_key(expiry, legs):
    """Same expiry + same strikes = same signal. Prices moving a bit is not a new signal."""
    if not legs:
        return f"{expiry:%Y%m%d%H}|NO_TRADE"
    return f"{expiry:%Y%m%d%H}|" + ",".join(sorted(o["symbol"] for _, o in legs))


def hours_since(ts):
    try:
        return (datetime.now(timezone.utc).timestamp() - float(ts)) / 3600
    except Exception:
        return 1e9


def log_signal(key, t, legs):
    """Append the signal to signals.json (read by paper_bot.py)."""
    if not legs:
        return
    try:
        with open(SIGNALS_FILE) as f:
            sigs = json.load(f)
    except Exception:
        sigs = []
    sigs.append({
        "key": key,
        "time": datetime.now(timezone.utc).isoformat(),
        "trend": t["trend"],
        "btc": t["price"],
        "legs": [{
            "label": label, "symbol": o["symbol"], "type": o["type"], "strike": o["strike"],
            "expiry": o["expiry"].isoformat(), "premium": o["premium"], "fee": o["fee"],
            "sl": o["sl"], "target": o["target"], "contract_value": o["contract_value"],
            "delta": o["delta"],
        } for label, o in legs],
    })
    with open(SIGNALS_FILE, "w") as f:
        json.dump(sigs[-SIGNALS_KEEP:], f, indent=1)


# ================================ MAIN ===================================
def main():
    # 1. Trend & volatility
    df = add_indicators(fetch_klines())
    t = analyse_trend(df)
    log.info("Trend: %s", t)

    # 2. Deribit sentiment: PCR, DVOL, 25-delta skew (non-fatal if it fails)
    p = deribit_sentiment()
    log.info("Deribit: pcr=%s dvol=%s skew=%s", p["pcr"], p["dvol"], p["skew"])

    # 3. Delta Exchange chain & strikes
    expiry, chain = fetch_delta_chain()
    hours_left = (expiry - datetime.now(timezone.utc)).total_seconds() / 3600
    iv = atm_iv(chain, t["price"])
    bounds = sd_boundaries(t["price"], t["atr"], hours_left, iv)
    log.info("Expiry %s, %d options, ATM IV %s, bounds %s", expiry, len(chain), iv, bounds)

    # which sides the trend allows
    allow_put = t["trend"] in ("Pure Long", "Sideways")
    allow_call = t["trend"] in ("Pure Short", "Sideways")
    notes = []

    # skew filter: don't sell the side the market is paying up to protect
    k = p.get("skew")
    if k and k["skew"] > SKEW_FEAR and allow_put:
        allow_put = False
        notes.append(f"Skew {k['skew']:+.1f}: puts in demand (fear), PUT side skipped.")
    if k and k["skew"] < SKEW_GREED and allow_call:
        allow_call = False
        notes.append(f"Skew {k['skew']:+.1f}: calls in demand (FOMO), CALL side skipped.")

    # DVOL filter: cheap IV = thin premiums for the risk
    d = p.get("dvol")
    if d and d["iv_rank"] < IV_RANK_LOW:
        if BLOCK_LOW_IV:
            allow_put = allow_call = False
            notes.append(f"IV Rank {d['iv_rank']:.0f} < {IV_RANK_LOW}: premiums too cheap, no trade.")
        else:
            notes.append(f"IV Rank {d['iv_rank']:.0f} is low: premiums are thin, size down.")

    legs = []
    if allow_put:
        put = pick_strike(chain, "put", bounds, t["price"])
        if put:
            legs.append(("SELL PUT", put))
    if allow_call:
        call = pick_strike(chain, "call", bounds, t["price"])
        if call:
            legs.append(("SELL CALL", call))
    # a strangle needs both legs; otherwise report what was found
    if t["trend"] == "Sideways" and len(legs) == 1:
        log.info("Only one strangle leg passed filters")

    # 4. Notify only when something changed (or on manual run / daily heartbeat)
    st = load_state()
    key = signal_key(expiry, legs)
    now_ts = datetime.now(timezone.utc).timestamp()
    if FORCE_SEND or key != st.get("last_key"):
        send_telegram(build_message(t, p, bounds, expiry, legs, notes))
        log_signal(key, t, legs)
        st.update(last_key=key, last_sent=now_ts)
        log.info("Sent new signal: %s", key)
    elif hours_since(st.get("last_sent")) >= HEARTBEAT_HOURS:
        send_telegram(f"💚 <b>Bot alive</b> | BTC ${t['price']:,.0f} | {t['trend']}\n"
                      f"No change since last signal. Last signal is still the latest.")
        st["last_sent"] = now_ts
        log.info("Heartbeat sent")
    else:
        log.info("Same signal as before (%s): not re-sending", key)
    st["last_run"] = now_ts
    save_state(st)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.exception("Bot failed")
        try:
            err = str(e).replace(TELEGRAM_BOT_TOKEN or "~~", "***")
            send_telegram(f"⚠️ <b>BTC Signal Bot error</b>\n<code>{html.escape(err)[:3500]}</code>")
        except Exception:
            pass
        sys.exit(1)
