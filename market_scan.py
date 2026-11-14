#!/usr/bin/env python3
"""
Daily market scan: (1) situational awareness gate, (2) top entry candidates.
Rules are taken from the "How Trading Really Works: Technical Analysis" guide.

Install:  pip install yfinance pandas numpy lxml requests
Run:      python market_scan.py --account 50000 --risk-pct 0.5
          python market_scan.py --universe file --tickers-file my_list.txt
Best run after the close (signals use the last completed daily bar) or
pre-market, to build the plan for the next session.

Not financial advice. Rules marked [ASSUMPTION] were not numerically
specified in the guide and are tunable via flags.
"""
import argparse, sys, io, os, time, pickle
import datetime as dt
import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    sys.exit("Missing dependency: pip install yfinance pandas numpy lxml requests")

# Pydroid has no command line: put options here, e.g. ["--universe", "watchlist", "--account", "20000"]
PYDROID_ARGS = []

MARKET = ["QQQ", "IWM", "^VIX"]
INDUSTRY_ETFS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE",
                 "XLC", "SMH", "IGV", "XBI", "ITB", "KRE", "XME", "XRT", "IBB", "ARKK"]
FALLBACK_UNIVERSE = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "AMD",
                     "NFLX", "CRM", "UBER", "PLTR", "SHOP", "COIN", "MU", "ANET", "PANW",
                     "CRWD", "NOW", "ABNB", "DASH", "SNOW", "DDOG", "NET", "TTD", "APP", "HOOD"]


# ------------------------------------------------------------------ helpers
def ema(s, n): return s.ewm(span=n, adjust=False).mean()
def sma(s, n): return s.rolling(n).mean()


def adr_pct(df, n=20):
    """Average Daily Range % = 100 * (mean(High/Low) - 1)."""
    return 100 * ((df.High / df.Low).iloc[-n:].mean() - 1)


def pivots(series, n, kind):
    vals, out = series.values, []
    for i in range(n, len(vals) - n):
        w = vals[i - n:i + n + 1]
        if (kind == "high" and vals[i] == w.max()) or (kind == "low" and vals[i] == w.min()):
            out.append(vals[i])
    return out


def structure_up(df, n):
    """Market structure: last swing high > prior swing high AND last swing low > prior swing low."""
    ph, pl = pivots(df.High, n, "high"), pivots(df.Low, n, "low")
    return len(ph) >= 2 and len(pl) >= 2 and ph[-1] > ph[-2] and pl[-1] > pl[-2]


def weekly(df):
    return df.resample("W").agg({"Open": "first", "High": "max", "Low": "min",
                                 "Close": "last", "Volume": "sum"}).dropna()


def trend_state(df):
    """QQQ rule: 10/20 EMA slope up, or at the very least price above the 20 EMA."""
    c = df.Close; e10, e20 = ema(c, 10), ema(c, 20)
    above20 = c.iloc[-1] > e20.iloc[-1]
    slope_up = e10.iloc[-1] > e10.iloc[-4] and e20.iloc[-1] > e20.iloc[-4]
    return "STRONG" if (above20 and slope_up) else "OK" if above20 else "FAIL"


def day_chg(df): return (df.Close.iloc[-1] / df.Close.iloc[-2] - 1) * 100


# ------------------------------------------------------------------ data
def get_universe(kind, path):
    if kind == "file":
        return sorted({x.strip().upper() for x in open(path).read().replace(",", "\n").split() if x.strip()})
    if kind == "watchlist":
        return FALLBACK_UNIVERSE
    try:
        import requests
        hdr = {"User-Agent": "Mozilla/5.0"}
        sp = pd.read_html(io.StringIO(requests.get(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", headers=hdr, timeout=20).text))[0]
        tk = set(sp["Symbol"].astype(str))
        try:
            nd = pd.read_html(io.StringIO(requests.get(
                "https://en.wikipedia.org/wiki/Nasdaq-100", headers=hdr, timeout=20).text))
            for t in nd:
                if "Ticker" in t.columns:
                    tk |= set(t["Ticker"].astype(str))
        except Exception:
            pass
        return sorted(x.replace(".", "-") for x in tk)
    except Exception as e:
        print(f"[warn] could not fetch index lists ({e}); using built-in watchlist")
        return FALLBACK_UNIVERSE


CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     f"scan_cache_{dt.date.today():%Y%m%d}.pkl")


def _fetch(part):
    raw = yf.download(part, period="2y", interval="1d", auto_adjust=True,
                      group_by="ticker", progress=False, threads=False)
    out = {}
    for t in part:
        try:
            d = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
            d = d[["Open", "High", "Low", "Close", "Volume"]].dropna()
            if len(d) > 30:
                d.index = pd.to_datetime(d.index).tz_localize(None)
                out[t] = d
        except Exception:
            pass
    return out


def load(tickers, chunk=20, pause=3.0):
    """Small batches + pauses + retries + daily disk cache (Yahoo rate-limits fast downloads)."""
    out = {}
    if os.path.exists(CACHE):
        try:
            out = pickle.load(open(CACHE, "rb")); print(f"Cache: {len(out)} symbols already saved today")
        except Exception:
            out = {}
    todo = [t for t in tickers if t not in out]
    for attempt in range(4):
        if not todo: break
        if attempt:
            wait = 30 * attempt
            print(f"\nRate limited / missing {len(todo)} symbols - waiting {wait}s then retrying ...")
            time.sleep(wait)
        for i in range(0, len(todo), chunk):
            try:
                out.update(_fetch(todo[i:i + chunk]))
            except Exception as e:
                print(f"[warn] {e}")
            time.sleep(pause)
            print(f"  {min(i + chunk, len(todo))}/{len(todo)} requested, {len(out)} loaded", end="\r")
            pickle.dump(out, open(CACHE, "wb"))
        todo = [t for t in todo if t not in out]
        if len(todo) < 0.03 * len(tickers): break   # remaining are likely delisted/odd tickers
    print(f"\nLoaded {len(out)} symbols, {len(todo)} unavailable.")
    return out


# ------------------------------------------------------------------ market context
def breakout_stats(data):
    """Are breakouts working? Fresh 20d-high breakouts in the last ~10 sessions that
    closed back below the level within 3 bars count as failed (rejection)."""
    ok = fail = 0
    for df in data.values():
        if len(df) < 60: continue
        c, h = df.Close.values, df.High.values
        for i in range(len(c) - 13, len(c) - 3):
            lvl = h[i - 20:i].max()
            if c[i] > lvl and c[i - 1] <= lvl:
                if (c[i + 1:i + 4] < lvl).any(): fail += 1
                else: ok += 1
    return ok, fail


def market_context(data, universe, a):
    checks, hard_fail = [], False

    def add(name, status, detail):
        checks.append((name, status, detail))

    # QQQ (hard rule)
    q = data.get("QQQ")
    if q is None: add("QQQ", "FAIL", "no data"); hard_fail = True
    else:
        st = trend_state(q); e20 = ema(q.Close, 20).iloc[-1]
        add("QQQ trend", {"STRONG": "PASS", "OK": "WARN", "FAIL": "FAIL"}[st],
            f"{st}: close {q.Close.iloc[-1]:.2f} vs 20EMA {e20:.2f}, day {day_chg(q):+.2f}%")
        hard_fail |= st == "FAIL"
    # VIX (hard rule: <15 or no entries today)
    v = data.get("^VIX")
    if v is None: add("VIX", "FAIL", "no data"); hard_fail = True
    else:
        vx = v.Close.iloc[-1]
        add("VIX < %.0f" % a.vix_max, "PASS" if vx < a.vix_max else "FAIL", f"VIX {vx:.2f}")
        hard_fail |= vx >= a.vix_max
    # IWM
    r = data.get("IWM")
    if r is not None:
        st = trend_state(r)
        add("IWM risk appetite", "PASS" if st == "STRONG" else "WARN",
            f"{st}, day {day_chg(r):+.2f}%")
    # Breadth: % of stocks above 50SMA  [ASSUMPTION threshold]
    ab = n = adv = 0
    for t in universe:
        d = data.get(t)
        if d is None or len(d) < 60: continue
        n += 1; ab += d.Close.iloc[-1] > sma(d.Close, 50).iloc[-1]; adv += day_chg(d) > 0
    if n:
        pct = 100 * ab / n
        add("Breadth (>50SMA)", "PASS" if pct >= a.breadth_min else "WARN",
            f"{pct:.0f}% above 50SMA, {100 * adv / n:.0f}% advancing ({n} stocks)")
    # Industries: how many up/down
    up = a20 = m = 0
    for t in INDUSTRY_ETFS:
        d = data.get(t)
        if d is None: continue
        m += 1; up += day_chg(d) > 0; a20 += d.Close.iloc[-1] > ema(d.Close, 20).iloc[-1]
    if m:
        good = up >= m / 2 and a20 >= m / 2
        add("Industries", "PASS" if good else "WARN", f"{up}/{m} up today, {a20}/{m} above 20EMA")
    # Are breakouts working?
    ok, fail = breakout_stats({t: data[t] for t in universe if t in data})
    if ok + fail >= 10:
        rate = 100 * fail / (ok + fail)
        add("Breakouts working", "PASS" if rate <= 40 else "WARN",
            f"{fail}/{ok + fail} recent breakouts failed ({rate:.0f}%)")
    else:
        add("Breakouts working", "WARN", f"only {ok + fail} recent breakouts - not enough sample")

    warns = sum(1 for c in checks if c[1] == "WARN")
    verdict = "NO-GO" if hard_fail else ("GO" if warns <= 1 else "SELECTIVE")
    return verdict, checks


# ------------------------------------------------------------------ setups
def analyze(t, df, a):
    """Return (candidates, extended_record). Pure daily-bar logic."""
    if len(df) < 260: return [], None
    c, h, l, o, v = df.Close, df.High, df.Low, df.Open, df.Volume
    e10, e20, e50 = ema(c, 10), ema(c, 20), ema(c, 50)
    s50, s200 = sma(c, 50), sma(c, 200)
    last = c.iloc[-1]
    if not last > s50.iloc[-1]: return [], None            # NEVER trade below the 50SMA
    adr = adr_pct(df)
    if not (a.adr_min <= adr <= a.adr_max): return [], None  # ADR 4-8%
    if (c * v).rolling(20).mean().iloc[-1] < a.min_dollar_vol: return [], None
    adr_d = last * adr / 100

    # --- extension (section 6): 3x/7x ADR from 20/50 EMA, 2x price vs 200SMA, 30% run  [ASSUMPTION: ADR multiples]
    ext20 = (last - e20.iloc[-1]) / adr_d
    ext50 = (last - e50.iloc[-1]) / adr_d
    r200 = last / s200.iloc[-1]
    run = last / c.iloc[-20:].min() - 1
    if ext20 >= a.ext20 or ext50 >= a.ext50 or r200 >= a.ext200 or run >= 0.30:
        return [], dict(ticker=t, close=last, adr=adr, ext20=ext20, ext50=ext50, r200=r200, run20=run * 100)

    stage2 = s50.iloc[-1] > s200.iloc[-1] and s200.iloc[-1] > s200.iloc[-21]
    rng = (h - l) / c * 100
    contraction = rng.iloc[-5:].mean() <= 0.75 * adr
    vol_dry = v.iloc[-5:].mean() < v.iloc[-20:].mean()
    d_up = structure_up(df.iloc[-120:], 3)
    w_up = structure_up(weekly(df).iloc[-60:], 2)
    macd = ema(c, 3) - ema(c, 6)
    lo, hi = macd.rolling(50).min().iloc[-1], macd.rolling(50).max().iloc[-1]
    smackd = 100 * (macd.iloc[-1] - lo) / (hi - lo) if hi > lo else 50  # [ASSUMPTION] normalised 0-100
    dip = smackd < 30

    def plan(trigger, stop):
        risk = trigger - stop
        if risk <= 0: return None
        rr = adr_d / risk
        if rr < a.min_rr: return None
        return dict(trigger=trigger, stop=stop, risk=risk, rr=rr, target=trigger + risk,
                    shares=int(a.account * a.risk_pct / 100 / risk))

    cands = []
    # ---- BREAKOUT: tight 10-week base, volume expansion on the move, 10/20 caught up
    base_hi = h.iloc[-51:-1].max()
    depth = (base_hi / l.iloc[-50:].min() - 1) * 100
    avgv = v.iloc[-51:-1].mean()
    dist = (base_hi / last - 1) * 100
    if (depth <= a.base_depth and last > e10.iloc[-1] and last > e20.iloc[-1]
            and contraction and v.iloc[-10:].mean() < avgv and 0 <= dist <= a.near_pct):
        p = plan(base_hi, l.iloc[-1])
        if p:
            sc = 2 * contraction + 2 * vol_dry + d_up + w_up + min(3, max(0, p["rr"] - 1)) + (2 if dist <= 1.5 else 0)
            cands.append(dict(setup="BREAKOUT", ticker=t, score=sc, close=last, adr=adr, dist=dist,
                              d_up=d_up, w_up=w_up, dip=dip, **p,
                              note=f"base {depth:.0f}% deep/50d, {dist:.1f}% below pivot; need volume >= {a.vol_expand}x on break"))
    # ---- PULLBACK: stage-2 uptrend, orderly low-volume decline to rising 10/20, tightening
    if stage2:
        rh = h.iloc[-15:].max()
        pb_adr = (rh - last) / adr_d
        near_ma = abs(last - e20.iloc[-1]) / adr_d <= 1.0 or e20.iloc[-1] <= last <= e10.iloc[-1] * 1.01
        ma_rising = e20.iloc[-1] > e20.iloc[-6] and e10.iloc[-1] > e10.iloc[-6]
        tight = rng.iloc[-3:].mean() < rng.iloc[-6:-3].mean()
        if 0.5 <= pb_adr <= 4 and near_ma and ma_rising and vol_dry and tight:
            trig = max(h.iloc[-3:].max(), h.iloc[-1])
            p = plan(trig, l.iloc[-1])
            if p:
                sc = 2 * contraction + 2 * vol_dry + d_up + w_up + 1 * dip + min(3, max(0, p["rr"] - 1)) + \
                     (2 if (trig / last - 1) * 100 <= 1.5 else 0)
                cands.append(dict(setup="PULLBACK", ticker=t, score=sc, close=last, adr=adr,
                                  dist=(trig / last - 1) * 100, d_up=d_up, w_up=w_up, dip=dip, **p,
                                  note=f"{pb_adr:.1f} ADR off 15d high, near 10/20EMA, vol drying, SMACK-D~{smackd:.0f}"))
    return cands, None


# ------------------------------------------------------------------ HTML dashboard
from html import escape as esc

CSS = """
:root{--bg:#0b0f14;--card:#141b24;--line:#223041;--tx:#e6edf3;--mut:#8b9bb0;--ok:#22c55e;--warn:#f59e0b;--bad:#ef4444;--acc:#38bdf8}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--tx);font:15px/1.4 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;padding:14px 14px 40px;max-width:640px;margin:auto}
h2{font-size:13px;letter-spacing:.08em;text-transform:uppercase;color:var(--mut);margin:26px 2px 10px}
.top{display:flex;justify-content:space-between;color:var(--mut);font-size:12px;margin-bottom:10px}
.banner{border-radius:16px;padding:20px;border:1px solid var(--c);background:linear-gradient(135deg,color-mix(in srgb,var(--c) 22%,var(--card)),var(--card))}
.banner b{display:block;font-size:34px;letter-spacing:.02em;color:var(--c)}.banner span{color:var(--tx);opacity:.85}
.chk{display:grid;grid-template-columns:1fr auto;gap:2px 10px;background:var(--card);border:1px solid var(--line);border-left:4px solid var(--c);border-radius:12px;padding:10px 12px;margin-bottom:8px}
.chk span{grid-column:1;color:var(--mut);font-size:13px}.chk i{grid-row:1/3;grid-column:2;align-self:center;font-style:normal;font-size:11px;font-weight:700;color:var(--c)}
.pass{--c:var(--ok)}.warn{--c:var(--warn)}.fail{--c:var(--bad)}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:12px;margin-bottom:10px}
.hd{display:flex;align-items:center;gap:8px}.tk{font-size:20px;font-weight:800}
.pill{font-size:11px;font-weight:700;padding:2px 8px;border-radius:99px;background:#1e293b;color:var(--acc)}.sc{margin-left:auto;color:var(--mut);font-size:12px}
.g{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin:10px 0 6px}.g div{background:#0f151d;border-radius:8px;padding:6px 8px}
.g small{display:block;color:var(--mut);font-size:10px;text-transform:uppercase}.g b{font-size:15px}
.note{color:var(--mut);font-size:12px}.tag{color:var(--ok);font-size:11px;font-weight:700;margin-right:6px}
.ext{display:flex;justify-content:space-between;gap:8px;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:8px 12px;margin-bottom:6px;font-size:13px}
.ext span{color:var(--mut);font-size:12px;text-align:right}.rules{color:var(--mut);font-size:12.5px;padding-left:18px}.rules li{margin-bottom:5px}
.empty{background:var(--card);border:1px dashed var(--line);border-radius:12px;padding:16px;color:var(--mut)}
"""


def html_path(p):
    if p != "auto": return p
    for d in ("/storage/emulated/0/Download", os.path.expanduser("~/storage/downloads")):
        if os.path.isdir(d) and os.access(d, os.W_OK):
            return os.path.join(d, "scan_report.html")
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "scan_report.html")


def render_html(verdict, checks, cands, ext, asof, show, a):
    col = {"GO": "#22c55e", "SELECTIVE": "#f59e0b", "NO-GO": "#ef4444"}[verdict]
    msg = {"GO": "Conditions supportive. Look at setups.",
           "SELECTIVE": "Multiple warnings. Only the best setups, reduce size.",
           "NO-GO": "Hard rule failed (QQQ weak or VIX high). No new entries."}[verdict]
    chk = "".join(f'<div class="chk {s.lower()}"><b>{esc(n)}</b><i>{s}</i><span>{esc(d)}</span></div>'
                  for n, s, d in checks)
    if show and cands:
        cards = ""
        for i, x in enumerate(cands[:a.top], 1):
            flags = ("<span class='tag'>D1 HH/HL</span>" if x["d_up"] else "") + \
                    ("<span class='tag'>W1 HH/HL</span>" if x["w_up"] else "")
            cards += (f'<div class="card"><div class="hd"><span class="tk">{esc(x["ticker"])}</span>'
                      f'<span class="pill">{x["setup"]}</span><span class="sc">score {x["score"]:.1f}</span></div>'
                      f'<div class="g"><div><small>Trigger</small><b>{x["trigger"]:.2f}</b></div>'
                      f'<div><small>Stop</small><b>{x["stop"]:.2f}</b></div>'
                      f'<div><small>1:1 target</small><b>{x["target"]:.2f}</b></div>'
                      f'<div><small>Shares</small><b>{x["shares"]}</b></div>'
                      f'<div><small>Risk/sh</small><b>{x["risk"]:.2f}</b></div>'
                      f'<div><small>ADR</small><b>{x["adr"]:.1f}%</b></div></div>'
                      f'<div class="note">{flags}Close {x["close"]:.2f} &middot; buy before +{x["adr"] / 2:.1f}% from open '
                      f'&middot; {esc(x["note"])}</div></div>')
        title = "Top entry candidates" if verdict != "NO-GO" else "Watchlist only - NO-GO day, do not enter"
    else:
        cards = ('<div class="empty">Setups hidden on NO-GO days (situational awareness first). '
                 'Run with <code>--force</code> for a watchlist.</div>' if not show else
                 '<div class="empty">No setups meet all rules today. Patience is a position.</div>')
        title = "Entry candidates"
    exts = "".join(f'<div class="ext"><b>{esc(e["ticker"])}</b><span>{e["ext20"]:.1f}x ADR from 20EMA &middot; '
                   f'{e["ext50"]:.1f}x from 50EMA &middot; {e["r200"]:.2f}x 200SMA &middot; +{e["run20"]:.0f}%</span></div>'
                   for e in ext[:10]) or '<div class="empty">None flagged.</div>'
    rules = ("<li>Enter only on a break of Trigger (prev-day / consolidation high), ideally on rising volume.</li>"
             "<li>Buy before the stock has made 50% of its ADR from today's open.</li>"
             "<li>Stop = prior day's low (use low of day once available). Size from entry-to-stop distance.</li>"
             "<li>Take 20-50% off at 1:1, trail the rest on 20MA / previous bar low / pivot low.</li>"
             "<li>A breakout that falls back inside its range has failed: exit.</li>")
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    return (f'<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<meta name="theme-color" content="#0b0f14"><title>Market Scan</title><style>{CSS}</style></head><body>'
            f'<div class="top"><span>Market scan &middot; data to {asof}</span><span>generated {now}</span></div>'
            f'<div class="banner" style="--c:{col}"><b>{verdict}</b><span>{msg}</span></div>'
            f'<h2>Market mood</h2>{chk}<h2>{title}</h2>{cards}'
            f'<h2>Extended - do not buy</h2>{exts}<h2>Execution rules</h2><ul class="rules">{rules}</ul>'
            f'<p class="note" style="margin-top:22px">Not financial advice. Verify levels on a live chart.</p></body></html>')


# ------------------------------------------------------------------ report
def main():
    ap = argparse.ArgumentParser(description="Daily market mood scan + top entries")
    ap.add_argument("--universe", choices=["sp500", "watchlist", "file"], default="sp500")
    ap.add_argument("--tickers-file")
    ap.add_argument("--account", type=float, default=50000, help="account size for position sizing")
    ap.add_argument("--risk-pct", type=float, default=0.5, help="[ASSUMPTION] %% of account risked per trade")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--csv", help="write candidates to CSV")
    ap.add_argument("--hide-nogo", action="store_true",
                    help="dashboard hides setups on NO-GO days (default: shows them as a watchlist)")
    ap.add_argument("--chunk", type=int, default=20, help="symbols per download batch")
    ap.add_argument("--pause", type=float, default=3.0, help="seconds between batches")
    ap.add_argument("--html", default="auto", help="dashboard path ('auto' = Downloads folder), or 'off'")
    ap.add_argument("--force", action="store_true", help="list setups even on NO-GO days (watchlist only)")
    ap.add_argument("--vix-max", type=float, default=15.0)
    ap.add_argument("--breadth-min", type=float, default=50.0, help="[ASSUMPTION] min %% above 50SMA")
    ap.add_argument("--adr-min", type=float, default=4.0)
    ap.add_argument("--adr-max", type=float, default=8.0)
    ap.add_argument("--min-dollar-vol", type=float, default=20e6, help="[ASSUMPTION] 20d avg $ volume")
    ap.add_argument("--base-depth", type=float, default=30.0, help="[ASSUMPTION] max base depth %%")
    ap.add_argument("--near-pct", type=float, default=3.0, help="[ASSUMPTION] max %% below pivot to be 'ready'")
    ap.add_argument("--min-rr", type=float, default=1.5, help="[ASSUMPTION] min ADR$/risk$ (tight stop)")
    ap.add_argument("--vol-expand", type=float, default=1.5)
    ap.add_argument("--ext20", type=float, default=3.0)
    ap.add_argument("--ext50", type=float, default=7.0)
    ap.add_argument("--ext200", type=float, default=2.0)
    a = ap.parse_args(sys.argv[1:] or PYDROID_ARGS)

    universe = get_universe(a.universe, a.tickers_file)
    print(f"Loading {len(universe) + len(MARKET) + len(INDUSTRY_ETFS)} symbols ...")
    # market-context symbols first so the verdict survives even if the stock list is throttled
    data = load(MARKET + INDUSTRY_ETFS, a.chunk, a.pause)
    data.update(load(sorted(set(universe) - set(data)), a.chunk, a.pause))
    if "QQQ" not in data:
        sys.exit("No market data (check internet access).")
    print(f"\n=== MARKET MOOD  (last bar: {data['QQQ'].index[-1].date()}) ===")
    verdict, checks = market_context(data, universe, a)
    icon = {"PASS": "[ OK ]", "WARN": "[WARN]", "FAIL": "[FAIL]"}
    for n, s, d in checks:
        print(f"{icon[s]} {n:<20} {d}")
    msg = {"GO": "GO - conditions supportive, look at setups.",
           "SELECTIVE": "SELECTIVE - multiple warnings: only the best setups, reduce size.",
           "NO-GO": "NO-GO - hard rule failed (QQQ weak or VIX high): NO new entries today."}[verdict]
    print(f"\nVERDICT: {msg}")

    allc, ext = [], []
    for t in universe:
        if t in data:
            c, e = analyze(t, data[t], a)
            allc += c
            if e: ext.append(e)
    allc.sort(key=lambda x: -x["score"])

    title = "TOP ENTRY CANDIDATES" if verdict != "NO-GO" else "WATCHLIST ONLY (do not enter)"
    if verdict == "NO-GO" and not a.force:
        print("\nSetups hidden on NO-GO days (guide: situational awareness first). Use --force for a watchlist.")
    else:
        print(f"\n=== {title} ===")
        print(f"{'#':>2} {'Tkr':<6}{'Setup':<9}{'Scr':>4}{'Close':>8}{'ADR%':>6}{'Trigger':>9}{'Stop':>8}"
              f"{'Risk$':>7}{'1:1 tgt':>9}{'Shs':>6}  Notes")
        for i, x in enumerate(allc[:a.top], 1):
            print(f"{i:>2} {x['ticker']:<6}{x['setup']:<9}{x['score']:>4.1f}{x['close']:>8.2f}{x['adr']:>6.1f}"
                  f"{x['trigger']:>9.2f}{x['stop']:>8.2f}{x['risk']:>7.2f}{x['target']:>9.2f}{x['shares']:>6}  "
                  f"{'D1HH/HL ' if x['d_up'] else ''}{'W1HH/HL ' if x['w_up'] else ''}{x['note']}")
        if not allc: print("No setups meet all rules today. Patience is a position.")
        print("\nEXECUTION RULES (from the guide):"
              "\n - Enter ONLY on a break of Trigger (prev-day high / consolidation high), ideally with rising volume."
              "\n - Buy before the stock has made 50% of its ADR from today's open (cap = open x (1 + ADR%/2))."
              "\n - Stop = Stop column (low of previous day; use low of day once available). Size = risk / (entry-stop)."
              "\n - Take 20-50% off at 1:1 (1:1 tgt), trail the rest on 20MA / previous bar low / pivot low."
              "\n - Breakouts need volume expansion on the break; failed breakout back inside range = exit (rejection).")

    ext.sort(key=lambda x: -x["ext20"])
    if ext:
        print("\n=== EXTENDED - DO NOT BUY (expect pullback / reversion 'snacks' only) ===")
        for e in ext[:10]:
            print(f" {e['ticker']:<6} close {e['close']:.2f}  ADR {e['adr']:.1f}%  "
                  f"{e['ext20']:.1f}x ADR from 20EMA, {e['ext50']:.1f}x from 50EMA, "
                  f"{e['r200']:.2f}x of 200SMA, +{e['run20']:.0f}% off 20d low")

    if a.csv and allc:
        pd.DataFrame(allc).to_csv(a.csv, index=False); print(f"\nSaved {a.csv}")
    if a.html != "off":
        path = html_path(a.html)
        show = verdict != "NO-GO" or not a.hide_nogo or a.force
        with open(path, "w", encoding="utf-8") as f:
            f.write(render_html(verdict, checks, allc, ext, str(data["QQQ"].index[-1].date()), show, a))
        print(f"\nDashboard saved: {path}")
        try:
            import webbrowser; webbrowser.open("file://" + path)
        except Exception:
            pass
    print("\nNot financial advice. Verify levels on a live chart before acting.")


if __name__ == "__main__":
    main()
