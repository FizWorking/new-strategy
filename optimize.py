import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from binance.client import Client
import json
import os

SYMBOL = "BTCUSDT"
QUANTITY = 0.001
DAYS = 180
CACHE_5M = "cache_5m.csv"
CACHE_1H = "cache_1h.csv"


def calc_cpr(h, l, c):
    p = (h + l + c) / 3
    bc = (h + l) / 2
    tc = 2 * p - bc
    r1 = 2 * p - l
    r2 = p + (h - l)
    return {"p": p, "bc": bc, "tc": tc, "r1": r1, "r2": r2}


def load_data():
    if os.path.exists(CACHE_5M) and os.path.exists(CACHE_1H):
        print("Loading cached data...")
        df5 = pd.read_csv(CACHE_5M, parse_dates=["time"])
        df1 = pd.read_csv(CACHE_1H, parse_dates=["time"])
        return df1, df5
    client = Client()
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=DAYS)
    print(f"Downloading {DAYS} days...")
    cols = ["time","open","high","low","close","volume","close_time","qav","trades","tbb","tbq","ign"]
    def dl(interval):
        k = client.get_historical_klines(SYMBOL, interval, start.strftime("%d %b %Y"), end.strftime("%d %b %Y"))
        df = pd.DataFrame(k, columns=cols)
        df["time"] = pd.to_datetime(df["time"], unit="ms")
        for c in ["open","high","low","close"]: df[c] = df[c].astype(float)
        return df
    df1 = dl(Client.KLINE_INTERVAL_1HOUR)
    df5 = dl(Client.KLINE_INTERVAL_5MINUTE)
    df1.to_csv(CACHE_1H, index=False)
    df5.to_csv(CACHE_5M, index=False)
    print(f"1H:{len(df1)} 5M:{len(df5)}")
    return df1, df5


def precompute(df1, df5):
    """Pre-compute all signals at each 5min candle position"""
    print("Pre-computing signals...")

    df1["week"] = df1["time"].dt.strftime("%Y-W%V")

    # Pre-compute daily CPR from 1H data
    df1["day"] = df1["time"].dt.date
    daily = df1.groupby("day").agg({"high":"max","low":"min","close":"last"}).reset_index()
    daily["prev_high"] = daily["high"].shift(1)
    daily["prev_low"] = daily["low"].shift(1)
    daily["prev_close"] = daily["close"].shift(1)

    # For each 5min candle, compute:
    # - weekly CPR (from 1H data up to this time in the same ISO week)
    # - daily CPR (from prev day's data)
    # - ATR

    n = len(df5)
    p_weekly = np.full(n, np.nan)
    p_r1 = np.full(n, np.nan)
    p_r2 = np.full(n, np.nan)
    p_prev_high = np.full(n, np.nan)
    p_tc = np.full(n, np.nan)

    # Group 1H by week for faster lookup
    week_groups = {w: g for w, g in df1.groupby("week")}

    # Map date->daily row
    daily_map = {r["day"]: r for _, r in daily.iterrows()}

    for i in range(n):
        t = df5.iloc[i]["time"]
        wk = t.strftime("%Y-W%V")

        # Weekly CPR
        if wk in week_groups:
            gw = week_groups[wk]
            gw_up_to = gw[gw["time"] <= t]
            if len(gw_up_to) >= 3:
                wh = gw_up_to["high"].max()
                wl = gw_up_to["low"].min()
                wc = gw_up_to["close"].iloc[-1]
                cpr = calc_cpr(wh, wl, wc)
                p_weekly[i] = cpr["tc"]

        # Daily CPR from previous day
        today = t.date()
        yesterday = today - timedelta(days=1)
        if yesterday in daily_map:
            dr = daily_map[yesterday]
            cpr = calc_cpr(dr["prev_high"], dr["prev_low"], dr["prev_close"])
            p_r1[i] = cpr["r1"]
            p_r2[i] = cpr["r2"]
            p_prev_high[i] = dr["prev_high"]
            p_tc[i] = cpr["tc"]

    # ATR
    h = df5["high"].values
    l = df5["low"].values
    c = df5["close"].values
    tr = np.maximum(h[1:] - l[1:],
                    np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
    atr = np.full(n, np.nan)
    atr[14] = np.mean(tr[:14])
    for i in range(15, n):
        atr[i] = (atr[i-1] * 13 + tr[i-1]) / 14

    return {
        "time": df5["time"].values,
        "open": df5["open"].values,
        "high": h,
        "low": l,
        "close": c,
        "weekly_tc": p_weekly.astype(float),
        "daily_r1": p_r1.astype(float),
        "daily_r2": p_r2.astype(float),
        "daily_prev_high": p_prev_high.astype(float),
        "daily_tc": p_tc.astype(float),
        "atr": atr,
    }


def simulate(data, params):
    n = len(data["close"])
    legs = []
    pos_entry = None
    pos_sl = None
    pos_tp1 = None
    pos_tp2 = None
    tp1_hit = False
    qty_remain = 0.0
    pos_entry_time = None
    trailing = False
    highest = 0.0
    breakeven = 0.0

    is_atr = "atr" in params["method"]
    close_cfm = params.get("close_confirm", True)

    for i in range(n):
        hi = data["high"][i]
        lo = data["low"][i]
        cl = data["close"][i]

        if pos_entry is not None:
            if hi > highest:
                highest = hi

            if trailing:
                if is_atr:
                    tr_pct = min(data["atr"][i] * params.get("trail_atr_mult", 0) / pos_entry, 0.05) if not np.isnan(data["atr"][i]) else 0
                else:
                    tr_pct = params.get("trail_pct", 0)
                new_sl = max(highest * (1 - tr_pct), breakeven) if tr_pct > 0 else breakeven
                if new_sl > pos_sl:
                    pos_sl = new_sl

            if lo <= pos_sl:
                pnl = (pos_sl - pos_entry) / pos_entry * 100
                legs.append((str(pos_entry_time), str(data["time"][i]), "SL", qty_remain/QUANTITY*100, pnl))
                pos_entry = None
                continue

            if hi >= pos_tp2:
                pnl = (pos_tp2 - pos_entry) / pos_entry * 100
                legs.append((str(pos_entry_time), str(data["time"][i]), "TP2", qty_remain/QUANTITY*100, pnl))
                pos_entry = None
                continue

            if hi >= pos_tp1 and not tp1_hit:
                tp1_hit = True
                half = qty_remain * 0.5
                pnl = (pos_tp1 - pos_entry) / pos_entry * 100
                legs.append((str(pos_entry_time), str(data["time"][i]), "TP1", half/QUANTITY*100, pnl))
                qty_remain -= half
                trailing = True
                breakeven = pos_entry * 1.001
                if breakeven > pos_sl:
                    pos_sl = breakeven
                if lo <= pos_sl:
                    pnl = (pos_sl - pos_entry) / pos_entry * 100
                    legs.append((str(pos_entry_time), str(data["time"][i]), "SL", qty_remain/QUANTITY*100, pnl))
                    pos_entry = None
                    continue
            continue

        # Entry logic
        if np.isnan(data["weekly_tc"][i]) or np.isnan(data["daily_tc"][i]):
            continue
        if np.isnan(data["daily_r1"][i]) or np.isnan(data["daily_prev_high"][i]):
            continue

        if not (cl > data["weekly_tc"][i] and cl > data["daily_tc"][i]):
            continue
        if not (hi > data["daily_r1"][i] and hi > data["daily_prev_high"][i]):
            continue
        if close_cfm and cl <= data["daily_r1"][i]:
            continue

        entry = cl

        if is_atr:
            a = data["atr"][i]
            if np.isnan(a) or a <= 0:
                continue
            sl = entry - a * params["sl_atr_mult"]
            tp1 = entry + a * params["tp1_atr_mult"]
            tp2 = entry + a * params["tp2_atr_mult"]
        else:
            sl_level = min(data["daily_r1"][i], data["daily_prev_high"][i])
            sl = sl_level * (1 - params["sl_buffer"])
            if entry <= sl:
                continue
            dr = data["daily_r2"][i] - data["daily_r1"][i]
            tp1 = data["daily_r1"][i] + dr * params["tp1_ratio"]
            tp2 = data["daily_r1"][i] + dr * params["tp2_ratio"]

        pos_entry = entry
        pos_sl = sl
        pos_tp1 = tp1
        pos_tp2 = tp2
        tp1_hit = False
        qty_remain = QUANTITY
        pos_entry_time = data["time"][i]
        trailing = False
        highest = entry
        breakeven = 0.0

    if not legs:
        return None

    df = pd.DataFrame(legs, columns=["entry_time","exit_time","side","qty_pct","pnl_pct"])
    total = (df["qty_pct"] / 100 * df["pnl_pct"]).sum()
    trades = df["entry_time"].nunique()
    trade_pnl = df.groupby("entry_time")["pnl_pct"].apply(lambda x: sum((df[df["entry_time"]==x.name]["qty_pct"]/100 * df[df["entry_time"]==x.name]["pnl_pct"])))
    # Correct approach
    per_trade = df.groupby("entry_time").apply(
        lambda g: (g["qty_pct"]/100 * g["pnl_pct"]).sum(), include_groups=False
    )
    wins = (per_trade > 0).sum()
    win_rate = wins / trades * 100 if trades else 0
    avg_w = df[df["pnl_pct"]>0]["pnl_pct"].mean() if len(df[df["pnl_pct"]>0]) else 0
    avg_l = df[df["pnl_pct"]<=0]["pnl_pct"].mean() if len(df[df["pnl_pct"]<=0]) else 0

    return {"trades": trades, "win_rate": round(win_rate,1), "pnl": round(total,2),
            "avg_win": round(avg_w,2), "avg_loss": round(avg_l,2)}


def main():
    df1, df5 = load_data()
    data = precompute(df1, df5)
    print(f"Precomputed {len(data['close'])} candles. Running simulations...\n")

    variants = [
        # Top 2 ATR winners — verify on 180-day data
        {"method": "atr", "close_confirm": True,  "sl_atr_mult": 2.0,  "tp1_atr_mult": 0.5,  "tp2_atr_mult": 1.5},
        {"method": "atr", "close_confirm": True,  "sl_atr_mult": 1.5,  "tp1_atr_mult": 0.5,  "tp2_atr_mult": 1.5},
    ]

    results = []
    for i, v in enumerate(variants):
        r = simulate(data, v)
        if r:
            results.append({**v, **r})
        print(f"  {i+1}/{len(variants)} done — {r['pnl']:.1f}% ({v['method']})" if r else f"  {i+1}/{len(variants)} done — no trades ({v['method']})")

    results.sort(key=lambda x: x["pnl"], reverse=True)

    print(f"\n{'='*90}")
    print(f"TOP 20 — CPR Strategy Optimization on {SYMBOL} (90 days)")
    print(f"{'='*90}")
    print(f"{'Rank':<5} {'Method':<12} {'Params':<45} {'Trades':<7} {'Win%':<6} {'PnL%':<9} {'AvgW':<6} {'AvgL':<6}")
    print(f"{'-'*90}")
    for i, r in enumerate(results[:20]):
        if "atr" in r["method"]:
            tr = f"{r.get('trail_atr_mult',0)}ATR" if "trail" in r["method"] else "none"
            p = f"SL={r['sl_atr_mult']}A TP1={r['tp1_atr_mult']}A TP2={r['tp2_atr_mult']}A Tr={tr}"
        else:
            tr = f"{r.get('trail_pct',0)}" if "trail" in r["method"] else "none"
            p = f"SL={r['sl_buffer']} TP1={r['tp1_ratio']} TP2={r['tp2_ratio']} Tr={tr} Cfm={r.get('close_confirm',True)}"
        print(f"{i+1:<5} {r['method']:<12} {p:<45} {r['trades']:<7} {r['win_rate']:<6} {r['pnl']:<9} {r['avg_win']:<6} {r['avg_loss']:<6}")

    with open("optimization_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to optimization_results.json")


if __name__ == "__main__":
    main()
