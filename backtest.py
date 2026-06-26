import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from binance.client import Client
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("backtest")

SYMBOL = "BTCUSDT"
QUANTITY = 0.001
DAYS_OF_DATA = 180

SL_ATR_MULT = 2.0
TP1_ATR_MULT = 0.5
TP2_ATR_MULT = 1.5
ATR_PERIOD = 14


def calc_cpr(h, l, c):
    p = (h + l + c) / 3
    bc = (h + l) / 2
    tc = 2 * p - bc
    r1 = 2 * p - l
    r2 = p + (h - l)
    return {"p": p, "bc": bc, "tc": tc, "r1": r1, "r2": r2}


def df_from_klines(klines):
    df = pd.DataFrame(
        klines,
        columns=[
            "time", "open", "high", "low", "close", "volume",
            "close_time", "qav", "trades", "tbb", "tbq", "ign",
        ],
    )
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
    for c in ["open", "high", "low", "close"]:
        df[c] = df[c].astype(float)
    return df


def compute_atr(df, period=14):
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    tr = np.maximum(
        h[1:] - l[1:],
        np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])),
    )
    atr = np.full(len(df), np.nan)
    atr[period] = np.mean(tr[:period])
    for i in range(period + 1, len(df)):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i - 1]) / period
    return atr


def determine_trend(cpr, price):
    if price > cpr["tc"]:
        return "up"
    elif price < cpr["bc"]:
        return "down"
    return "neutral"


def run():
    client = Client()

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=DAYS_OF_DATA)

    logger.info(f"Downloading {DAYS_OF_DATA} days of data...")
    df_1h = df_from_klines(
        client.get_historical_klines(
            SYMBOL, Client.KLINE_INTERVAL_1HOUR,
            start.strftime("%d %b %Y"), end.strftime("%d %b %Y"),
        )
    )
    df_5m = df_from_klines(
        client.get_historical_klines(
            SYMBOL, Client.KLINE_INTERVAL_5MINUTE,
            start.strftime("%d %b %Y"), end.strftime("%d %b %Y"),
        )
    )
    logger.info(f"Downloaded 1H:{len(df_1h)} candles, 5M:{len(df_5m)} candles")

    df_1h["week_label"] = df_1h["time"].dt.strftime("%Y-W%V")
    atr_vals = compute_atr(df_5m, ATR_PERIOD)

    legs = []
    pos_entry = None
    pos_sl = None
    pos_tp1 = None
    pos_tp2 = None
    tp1_hit = False
    qty_remain = 0.0
    pos_entry_time = None

    for i in range(len(df_5m)):
        row = df_5m.iloc[i]
        t = row["time"]

        if pos_entry is not None:
            if row["low"] <= pos_sl:
                pnl = (pos_sl - pos_entry) / pos_entry * 100
                legs.append({
                    "entry_time": pos_entry_time, "exit_time": t,
                    "side": "SL", "qty_pct": qty_remain / QUANTITY * 100,
                    "pnl_pct": round(pnl, 2),
                })
                pos_entry = None
                continue

            if row["high"] >= pos_tp2:
                pnl = (pos_tp2 - pos_entry) / pos_entry * 100
                legs.append({
                    "entry_time": pos_entry_time, "exit_time": t,
                    "side": "TP2", "qty_pct": qty_remain / QUANTITY * 100,
                    "pnl_pct": round(pnl, 2),
                })
                pos_entry = None
                continue

            if row["high"] >= pos_tp1 and not tp1_hit:
                tp1_hit = True
                half_qty = qty_remain * 0.5
                pnl = (pos_tp1 - pos_entry) / pos_entry * 100
                legs.append({
                    "entry_time": pos_entry_time, "exit_time": t,
                    "side": "TP1", "qty_pct": half_qty / QUANTITY * 100,
                    "pnl_pct": round(pnl, 2),
                })
                qty_remain -= half_qty
                breakeven = pos_entry * 1.001
                if breakeven > pos_sl:
                    pos_sl = breakeven
                if row["low"] <= pos_sl:
                    pnl = (pos_sl - pos_entry) / pos_entry * 100
                    legs.append({
                        "entry_time": pos_entry_time, "exit_time": t,
                        "side": "SL", "qty_pct": qty_remain / QUANTITY * 100,
                        "pnl_pct": round(pnl, 2),
                    })
                    pos_entry = None
                    continue
            continue

        wk_label = t.strftime("%Y-W%V")
        wh = df_1h[df_1h["week_label"] == wk_label]
        if len(wh) < 5:
            continue
        wh_upto = wh[wh["time"] <= t]
        if len(wh_upto) < 3:
            continue

        w_cpr = calc_cpr(
            wh_upto["high"].max(), wh_upto["low"].min(), wh_upto["close"].iloc[-1],
        )

        yesterday = t - timedelta(days=1)
        yd_start = yesterday.replace(hour=0, minute=0, second=0, microsecond=0)
        yd_end = yd_start + timedelta(days=1)
        yd_df = df_1h[(df_1h["time"] >= yd_start) & (df_1h["time"] < yd_end)]
        if len(yd_df) == 0:
            continue

        prev_high = yd_df["high"].max()
        prev_low = yd_df["low"].min()
        prev_close = yd_df["close"].iloc[-1]
        d_cpr = calc_cpr(prev_high, prev_low, prev_close)
        d_cpr["prev_day_high"] = prev_high

        price = row["close"]

        if determine_trend(w_cpr, price) != "up":
            continue
        if determine_trend(d_cpr, price) != "up":
            continue

        entry = row["close"]
        candle_high = row["high"]

        if not (candle_high > d_cpr["r1"] and candle_high > d_cpr["prev_day_high"] and entry > d_cpr["r1"]):
            continue

        current_atr = atr_vals[i]
        if np.isnan(current_atr) or current_atr <= 0:
            continue

        sl = entry - current_atr * SL_ATR_MULT
        if entry <= sl:
            continue

        tp1 = entry + current_atr * TP1_ATR_MULT
        tp2 = entry + current_atr * TP2_ATR_MULT

        pos_entry = entry
        pos_sl = sl
        pos_tp1 = tp1
        pos_tp2 = tp2
        tp1_hit = False
        qty_remain = QUANTITY
        pos_entry_time = t

    logger.info("Backtest complete!\n")

    if not legs:
        logger.info("No trades found.")
        return

    df_legs = pd.DataFrame(legs)

    total_pnl = (df_legs["qty_pct"] / 100 * df_legs["pnl_pct"]).sum()
    wins = df_legs[df_legs["pnl_pct"] > 0]
    losses = df_legs[df_legs["pnl_pct"] <= 0]

    trade_pnl = df_legs.groupby("entry_time").apply(
        lambda g: (g["qty_pct"] / 100 * g["pnl_pct"]).sum(), include_groups=False,
    )
    total_trades = len(trade_pnl)
    won_trades = (trade_pnl > 0).sum()
    win_rate = won_trades / total_trades * 100 if total_trades else 0

    tp1_legs = df_legs[df_legs["side"] == "TP1"]
    tp2_legs = df_legs[df_legs["side"] == "TP2"]
    sl_legs = df_legs[df_legs["side"] == "SL"]

    print(f"{'='*55}")
    print(f"  BACKTEST RESULTS — {SYMBOL}")
    print(f"  Period: {start.date()} to {end.date()} ({DAYS_OF_DATA} days)")
    print(f"  Params: SL={SL_ATR_MULT}×ATR, TP1={TP1_ATR_MULT}×ATR, TP2={TP2_ATR_MULT}×ATR")
    print(f"{'='*55}")
    print(f"  Total trades          : {total_trades}")
    print(f"  Winning trades        : {won_trades}")
    print(f"  Win rate              : {win_rate:.1f}%")
    print(f"  Total legs            : {len(df_legs)}")
    print(f"  Total P&L             : {total_pnl:.2f}%")
    print(f"  Avg win per leg       : {wins['pnl_pct'].mean():.2f}%" if len(wins) else "  N/A")
    print(f"  Avg loss per leg      : {losses['pnl_pct'].mean():.2f}%" if len(losses) else "  N/A")
    print(f"")
    print(f"  TP1 hits (50% close)  : {len(tp1_legs)}")
    print(f"  TP2 hits (close)      : {len(tp2_legs)}")
    print(f"  SL hits  (close)      : {len(sl_legs)}")
    print(f"{'='*55}")

    df_legs.to_csv("backtest_results.csv", index=False)
    logger.info("Detailed results saved to backtest_results.csv")


if __name__ == "__main__":
    run()
