import logging
import numpy as np

logger = logging.getLogger(__name__)


def determine_trend(cpr: dict, current_price: float) -> str:
    if current_price > cpr["tc"]:
        return "up"
    elif current_price < cpr["bc"]:
        return "down"
    return "neutral"


def is_buy_setup(weekly_cpr: dict, daily_cpr: dict, current_price: float) -> bool:
    weekly_trend = determine_trend(weekly_cpr, current_price)
    daily_trend = determine_trend(daily_cpr, current_price)
    logger.info(f"Weekly trend: {weekly_trend}, Daily trend: {daily_trend}")
    return weekly_trend == "up" and daily_trend == "up"


def check_buy_entry(
    closed_candle: list,
    daily_cpr: dict,
    current_atr: float,
    sl_atr_mult: float = 2.0,
    tp1_atr_mult: float = 0.5,
    tp2_atr_mult: float = 1.5,
) -> dict | None:
    candle_high = float(closed_candle[2])
    candle_close = float(closed_candle[4])

    r1 = daily_cpr["r1"]
    prev_high = daily_cpr["prev_day_high"]

    entry_price = candle_close

    if not (candle_high > r1 and candle_high > prev_high and entry_price > r1):
        return None

    if current_atr <= 0:
        return None

    stop_loss = entry_price - current_atr * sl_atr_mult

    if entry_price <= stop_loss:
        logger.info(f"BUY signal rejected — entry {entry_price:.2f} below SL {stop_loss:.2f}")
        return None

    tp1 = entry_price + current_atr * tp1_atr_mult
    tp2 = entry_price + current_atr * tp2_atr_mult

    logger.info(
        f"BUY signal — Entry: {entry_price:.2f}, SL: {stop_loss:.2f}, "
        f"TP1: {tp1:.2f}, TP2: {tp2:.2f} (ATR={current_atr:.2f})"
    )
    return {
        "entry": entry_price,
        "stop_loss": stop_loss,
        "take_profit_1": tp1,
        "take_profit_2": tp2,
    }


def calc_atr(klines: list, period: int = 14) -> float:
    if len(klines) < period + 1:
        return 0.0

    closes = [float(k[4]) for k in klines]
    highs = [float(k[2]) for k in klines]
    lows = [float(k[3]) for k in klines]

    tr_values = []
    for i in range(1, len(klines)):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i - 1])
        lc = abs(lows[i] - closes[i - 1])
        tr_values.append(max(hl, hc, lc))

    first_atr = sum(tr_values[:period]) / period
    atr = first_atr
    for i in range(period, len(tr_values)):
        atr = (atr * (period - 1) + tr_values[i]) / period

    return atr


def get_5min_closed_candles(
    klines: list, last_candle_time: int, current_time_ms: int
) -> list:
    new_candles = []
    for k in klines:
        k_time = int(k[0])
        close_time = int(k[6])
        if k_time > last_candle_time and close_time < current_time_ms:
            new_candles.append(k)
    return new_candles
