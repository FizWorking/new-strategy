import pandas as pd
import logging

logger = logging.getLogger(__name__)


def calculate_cpr(high: float, low: float, close: float) -> dict:
    pivot = (high + low + close) / 3
    bc = (high + low) / 2
    tc = 2 * pivot - bc
    r1 = 2 * pivot - low
    r2 = pivot + (high - low)
    s1 = 2 * pivot - high
    s2 = pivot - (high - low)

    return {
        "p": pivot,
        "bc": bc,
        "tc": tc,
        "r1": r1,
        "r2": r2,
        "s1": s1,
        "s2": s2,
    }


def klines_to_dataframe(klines: list) -> pd.DataFrame:
    df = pd.DataFrame(
        klines,
        columns=[
            "time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "trades",
            "taker_buy_base", "taker_buy_quote", "ignore",
        ],
    )
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    return df


def compute_weekly_cpr_from_klines(klines: list) -> dict | None:
    if not klines:
        return None
    df = klines_to_dataframe(klines)
    high = df["high"].max()
    low = df["low"].min()
    close = df["close"].iloc[-1]
    logger.info(
        f"Weekly CPR — High: {high:.2f}, Low: {low:.2f}, Close: {close:.2f}"
    )
    return calculate_cpr(high, low, close)


def compute_daily_cpr(
    prev_day_high: float, prev_day_low: float, prev_day_close: float
) -> dict:
    cpr = calculate_cpr(prev_day_high, prev_day_low, prev_day_close)
    cpr["prev_day_high"] = prev_day_high
    cpr["prev_day_low"] = prev_day_low
    logger.info(
        f"Daily CPR — P: {cpr['p']:.2f}, BC: {cpr['bc']:.2f}, TC: {cpr['tc']:.2f}, "
        f"R1: {cpr['r1']:.2f}, R2: {cpr['r2']:.2f}, "
        f"S1: {cpr['s1']:.2f}, S2: {cpr['s2']:.2f}, "
        f"Prev High: {cpr['prev_day_high']:.2f}, Prev Low: {cpr['prev_day_low']:.2f}"
    )
    return cpr
