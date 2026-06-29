import os
import time
import json
import logging
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
from binance.client import Client

from exchange import BinanceExchange
from cpr import compute_weekly_cpr_from_klines, compute_daily_cpr
from strategy import (
    is_buy_setup,
    check_buy_entry,
    calc_atr,
    get_5min_closed_candles,
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("trading_bot.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("main")

SYMBOL = os.getenv("SYMBOL", "BTCUSDT")
USE_TESTNET = os.getenv("USE_TESTNET", "true").lower() == "true"
CHECK_INTERVAL = 10
DAILY_CPR_CACHE_SECONDS = 600
WEEKLY_CPR_CACHE_SECONDS = 300

SL_ATR_MULT = float(os.getenv("SL_ATR_MULT", "2.0"))
TP1_ATR_MULT = float(os.getenv("TP1_ATR_MULT", "0.5"))
TP2_ATR_MULT = float(os.getenv("TP2_ATR_MULT", "1.5"))
ATR_PERIOD = int(os.getenv("ATR_PERIOD", "14"))

POSITION_FILE = "position.json"


def _load_quantity() -> float:
    raw = os.getenv("QUANTITY", "0.001")
    try:
        qty = float(raw)
        if qty <= 0:
            raise ValueError("must be positive")
        return qty
    except ValueError as e:
        logger.error(f"Invalid QUANTITY '{raw}': {e}")
        raise


try:
    QUANTITY = _load_quantity()
except ValueError:
    logger.error("FATAL: Invalid QUANTITY in .env. Bot exiting.")
    exit(1)


def load_position() -> dict | None:
    try:
        with open(POSITION_FILE, "r") as f:
            data = json.load(f)
            logger.info(f"Loaded existing position: {data}")
            return data
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def save_position(position: dict):
    with open(POSITION_FILE, "w") as f:
        json.dump(position, f, indent=2)


def clear_position():
    if os.path.exists(POSITION_FILE):
        os.remove(POSITION_FILE)


def buy_to_close(exchange, quantity: float):
    exchange.create_market_order(SYMBOL, Client.SIDE_SELL, quantity)


def manage_position(exchange, position: dict, current_price: float) -> dict | None:
    remaining = position.get("remaining_qty", QUANTITY)

    if current_price <= position["stop_loss"]:
        logger.info(f"SL hit at {current_price:.2f}")
        buy_to_close(exchange, remaining)
        clear_position()
        return None

    if current_price >= position["take_profit_2"]:
        logger.info(f"TP2 hit at {current_price:.2f}")
        buy_to_close(exchange, remaining)
        clear_position()
        return None

    if current_price >= position["take_profit_1"] and not position.get("tp1_hit"):
        logger.info(f"TP1 hit at {current_price:.2f}, closing 50%")
        half = remaining * 0.5
        buy_to_close(exchange, half)
        position["tp1_hit"] = True
        position["remaining_qty"] = remaining - half
        breakeven_sl = position["entry"] * 1.001
        if breakeven_sl > position["stop_loss"]:
            position["stop_loss"] = breakeven_sl
        save_position(position)

    return position


def main():
    logger.info("=" * 50)
    logger.info("CPR + ATR Spot Trading Bot Starting...")
    logger.info(f"Symbol: {SYMBOL}, Quantity: {QUANTITY}, Testnet: {USE_TESTNET}")
    logger.info(f"Params: SL={SL_ATR_MULT}×ATR, TP1={TP1_ATR_MULT}×ATR, TP2={TP2_ATR_MULT}×ATR, ATR period={ATR_PERIOD}")
    logger.info("=" * 50)

    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")
    if not api_key or not api_secret:
        logger.error("BINANCE_API_KEY and BINANCE_API_SECRET must be set in .env")
        return

    exchange = BinanceExchange(api_key, api_secret, testnet=USE_TESTNET)

    position = load_position()

    init_klines = exchange.get_klines(
        SYMBOL, Client.KLINE_INTERVAL_5MINUTE, limit=3
    )
    if init_klines and len(init_klines) >= 2:
        last_checked_candle = int(init_klines[-2][0])
    else:
        last_checked_candle = 0

    weekly_cpr_cache = None
    daily_cpr_cache = None
    weekly_cpr_last_fetch = 0.0
    daily_cpr_last_fetch = 0.0

    logger.info("Bot is running. Press Ctrl+C to stop.")

    while True:
        try:
            now = datetime.now(timezone.utc)
            now_ts = now.timestamp()

            current_price = exchange.get_symbol_ticker(SYMBOL)
            if current_price is None:
                time.sleep(CHECK_INTERVAL)
                continue

            if position is not None:
                position = manage_position(exchange, position, current_price)
                time.sleep(CHECK_INTERVAL)
                continue

            five_min_klines = exchange.get_klines(
                SYMBOL, Client.KLINE_INTERVAL_5MINUTE, limit=max(ATR_PERIOD + 5, 50)
            )
            if five_min_klines is None:
                time.sleep(CHECK_INTERVAL)
                continue

            now_ms = int(now_ts * 1000)
            new_candles = get_5min_closed_candles(
                five_min_klines, last_checked_candle, now_ms
            )
            if not new_candles:
                time.sleep(CHECK_INTERVAL)
                continue

            latest_candle = new_candles[-1]
            last_checked_candle = int(latest_candle[0])

            current_atr = calc_atr(five_min_klines, ATR_PERIOD)

            if (
                weekly_cpr_cache is None
                or (now_ts - weekly_cpr_last_fetch) > WEEKLY_CPR_CACHE_SECONDS
            ):
                week_start = (now - timedelta(days=now.weekday())).replace(
                    hour=0, minute=0, second=0, microsecond=0
                )
                weekly_klines = exchange.get_historical_klines(
                    SYMBOL,
                    Client.KLINE_INTERVAL_1HOUR,
                    start_str=int(week_start.timestamp() * 1000),
                )
                weekly_cpr_cache = compute_weekly_cpr_from_klines(weekly_klines)
                if weekly_cpr_cache is None:
                    time.sleep(CHECK_INTERVAL)
                    continue
                weekly_cpr_last_fetch = now_ts

            if (
                daily_cpr_cache is None
                or (now_ts - daily_cpr_last_fetch) > DAILY_CPR_CACHE_SECONDS
            ):
                daily_klines = exchange.get_klines(
                    SYMBOL, Client.KLINE_INTERVAL_1DAY, limit=3
                )
                if daily_klines and len(daily_klines) >= 2:
                    prev = daily_klines[-2]
                    prev_high = float(prev[2])
                    prev_low = float(prev[3])
                    prev_close = float(prev[4])
                    daily_cpr_cache = compute_daily_cpr(
                        prev_high, prev_low, prev_close
                    )
                    daily_cpr_last_fetch = now_ts
                else:
                    logger.warning("Not enough daily klines for CPR")
                    time.sleep(CHECK_INTERVAL)
                    continue

            if not is_buy_setup(weekly_cpr_cache, daily_cpr_cache, current_price):
                time.sleep(CHECK_INTERVAL)
                continue

            signal = check_buy_entry(
                latest_candle, daily_cpr_cache, current_atr,
                SL_ATR_MULT, TP1_ATR_MULT, TP2_ATR_MULT,
            )
            if signal:
                order = exchange.create_market_order(
                    SYMBOL, Client.SIDE_BUY, QUANTITY
                )
                if order:
                    position = {
                        "side": "BUY",
                        "entry": signal["entry"],
                        "stop_loss": signal["stop_loss"],
                        "take_profit_1": signal["take_profit_1"],
                        "take_profit_2": signal["take_profit_2"],
                        "tp1_hit": False,
                        "remaining_qty": QUANTITY,
                        "entry_time": now.isoformat(),
                    }
                    save_position(position)
                    logger.info(
                        f"BUY position opened — Entry: {position['entry']:.2f}, "
                        f"SL: {position['stop_loss']:.2f}, "
                        f"TP1: {position['take_profit_1']:.2f}, "
                        f"TP2: {position['take_profit_2']:.2f}"
                    )

        except KeyboardInterrupt:
            logger.info("Bot stopped by user.")
            break
        except Exception as e:
            logger.exception(f"Unexpected error: {e}")
            time.sleep(CHECK_INTERVAL)

    logger.info("Bot shutting down.")


class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"status": "running"}')
    def log_message(self, format, *args):
        logger.debug(f"Health: {format % args}")


def run_health_server():
    port = int(os.getenv("PORT", "10000"))
    server = HTTPServer(("0.0.0.0", port), HealthHandler)
    logger.info(f"Health server listening on port {port}")
    server.serve_forever()


thread = threading.Thread(target=run_health_server, daemon=True)
thread.start()

if __name__ == "__main__":
    main()
