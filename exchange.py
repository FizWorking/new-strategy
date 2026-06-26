import logging
import time
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceRequestException

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT = 30


class BinanceExchange:
    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        self.client = Client(
            api_key,
            api_secret,
            testnet=testnet,
            requests_params={"timeout": REQUEST_TIMEOUT},
        )
        self.testnet = testnet
        logger.info(f"Binance client initialized (testnet={testnet})")

    def _call(self, fn, *args, **kwargs):
        for attempt in range(3):
            try:
                return fn(*args, **kwargs)
            except (BinanceAPIException, BinanceRequestException) as e:
                logger.warning(f"API call attempt {attempt+1} failed: {e}")
                if attempt < 2:
                    time.sleep(2**attempt)
        logger.error("API call failed after 3 attempts")
        return None

    def get_klines(self, symbol: str, interval: str, limit: int = 100):
        return self._call(
            self.client.get_klines, symbol=symbol, interval=interval, limit=limit
        )

    def get_historical_klines(
        self, symbol: str, interval: str, start_str, end_str=None
    ):
        return self._call(
            self.client.get_historical_klines,
            symbol=symbol,
            interval=interval,
            start_str=start_str,
            end_str=end_str,
        )

    def get_symbol_ticker(self, symbol: str) -> float | None:
        result = self._call(self.client.get_symbol_ticker, symbol=symbol)
        return float(result["price"]) if result else None

    def create_market_order(
        self, symbol: str, side: str, quantity: float
    ) -> dict | None:
        result = self._call(
            self.client.create_order,
            symbol=symbol,
            side=side,
            type=Client.ORDER_TYPE_MARKET,
            quantity=quantity,
        )
        if result:
            logger.info(f"Market order executed: {side} {quantity} {symbol}")
        return result


