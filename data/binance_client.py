"""
Binance Futures client wrapper for V17.
Provides a singleton client with futures-specific helpers.
"""
import logging
import time
from binance.client import Client
from config.settings import API_KEY, API_SECRET

logger = logging.getLogger("BinanceClient")

_client_instance = None


def get_client() -> Client:
    """Return the singleton Binance Futures client, initialising it on first call."""
    global _client_instance
    if _client_instance is None:
        _client_instance = _create_client()
    return _client_instance


def _create_client() -> Client:
    c = Client(API_KEY, API_SECRET)
    c.API_URL = "https://fapi.binance.com"
    logger.info("✅ Binance Futures client initialised")
    return c


def fetch_futures_klines(symbol: str, interval: str, limit: int = 500):
    """Fetch klines from Binance Futures REST API.

    Returns a list of raw kline lists as returned by python-binance.
    """
    c = get_client()
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return c.futures_klines(symbol=symbol, interval=interval, limit=limit)
        except Exception as e:
            logger.warning(f"fetch_futures_klines {symbol} {interval} attempt {attempt+1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    return []


def fetch_exchange_info():
    """Return Binance Futures exchange info."""
    c = get_client()
    try:
        return c.futures_exchange_info()
    except Exception as e:
        logger.error(f"fetch_exchange_info: {e}")
        return {}


def fetch_futures_ticker():
    """Return all Binance Futures 24h tickers."""
    c = get_client()
    try:
        return c.futures_ticker()
    except Exception as e:
        logger.error(f"fetch_futures_ticker: {e}")
        return []


def place_futures_order(symbol: str, side: str, order_type: str = "MARKET",
                        quantity: float = 0.0, reduce_only: bool = False,
                        stop_price: float = None, time_in_force: str = "GTC"):
    """Place a real Binance Futures order.

    Only called when PAPER_TRADING is False.
    side: 'BUY' or 'SELL'
    order_type: 'MARKET', 'LIMIT', 'STOP_MARKET', 'TAKE_PROFIT_MARKET'
    """
    c = get_client()
    params = {
        "symbol": symbol,
        "side": side,
        "type": order_type,
        "quantity": quantity,
    }
    if reduce_only:
        params["reduceOnly"] = True
    if stop_price is not None:
        params["stopPrice"] = stop_price
    if order_type == "LIMIT":
        params["timeInForce"] = time_in_force
    try:
        result = c.futures_create_order(**params)
        logger.info(f"✅ Order placed: {symbol} {side} {order_type} qty={quantity}")
        return result
    except Exception as e:
        logger.error(f"place_futures_order {symbol} {side}: {e}")
        return None
