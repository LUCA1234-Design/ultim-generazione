"""
Binance Futures client wrapper for V17.
Provides a singleton client with futures-specific helpers.
"""
import logging
import time
from typing import Optional, Tuple
from binance.client import Client
from config.settings import API_KEY, API_SECRET

logger = logging.getLogger("BinanceClient")

_client_instance = None

# Rate-limit back-off: minimum sleep after a 429/418 response
_RATE_LIMIT_BACKOFF = 60.0   # seconds to wait after a 429 (rate-limited)
_IP_BAN_BACKOFF = 300.0      # seconds to wait after a 418 (temporary IP ban)


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


def _is_rate_limit_error(exc: Exception) -> bool:
    """Return True if the exception signals HTTP 429 (rate-limited)."""
    msg = str(exc)
    return "429" in msg or "Too Many Requests" in msg.lower()


def _is_ip_ban_error(exc: Exception) -> bool:
    """Return True if the exception signals HTTP 418 (temporary IP ban)."""
    return "418" in str(exc)


def fetch_futures_klines(symbol: str, interval: str, limit: int = 500,
                         start_time: Optional[int] = None,
                         end_time: Optional[int] = None):
    """Fetch klines from Binance Futures REST API.

    Handles HTTP 429 (rate-limited) and 418 (IP ban) responses with
    appropriate back-off to avoid making the situation worse.

    Returns a list of raw kline lists as returned by python-binance.
    """
    c = get_client()
    max_retries = 4
    for attempt in range(max_retries):
        try:
            params = {"symbol": symbol, "interval": interval, "limit": limit}
            if start_time is not None:
                params["startTime"] = int(start_time)
            if end_time is not None:
                params["endTime"] = int(end_time)
            return c.futures_klines(**params)
        except Exception as e:
            if _is_ip_ban_error(e):
                logger.error(
                    f"fetch_futures_klines: IP temporarily banned (418). "
                    f"Sleeping {_IP_BAN_BACKOFF}s before retry."
                )
                time.sleep(_IP_BAN_BACKOFF)
                continue
            if _is_rate_limit_error(e):
                logger.warning(
                    f"fetch_futures_klines: rate limited (429). "
                    f"Sleeping {_RATE_LIMIT_BACKOFF}s before retry."
                )
                time.sleep(_RATE_LIMIT_BACKOFF)
                continue
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
        if _is_rate_limit_error(e) or _is_ip_ban_error(e):
            logger.error(
                f"place_futures_order rate-limited/banned for {symbol} {side}: {e}"
            )
        else:
            logger.error(f"place_futures_order {symbol} {side}: {e}")
        return None


def place_limit_order(symbol: str, side: str, quantity: float, price: float,
                      time_in_force: str = "GTC") -> Optional[dict]:
    c = get_client()
    try:
        res = c.futures_create_order(
            symbol=symbol,
            side=side,
            type="LIMIT",
            quantity=quantity,
            price=price,
            timeInForce=time_in_force,
        )
        return res
    except Exception as e:
        logger.error(f"place_limit_order {symbol} {side}: {e}")
        return None


def cancel_order(symbol: str, order_id) -> bool:
    c = get_client()
    try:
        c.futures_cancel_order(symbol=symbol, orderId=order_id)
        return True
    except Exception as e:
        logger.debug(f"cancel_order {symbol} {order_id}: {e}")
        return False


def get_order_status(symbol: str, order_id) -> Optional[dict]:
    c = get_client()
    try:
        return c.futures_get_order(symbol=symbol, orderId=order_id)
    except Exception as e:
        logger.debug(f"get_order_status {symbol} {order_id}: {e}")
        return None


def get_best_bid_ask(symbol: str) -> Tuple[float, float]:
    c = get_client()
    try:
        ob = c.futures_order_book(symbol=symbol, limit=5)
        bids = ob.get("bids", [])
        asks = ob.get("asks", [])
        best_bid = float(bids[0][0]) if bids else 0.0
        best_ask = float(asks[0][0]) if asks else 0.0
        return best_bid, best_ask
    except Exception as e:
        logger.debug(f"get_best_bid_ask {symbol}: {e}")
        return 0.0, 0.0
