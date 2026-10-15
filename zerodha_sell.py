import logging
from datetime import datetime
from decimal import Decimal, InvalidOperation, ROUND_DOWN
from zoneinfo import ZoneInfo

from kiteconnect import KiteConnect
from zerodha_session import (
    get_static_ip_proxies,
    get_todays_access_token,
    load_secrets,
)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
logger.propagate = False

NSE_TIMEZONE = ZoneInfo("Asia/Kolkata")


def get_kite_client() -> KiteConnect:
    """Create an authenticated Kite client with its own static-IP proxy."""
    api_key, _ = load_secrets()
    access_token = get_todays_access_token()
    if not access_token:
        raise RuntimeError(
            "No valid Zerodha access token for today. Complete today's login "
            "through the portfolio app, then try again."
        )

    kite_client = KiteConnect(api_key=api_key, proxies=get_static_ip_proxies())
    kite_client.set_access_token(access_token)
    return kite_client


# =====================================================================
# 3. AUTO-DETECT ORDER VARIETY
# =====================================================================
def get_order_variety(kite_client: KiteConnect | None = None) -> str:
    """
    Returns kite.VARIETY_REGULAR or kite.VARIETY_AMO based on current NSE time.
    NSE Regular Hours: Mon-Fri, 9:15 AM to 3:30 PM.
    """
    kite_client = kite_client or get_kite_client()
    now = datetime.now(NSE_TIMEZONE)
    weekday = now.weekday()  # 0 = Monday, 6 = Sunday
    current_time = now.time()

    # Define strict market boundaries
    market_open = datetime.strptime("09:15:00", "%H:%M:%S").time()
    market_close = datetime.strptime("15:30:00", "%H:%M:%S").time()

    # Weekend check -> Always AMO
    if weekday >= 5:
        logger.info("NSE time is %s; routing order as AMO.", now.isoformat())
        return kite_client.VARIETY_AMO

    # Weekday check -> Regular hours vs Off hours
    if market_open <= current_time <= market_close:
        logger.info("NSE time is %s; routing order as REGULAR.", now.isoformat())
        return kite_client.VARIETY_REGULAR
    else:
        logger.info("NSE time is %s; routing order as AMO.", now.isoformat())
        return kite_client.VARIETY_AMO

# =====================================================================
# 3. PLACE A SELL ORDER (Example: NSE Cash / Equity Market)
# =====================================================================
def place_equity_sell_order(
    symbol: str,
    qty: int,
    price: float | None = None,
    order_type: str = "MARKET",
    *,
    kite_client: KiteConnect | None = None,
) -> str:
    """
    Places a Sell Order for Indian Equities via Kite API.
    - For regular instant selling of your holdings, product='CNC' is used.
    - Set order_type="LIMIT" and provide a price if you don't want a MARKET order.
    """
    if not symbol or not symbol.strip():
        raise ValueError("A trading symbol is required.")
    if isinstance(qty, bool) or not isinstance(qty, int) or qty <= 0:
        raise ValueError("The sell quantity must be a positive whole number.")
    if order_type not in {"MARKET", "LIMIT"}:
        raise ValueError("Order type must be MARKET or LIMIT.")
    if order_type == "LIMIT":
        try:
            limit_price = Decimal(str(price))
        except (InvalidOperation, TypeError, ValueError) as error:
            raise ValueError("A positive limit price is required for a limit order.") from error
        if not limit_price.is_finite() or limit_price <= 0:
            raise ValueError("A positive limit price is required for a limit order.")
        price = float(limit_price)

    client = kite_client or get_kite_client()
    variety = get_order_variety(client)
    order_payload = {
        "variety": variety,
        "exchange": client.EXCHANGE_NSE,
        "tradingsymbol": symbol.strip().upper(),
        "transaction_type": client.TRANSACTION_TYPE_SELL,
        "quantity": qty,
        "product": client.PRODUCT_CNC,
        "order_type": (
            client.ORDER_TYPE_MARKET if order_type == "MARKET" else client.ORDER_TYPE_LIMIT
        ),
        "price": price if order_type == "LIMIT" else 0,
        "validity": client.VALIDITY_DAY,
    }
    logger.info("Submitting complete Zerodha sell order: %s", order_payload)

    try:
        order_id = client.place_order(**order_payload)
    except Exception:
        logger.exception("Zerodha sell order failed for %s", symbol.strip().upper())
        raise
    logger.info("Zerodha sell order accepted. Order ID: %s", order_id)
    return order_id


def place_holding_sell_order(
    symbol: str,
    qty: int,
    *,
    kite_client: KiteConnect | None = None,
) -> tuple[str, Decimal]:
    """Sell a holding at a limit price 1% below fresh LTP.

    This operation is independent of the portfolio UI so it can also be reused
    by a future automated sell workflow.
    """
    normalized_symbol = symbol.strip().upper()
    if not normalized_symbol:
        raise ValueError("A trading symbol is required.")
    if isinstance(qty, bool) or not isinstance(qty, int) or qty <= 0:
        raise ValueError("The holding quantity must be a positive whole number.")

    logger.info(
        "Received sell request: symbol=%s quantity=%d order_type=LIMIT "
        "limit_offset=1%% below fresh LTP",
        normalized_symbol,
        qty,
    )
    client = kite_client or get_kite_client()
    instrument = f"NSE:{normalized_symbol}"
    try:
        quote_data = client.ltp(instrument)
        last_price = Decimal(str(quote_data[instrument]["last_price"]))
    except (KeyError, InvalidOperation, TypeError) as error:
        logger.exception("Unable to obtain a valid LTP for %s", normalized_symbol)
        raise ValueError(f"Kite returned an invalid LTP for {normalized_symbol}.") from error
    except Exception:
        logger.exception("Kite LTP request failed for %s", normalized_symbol)
        raise
    if not last_price.is_finite() or last_price <= 0:
        raise ValueError(f"Kite returned an invalid LTP for {normalized_symbol}.")

    tick_size = Decimal("0.05")
    limit_price = (
        (last_price * Decimal("0.99") / tick_size).to_integral_value(
            rounding=ROUND_DOWN
        )
        * tick_size
    )
    if limit_price <= 0:
        raise ValueError(f"The calculated limit price for {normalized_symbol} is invalid.")

    logger.info(
        "Sell request priced: symbol=%s quantity=%d LTP=%s limit_price=%s",
        normalized_symbol,
        qty,
        last_price,
        limit_price,
    )
    order_id = place_equity_sell_order(
        symbol=normalized_symbol,
        qty=qty,
        price=float(limit_price),
        order_type="LIMIT",
        kite_client=client,
    )
    return order_id, limit_price

# =====================================================================
# 4. EXECUTION EXAMPLES
# =====================================================================
if __name__ == "__main__":
    # Example A: Market Sell Order (Executes instantly at current market price)
    # This sells 10 shares of Tata Motors from your existing CNC holdings
    print("\nAttempting Market Sell Order...")
    place_equity_sell_order(symbol="TATAMOTORS", qty=10, order_type="MARKET")
    
    # Example B: Limit Sell Order (Executes only when price hits target or higher)
    # This places a pending order to sell 5 shares of ITC if it touches ₹500
    print("\nAttempting Limit Sell Order...")
    place_equity_sell_order(symbol="ITC", qty=5, price=500.00, order_type="LIMIT")
