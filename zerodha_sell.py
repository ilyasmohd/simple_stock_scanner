import csv
import os
from pathlib import Path
from urllib.parse import quote

import requests
from datetime import datetime
from kiteconnect import KiteConnect
from zerodha_session import get_todays_access_token, load_secrets


def load_static_ip_credentials():
    credentials_file = Path(__file__).resolve().parent / "static_ip" / "my_ips.csv"
    with credentials_file.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file, skipinitialspace=True)
        if reader.fieldnames is None:
            raise ValueError(f"Static IP credentials file is empty: {credentials_file}")

        required_columns = {
            "brokername",
            "userid",
            "password",
            "hostname",
            "portno",
            "status",
        }
        columns = {column.strip().lower() for column in reader.fieldnames}
        missing_columns = required_columns - columns
        if missing_columns:
            raise ValueError(
                f"Static IP credentials file is missing columns: "
                f"{', '.join(sorted(missing_columns))}"
            )

        matches = [
            {
                (key or "").strip().lower(): (value or "").strip()
                for key, value in row.items()
                if key is not None
            }
            for row in reader
        ]
        active_credentials = [
            row
            for row in matches
            if row.get("brokername", "").lower() == "zerodha"
            and row.get("status", "").lower() == "active"
        ]

    if len(active_credentials) != 1:
        raise ValueError(
            "Expected exactly one active zerodha row in "
            f"{credentials_file}, found {len(active_credentials)}"
        )

    credentials = active_credentials[0]
    username = credentials.get("userid", "")
    password = credentials.get("password", "")
    hostname = credentials.get("hostname", "")
    port = credentials.get("portno", "")
    if not username or not password or not hostname or not port:
        raise ValueError(
            f"The active zerodha row in {credentials_file} has an empty "
            "userid, password, hostname, or portno"
        )
    return username, password, hostname, port


# =====================================================================
# 1. CONFIGURE STATICIP.IN PROXY (Must be done before calling Kite APIs)
# =====================================================================
(
    STATICIP_USERNAME,
    STATICIP_PASSWORD,
    STATICIP_HOSTNAME,
    STATICIP_PORT,
) = load_static_ip_credentials()

# Construct the Proxy URL string
PROXY_URL = (
    f"http://{quote(STATICIP_USERNAME, safe='')}:{quote(STATICIP_PASSWORD, safe='')}"
    f"@{STATICIP_HOSTNAME}:{STATICIP_PORT}"
)

# Inject the proxy into the Python environment variables
for var in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
    os.environ[var] = PROXY_URL

# Network Check: Print the active outgoing IP to confirm it matches your StaticIP.in dashboard
try:
    current_ip = requests.get("https://api64.ipify.org", timeout=5).text
    print(f"[SUCCESS] Proxy Connected! Traffic is routing from Static IP: {current_ip}")
except Exception as e:
    print(f"[ERROR] Proxy routing failed. Check credentials. Detail: {e}")


# =====================================================================
# 2. INITIALIZE ZERODHA KITE CONNECT
# =====================================================================
API_KEY, _ = load_secrets()
ACCESS_TOKEN = get_todays_access_token()
if not ACCESS_TOKEN:
    raise RuntimeError(
        "No valid Zerodha access token for today. Complete today's login through "
        "the portfolio app, then run this script again."
    )

kite = KiteConnect(api_key=API_KEY)
kite.set_access_token(ACCESS_TOKEN)


# =====================================================================
# 3. AUTO-DETECT ORDER VARIETY
# =====================================================================
def get_order_variety():
    """
    Returns kite.VARIETY_REGULAR or kite.VARIETY_AMO based on current NSE time.
    NSE Regular Hours: Mon-Fri, 9:15 AM to 3:30 PM.
    """
    now = datetime.now()
    weekday = now.weekday()  # 0 = Monday, 6 = Sunday
    current_time = now.time()

    # Define strict market boundaries
    market_open = datetime.strptime("09:15:00", "%H:%M:%S").time()
    market_close = datetime.strptime("15:30:00", "%H:%M:%S").time()

    # Weekend check -> Always AMO
    if weekday >= 5:
        return kite.VARIETY_AMO

    # Weekday check -> Regular hours vs Off hours
    if market_open <= current_time <= market_close:
        return kite.VARIETY_REGULAR
    else:
        return kite.VARIETY_AMO

# =====================================================================
# 3. PLACE A SELL ORDER (Example: NSE Cash / Equity Market)
# =====================================================================
def place_equity_sell_order(symbol, qty, price=None, order_type="MARKET"):
    """
    Places a Sell Order for Indian Equities via Kite API.
    - For regular instant selling of your holdings, product='CNC' is used.
    - Set order_type="LIMIT" and provide a price if you don't want a MARKET order.
    """
    try:

        detected_variety = get_order_variety()
        variety_label = "AMO" if detected_variety == kite.VARIETY_AMO else "REGULAR"
        print(f"[INFO] Routing order as {variety_label} based on current time.")

        order_id = kite.place_order(
            variety=detected_variety,
            exchange=kite.EXCHANGE_NSE,
            tradingsymbol=symbol,               # e.g., "RELIANCE", "TCS", "INFY"
            transaction_type=kite.TRANSACTION_TYPE_SELL,
            quantity=qty,
            product=kite.PRODUCT_CNC,           # CNC = Cash & Carry (for long term equity holdings)
            order_type=order_type if order_type == "MARKET" else kite.ORDER_TYPE_LIMIT,
            price=price if order_type == "LIMIT" else 0,
            validity=kite.VALIDITY_DAY
        )
        print(f"[SUCCESS] Sell order placed successfully! Order ID: {order_id}")
        return order_id
        
    except Exception as e:
        print(f"[ORDER FAILED] Error placing sell order for {symbol}: {e}")
        return None

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
