"""Read holdings from a Zerodha Kite Connect account.

Install the client first:
    python -m pip install kiteconnect

Store the API credentials in zerodha_secrets.json beside this script.
The request token is obtained from the redirect URL after opening the Kite
login URL printed by this script. It is short-lived and should not be saved.
"""

import json
from pathlib import Path

from kiteconnect import KiteConnect


SECRETS_FILE = Path(__file__).with_name("zerodha_secrets.json")


def load_credentials() -> tuple[str, str]:
    """Load and validate Kite API credentials from the local secrets file."""
    try:
        with SECRETS_FILE.open(encoding="utf-8") as secrets_file:
            secrets = json.load(secrets_file)
        # The existing secrets file uses these two field names in reverse:
        # secret_key contains the Kite API key, and api_key contains its secret.
        api_key = secrets["secret_key"].strip()
        api_secret = secrets["api_key"].strip()
    except FileNotFoundError as error:
        raise RuntimeError(
            f"Create {SECRETS_FILE.name} beside this script first."
        ) from error
    except (KeyError, TypeError, AttributeError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"{SECRETS_FILE.name} must contain non-empty api_key and secret_key values."
        ) from error

    if not api_key or not api_secret:
        raise RuntimeError(
            f"{SECRETS_FILE.name} must contain non-empty api_key and secret_key values."
        )

    return api_key, api_secret


def read_portfolio() -> None:
    """Authenticate with Kite Connect and print the account holdings."""
    api_key, api_secret = load_credentials()

    kite = KiteConnect(api_key=api_key)
    print("\nOpen this URL in your browser and complete the Kite login:")
    print(kite.login_url())
    request_token = input(
        "\nPaste the request_token from the redirect URL: "
    ).strip()

    session_data = kite.generate_session(
        request_token,
        api_secret=api_secret,
    )
    kite.set_access_token(session_data["access_token"])

    holdings = kite.holdings()
    if not holdings:
        print("No holdings found.")
        return

    print(f"\nHoldings ({len(holdings)}):")
    for holding in holdings:
        symbol = holding.get("tradingsymbol", "Unknown")
        quantity = holding.get("quantity", 0)
        average_price = holding.get("average_price", 0)
        last_price = holding.get("last_price", 0)
        print(
            f"{symbol:20} quantity={quantity:<8} "
            f"average_price={average_price:<10} last_price={last_price}"
        )


if __name__ == "__main__":
    read_portfolio()