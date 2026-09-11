"""Main FastAPI page for the Zerodha portfolio."""

from __future__ import annotations

import html
import threading
import webbrowser

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from kiteconnect import KiteConnect

from stock_indicators import get_stock_indicators
from zerodha_session import (
    get_todays_access_token,
    load_secrets,
    router as session_router,
)


APP_HOST = "127.0.0.1"
APP_PORT = 8000
APP_URL = f"http://{APP_HOST}:{APP_PORT}"
app = FastAPI(title="Zerodha Portfolio")
app.include_router(session_router)


@app.get("/", response_class=HTMLResponse)
def home() -> Response:
    """Read today's token, fetch holdings, or start the login flow."""
    access_token = get_todays_access_token()
    if not access_token:
        return RedirectResponse("/login", status_code=303)

    try:
        api_key, _ = load_secrets()
        kite = KiteConnect(api_key=api_key)
        kite.set_access_token(access_token)
        holdings = kite.holdings()
    except Exception:
        return RedirectResponse("/login", status_code=303)

    for holding in holdings:
        symbol = holding.get("tradingsymbol", "")
        clean_symbol = symbol.removesuffix("-BE")
        holding["tradingsymbol"] = clean_symbol
        holding["indicators"] = get_stock_indicators(
            clean_symbol,
            current_price=holding.get("last_price"),
        )

    return render_holdings(holdings)


def render_holdings(holdings: list[dict]) -> HTMLResponse:
    """Render holdings and technical indicators in the portfolio page."""
    rows = "".join(_holding_row(holding) for holding in holdings)
    if not rows:
        rows = "<tr><td colspan='15'>No holdings found.</td></tr>"

    return HTMLResponse(
        f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Zerodha Portfolio</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 32px; color: #17202a; }}
table {{ border-collapse: collapse; white-space: nowrap; }}
th, td {{ border: 1px solid #d7dee3; padding: 8px; text-align: right; }}
th {{ background: #387ed1; color: white; }}
th:first-child, td:first-child {{ text-align: left; }}
</style></head><body>
<h2>Your Holdings ({len(holdings)})</h2>
<table><thead><tr>
<th>Symbol</th><th>Qty</th><th>Avg Price</th><th>LTP</th><th>P&amp;L</th>
<th>RSI (14)</th><th>Weekly RSI (14)</th><th>Monthly RSI (14)</th>
<th>EMA 10</th><th>EMA 20</th><th>EMA 50</th><th>EMA 200</th>
<th>MACD</th><th>MACD Signal</th><th>MACD Histogram</th>
</tr></thead><tbody>{rows}</tbody></table>
</body></html>"""
    )


def _display_value(value: object) -> str:
    """Escape a value before placing it in the HTML response."""
    return "N/A" if value is None or value == "" else html.escape(str(value))


def _holding_row(holding: dict) -> str:
    """Render one holding and its latest technical indicator values."""
    indicators = holding.get("indicators", {})
    indicator_keys = (
        "rsi",
        "weekly_rsi_14",
        "monthly_rsi_14",
        "ema_10",
        "ema_20",
        "ema_50",
        "ema_200",
        "macd",
        "macd_signal",
        "macd_histogram",
    )
    if indicators.get("indicator_error"):
        indicator_cells = "<td colspan='10'>N/A</td>"
    else:
        indicator_cells = "".join(
            f"<td>{_display_value(indicators.get(key))}</td>"
            for key in indicator_keys
        )

    return (
        "<tr>"
        f"<td>{_display_value(holding.get('tradingsymbol'))}</td>"
        f"<td>{_display_value(holding.get('quantity'))}</td>"
        f"<td>{_display_value(holding.get('average_price'))}</td>"
        f"<td>{_display_value(holding.get('last_price'))}</td>"
        f"<td>{_display_value(holding.get('pnl'))}</td>"
        f"{indicator_cells}</tr>"
    )


if __name__ == "__main__":
    print(f"Starting portfolio app at {APP_URL}")
    print(f"Kite redirect URL: {APP_URL}/callback")
    threading.Timer(1.0, lambda: webbrowser.open(APP_URL)).start()
    uvicorn.run(app, host=APP_HOST, port=APP_PORT)
