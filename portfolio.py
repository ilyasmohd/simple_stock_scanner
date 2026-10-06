"""Main FastAPI page for the Zerodha portfolio."""

from __future__ import annotations

import html
import math
import threading
import webbrowser
from typing import Any

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from kiteconnect import KiteConnect

from automated_selection import DEFAULT_YEARS, SIDEWAYS_CONFIG, evaluate_symbol
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
        _evaluate_holding(holding)

    return render_holdings(holdings)


def _evaluate_holding(holding: dict[str, Any]) -> None:
    """Attach the shared automated-selection decision to a Kite holding."""
    symbol = str(holding.get("tradingsymbol", "")).strip().upper().removesuffix("-BE")
    holding["tradingsymbol"] = symbol
    try:
        result, status = evaluate_symbol(
            symbol,
            company_name="",
            years=DEFAULT_YEARS,
            config=SIDEWAYS_CONFIG,
        )
    except Exception as error:
        result, status = None, f"SKIP: {error}"

    holding["selection_result"] = result
    holding["selection_status"] = status
    holding["qualifies"] = result is not None


def _format_number(value: Any, decimals: int = 2) -> str:
    """Format a finite number for display, or show N/A for unavailable data."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not math.isfinite(number):
        return "N/A"
    return f"{number:,.{decimals}f}"


def _holding_row(holding: dict[str, Any]) -> str:
    """Render one holding, marking any failed scanner condition in red."""
    qualifies = bool(holding.get("qualifies"))
    selection = holding.get("selection_result") or {}
    row_class = "qualifies" if qualifies else "flagged"
    decision = "QUALIFIES" if qualifies else "FLAGGED"

    if qualifies:
        criteria = (
            f"RSI {selection['Daily RSI(14)']:.2f}; "
            f"MACD histogram {selection['MACD Histogram']:.4f}; not sideways"
        )
    else:
        criteria = str(holding.get("selection_status", "Did not meet scanner criteria"))

    values = (
        holding.get("tradingsymbol", ""),
        _format_number(holding.get("quantity"), 0),
        _format_number(holding.get("average_price")),
        _format_number(holding.get("last_price")),
        _format_number(holding.get("pnl")),
        _format_number(selection.get("Daily RSI(14)")),
        _format_number(selection.get("MACD Histogram"), 4),
        decision,
        criteria,
    )
    cells = "".join(f"<td>{html.escape(str(value), quote=True)}</td>" for value in values)
    return f'<tr class="{row_class}">{cells}</tr>'


def render_holdings(holdings: list[dict]) -> HTMLResponse:
    """Render portfolio holdings with their automated-selection decisions."""
    rows = "".join(_holding_row(holding) for holding in holdings)
    if not rows:
        rows = "<tr><td colspan='9' class='empty'>No holdings found.</td></tr>"

    qualifying_count = sum(bool(holding.get("qualifies")) for holding in holdings)
    flagged_count = len(holdings) - qualifying_count

    return HTMLResponse(
        f"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Portfolio Dashboard</title>
<style>
body {{ margin: 0; background: #f4f6f7; color: #1f2933; font: 14px/1.45 system-ui, sans-serif; }}
main {{ max-width: 1500px; margin: 0 auto; padding: 28px 24px; }}
header {{ display: flex; align-items: baseline; justify-content: space-between; gap: 16px; }}
h1 {{ margin: 0; font-size: 24px; }}
.refresh {{ color: #145f77; }}
.summary {{ display: flex; gap: 20px; margin: 12px 0 20px; color: #52616b; }}
.summary strong {{ color: #1f2933; }}
.table-wrap {{ overflow-x: auto; background: #fff; border: 1px solid #d5dde1; }}
table {{ width: 100%; border-collapse: collapse; white-space: nowrap; }}
th, td {{ border-bottom: 1px solid #e0e5e8; padding: 10px 12px; text-align: right; }}
th {{ position: sticky; top: 0; background: #244b5a; color: #fff; font-weight: 600; }}
th:first-child, td:first-child {{ text-align: left; }}
td:last-child {{ text-align: left; white-space: normal; min-width: 280px; }}
tr.flagged td {{ background: #fff0ef; color: #782b25; }}
tr.qualifies td {{ background: #eff8f1; }}
tr.flagged td:nth-last-child(2) {{ font-weight: 700; }}
.empty {{ text-align: center !important; padding: 28px; color: #52616b; }}
@media (max-width: 640px) {{ main {{ padding: 18px 12px; }} header {{ align-items: flex-start; }} h1 {{ font-size: 20px; }} .summary {{ flex-wrap: wrap; gap: 8px 16px; }} }}
</style></head><body>
<main>
<header><h1>Portfolio Dashboard</h1><a class="refresh" href="/">Refresh analysis</a></header>
<div class="summary"><span>Holdings <strong>{len(holdings)}</strong></span>
<span>Qualifying <strong>{qualifying_count}</strong></span>
<span>Flagged <strong>{flagged_count}</strong></span></div>
<div class="table-wrap">
<table><thead><tr>
<th>Symbol</th><th>Qty</th><th>Avg Price</th><th>LTP</th><th>P&amp;L</th>
<th>Daily RSI (14)</th><th>MACD Histogram</th><th>Decision</th><th>Criteria</th>
</tr></thead><tbody>{rows}</tbody></table>
</div>
</main>
</body></html>"""
    )

if __name__ == "__main__":
    print(f"Starting portfolio app at {APP_URL}")
    print(f"Kite redirect URL: {APP_URL}/callback")
    threading.Timer(1.0, lambda: webbrowser.open(APP_URL)).start()
    uvicorn.run(app, host=APP_HOST, port=APP_PORT)
