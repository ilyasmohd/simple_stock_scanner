"""Main FastAPI page for the Zerodha portfolio."""

from __future__ import annotations

import html
import math
import threading
import webbrowser
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pandas as pd
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, RedirectResponse, Response

from automated_selection import DEFAULT_YEARS, SIDEWAYS_CONFIG, evaluate_symbol
from zerodha_sell import get_kite_client, place_holding_sell_order
from zerodha_session import (
    router as session_router,
)


APP_HOST = "127.0.0.1"
APP_PORT = 8000
APP_URL = f"http://{APP_HOST}:{APP_PORT}"
SMALL_CASE_FILE = Path(__file__).resolve().parent / "masterdata_excels" / "Small_case.csv"
VOLUME_SIGNAL_CLASSES = {
    "Blue": "volume-blue",
    "Green": "volume-green",
    "Red": "volume-red",
    "Grey (Noise)": "volume-grey-noise",
}
app = FastAPI(title="Zerodha Portfolio")
app.include_router(session_router)


@app.get("/", response_class=HTMLResponse)
def home() -> Response:
    """Read today's token, fetch holdings, or start the login flow."""
    try:
        kite = get_kite_client()
        holdings = kite.holdings()
    except Exception:
        return RedirectResponse("/login", status_code=303)

    smallcase_symbols = load_smallcase_symbols(SMALL_CASE_FILE)
    regular_holdings = []
    smallcase_holdings = []
    for holding in holdings:
        symbol = _normalize_symbol(holding.get("tradingsymbol", ""))
        group = smallcase_holdings if symbol in smallcase_symbols else regular_holdings
        group.append(holding)

    for holding in regular_holdings + smallcase_holdings:
        _evaluate_holding(holding)

    return render_holdings(regular_holdings, smallcase_holdings)


def _sell_result(message: str, status_code: int) -> HTMLResponse:
    """Render an escaped sell-order result with a link back to holdings."""
    return HTMLResponse(
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Sell order</title></head><body>"
        f"<p>{html.escape(message, quote=True)}</p>"
        "<p><a href='/'>Return to portfolio</a></p></body></html>",
        status_code=status_code,
    )


@app.post("/sell/{symbol}", response_class=HTMLResponse)
def sell_holding(symbol: str) -> HTMLResponse:
    """Sell the full current holding through the reusable Zerodha sell service."""
    requested_symbol = _normalize_symbol(symbol)
    if not requested_symbol:
        return _sell_result("A valid trading symbol is required.", 400)

    try:
        kite = get_kite_client()
        holding = next(
            (
                item
                for item in kite.holdings()
                if _normalize_symbol(item.get("tradingsymbol", "")) == requested_symbol
            ),
            None,
        )
        if holding is None:
            return _sell_result(
                f"{requested_symbol} is not present in current Kite holdings.", 404
            )

        try:
            quantity_value = Decimal(str(holding.get("quantity", "")))
        except (InvalidOperation, TypeError, ValueError) as error:
            raise ValueError("Kite returned an invalid holding quantity.") from error
        if (
            not quantity_value.is_finite()
            or quantity_value <= 0
            or quantity_value != quantity_value.to_integral_value()
        ):
            return _sell_result(
                f"{requested_symbol} has no valid whole-share quantity to sell.", 400
            )

        quantity = int(quantity_value)
        order_id, limit_price = place_holding_sell_order(
            requested_symbol,
            quantity,
            kite_client=kite,
        )
    except Exception as error:
        return _sell_result(f"Sell order failed: {error}", 502)

    return _sell_result(
        f"Sell order placed for all {quantity} {requested_symbol} shares at "
        f"limit {limit_price:.2f}. Order ID: {order_id}",
        200,
    )


def _normalize_symbol(symbol: Any) -> str:
    """Normalize Kite and Yahoo symbol suffixes to the NSE base symbol."""
    normalized = str(symbol or "").strip().upper()
    for suffix in (".NS", "-BE"):
        normalized = normalized.removesuffix(suffix)
    return normalized


def load_smallcase_symbols(input_path: Path) -> set[str]:
    """Read and normalize the symbols listed in Small_case.csv."""
    symbols = pd.read_csv(input_path, usecols=["SYMBOL"], encoding="utf-8-sig")["SYMBOL"]
    return {
        _normalize_symbol(symbol)
        for symbol in symbols.dropna()
        if _normalize_symbol(symbol)
    }


def _evaluate_holding(holding: dict[str, Any]) -> None:
    """Attach the shared automated-selection decision to a Kite holding."""
    symbol = _normalize_symbol(holding.get("tradingsymbol", ""))
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
    holding["qualifies"] = status == "PASS"
    if result is not None:
        holding["sideways"] = result["Sideways"]
    elif status.startswith("SKIP:"):
        holding["sideways"] = None
    else:
        holding["sideways"] = "stock is sideways" in status


def _format_number(value: Any, decimals: int = 2) -> str:
    """Format a finite number for display, or show N/A for unavailable data."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not math.isfinite(number):
        return "N/A"
    return f"{number:,.{decimals}f}"


def _holding_row(holding: dict[str, Any], allow_sell: bool = False) -> str:
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

    signal = selection.get("Volume Signal")
    volume_signal = "N/A" if signal in (None, "None") else str(signal)
    values = (
        holding.get("tradingsymbol", ""),
        _format_number(holding.get("quantity"), 0),
        _format_number(holding.get("average_price")),
        _format_number(holding.get("last_price")),
        _format_number(holding.get("pnl")),
        _format_number(selection.get("Daily RSI(14)")),
        _format_number(selection.get("MACD Histogram"), 4),
        str(holding["sideways"]) if holding.get("sideways") is not None else "N/A",
        volume_signal,
        decision,
        criteria,
    )
    cells = []
    for index, value in enumerate(values):
        signal_class = VOLUME_SIGNAL_CLASSES.get(str(value)) if index == 8 else None
        class_attribute = f' class="{signal_class}"' if signal_class else ""
        cells.append(
            f"<td{class_attribute}>{html.escape(str(value), quote=True)}</td>"
        )
    if allow_sell:
        symbol = str(holding.get("tradingsymbol", "")).strip()
        try:
            quantity = Decimal(str(holding.get("quantity", "")))
            can_sell = (
                quantity.is_finite()
                and quantity > 0
                and quantity == quantity.to_integral_value()
            )
        except (InvalidOperation, TypeError, ValueError):
            can_sell = False
        if can_sell and symbol:
            action = (
                f'<form method="post" action="/sell/{quote(symbol, safe="")}" '
                'onsubmit="return confirm(\'Sell all current holdings at a limit '
                'price 1% below fresh LTP?\')">'
                '<button class="sell-button" type="submit">Sell all</button></form>'
            )
        else:
            action = '<button class="sell-button" type="button" disabled>Sell all</button>'
        cells.append(f"<td>{action}</td>")
    return f'<tr class="{row_class}">{"".join(cells)}</tr>'


def _render_holding_grid(
    title: str, holdings: list[dict], allow_sell: bool = False
) -> str:
    """Render one portfolio group as a separate holdings grid."""
    rows = "".join(_holding_row(holding, allow_sell) for holding in holdings)
    if not rows:
        column_count = 12 if allow_sell else 11
        rows = f"<tr><td colspan='{column_count}' class='empty'>No holdings found.</td></tr>"
    sell_header = "<th>Action</th>" if allow_sell else ""

    return f"""
<section class="portfolio-group">
<h2>{html.escape(title)} <span>{len(holdings)}</span></h2>
<div class="table-wrap">
<table><thead><tr>
<th>Symbol</th><th>Qty</th><th>Avg Price</th><th>LTP</th><th>P&amp;L</th>
<th>Daily RSI (14)</th><th>MACD Histogram</th><th>Sideways</th><th>Volume Signal</th><th>Decision</th><th>Criteria</th>{sell_header}
</tr></thead><tbody>{rows}</tbody></table>
</div>
</section>"""


def render_holdings(
    regular_holdings: list[dict], smallcase_holdings: list[dict] | None = None
) -> HTMLResponse:
    """Render regular and Smallcase holdings in separate grids."""
    smallcase_holdings = smallcase_holdings or []
    all_holdings = regular_holdings + smallcase_holdings
    qualifying_count = sum(bool(holding.get("qualifies")) for holding in all_holdings)
    flagged_count = len(all_holdings) - qualifying_count
    
    # Swapped variable assignments to fix the UI grouping bug
    regular_grid = _render_holding_grid(
        "Kite Holdings", regular_holdings, allow_sell=True
    )
    smallcase_grid = _render_holding_grid(
        "Small Case Stocks", smallcase_holdings, allow_sell=True
    )

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
.portfolio-group {{ margin-top: 24px; }}
.portfolio-group h2 {{ margin: 0 0 10px; font-size: 18px; }}
.portfolio-group h2 span {{ color: #52616b; font-size: 14px; font-weight: 400; }}
.table-wrap {{ overflow-x: auto; background: #fff; border: 1px solid #d5dde1; }}
table {{ width: 100%; border-collapse: collapse; white-space: nowrap; }}
th, td {{ border-bottom: 1px solid #e0e5e8; padding: 10px 12px; text-align: right; }}
th {{ position: sticky; top: 0; background: #244b5a; color: #fff; font-weight: 600; }}
th:first-child, td:first-child {{ text-align: left; }}
td:nth-child(11) {{ text-align: left; white-space: normal; min-width: 280px; }}
td:nth-child(12) {{ text-align: center; }}
.sell-button {{ border: 0; border-radius: 4px; padding: 7px 12px; background: #b42318; color: #fff; cursor: pointer; }}
.sell-button:disabled {{ background: #9aa4ad; cursor: not-allowed; }}
td.volume-blue {{ background: #d9eaf7 !important; color: #1f4e78 !important; font-weight: 700; }}
td.volume-green {{ background: #c6efce !important; color: #006100 !important; font-weight: 700; }}
td.volume-red {{ background: #ffc7ce !important; color: #9c0006 !important; font-weight: 700; }}
td.volume-grey-noise {{ background: #d9d9d9 !important; color: #404040 !important; font-weight: 700; }}
tr.flagged td {{ background: #fff0ef; color: #782b25; }}
tr.qualifies td {{ background: #eff8f1; }}
tr.flagged td:nth-child(10) {{ font-weight: 700; }}
.empty {{ text-align: center !important; padding: 28px; color: #52616b; }}
@media (max-width: 640px) {{ main {{ padding: 18px 12px; }} header {{ align-items: flex-start; }} h1 {{ font-size: 20px; }} .summary {{ flex-wrap: wrap; gap: 8px 16px; }} }}
</style></head><body>
<main>
<header><h1>Portfolio Dashboard</h1><a class="refresh" href="/">Refresh analysis</a></header>
<div class="summary"><span>Holdings <strong>{len(all_holdings)}</strong></span>
<span>Qualifying <strong>{qualifying_count}</strong></span>
<span>Flagged <strong>{flagged_count}</strong></span></div>
{smallcase_grid}
{regular_grid}
</main>
</body></html>"""
    )

if __name__ == "__main__":
    print(f"Starting portfolio app at {APP_URL}")
    print(f"Kite redirect URL: {APP_URL}/callback")
    threading.Timer(1.0, lambda: webbrowser.open(APP_URL)).start()
    uvicorn.run(app, host=APP_HOST, port=APP_PORT)
