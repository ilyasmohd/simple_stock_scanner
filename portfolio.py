"""Main FastAPI page for the Zerodha portfolio."""

from __future__ import annotations

import html
import importlib.util
import threading
import webbrowser
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from kiteconnect import KiteConnect

from daily_scan_insertion import CSV_FILE, get_scan_history, insert_daily_scan
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


def _load_chartink_scanner():
    """Load the hyphenated scanner module from this project directory."""
    scanner_path = Path(__file__).with_name("chartink-scanner.py")
    module_spec = importlib.util.spec_from_file_location(
        "chartink_scanner", scanner_path
    )
    if module_spec is None or module_spec.loader is None:
        raise ImportError(f"Could not load {scanner_path.name}")
    scanner = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(scanner)
    return scanner


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


@app.get("/chartink-scanner", response_class=HTMLResponse)
def chartink_scanner_dashboard() -> HTMLResponse:
    """Display Chartink CSV symbols with the same technical indicators."""
    try:
        scanner = _load_chartink_scanner()
        rows = scanner.scan_chartink_symbols()
        latest_scan, scan_statuses, dropped_rows = get_scan_history()
    except Exception as error:
        return HTMLResponse(
            f"<h3>Chartink scanner failed: {_display_value(error)}</h3>",
            status_code=500,
        )
    status_by_symbol = {row["symbol"]: row for row in scan_statuses}
    for row in rows:
        status = status_by_symbol.get(row["symbol"], {})
        row["scan_status"] = status.get("status")
        row["current_streak"] = status.get("current_streak")
    return render_chartink_dashboard(rows, latest_scan, dropped_rows)


@app.post("/upload-daily-scan")
async def upload_daily_scan(file: UploadFile = File(...)) -> Response:
    """Replace the scanner CSV, store the scan, and open the scanner page."""
    filename = file.filename or ""
    if not filename.lower().endswith(".csv"):
        return HTMLResponse("<h3>Please upload a .csv file.</h3>", status_code=400)

    contents = await file.read()
    if not contents.strip():
        return HTMLResponse("<h3>The uploaded CSV is empty.</h3>", status_code=400)

    CSV_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = CSV_FILE.with_name(f".{CSV_FILE.name}.uploading")
    temporary_file.write_bytes(contents)
    temporary_file.replace(CSV_FILE)
    try:
        insert_daily_scan()
    except Exception as error:
        return HTMLResponse(
            f"<h3>Daily scan upload failed: {_display_value(error)}</h3>",
            status_code=400,
        )
    return RedirectResponse("/chartink-scanner", status_code=303)


def render_holdings(holdings: list[dict]) -> HTMLResponse:
    """Render holdings and technical indicators in the portfolio page."""
    rows = "".join(_holding_row(holding) for holding in holdings)
    if not rows:
        rows = "<tr><td colspan='13'>No holdings found.</td></tr>"

    return HTMLResponse(
        f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Zerodha Portfolio</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 32px; color: #17202a; }}
table {{ border-collapse: collapse; white-space: nowrap; }}
th, td {{ border: 1px solid #d7dee3; padding: 8px; text-align: right; }}
th {{ background: #387ed1; color: white; }}
th:first-child, td:first-child {{ text-align: left; }}
form {{ margin: 16px 0; }}
button {{ padding: 7px 12px; border: 1px solid #9fb3c5; border-radius: 3px; background: white; cursor: pointer; }}
</style></head><body>
<h2>Your Holdings ({len(holdings)})</h2>
<p><a href="/chartink-scanner">Chartink Scanner</a></p>
<form action="/upload-daily-scan" method="post" enctype="multipart/form-data">
<input type="file" name="file" accept=".csv,text/csv" required>
<button type="submit">Upload CSV and open scanner</button>
</form>
<table><thead><tr>
<th>Symbol</th><th>Qty</th><th>Avg Price</th><th>LTP</th><th>P&amp;L</th>
<th>RSI (14)</th><th>Weekly RSI (14)</th><th>Monthly RSI (14)</th>
<th>EMA 10</th><th>EMA 20</th><th>EMA 50</th><th>EMA 200</th>
<th>MACD Histogram</th>
</tr></thead><tbody>{rows}</tbody></table>
</body></html>"""
    )


def render_chartink_dashboard(
    rows: list[dict], latest_scan: str | None, dropped_rows: list[dict]
) -> HTMLResponse:
    """Render enriched Chartink rows in a portfolio-style dashboard."""
    table_rows = "".join(_chartink_row(row) for row in rows)
    if not table_rows:
        table_rows = "<tr><td colspan='18'>No Chartink symbols found.</td></tr>"
    status_counts = {"NEW": 0, "CONTINUING": 0}
    for row in rows:
        status = row.get("scan_status")
        if status in status_counts:
            status_counts[status] += 1
    dropped_table_rows = "".join(_dropped_row(row) for row in dropped_rows)
    if not dropped_table_rows:
        dropped_table_rows = "<tr><td colspan='5'>None since the previous scan.</td></tr>"
    scan_label = latest_scan or "No database scan yet"
    sectors = sorted({str(row.get("sector") or "") for row in rows if row.get("sector")})
    sector_options = "".join(
        f"<option value=\"{html.escape(sector, quote=True)}\">{_display_value(sector)}</option>"
        for sector in sectors
    )
    return HTMLResponse(
        f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Chartink Scanner</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 32px; color: #17202a; }}
table {{ border-collapse: separate; border-spacing: 0; white-space: nowrap; }}
th, td {{ border: 1px solid #d7dee3; padding: 8px; text-align: right; }}
th {{ background: #387ed1; color: white; position: sticky; top: 0; z-index: 2; }}
thead tr.filter-row th {{ background: #eaf1f8; padding: 5px; top: 38px; z-index: 3; }}
thead input, thead select {{ box-sizing: border-box; min-width: 92px; width: 100%; padding: 6px; border: 1px solid #b7c7d6; border-radius: 3px; background: white; color: #17202a; }}
thead select {{ min-width: 120px; }}
thead .range-filter {{ display: flex; gap: 4px; min-width: 150px; }}
thead .range-filter input {{ min-width: 0; width: 50%; }}
th:first-child, td:first-child, td:nth-child(2) {{ text-align: left; }}
a {{ display: inline-block; margin-bottom: 16px; }}
.summary {{ display: flex; gap: 24px; margin: 16px 0; }}
.summary strong {{ font-size: 1.25rem; }}
h3 {{ margin-top: 32px; }}
.grid-toolbar {{ display: flex; align-items: center; gap: 12px; margin: 12px 0 8px; }}
.grid-toolbar button {{ padding: 7px 12px; border: 1px solid #9fb3c5; border-radius: 3px; background: white; cursor: pointer; }}
.grid-scroll {{ max-height: 68vh; overflow: auto; border: 1px solid #b7c7d6; }}
.grid-scroll table {{ min-width: 1600px; width: 100%; }}
.grid-scroll tbody tr:nth-child(even) {{ background: #f7f9fb; }}
.grid-scroll tbody tr:hover {{ background: #e8f1fa; }}
</style></head><body>
<h2>Chartink Scanner ({len(rows)})</h2>
<p>Latest stored scan: <strong>{_display_value(scan_label)}</strong></p>
<div class="summary">
<div><strong>{status_counts['NEW']}</strong><br>New</div>
<div><strong>{status_counts['CONTINUING']}</strong><br>Continuing</div>
<div><strong>{len(dropped_rows)}</strong><br>Dropped</div>
</div>
<p><a href="/">Back to Portfolio</a></p>
<div class="grid-toolbar">
<strong id="visible-count">Showing {len(rows)} of {len(rows)}</strong>
<button type="button" id="clear-filters">Clear filters</button>
</div>
<div class="grid-scroll">
<table id="scanner-grid"><thead><tr>
<th>Symbol</th><th>Stock Name</th><th>Status</th><th>Streak</th><th>Close</th><th>Change %</th><th>Volume</th>
<th>Sector</th><th>Industry</th><th>Close vs EMA 10 %</th><th>RSI (14)</th><th>Weekly RSI (14)</th>
<th>Monthly RSI (14)</th><th>EMA 10</th><th>EMA 20</th><th>EMA 50</th>
<th>EMA 200</th><th>MACD Histogram</th>
</tr><tr class="filter-row">
<th><input type="search" data-filter-column="0" placeholder="Filter"></th>
<th><input type="search" data-filter-column="1" placeholder="Filter"></th>
<th><input type="search" data-filter-column="2" placeholder="Filter"></th>
<th><input type="search" data-filter-column="3" placeholder="Filter"></th>
<th><input type="search" data-filter-column="4" placeholder="Filter"></th>
<th><input type="search" data-filter-column="5" placeholder="Filter"></th>
<th><input type="search" data-filter-column="6" placeholder="Filter"></th>
<th><select data-filter-column="7"><option value="">All sectors</option>{sector_options}</select></th>
<th><input type="search" data-filter-column="8" placeholder="Filter"></th>
<th><input type="search" data-filter-column="9" placeholder="Filter"></th>
<th><div class="range-filter"><input type="number" step="any" data-filter-column="10" data-filter-bound="min" placeholder="From" aria-label="RSI minimum"><input type="number" step="any" data-filter-column="10" data-filter-bound="max" placeholder="To" aria-label="RSI maximum"></div></th>
<th><div class="range-filter"><input type="number" step="any" data-filter-column="11" data-filter-bound="min" placeholder="From" aria-label="Weekly RSI minimum"><input type="number" step="any" data-filter-column="11" data-filter-bound="max" placeholder="To" aria-label="Weekly RSI maximum"></div></th>
<th><div class="range-filter"><input type="number" step="any" data-filter-column="12" data-filter-bound="min" placeholder="From" aria-label="Monthly RSI minimum"><input type="number" step="any" data-filter-column="12" data-filter-bound="max" placeholder="To" aria-label="Monthly RSI maximum"></div></th>
<th><input type="search" data-filter-column="13" placeholder="Filter"></th>
<th><input type="search" data-filter-column="14" placeholder="Filter"></th>
<th><input type="search" data-filter-column="15" placeholder="Filter"></th>
<th><input type="search" data-filter-column="16" placeholder="Filter"></th>
<th><input type="search" data-filter-column="17" placeholder="Filter"></th>
</tr></thead><tbody>{table_rows}</tbody></table></div>
<h3>Dropped Since Previous Scan ({len(dropped_rows)})</h3>
<table><thead><tr><th>Symbol</th><th>Stock Name</th><th>Close</th><th>Sector</th><th>Industry</th></tr></thead>
<tbody>{dropped_table_rows}</tbody></table>
<script>
const grid = document.querySelector("#scanner-grid");
const bodyRows = Array.from(grid.tBodies[0].rows);
const filters = Array.from(grid.querySelectorAll("[data-filter-column]"));
const visibleCount = document.querySelector("#visible-count");
function applyFilters() {{
    let visible = 0;
    bodyRows.forEach((row) => {{
        const matches = filters.every((filter) => {{
            const cell = row.cells[Number(filter.dataset.filterColumn)];
            if (!filter.value) return true;
            if (filter.dataset.filterBound) {{
                const cellValue = Number.parseFloat(cell?.textContent ?? "");
                const filterValue = Number.parseFloat(filter.value);
                if (Number.isNaN(cellValue) || Number.isNaN(filterValue)) return false;
                return filter.dataset.filterBound === "min"
                    ? cellValue >= filterValue
                    : cellValue <= filterValue;
            }}
            return cell && cell.textContent.toLowerCase().includes(filter.value.toLowerCase());
        }});
        row.hidden = !matches;
        if (matches) visible += 1;
    }});
    visibleCount.textContent = `Showing ${{visible}} of ${{bodyRows.length}}`;
}}
filters.forEach((filter) => filter.addEventListener("input", applyFilters));
document.querySelector("#clear-filters").addEventListener("click", () => {{
    filters.forEach((filter) => {{ filter.value = ""; }});
    applyFilters();
}});
</script>
</body></html>"""
    )


def _chartink_row(row: dict) -> str:
    """Render one enriched Chartink result."""
    indicators = row.get("indicators", {})
    close_vs_ema_10 = _percentage_difference(row.get("close"), indicators.get("ema_10"))
    indicator_keys = (
        "rsi", "weekly_rsi_14", "monthly_rsi_14", "ema_10", "ema_20",
        "ema_50", "ema_200", "macd_histogram",
    )
    if indicators.get("indicator_error"):
        indicator_cells = "<td colspan='8'>N/A</td>"
    else:
        indicator_cells = "".join(
            f"<td>{_display_value(indicators.get(key))}</td>"
            for key in indicator_keys
        )
    return (
        "<tr>"
        f"<td>{_display_value(row.get('symbol'))}</td>"
        f"<td>{_display_value(row.get('stock_name'))}</td>"
        f"<td>{_display_value(row.get('scan_status'))}</td>"
        f"<td>{_display_value(row.get('current_streak'))}</td>"
        f"<td>{_display_value(row.get('close'))}</td>"
        f"<td>{_display_value(row.get('change'))}</td>"
        f"<td>{_display_value(row.get('volume'))}</td>"
        f"<td>{_display_value(row.get('sector'))}</td>"
        f"<td>{_display_value(row.get('industry'))}</td>"
        f"<td>{_display_value(close_vs_ema_10)}</td>"
        f"{indicator_cells}</tr>"
    )


def _percentage_difference(value: object, reference: object) -> str | None:
    """Return percentage difference between a value and its reference."""
    try:
        numeric_value = float(value)
        numeric_reference = float(reference)
    except (TypeError, ValueError):
        return None
    if numeric_reference == 0:
        return None
    return f"{((numeric_value - numeric_reference) / numeric_reference) * 100:.2f}%"


def _dropped_row(row: dict) -> str:
    """Render one symbol absent from the latest scan."""
    return (
        "<tr>"
        f"<td>{_display_value(row.get('symbol'))}</td>"
        f"<td>{_display_value(row.get('name'))}</td>"
        f"<td>{_display_value(row.get('close'))}</td>"
        f"<td>{_display_value(row.get('sector'))}</td>"
        f"<td>{_display_value(row.get('industry'))}</td>"
        "</tr>"
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
        "macd_histogram",
    )
    if indicators.get("indicator_error"):
        indicator_cells = "<td colspan='8'>N/A</td>"
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
