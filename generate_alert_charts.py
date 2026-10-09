"""Generate full-HD sideways charts for symbols in an automated alert workbook."""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd

import sideways


PROJECT_ROOT = Path(__file__).resolve().parent
ALERTS_DIR = PROJECT_ROOT / "daily_nse_scans_alerts"
DEFAULT_YEARS = 2.0
CHART_FIGSIZE = (19.2, 10.8)
CHART_DPI = 100


def latest_alert_workbook(directory: Path) -> Path:
    """Return the most recently modified automated alert workbook."""
    workbooks = list(directory.glob("automated_alert_*.xlsx"))
    if not workbooks:
        raise FileNotFoundError(
            f"No automated_alert_*.xlsx workbook found in {directory}"
        )
    return max(workbooks, key=lambda workbook: workbook.stat().st_mtime)


def generate_charts(
    input_path: Path,
    output_dir: Path,
    years: float = DEFAULT_YEARS,
) -> tuple[int, int]:
    """Generate one full-HD sideways chart for each unique workbook symbol."""
    alerts = pd.read_excel(input_path)
    if "Symbol" not in alerts.columns:
        raise ValueError(f"{input_path} must contain a 'Symbol' column")

    symbols = (
        alerts["Symbol"]
        .dropna()
        .astype(str)
        .str.strip()
    )
    symbols = list(dict.fromkeys(symbol for symbol in symbols if symbol))
    if not symbols:
        raise ValueError(f"No symbols found in {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    succeeded = 0
    failed = 0
    for position, symbol in enumerate(symbols, start=1):
        chart_path = output_dir / f"{symbol}_{input_path.stem}_sideways.png"
        try:
            result = sideways.analyze_symbol(symbol, years=years)
            sideways.plot_result(
                result,
                title=f"{symbol} sideways detection",
                save=str(chart_path),
                figsize=CHART_FIGSIZE,
                dpi=CHART_DPI,
            )
        except Exception as error:
            failed += 1
            print(
                f"[{position}/{len(symbols)}] {symbol}: FAILED: {error}",
                file=sys.stderr,
            )
            continue

        succeeded += 1
        print(f"[{position}/{len(symbols)}] {symbol}: saved {chart_path}")

    return succeeded, failed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate full-HD sideways charts for an automated alert workbook."
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Alert workbook (defaults to the latest automated_alert_*.xlsx)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for charts (default: daily_nse_scans_alerts/<today's date>)",
    )
    parser.add_argument(
        "--years",
        type=float,
        default=DEFAULT_YEARS,
        help="Daily history to plot (default: 2 years)",
    )
    args = parser.parse_args()
    if args.years <= 0:
        parser.error("--years must be greater than 0")

    try:
        input_path = args.input or latest_alert_workbook(ALERTS_DIR)
        output_dir = args.output_dir or ALERTS_DIR / date.today().isoformat()
        succeeded, failed = generate_charts(input_path, output_dir, args.years)
    except (FileNotFoundError, ValueError, OSError) as error:
        parser.error(str(error))

    print(f"Generated {succeeded} chart(s); {failed} symbol(s) failed.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
