"""Load Chartink symbols and enrich them with portfolio indicators."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from stock_indicators import get_stock_indicators


CSV_FILE = Path(__file__).parent / "chartink_downloads" / "monthly_weekly_daily_rsi.csv"


def _number(value: str) -> float | None:
	"""Convert Chartink formatted numbers such as ``1,002.8`` or ``17.44%``."""
	cleaned = value.strip().replace(",", "").replace("%", "")
	if not cleaned:
		return None
	try:
		return float(cleaned)
	except ValueError:
		return None


def scan_chartink_symbols() -> list[dict[str, Any]]:
	"""Read Chartink symbols and calculate indicators for each one."""
	if not CSV_FILE.exists():
		raise FileNotFoundError(f"Chartink CSV not found: {CSV_FILE}")

	results: list[dict[str, Any]] = []
	with CSV_FILE.open(newline="", encoding="utf-8-sig") as csv_file:
		for row in csv.DictReader(csv_file):
			symbol = row.get("Symbol", "").strip()
			if not symbol:
				continue

			indicators = get_stock_indicators(
				symbol,
				current_price=_number(row.get("close", "")),
			)
			results.append(
				{
					"symbol": symbol.removesuffix("-BE"),
					"stock_name": row.get("Stock Name", "").strip(),
					"close": _number(row.get("close", "")),
					"change": _number(row.get("%_change", "")),
					"volume": row.get("volume", "").strip(),
					"sector": row.get("sector", "").strip(),
					"industry": row.get("industry", "").strip(),
					"indicators": indicators,
				}
			)
	return results
