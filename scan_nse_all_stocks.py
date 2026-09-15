"""Scan all NSE symbols from EQUITY_L.csv using Yahoo Finance history."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf


PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_FILE = PROJECT_ROOT / "chartink_downloads" / "EQUITY_L.csv"
OUTPUT_FILE = PROJECT_ROOT / "chartink_downloads" / "nse_rsi_scan.csv"
RSI_PERIOD = 14
WEEKLY_RSI_MIN = 59.0
DAILY_RSI_MIN = 50.0
DAILY_RSI_MAX = 100.0
HISTORY_PERIOD = "5y"


def calculate_rsi(series: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
	"""Calculate Wilder RSI, matching the existing portfolio calculation."""
	if period <= 0:
		raise ValueError("RSI period must be greater than 0")

	values = pd.to_numeric(series, errors="coerce")
	delta = values.diff()
	gain = delta.clip(lower=0)
	loss = -delta.clip(upper=0)
	average_gain = gain.ewm(
		alpha=1 / period, adjust=False, min_periods=period
	).mean()
	average_loss = loss.ewm(
		alpha=1 / period, adjust=False, min_periods=period
	).mean()

	if len(gain) > period:
		average_gain.iloc[period] = gain.iloc[1 : period + 1].mean()
		average_loss.iloc[period] = loss.iloc[1 : period + 1].mean()
		for index in range(period + 1, len(values)):
			average_gain.iloc[index] = (
				(average_gain.iloc[index - 1] * (period - 1) + gain.iloc[index])
				/ period
			)
			average_loss.iloc[index] = (
				(average_loss.iloc[index - 1] * (period - 1) + loss.iloc[index])
				/ period
			)

	relative_strength = average_gain / average_loss.replace(0, np.nan)
	rsi = 100 - (100 / (1 + relative_strength))
	positive_only = average_loss.eq(0) & average_gain.gt(0)
	return rsi.mask(positive_only, 100).fillna(50)


def calculate_macd(close: pd.Series) -> pd.DataFrame:
	"""Calculate standard 12/26/9 MACD values."""
	fast = close.ewm(span=12, adjust=False).mean()
	slow = close.ewm(span=26, adjust=False).mean()
	macd = fast - slow
	signal = macd.ewm(span=9, adjust=False).mean()
	return pd.DataFrame(
		{"macd": macd, "signal": signal, "histogram": macd - signal}
	)


def _latest_number(value: Any) -> float | None:
	"""Convert a pandas/numpy value to a regular rounded float."""
	if value is None or pd.isna(value):
		return None
	return round(float(value), 2)


def _load_symbols() -> list[dict[str, str]]:
	"""Load non-empty symbols and metadata from the NSE master CSV."""
	with INPUT_FILE.open(newline="", encoding="utf-8-sig") as csv_file:
		reader = csv.DictReader(csv_file, skipinitialspace=True)
		if "SYMBOL" not in (reader.fieldnames or []):
			raise ValueError("EQUITY_L.csv must contain a SYMBOL column")
		return [
			row
			for row in reader
			if (row.get("SYMBOL") or "").strip()
		]


def _get_market_history(symbol: str) -> tuple[pd.Series, pd.Series]:
	"""Fetch one NSE symbol's daily close and volume history."""
	history = yf.Ticker(f"{symbol}.NS").history(
		period=HISTORY_PERIOD,
		auto_adjust=False,
	)
	if history.empty or "Close" not in history or "Volume" not in history:
		raise ValueError("No closing price and volume history found")
	close = pd.to_numeric(history["Close"], errors="coerce").dropna()
	if close.empty:
		raise ValueError("No closing prices found")
	volume = pd.to_numeric(history["Volume"], errors="coerce").reindex(close.index)
	return close, volume


def _calculate_values(
	close: pd.Series, volume: pd.Series
) -> dict[str, float | None]:
	"""Calculate the values used by the scan for one market history."""
	weekly_close = close.resample("W-FRI").last().dropna()
	daily_rsi = calculate_rsi(close).iloc[-1]
	weekly_rsi = calculate_rsi(weekly_close).iloc[-1]
	ema_20 = close.ewm(span=20, adjust=False).mean().iloc[-1]
	histogram = calculate_macd(close)["histogram"].iloc[-1]
	return {
		"close": _latest_number(close.iloc[-1]),
		"volume": _latest_number(volume.iloc[-1]),
		"daily_rsi_14": _latest_number(daily_rsi),
		"weekly_rsi_14": _latest_number(weekly_rsi),
		"ema_20": _latest_number(ema_20),
		"macd_histogram": _latest_number(histogram),
	}


def _matches_scan(values: dict[str, float | None]) -> bool:
	"""Return whether one symbol meets all requested technical conditions."""
	daily_rsi = values["daily_rsi_14"]
	weekly_rsi = values["weekly_rsi_14"]
	close = values["close"]
	ema_20 = values["ema_20"]
	histogram = values["macd_histogram"]
	return (
		daily_rsi is not None
		and weekly_rsi is not None
		and close is not None
		and ema_20 is not None
		and histogram is not None
		and DAILY_RSI_MIN <= daily_rsi <= DAILY_RSI_MAX
		and weekly_rsi >= WEEKLY_RSI_MIN
		and close >= ema_20
		#and histogram > 0 -- ignoring histogram filter for now
	)


def scan_symbols() -> list[dict[str, str | float | None]]:
	"""Fetch every NSE symbol and return the rows matching the scan."""
	matches: list[dict[str, str | float | None]] = []
	symbols = _load_symbols()
	for position, source_row in enumerate(symbols, start=1):
		symbol = (source_row["SYMBOL"] or "").strip().upper()
		try:
			close, volume = _get_market_history(symbol)
			values = _calculate_values(close, volume)
		except Exception as error:
			print(f"[{position}/{len(symbols)}] {symbol}: skipped ({error})")
			continue

		print(f"[{position}/{len(symbols)}] {symbol}: checked")
		if _matches_scan(values):
			matches.append(
				{
					"SYMBOL": symbol,
					"NAME OF COMPANY": source_row.get("NAME OF COMPANY", "").strip(),
					**values,
				}
			)
	matches.sort(key=lambda row: float(row["weekly_rsi_14"]))
	return matches


def save_matches(matches: list[dict[str, str | float | None]]) -> None:
	"""Write matching symbols and their calculated values beside the input CSV."""
	fieldnames = [
		"SYMBOL",
		"NAME OF COMPANY",
		"close",
		"volume",
		"daily_rsi_14",
		"weekly_rsi_14",
		"ema_20",
		"macd_histogram",
	]
	with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as csv_file:
		writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(matches)


def main() -> None:
	"""Run the NSE scan and save its matching symbols."""
	matches = scan_symbols()
	save_matches(matches)
	print(f"Saved {len(matches)} matching symbols to {OUTPUT_FILE}")


if __name__ == "__main__":
	main()
