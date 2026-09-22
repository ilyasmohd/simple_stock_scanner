"""Scan NSE equity symbols and write the weekly-RSI stock alert workbook."""

from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_FILE = PROJECT_ROOT / "chartink_downloads" / "EQUITY_L.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "daily_nse_scans"
HISTORY_PERIOD = "5y"
RSI_PERIOD = 14
WEEKLY_RSI_MIN = 59.0
MAX_WORKERS = 1


def normalize_symbol(symbol: str) -> str:
    """Return a plain NSE symbol suitable for Yahoo Finance."""
    normalized = symbol.strip().upper()
    if normalized.endswith(".NS"):
        normalized = normalized[:-3]
    if normalized.endswith("-BE"):
        normalized = normalized[:-3]
    return normalized


def calculate_rsi(series: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """Calculate Wilder RSI, matching the existing scanner implementation."""
    if period <= 0:
        raise ValueError("RSI period must be greater than 0")

    values = pd.to_numeric(series, errors="coerce")
    delta = values.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    average_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    average_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    if len(values) > period:
        average_gain.iloc[period] = gain.iloc[1 : period + 1].mean()
        average_loss.iloc[period] = loss.iloc[1 : period + 1].mean()
        for index in range(period + 1, len(values)):
            average_gain.iloc[index] = (
                average_gain.iloc[index - 1] * (period - 1) + gain.iloc[index]
            ) / period
            average_loss.iloc[index] = (
                average_loss.iloc[index - 1] * (period - 1) + loss.iloc[index]
            ) / period

    relative_strength = average_gain / average_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + relative_strength))
    positive_only = average_loss.eq(0) & average_gain.gt(0)
    return rsi.mask(positive_only, 100).fillna(50)


def calculate_macd_histogram(close: pd.Series) -> pd.Series:
    """Calculate TradingView's standard 12/26/9 EMA MACD histogram."""
    fast = close.ewm(span=12, adjust=False).mean()
    slow = close.ewm(span=26, adjust=False).mean()
    macd = fast - slow
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd - signal


def calculate_ema_slope(close: pd.Series, period: int) -> float:
    """Return the latest one-day EMA change in price units."""
    ema = close.ewm(span=period, adjust=False).mean()
    return round(float(ema.iloc[-1] - ema.iloc[-2]), 4)


def completed_period_close(daily_close: pd.Series, rule: str) -> pd.Series:
    """Resample closes, including the currently forming period."""
    return daily_close.resample(rule).last().dropna()


def format_values(values: pd.Series, count: int) -> str:
    """Format the requested values oldest first and newest last."""
    return ", ".join(f"{float(value):.2f}" for value in values.iloc[-count:])


def load_symbol_rows(path: Path) -> list[dict[str, str]]:
    """Load only EQ-series symbols and names from the NSE equity master."""
    with path.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file, skipinitialspace=True)
        required = {"SYMBOL", "NAME OF COMPANY", "SERIES"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"EQUITY_L.csv is missing columns: {', '.join(sorted(missing))}")
        return [
            {
                "symbol": normalize_symbol(row["SYMBOL"]),
                "name": (row["NAME OF COMPANY"] or "").strip(),
            }
            for row in reader
            if (row.get("SERIES") or "").strip().upper() == "EQ"
            and (row.get("SYMBOL") or "").strip()
        ]


def fetch_history(symbol: str, period: str) -> pd.DataFrame | None:
    """Download daily OHLCV candles, including today's candle when available."""
    try:
        history = yf.Ticker(f"{normalize_symbol(symbol)}.NS").history(
            period=period, interval="1d", auto_adjust=False
        )
    except Exception as error:
        print(f"[WARN] {symbol}: download failed ({error})", file=sys.stderr)
        return None

    required = {"Open", "Close", "Volume"}
    if history.empty or not required.issubset(history.columns):
        print(f"[WARN] {symbol}: no OHLCV history", file=sys.stderr)
        return None
    history = history.dropna(subset=list(required)).copy()
    if history.empty:
        return None
    return history


def scan_symbol(symbol_row: dict[str, str], period: str) -> dict[str, object] | None:
    """Calculate one output row when the latest weekly RSI is above 59."""
    data = fetch_history(symbol_row["symbol"], period)
    if data is None:
        return None

    if data.index.tz is None:
        today = pd.Timestamp.now().normalize()
    else:
        today = pd.Timestamp.now(tz=data.index.tz).normalize()
    completed_data = data[data.index.normalize() < today]
    if len(completed_data) < 220:
        return None

    close = pd.to_numeric(data["Close"], errors="coerce")
    daily_rsi = calculate_rsi(close)
    weekly_close = completed_period_close(close, "W-FRI")
    monthly_close = completed_period_close(close, "ME")
    if len(weekly_close) < RSI_PERIOD + 6 or len(monthly_close) < RSI_PERIOD + 6:
        return None

    weekly_rsi = calculate_rsi(weekly_close)
    monthly_rsi = calculate_rsi(monthly_close)
    if float(weekly_rsi.iloc[-1]) <= WEEKLY_RSI_MIN:
        return None

    macd_histogram = calculate_macd_histogram(close)
    ema_10_slope = calculate_ema_slope(close, 10)
    ema_20_slope = calculate_ema_slope(close, 20)
    ema_50_slope = calculate_ema_slope(close, 50)
    latest_close = float(close.iloc[-1])

    return {
        "symbol": f"{symbol_row['symbol']},",
        "name": symbol_row["name"],
        "past 12 days return (%)": round((latest_close / float(close.iloc[-13]) - 1) * 100, 2),
        "past 10 days return (%)": round((latest_close / float(close.iloc[-11]) - 1) * 100, 2),
        "past 7 days return (%)": round((latest_close / float(close.iloc[-8]) - 1) * 100, 2),
        "past 5 days return (%)": round((latest_close / float(close.iloc[-6]) - 1) * 100, 2),
        "past 3 days return (%)": round((latest_close / float(close.iloc[-4]) - 1) * 100, 2),
        "day change (%)": round(float(close.pct_change().iloc[-1] * 100), 2),
        "ema 10 slope (daily)": ema_10_slope,
        "ema 20 slope (daily)": ema_20_slope,
        "ema 50 slope (daily)": ema_50_slope,
        "macd histogram (current and past 7 values)": format_values(macd_histogram, 8),
        "daily rsi": round(float(daily_rsi.iloc[-1]), 2),
        "daily rsi (past 7 values)": format_values(daily_rsi, 8),
        "weekly rsi": round(float(weekly_rsi.iloc[-1]), 2),
        "weekly rsi (past 5 values)": format_values(weekly_rsi.iloc[:-1], 5),
        "monthly rsi": round(float(monthly_rsi.iloc[-1]), 2),
        "monthly rsi (past 5 values)": format_values(monthly_rsi.iloc[:-1], 5),
    }


def scan_universe(symbol_rows: list[dict[str, str]], period: str, workers: int) -> list[dict[str, object]]:
    """Scan the EQ universe and return weekly-RSI matches."""
    rows: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(scan_symbol, symbol_row, period): symbol_row["symbol"]
            for symbol_row in symbol_rows
        }
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                row = future.result()
            except Exception as error:
                print(f"[ERROR] {symbol}: {error}", file=sys.stderr)
                continue
            if row is not None:
                rows.append(row)
            print(f"[CHECKED] {symbol}")
    return sorted(rows, key=lambda row: float(row["ema 10 slope (daily)"]), reverse=True)


def write_report(rows: list[dict[str, object]], output_dir: Path) -> Path:
    """Write the single-sheet dated Excel report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"stock_alerts-{pd.Timestamp.now().date().isoformat()}.xlsx"
    columns = [
        "symbol", "name", "past 12 days return (%)", "past 10 days return (%)",
        "past 7 days return (%)", "past 5 days return (%)", "past 3 days return (%)",
        "day change (%)", "ema 10 slope (daily)", "ema 20 slope (daily)",
        "ema 50 slope (daily)", "macd histogram (current and past 7 values)",
        "daily rsi", "daily rsi (past 7 values)", "weekly rsi",
        "weekly rsi (past 5 values)", "monthly rsi", "monthly rsi (past 5 values)",
    ]
    pd.DataFrame(rows, columns=columns).to_excel(output_path, index=False, sheet_name="Stock Alerts")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NSE weekly RSI stock alert scanner")
    parser.add_argument("--period", default=HISTORY_PERIOD, help="Yahoo history period, default: 5y")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS, help="Download workers")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Excel output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    symbol_rows = load_symbol_rows(INPUT_FILE)
    workers = max(1, args.workers)
    print(f"Scanning {len(symbol_rows)} EQ symbols from {INPUT_FILE}")
    rows = scan_universe(symbol_rows, args.period, workers)
    output_path = write_report(rows, Path(args.output_dir))
    print(f"Saved {len(rows)} matching symbols to {output_path}")


if __name__ == "__main__":
    main()
