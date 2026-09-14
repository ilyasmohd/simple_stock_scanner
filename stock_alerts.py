"""
Stock alerts for NSE uptrend candidates.

What this script implements:
1. Downloads daily NSE price history from Yahoo Finance.
2. Detects recent bullish EMA crossovers for EMA10 > EMA20 and EMA20 > EMA50.
3. Checks whether the latest completed daily RSI(14) is above 50 and the
   latest completed weekly RSI(14) is above 60, matching the confirmation
   rules used by scan_nse_all_stocks.py.
4. Alerts when daily RSI(14) crossed above 50 within the recent daily
    lookback, with weekly RSI confirmation.
5. Alerts when a confirmed EMA crossover happened within the last three
   completed daily bars.
6. Alerts when weekly RSI(14) crossed above 60 within the last three completed
   weekly bars, with daily RSI confirmation.
7. Prints alerts to the console and can optionally save them to a CSV file.

The script is a one-shot scanner. Run it once after the market data is updated,
or schedule it with Windows Task Scheduler for recurring alerts.

Examples:
    python stock_alerts.py --symbols SHYAMMETL CLEANMAX
    python stock_alerts.py --symbols SHYAMMETL --lookback 3
    python stock_alerts.py --file chartink_downloads/EQUITY_L.csv --output alerts.csv

Requires: yfinance, pandas, numpy
"""

from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import yfinance as yf

from scan_nse_all_stocks import calculate_rsi


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SYMBOLS = ["SHYAMMETL", "CLEANMAX"]
HISTORY_PERIOD = "2y"
RSI_PERIOD = 14
DAILY_RSI_MIN = 50.0
WEEKLY_RSI_MIN = 60.0
EMA_PAIRS = (
    ("EMA10", "EMA20", "EMA10>EMA20"),
    ("EMA20", "EMA50", "EMA20>EMA50"),
)


@dataclass
class Alert:
    symbol: str
    alert_type: str
    signal_date: str
    bars_ago: int
    close: float
    daily_rsi_14: float
    weekly_rsi_14: float
    crossover_type: str = ""


def normalize_symbol(symbol: str) -> str:
    """Return a plain NSE symbol suitable for Yahoo Finance."""
    normalized = symbol.strip().upper()
    if normalized.endswith(".NS"):
        normalized = normalized[:-3]
    if normalized.endswith("-BE"):
        normalized = normalized[:-3]
    return normalized


def fetch_completed_daily_close(symbol: str, period: str) -> pd.Series | None:
    """Download daily closes and remove today's possibly incomplete candle."""
    yahoo_symbol = f"{normalize_symbol(symbol)}.NS"
    try:
        history = yf.Ticker(yahoo_symbol).history(
            period=period,
            interval="1d",
            auto_adjust=False,
        )
    except Exception as error:
        print(f"[WARN] {symbol}: download failed ({error})", file=sys.stderr)
        return None

    if history.empty or "Close" not in history:
        print(f"[WARN] {symbol}: no closing history", file=sys.stderr)
        return None

    close = pd.to_numeric(history["Close"], errors="coerce").dropna()
    if close.empty:
        return None

    if close.index.tz is None:
        today = pd.Timestamp.now().normalize()
    else:
        today = pd.Timestamp.now(tz=close.index.tz).normalize()
    close = close[close.index.normalize() < today]
    return close if not close.empty else None


def add_emas(close: pd.Series) -> pd.DataFrame:
    """Calculate the EMA columns used by the crossover alerts."""
    return pd.DataFrame(
        {
            "Close": close,
            "EMA10": close.ewm(span=10, adjust=False).mean(),
            "EMA20": close.ewm(span=20, adjust=False).mean(),
            "EMA50": close.ewm(span=50, adjust=False).mean(),
        }
    )


def completed_weekly_close(daily_close: pd.Series) -> pd.Series:
    """Resample daily closes and exclude the current incomplete week."""
    weekly = daily_close.resample("W-FRI").last().dropna()
    if weekly.empty:
        return weekly
    latest_daily_date = daily_close.index[-1].normalize()
    return weekly[weekly.index.normalize() <= latest_daily_date]


def recent_cross_indices(
    fast: pd.Series,
    slow: pd.Series,
    lookback: int,
) -> list[tuple[int, int]]:
    """Return (index, bars_ago) for upward crosses in the recent bars."""
    if lookback <= 0:
        raise ValueError("Lookback must be greater than 0")

    crosses: list[tuple[int, int]] = []
    start = max(1, len(fast) - lookback)
    for index in range(start, len(fast)):
        crossed_up = fast.iloc[index - 1] <= slow.iloc[index - 1] and fast.iloc[index] > slow.iloc[index]
        if crossed_up:
            crosses.append((index, len(fast) - 1 - index))
    return crosses


def scan_symbol(
    symbol: str,
    period: str,
    daily_lookback: int,
    weekly_lookback: int,
) -> list[Alert]:
    """Return confirmed EMA and RSI alerts for one symbol."""
    symbol = normalize_symbol(symbol)
    daily_close = fetch_completed_daily_close(symbol, period)
    if daily_close is None or len(daily_close) < 60:
        return []

    daily = add_emas(daily_close)
    daily_rsi = calculate_rsi(daily_close, RSI_PERIOD)
    weekly_close = completed_weekly_close(daily_close)
    if len(weekly_close) <= RSI_PERIOD:
        return []
    weekly_rsi = calculate_rsi(weekly_close, RSI_PERIOD)

    latest_daily_rsi = float(daily_rsi.iloc[-1])
    latest_weekly_rsi = float(weekly_rsi.iloc[-1])
    if latest_daily_rsi <= DAILY_RSI_MIN or latest_weekly_rsi <= WEEKLY_RSI_MIN:
        return []

    alerts: list[Alert] = []
    daily_rsi_crosses = recent_cross_indices(
        daily_rsi,
        pd.Series(DAILY_RSI_MIN, index=daily_rsi.index),
        daily_lookback,
    )
    for index, bars_ago in daily_rsi_crosses:
        alerts.append(
            Alert(
                symbol=symbol,
                alert_type="CONFIRMED_DAILY_RSI_CROSS_ABOVE_50",
                signal_date=daily_rsi.index[index].date().isoformat(),
                bars_ago=bars_ago,
                close=round(float(daily_close.iloc[index]), 2),
                daily_rsi_14=round(latest_daily_rsi, 2),
                weekly_rsi_14=round(latest_weekly_rsi, 2),
            )
        )

    for fast_name, slow_name, crossover_type in EMA_PAIRS:
        fast = daily[fast_name]
        slow = daily[slow_name]
        for index, bars_ago in recent_cross_indices(fast, slow, daily_lookback):
            alerts.append(
                Alert(
                    symbol=symbol,
                    alert_type="CONFIRMED_DAILY_GOLDEN_CROSS",
                    signal_date=daily.index[index].date().isoformat(),
                    bars_ago=bars_ago,
                    close=round(float(daily["Close"].iloc[index]), 2),
                    daily_rsi_14=round(latest_daily_rsi, 2),
                    weekly_rsi_14=round(latest_weekly_rsi, 2),
                    crossover_type=crossover_type,
                )
            )

    weekly_crosses = recent_cross_indices(
        pd.Series(weekly_rsi.values, index=weekly_rsi.index),
        pd.Series(WEEKLY_RSI_MIN, index=weekly_rsi.index),
        weekly_lookback,
    )
    for index, bars_ago in weekly_crosses:
        alerts.append(
            Alert(
                symbol=symbol,
                alert_type="CONFIRMED_WEEKLY_RSI_CROSS_ABOVE_60",
                signal_date=weekly_rsi.index[index].date().isoformat(),
                bars_ago=bars_ago,
                close=round(float(daily_close.iloc[-1]), 2),
                daily_rsi_14=round(latest_daily_rsi, 2),
                weekly_rsi_14=round(latest_weekly_rsi, 2),
            )
        )
    return alerts


def load_symbols(path: str) -> list[str]:
    """Load symbols from either SYMBOL or symbol, or the first CSV column."""
    with Path(path).open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        fields = reader.fieldnames or []
        column = "SYMBOL" if "SYMBOL" in fields else "symbol" if "symbol" in fields else fields[0]
        return [normalize_symbol(row[column]) for row in reader if row.get(column, "").strip()]


def run_alert_scan(
    symbols: list[str],
    period: str,
    daily_lookback: int,
    weekly_lookback: int,
    workers: int,
) -> list[Alert]:
    """Scan symbols concurrently and print every alert as it is found."""
    alerts: list[Alert] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                scan_symbol,
                symbol,
                period,
                daily_lookback,
                weekly_lookback,
            ): symbol
            for symbol in symbols
        }
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                symbol_alerts = future.result()
            except Exception as error:
                print(f"[ERROR] {symbol}: {error}", file=sys.stderr)
                continue
            alerts.extend(symbol_alerts)
            for alert in symbol_alerts:
                print(
                    f"[ALERT] {alert.symbol} {alert.alert_type} "
                    f"on {alert.signal_date} ({alert.bars_ago} bars ago) | "
                    f"daily RSI={alert.daily_rsi_14:.2f}, "
                    f"weekly RSI={alert.weekly_rsi_14:.2f}"
                    + (f", {alert.crossover_type}" if alert.crossover_type else "")
                )
    return sorted(alerts, key=lambda alert: (alert.symbol, alert.signal_date, alert.alert_type))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NSE EMA and RSI stock alerts")
    parser.add_argument("--symbols", nargs="+", help="NSE symbols, e.g. SHYAMMETL CLEANMAX")
    parser.add_argument("--file", help="CSV containing a SYMBOL or symbol column")
    parser.add_argument("--period", default=HISTORY_PERIOD, help="Yahoo history period, default: 2y")
    parser.add_argument("--lookback", type=int, default=3, help="Recent completed daily bars, default: 3")
    parser.add_argument("--weekly-lookback", type=int, default=3, help="Recent completed weekly bars, default: 3")
    parser.add_argument("--workers", type=int, default=6, help="Parallel downloads, default: 6")
    parser.add_argument("--output", help="Optional CSV path for the alerts")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.symbols:
        symbols = [normalize_symbol(symbol) for symbol in args.symbols]
    elif args.file:
        symbols = load_symbols(args.file)
    else:
        symbols = DEFAULT_SYMBOLS
        print(f"No symbols supplied; using {', '.join(symbols)}")

    alerts = run_alert_scan(
        symbols,
        args.period,
        args.lookback,
        args.weekly_lookback,
        args.workers,
    )
    if not alerts:
        print("No confirmed alerts found.")
        return

    if args.output:
        pd.DataFrame([asdict(alert) for alert in alerts]).to_csv(args.output, index=False)
        print(f"Saved {len(alerts)} alert(s) to {args.output}")
    else:
        print(f"Found {len(alerts)} confirmed alert(s).")


if __name__ == "__main__":
    main()
