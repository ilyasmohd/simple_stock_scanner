"""
EMA Golden Crossover Screener for NSE Stocks
==============================================

Detects two types of "golden crossovers" on the daily timeframe:
  1. EMA10 crossing above EMA20   (short-term golden cross)
  2. EMA20 crossing above EMA50   (medium-term golden cross)

A crossover is flagged when, comparing the most recent completed candle
to the one before it:
    fast_ema was <= slow_ema  ->  fast_ema is now > slow_ema

Data source: yfinance, NSE symbols with the ".NS" suffix, auto_adjust=False
(unadjusted prices) to match Chartink-style calculations.

Usage
-----
    python ema_golden_crossover.py                     # scan default watchlist
    python ema_golden_crossover.py --symbols RELIANCE TCS INFY
    python ema_golden_crossover.py --file symbols.csv   # CSV with a 'symbol' column
    python ema_golden_crossover.py --lookback 3         # flag crossovers in last 3 bars
    python ema_golden_crossover.py --period 1y --workers 8

Requires: yfinance, pandas
    pip install yfinance pandas
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import yfinance as yf

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

# Minimum bars needed to calculate the slowest EMA used by the screener.
MIN_BARS_REQUIRED = 50

DEFAULT_WATCHLIST = [
    "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
    "SBIN", "LT", "ITC", "AXISBANK", "KOTAKBANK",
]


@dataclass
class CrossoverResult:
    symbol: str
    crossover_type: str      # "EMA10>EMA20" or "EMA20>EMA50"
    cross_date: pd.Timestamp
    close: float
    fast_ema: float
    slow_ema: float
    bars_ago: int             # 0 = happened on the latest completed candle


# --------------------------------------------------------------------------
# Data fetch + indicator calculation
# --------------------------------------------------------------------------

def fetch_ohlc(symbol: str, period: str = "2y") -> Optional[pd.DataFrame]:
    """Fetch daily OHLC data for an NSE symbol via yfinance."""
    ticker = f"{symbol.upper()}.NS"
    try:
        df = yf.download(
            ticker,
            period=period,
            interval="1d",
            auto_adjust=False,
            progress=False,
        )
    except Exception as exc:
        print(f"  [WARN] {symbol}: download failed ({exc})", file=sys.stderr)
        return None

    if df is None or df.empty:
        print(f"  [WARN] {symbol}: no data returned", file=sys.stderr)
        return None

    # yfinance sometimes returns a MultiIndex column set for a single ticker
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.dropna(subset=["Close"])
    return df


def add_emas(df: pd.DataFrame) -> pd.DataFrame:
    """Attach EMA10, EMA20, EMA50 columns (pandas' EMA = standard EWM)."""
    df = df.copy()
    df["EMA10"] = df["Close"].ewm(span=10, adjust=False).mean()
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
    return df


# --------------------------------------------------------------------------
# Crossover detection
# --------------------------------------------------------------------------

def find_crossovers(
    symbol: str,
    df: pd.DataFrame,
    lookback: int = 1,
) -> list[CrossoverResult]:
    """
    Scan the last `lookback` completed bars for a fast-EMA-crosses-above
    -slow-EMA event, for both the (10,20) and (20,50) EMA pairs.
    """
    results: list[CrossoverResult] = []
    pairs = [("EMA10", "EMA20", "EMA10>EMA20"), ("EMA20", "EMA50", "EMA20>EMA50")]

    n = len(df)
    if n < MIN_BARS_REQUIRED:
        return results

    for fast_col, slow_col, label in pairs:
        for bars_ago in range(lookback):
            idx_today = n - 1 - bars_ago
            idx_yday = idx_today - 1
            if idx_yday < 0:
                continue

            fast_today = df[fast_col].iloc[idx_today]
            slow_today = df[slow_col].iloc[idx_today]
            fast_yday = df[fast_col].iloc[idx_yday]
            slow_yday = df[slow_col].iloc[idx_yday]

            crossed_up = (fast_yday <= slow_yday) and (fast_today > slow_today)
            if crossed_up:
                results.append(
                    CrossoverResult(
                        symbol=symbol,
                        crossover_type=label,
                        cross_date=df.index[idx_today],
                        close=float(df["Close"].iloc[idx_today]),
                        fast_ema=float(fast_today),
                        slow_ema=float(slow_today),
                        bars_ago=bars_ago,
                    )
                )
    return results


def scan_symbol(symbol: str, period: str, lookback: int) -> list[CrossoverResult]:
    df = fetch_ohlc(symbol, period=period)
    if df is None:
        return []
    df = add_emas(df)
    return find_crossovers(symbol, df, lookback=lookback)


# --------------------------------------------------------------------------
# Screener orchestration
# --------------------------------------------------------------------------

def load_symbols_from_csv(path: str) -> list[str]:
    df = pd.read_csv(path)
    col = "symbol" if "symbol" in df.columns else df.columns[0]
    return [str(s).strip().upper() for s in df[col].dropna().tolist()]


def run_screener(
    symbols: list[str],
    period: str = "2y",
    lookback: int = 1,
    max_workers: int = 6,
) -> pd.DataFrame:
    all_results: list[CrossoverResult] = []

    print(f"Scanning {len(symbols)} symbols for EMA golden crossovers...\n")

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(scan_symbol, sym, period, lookback): sym for sym in symbols
        }
        for fut in as_completed(futures):
            sym = futures[fut]
            try:
                res = fut.result()
                if res:
                    all_results.extend(res)
                    for r in res:
                        print(
                            f"  [HIT] {r.symbol:<12} {r.crossover_type:<12} "
                            f"on {r.cross_date.date()} "
                            f"(close={r.close:.2f}, fast={r.fast_ema:.2f}, slow={r.slow_ema:.2f})"
                        )
            except Exception as exc:
                print(f"  [ERROR] {sym}: {exc}", file=sys.stderr)

    if not all_results:
        print("\nNo golden crossovers found.")
        return pd.DataFrame()

    out = pd.DataFrame([r.__dict__ for r in all_results])
    out = out.sort_values(["crossover_type", "cross_date"], ascending=[True, False])
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NSE EMA Golden Crossover Screener")
    p.add_argument(
        "--symbols", nargs="+", default=None,
        help="NSE symbols without .NS suffix, e.g. RELIANCE TCS INFY",
    )
    p.add_argument(
        "--file", type=str, default=None,
        help="CSV file with a 'symbol' column to load the watchlist from",
    )
    p.add_argument(
        "--period", type=str, default="2y",
        help="yfinance history period, e.g. 1y, 2y, 5y (default: 2y)",
    )
    p.add_argument(
        "--lookback", type=int, default=1,
        help="Flag crossovers within the last N completed bars (default: 1 = latest bar only)",
    )
    p.add_argument(
        "--workers", type=int, default=6,
        help="Number of parallel download threads (default: 6)",
    )
    p.add_argument(
        "--output", type=str, default=None,
        help="Optional path to save results as CSV",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.file:
        symbols = load_symbols_from_csv(args.file)
    elif args.symbols:
        symbols = [s.upper() for s in args.symbols]
    else:
        symbols = DEFAULT_WATCHLIST
        print(f"No --symbols/--file given, using default watchlist ({len(symbols)} symbols).\n")

    results_df = run_screener(
        symbols,
        period=args.period,
        lookback=args.lookback,
        max_workers=args.workers,
    )

    if not results_df.empty:
        print("\n=== Summary ===")
        print(results_df.to_string(index=False))

        if args.output:
            results_df.to_csv(args.output, index=False)
            print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()
