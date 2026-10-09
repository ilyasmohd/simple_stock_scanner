"""
hull_trend_strategy.py

Python reimplementation of the EXACT "Hull Trend Strategy" formula tree
found in Chartink's indicator builder (confirmed from the Formulas tab),
not the generic TradingView "Hull Suite" script.

Chartink formula tree (length = 24):

    hma  (=a) = Wma( 2*Wma(Close, 12) - Wma(Close, 24), 5 )
                # standard Hull MA(24) on Close
                # 12 = length/2, 5 = round(sqrt(24))

    hma3 (=b) = Wma( Wma(Close, 4)*3 - Wma(Close, 6) - Wma(Close, 12), 12 )
                # Triangular Hull MA, computed with n = length/2 = 12
                # 4 = n/3, 6 = n/2, 12 = n

    crossdn = a crosses above b   -> BUY  (bullish flip, green)
    crossup = b crosses above a   -> SELL (bearish flip, red)

This is a two-line Hull crossover system (fast HMA vs. smoother THMA),
NOT the single-Hull-vs-itself-lagged-by-2-bars version used by the
generic TradingView Hull Suite script. The two produce different
signals, which is why the earlier version didn't match your chart.

No look-ahead: every value at row i only uses data up to and including
row i, exactly like the indicator painting bar-by-bar on the chart.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf


# --------------------------------------------------------------------------
# Weighted Moving Average + Hull Moving Average
# --------------------------------------------------------------------------

def wma(series: pd.Series, length: int) -> pd.Series:
    """Linearly-weighted moving average (matches Pine Script's wma())."""
    length = max(int(round(length)), 1)
    weights = np.arange(1, length + 1, dtype=float)

    def _w(x: np.ndarray) -> float:
        return float(np.dot(x, weights) / weights.sum())

    return series.rolling(length, min_periods=length).apply(_w, raw=True)


def hull_ma(src: pd.Series, length: int) -> pd.Series:
    """
    Standard Hull Moving Average -> Chartink's "hma" / line "a".
        HMA(n) = WMA( 2*WMA(src, n/2) - WMA(src, n), round(sqrt(n)) )
    """
    half_len = length / 2.0
    sqrt_len = round(np.sqrt(length))
    raw = 2 * wma(src, half_len) - wma(src, length)
    return wma(raw, sqrt_len)


def triangular_hull_ma(src: pd.Series, length: int) -> pd.Series:
    """
    Triangular Hull MA -> Chartink's "hma3" / line "b".
    Chartink calls this with n = length/2 (so for indicator length=24,
    n=12, matching the #4 / #6 / #12 periods seen in the formula tree).
        THMA(n) = WMA( WMA(src, n/3)*3 - WMA(src, n/2) - WMA(src, n), n )
    """
    n = length / 2.0
    raw = wma(src, n / 3.0) * 3 - wma(src, n / 2.0) - wma(src, n)
    return wma(raw, n)


# --------------------------------------------------------------------------
# "Hull Trend Strategy" (Chartink formula tree, confirmed from Formulas tab)
# --------------------------------------------------------------------------

@dataclass
class HullConfig:
    length: int = 24          # matches the {24} shown on your chart
    src_col: str = "Close"


def hull_trend_strategy(df: pd.DataFrame, cfg: HullConfig = HullConfig()) -> pd.DataFrame:
    """
    Adds a (hma), b (hma3/thma), trend color, and BUY/SELL flip signals.

    trend  = green when b > a (THMA above HMA), red when a > b
    BUY    = b crosses above a  (Chartink's "crossup")
    SELL   = a crosses above b  (Chartink's "crossdn", despite the name)

    NOTE: empirically verified against live Chartink values — the
    THMA line (b) sitting ABOVE the HMA line (a) is the bullish/green
    state, not the reverse. (b > a) matched a continuous green stretch
    from ~2026-07-06 through present on MACPOWER weekly.

    df must have a 'Close' column (or whatever cfg.src_col points to),
    ordered oldest -> newest (like a normal OHLCV frame).
    """
    out = df.copy()
    src = out[cfg.src_col]

    out["a"] = hull_ma(src, cfg.length)
    out["b"] = triangular_hull_ma(src, cfg.length)

    out["trend"] = np.where(out["b"] > out["a"], "green", "red")

    a_prev = out["a"].shift(1)
    b_prev = out["b"].shift(1)
    crossdn = (out["a"] > out["b"]) & (a_prev < b_prev)   # a crosses above b -> bearish
    crossup = (out["b"] > out["a"]) & (b_prev < a_prev)   # b crosses above a -> bullish

    out["signal"] = np.select([crossup, crossdn], ["BUY", "SELL"], default="")
    return out


# --------------------------------------------------------------------------
# Convenience: fetch NSE weekly data + run the indicator
# --------------------------------------------------------------------------

def fetch_weekly(symbol: str, period: str = "3y") -> pd.DataFrame:
    """
    yfinance weekly OHLCV for an NSE symbol.
    Uses auto_adjust=False to keep raw close, matching Chartink.
    """
    ticker = symbol if symbol.upper().endswith(".NS") else f"{symbol.upper()}.NS"
    df = yf.download(
        ticker,
        period=period,
        interval="1wk",
        auto_adjust=False,
        progress=False,
    )
    if df.empty:
        raise ValueError(f"No data returned for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index.name = "Date"
    return df


def run(symbol: str, length: int = 24, period: str = "3y") -> pd.DataFrame:
    df = fetch_weekly(symbol, period=period)
    result = hull_trend_strategy(df, HullConfig(length=length))
    cols = ["Open", "High", "Low", "Close", "a", "b", "trend", "signal"]
    return result[cols]


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _print_recent(result: pd.DataFrame, n: int) -> None:
    view = result.tail(n).copy()
    view["a"] = view["a"].round(2)
    view["b"] = view["b"].round(2)
    view["Close"] = view["Close"].round(2)
    print(view[["Close", "a", "b", "trend", "signal"]].to_string())


def _print_signals(result: pd.DataFrame) -> None:
    sig = result[result["signal"] != ""]
    if sig.empty:
        print("No BUY/SELL flips in the fetched window.")
        return
    print("Trend flips:")
    for date, row in sig.iterrows():
        print(f"  {date.date()}  {row['signal']:<4}  close={row['Close']:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hull Trend Strategy (Hull Suite) in Python")
    parser.add_argument("symbol", nargs="?", default="MACPOWER", help="NSE symbol, e.g. MACPOWER")
    parser.add_argument("--length", type=int, default=24, help="Hull length (default 24, matches chart)")
    parser.add_argument("--period", default="3y", help="yfinance period, e.g. 2y, 3y, 5y")
    parser.add_argument("--rows", type=int, default=15, help="how many recent rows to print")
    args = parser.parse_args()

    result = run(args.symbol, length=args.length, period=args.period)
    print(f"\n== {args.symbol.upper()} weekly | Hull length={args.length} ==\n")
    _print_recent(result, args.rows)
    print()
    _print_signals(result)

    current = result.iloc[-1]
    print(f"\nCurrent trend: {current['trend'].upper()}  "
          f"(a={current['a']:.2f} vs b={current['b']:.2f})")
