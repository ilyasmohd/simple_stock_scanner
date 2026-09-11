"""Fetch NSE price history and calculate portfolio technical indicators."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf


EMA_PERIODS = (10, 20, 50, 200)


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI using Wilder-style exponential smoothing."""
    if period <= 0:
        raise ValueError("RSI period must be greater than 0")

    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    average_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    average_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    relative_strength = average_gain / average_loss.replace(0, np.nan)
    return (100 - (100 / (1 + relative_strength))).fillna(50)


def calculate_macd(close: pd.Series) -> pd.DataFrame:
    """Calculate the standard 12/26/9 EMA MACD values."""
    fast = close.ewm(span=12, adjust=False).mean()
    slow = close.ewm(span=26, adjust=False).mean()
    macd = fast - slow
    signal = macd.ewm(span=9, adjust=False).mean()
    return pd.DataFrame(
        {
            "macd": macd,
            "signal": signal,
            "histogram": macd - signal,
        }
    )


def _latest_number(value: Any) -> float | None:
    """Convert a pandas/numpy number to a JSON- and HTML-friendly float."""
    if value is None or pd.isna(value):
        return None
    return round(float(value), 2)


def get_stock_indicators(symbol: str, period: str = "1y") -> dict[str, Any]:
    """Return the latest RSI, EMA, and MACD values for an NSE symbol.

    Symbols from Kite are plain NSE symbols, while Yahoo Finance expects the
    ``.NS`` suffix. Existing suffixes are preserved.
    """
    yahoo_symbol = symbol if symbol.upper().endswith(".NS") else f"{symbol}.NS"
    try:
        history = yf.Ticker(yahoo_symbol).history(period=period, auto_adjust=False)
        if history.empty or "Close" not in history:
            return {"indicator_error": "No price history found"}

        close = history["Close"].dropna()
        if close.empty:
            return {"indicator_error": "No closing prices found"}

        indicators: dict[str, Any] = {
            "rsi": _latest_number(calculate_rsi(close).iloc[-1]),
        }
        for period_length in EMA_PERIODS:
            ema = close.ewm(span=period_length, adjust=False).mean()
            indicators[f"ema_{period_length}"] = _latest_number(ema.iloc[-1])

        macd = calculate_macd(close).iloc[-1]
        indicators["macd"] = _latest_number(macd["macd"])
        indicators["macd_signal"] = _latest_number(macd["signal"])
        indicators["macd_histogram"] = _latest_number(macd["histogram"])
        return indicators
    except Exception as error:
        return {"indicator_error": str(error)}


if __name__ == "__main__":
    print(get_stock_indicators("RELIANCE"))
