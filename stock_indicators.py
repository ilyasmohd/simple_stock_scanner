"""Fetch NSE price history and calculate portfolio technical indicators."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf


EMA_PERIODS = (10, 20, 50, 200)


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate TradingView-compatible Wilder RSI."""
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

    # Seed Wilder's RMA with the first simple average, as TradingView does.
    first_gain = gain.iloc[1 : period + 1].mean()
    first_loss = loss.iloc[1 : period + 1].mean()
    if len(gain) > period:
        average_gain.iloc[period] = first_gain
        average_loss.iloc[period] = first_loss
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


def get_stock_indicators(
    symbol: str,
    period: str = "5y",
    current_price: float | None = None,
) -> dict[str, Any]:
    """Return the latest RSI, EMA, and MACD values for an NSE symbol.

    Symbols from Kite are plain NSE symbols, while Yahoo Finance expects the
    ``.NS`` suffix. The Kite ``-BE`` series suffix is removed before lookup.
    """
    normalized_symbol = symbol.strip()
    has_nse_suffix = normalized_symbol.upper().endswith(".NS")
    base_symbol = normalized_symbol[:-3] if has_nse_suffix else normalized_symbol
    if base_symbol.upper().endswith("-BE"):
        base_symbol = base_symbol[:-3]
    yahoo_symbol = f"{base_symbol}.NS"
    try:
        history = yf.Ticker(yahoo_symbol).history(period=period, auto_adjust=False)
        if history.empty or "Close" not in history:
            return {"indicator_error": "No price history found"}

        close = history["Close"].dropna()
        if close.empty:
            return {"indicator_error": "No closing prices found"}

        if current_price is not None:
            live_price = float(current_price)
            if np.isfinite(live_price):
                if close.index.tz is None:
                    today = pd.Timestamp(datetime.now().date())
                else:
                    today = pd.Timestamp.now(tz=close.index.tz).normalize()
                latest_date = close.index[-1].normalize()
                if today > latest_date:
                    close.loc[today] = live_price
                    close = close.sort_index()

        weekly_close = close.resample("W-FRI").last().dropna()
        monthly_close = close.resample("ME").last().dropna()
        indicators: dict[str, Any] = {
            "rsi": _latest_number(calculate_rsi(close).iloc[-1]),
            "weekly_rsi_14": _latest_number(calculate_rsi(weekly_close).iloc[-1]),
            "monthly_rsi_14": _latest_number(
                calculate_rsi(monthly_close).iloc[-1]
            ),
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
