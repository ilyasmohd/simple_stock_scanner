import pandas as pd
import yfinance as yf   # pip install yfinance

def add_bands(df, length=55, mult=1.0):
    """Bollinger Bands on SMA (population std, same as TradingView)."""
    df = df.copy()
    df["mid"] = df["Close"].rolling(length).mean()
    sd = df["Close"].rolling(length).std(ddof=0)
    df["upper"] = df["mid"] + mult * sd
    df["lower"] = df["mid"] - mult * sd
    return df

df = yf.download("ROSSTECH.NS", period="2y", interval="1d", auto_adjust=False, progress=False)
df.columns = df.columns.get_level_values(0)   # flatten yfinance MultiIndex

df = add_bands(df, length=55, mult=1.0)
print(df[["Close", "upper", "mid", "lower"]].tail(10).round(2))