# Zerodha portfolio reader

## Setup

1. In the Kite developer console, set the app's redirect URL to this exact URL:

`http://127.0.0.1:8765/kite/callback`

Do not omit `:8765`. A URL without the port uses port 80, but this FastAPI app
listens on port 8765, so Kite would not reach the callback handler.

2. Keep the API credentials in `zerodha_secrets.json` beside `zerodha_portfolio.py`.
3. Install the dependencies if needed:

```powershell
.\venv-yfinz\Scripts\python.exe -m pip install fastapi uvicorn kiteconnect
```

CSV uploads from the portfolio page also require `python-multipart`:

```powershell
.\venv-yfinz\Scripts\python.exe -m pip install python-multipart
```

The portfolio indicator page also uses Yahoo Finance through `yfinance`:

```powershell
.\venv-yfinz\Scripts\python.exe -m pip install yfinance pandas numpy
```

4. Start the FastAPI app:

```powershell
.\venv-yfinz\Scripts\python.exe .\zerodha_portfolio.py
```

The app opens `http://127.0.0.1:8765` in the default browser. Select **Login
with Kite**, complete the Kite login, and Kite will redirect to the app's
callback. The app exchanges the short-lived request token automatically and
renders the holdings page. Each holding includes RSI (14), EMA 10/20/50/200,
and MACD, signal, and histogram values from [stock_indicators.py](stock_indicators.py).
API credentials and access tokens remain server-side.

The reusable Kite access token is stored in
`db_operations/stocks_data.db`, table `login_token`, with its insertion time
in `created_at`. When the stored token was created today, the app uses it
directly. When it is missing, from an earlier date, or rejected by Kite, the
app asks for a new Kite login and replaces the stored token and timestamp.
