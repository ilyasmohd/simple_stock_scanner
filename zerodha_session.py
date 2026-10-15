"""Kite login and callback routes used by the portfolio application."""

from __future__ import annotations

import csv
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from kiteconnect import KiteConnect


PROJECT_ROOT = Path(__file__).parent
SECRET_FILE = PROJECT_ROOT / "zerodha_secrets.json"
DATABASE_FILE = PROJECT_ROOT / "db_operations" / "stocks_data.db"
router = APIRouter()


def load_secrets() -> tuple[str, str]:
    """Read the Kite API key and secret from the local JSON file."""
    try:
        with SECRET_FILE.open(encoding="utf-8") as secrets_file:
            secrets = json.load(secrets_file)
        api_key = str(secrets["api_key"]).strip()
        api_secret = str(secrets["secret_key"]).strip()
    except (FileNotFoundError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"{SECRET_FILE.name} must contain api_key and secret_key values."
        ) from error

    if not api_key or not api_secret:
        raise RuntimeError(f"{SECRET_FILE.name} contains empty credentials.")
    return api_key, api_secret


def get_static_ip_proxies() -> dict[str, str]:
    """Return Zerodha's static-IP proxy settings without changing process env."""
    credentials_file = Path(__file__).resolve().parent / "static_ip" / "my_ips.csv"
    with credentials_file.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file, skipinitialspace=True)
        if reader.fieldnames is None:
            raise ValueError(f"Static IP credentials file is empty: {credentials_file}")

        required_columns = {
            "brokername",
            "userid",
            "password",
            "hostname",
            "portno",
            "status",
        }
        columns = {column.strip().lower() for column in reader.fieldnames}
        missing_columns = required_columns - columns
        if missing_columns:
            raise ValueError(
                f"Static IP credentials file is missing columns: "
                f"{', '.join(sorted(missing_columns))}"
            )

        rows = [
            {
                (key or "").strip().lower(): (value or "").strip()
                for key, value in row.items()
                if key is not None
            }
            for row in reader
        ]
        active_credentials = [
            row
            for row in rows
            if row.get("brokername", "").lower() == "zerodha"
            and row.get("status", "").lower() == "active"
        ]

    if len(active_credentials) != 1:
        raise ValueError(
            "Expected exactly one active zerodha row in "
            f"{credentials_file}, found {len(active_credentials)}"
        )

    credentials = active_credentials[0]
    username = credentials.get("userid", "")
    password = credentials.get("password", "")
    hostname = credentials.get("hostname", "")
    port = credentials.get("portno", "")
    if not username or not password or not hostname or not port:
        raise ValueError(
            f"The active zerodha row in {credentials_file} has an empty "
            "userid, password, hostname, or portno"
        )

    proxy_url = (
        f"http://{quote(username, safe='')}:{quote(password, safe='')}"
        f"@{hostname}:{port}"
    )
    return {"http": proxy_url, "https": proxy_url}


def ensure_login_token_table() -> None:
    """Create the login-token table when the database is not initialized."""
    DATABASE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DATABASE_FILE) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS login_token (
                token TEXT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )


def get_todays_access_token() -> str | None:
    """Return today's stored access token, or None when it is missing or old."""
    ensure_login_token_table()
    with sqlite3.connect(DATABASE_FILE) as connection:
        row = connection.execute(
            """
            SELECT token, created_at
            FROM login_token
            ORDER BY datetime(created_at) DESC
            LIMIT 1
            """
        ).fetchone()

    if row is None:
        return None
    try:
        created_date = datetime.fromisoformat(str(row[1])).date()
    except ValueError:
        return None
    return str(row[0]) if created_date == datetime.now().date() else None


def save_access_token(access_token: str) -> None:
    """Replace the stored token and record when the new token was inserted."""
    ensure_login_token_table()
    with sqlite3.connect(DATABASE_FILE) as connection:
        connection.execute("DELETE FROM login_token")
        connection.execute(
            "INSERT INTO login_token (token, created_at) VALUES (?, ?)",
            (access_token, datetime.now().isoformat(sep=" ", timespec="seconds")),
        )


@router.get("/login")
def login() -> RedirectResponse:
    """Redirect the browser to Kite's login page."""
    api_key, _ = load_secrets()
    return RedirectResponse(KiteConnect(api_key=api_key).login_url())


@router.get("/callback", response_class=HTMLResponse)
def callback(request: Request) -> Response:
    """Exchange Kite's request token, save it, then return to the portfolio."""
    request_token = request.query_params.get("request_token")
    if not request_token:
        message = request.query_params.get("message", "No request token received.")
        return HTMLResponse(f"<h3>Login failed: {message}</h3>", status_code=400)

    try:
        api_key, api_secret = load_secrets()
        kite = KiteConnect(api_key=api_key, proxies=get_static_ip_proxies())
        session_data = kite.generate_session(request_token, api_secret=api_secret)
        save_access_token(session_data["access_token"])
    except Exception as error:
        return HTMLResponse(f"<h3>Login failed: {error}</h3>", status_code=500)

    return RedirectResponse("/", status_code=303)
