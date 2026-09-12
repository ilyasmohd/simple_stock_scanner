"""Insert the daily Chartink scan into the stocks_scans table."""

from __future__ import annotations

import csv
import sqlite3
from datetime import date
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
CSV_FILE = PROJECT_ROOT / "chartink_downloads" / "monthly_weekly_daily_rsi.csv"
DATABASE_FILE = PROJECT_ROOT / "db_operations" / "stocks_data.db"

REQUIRED_COLUMNS = {
    "Stock Name",
    "Symbol",
    "close",
    "volume",
    "sector",
    "industry",
}


def _number(value: str) -> float | None:
    """Convert Chartink values such as ``1,002.80`` to numbers."""
    cleaned = value.strip().replace(",", "").replace("%", "")
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError as error:
        raise ValueError(f"Invalid numeric value: {value!r}") from error


def insert_daily_scan(scan_date: date | None = None) -> int:
    """Insert or update all rows from the Chartink CSV for one scan date."""
    if not CSV_FILE.exists():
        raise FileNotFoundError(f"Chartink CSV not found: {CSV_FILE}")
    if not DATABASE_FILE.exists():
        raise FileNotFoundError(
            f"Database not found: {DATABASE_FILE}. Run db_operations/create_table.py first."
        )

    scan_date = scan_date or date.today()
    rows: list[tuple[str, str, str, str, str, float | None, float | None]] = []

    with CSV_FILE.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        actual_columns = set(reader.fieldnames or [])
        missing_columns = REQUIRED_COLUMNS - actual_columns
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"CSV is missing required columns: {missing}")

        for row in reader:
            symbol = (row["Symbol"] or "").strip()
            if not symbol:
                continue

            rows.append(
                (
                    symbol.removesuffix("-BE"),
                    scan_date.isoformat(),
                    (row["Stock Name"] or "").strip(),
                    (row["sector"] or "").strip(),
                    (row["industry"] or "").strip(),
                    _number(row["close"] or ""),
                    _number(row["volume"] or ""),
                )
            )

    with sqlite3.connect(DATABASE_FILE) as connection:
        connection.executemany(
            """
            INSERT INTO stocks_scans
                (symbol, scan_date, name, sector, industry, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol, scan_date) DO UPDATE SET
                name = excluded.name,
                sector = excluded.sector,
                industry = excluded.industry,
                close = excluded.close,
                volume = excluded.volume
            """,
            rows,
        )

    return len(rows)


def get_scan_history() -> tuple[str | None, list[dict], list[dict]]:
    """Return the latest scan's streaks and symbols dropped from the prior scan."""
    if not DATABASE_FILE.exists():
        return None, [], []

    with sqlite3.connect(DATABASE_FILE) as connection:
        connection.row_factory = sqlite3.Row
        latest_scan = connection.execute(
            "SELECT MAX(scan_date) FROM stocks_scans"
        ).fetchone()[0]
        if latest_scan is None:
            return None, [], []

        current_rows = connection.execute(
            """
            WITH ranked AS (
                SELECT symbol, scan_date,
                       ROW_NUMBER() OVER (
                           PARTITION BY symbol ORDER BY scan_date
                       ) AS row_number
                FROM stocks_scans
            ), streak_groups AS (
                SELECT symbol, scan_date,
                       DATE(scan_date, '-' || row_number || ' days') AS group_date
                FROM ranked
            ), current_streaks AS (
                SELECT symbol, COUNT(*) AS current_streak,
                       MAX(scan_date) AS last_seen
                FROM streak_groups
                GROUP BY symbol, group_date
                HAVING MAX(scan_date) = ?
            )
            SELECT symbol, current_streak, last_seen,
                   CASE WHEN current_streak = 1
                        THEN 'NEW' ELSE 'CONTINUING' END AS status
            FROM current_streaks
            ORDER BY current_streak DESC, symbol
            """,
            (latest_scan,),
        ).fetchall()

        previous_scan = connection.execute(
            """
            SELECT MAX(scan_date)
            FROM stocks_scans
            WHERE scan_date < ?
            """,
            (latest_scan,),
        ).fetchone()[0]
        dropped_rows = []
        if previous_scan is not None:
            dropped_rows = connection.execute(
                """
                SELECT previous.symbol, previous.name, previous.close,
                       previous.sector, previous.industry
                FROM stocks_scans AS previous
                WHERE previous.scan_date = ?
                  AND NOT EXISTS (
                      SELECT 1
                      FROM stocks_scans AS current
                      WHERE current.scan_date = ?
                        AND current.symbol = previous.symbol
                  )
                ORDER BY previous.symbol
                """,
                (previous_scan, latest_scan),
            ).fetchall()

    return (
        str(latest_scan),
        [dict(row) for row in current_rows],
        [dict(row) for row in dropped_rows],
    )


if __name__ == "__main__":
    inserted_count = insert_daily_scan()
    print(f"Inserted or updated {inserted_count} rows for {date.today().isoformat()}.")