#!/usr/bin/env python3
"""
run_sideways_example.py - example of calling sideways.py and saving a CSV.

Put this file in the SAME folder as sideways.py, then run:

    python run_sideways_example.py                      # JAYKAY.NS, 3 years
    python run_sideways_example.py RELIANCE.NS --years 2
    python run_sideways_example.py JAYKAY.NS --min-duration 1 --out my_file.csv

Output CSV: one row per completed daily bar with
    sideways_status  "SIDEWAYS" / "NOT SIDEWAYS"   <- use this for decisions
    sideways         True / False (same thing, machine-friendly)
    sideways_backfilled  True / False  (chart-style shading; looks ahead, analysis only)
plus close, votes (0-5), each vote, and every indicator value.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import sideways as sw  # sideways.py must sit next to this file


def run(symbol: str = "JAYKAY.NS", years: float = 3.0, min_duration: int = 1, out_path: str | None = None):
    """Fetch data, detect sideways bars, save CSV, return the result DataFrame."""
    cfg = sw.SidewaysConfig(min_duration=min_duration)

    df = sw.fetch_nse_data(symbol, years=years)       # completed daily bars only
    res = sw.detect_sideways(df, cfg)

    res.insert(0, "sideways_status", res["sideways"].map({True: "SIDEWAYS", False: "NOT SIDEWAYS"}))

    out = Path(out_path) if out_path else Path(__file__).resolve().parent / f"{symbol.upper().replace('.NS', '')}_sideways.csv"
    res.index.name = "date"
    res.round(4).to_csv(out)

    last = res.iloc[-1]
    print(f"{symbol.upper()}  last completed bar: {res.index[-1].date()}  close: {last['close']:.2f}")
    print(f"Status now : {last['sideways_status']}  (votes {int(last['votes'])}/5, min_duration={min_duration})")
    print(f"Rows       : {len(res)}   sideways bars: {int(res['sideways'].sum())}")
    print(f"CSV saved  : {out}")

    periods = sw.sideways_periods(res)
    if not periods.empty:
        show = periods.copy()
        show["start"] = show["start"].dt.date
        show["end"] = show["end"].dt.date
        print("\nSideways periods:")
        print(show.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run sideways detection for one NSE symbol and save a CSV.")
    ap.add_argument("symbol", nargs="?", default="JAYKAY.NS", help="default: JAYKAY.NS")
    ap.add_argument("--years", type=float, default=3.0)
    ap.add_argument("--min-duration", type=int, default=1, help="bars a run must last before the flag turns on (default 1)")
    ap.add_argument("--out", help="CSV path (default: <SYMBOL>_sideways.csv next to this script)")
    args = ap.parse_args(argv)
    try:
        run(args.symbol, args.years, args.min_duration, args.out)
    except Exception as exc:  # noqa: BLE001 - show a clean message for the usual failures
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
