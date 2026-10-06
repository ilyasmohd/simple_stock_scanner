#!/usr/bin/env python3
"""
sideways.py - programmatic sideways / range-bound detection for daily bars.

Purpose
-------
Gate entries from trend-following signals (Hull trend, MACD, RSI ...) so they
are not taken while a stock is chopping inside a range.

Method (five independent votes, all computed on COMPLETED daily bars only)
-------------------------------------------------------------------------
1. Efficiency Ratio (Kaufman)      low  -> price is going nowhere per unit of travel
2. Whipsaw count                   many close/EMA crosses (or Hull flips) recently
3. Range & drift in ATR units      narrow range AND small net drift
4. Normalised EMA slope            flat EMA, measured in ATR so it is scale-free
5. ADX low / Choppiness Index high confirming vote (config selectable)

A bar is sideways when enough votes agree, with:
  * hysteresis      enter at `votes_enter`, leave only at `votes_exit` or fewer
  * min duration    a run must last `min_duration` bars before it is reported
  * structure veto  higher-highs+higher-lows (or lower-lows+lower-highs) is a
                    trend, never a range, whatever the votes say
  * breakout lift   a close beyond the prior range lifts the gate and starts a
                    short cool-down so the gate cannot flicker straight back on

Causality
---------
Every column except `sideways_backfilled` uses only data up to and including
the bar itself, so `sideways[t]` is safe to use for decisions at close of t.
`sideways_backfilled` re-labels the whole run once it is confirmed; it looks
ahead and is for charts / eyeballing against your own date ranges ONLY.

Usage
-----
    python sideways.py KOPRAN                      # fetch from Yahoo (.NS), print summary
    python sideways.py SSWL --years 2 --plot       # also show chart
    python sideways.py KOPRAN --csv kopran.csv     # dump every indicator column
    python sideways.py --selftest                  # synthetic-data test-suite

    from sideways import detect_sideways, fetch_nse_data, apply_gate
    df  = fetch_nse_data("KOPRAN", years=3)
    res = detect_sideways(df)                      # DataFrame of indicators + flags
    entries = my_entry_signal & ~res["sideways"]   # or apply_gate(my_entry_signal, res["sideways"])

Requires: numpy, pandas. Optional: yfinance (data fetch), matplotlib (plots).
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, replace
from datetime import time as dtime
from typing import Optional

import numpy as np
import pandas as pd

__all__ = [
    "SidewaysConfig",
    "detect_sideways",
    "sideways_periods",
    "apply_gate",
    "gate_report",
    "fetch_nse_data",
    "analyze_symbol",
    "plot_result",
]

_VOTE_COLS = ["v_er", "v_whipsaw", "v_range", "v_slope", "v_confirm"]


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SidewaysConfig:
    """All tunables in one place. Defaults are a sensible daily-bar starting point."""

    # look-backs (bars)
    er_window: int = 20
    whipsaw_window: int = 20
    ema_period: int = 20
    slope_lookback: int = 5
    range_window: int = 30
    atr_period: int = 14
    adx_period: int = 14
    chop_period: int = 14

    # vote thresholds
    er_max: float = 0.30            # ER <= this           -> choppy
    whipsaw_min: int = 4            # crosses >= this      -> choppy
    range_atr_max: float = 7.0      # (HH-LL)/ATR <= this  -> contained ...
    drift_atr_max: float = 2.5      # |net move|/ATR <= this (both needed)
    slope_atr_max: float = 0.60     # |EMA change over slope_lookback|/ATR <= this
    adx_max: float = 20.0           # ADX < this           -> no trend
    chop_min: float = 61.8          # Choppiness > this    -> range
    confirm_with: str = "adx"       # "adx" | "chop" | "any"

    # whipsaw source: close-vs-EMA crosses, or Hull trend flips (hull > hull[2])
    whipsaw_source: str = "ema"     # "ema" | "hull"
    hull_length: int = 21           # set to YOUR Hull length when source="hull"

    # state machine
    votes_enter: int = 3
    votes_exit: int = 1             # leave when votes <= this
    min_duration: int = 5
    breakout_lifts_gate: bool = True
    breakout_buffer_atr: float = 0.25
    cooldown_bars: int = 5

    # structure veto (HH+HL / LL+LH over two consecutive half-windows)
    structure_veto: bool = True
    structure_window: int = 30
    structure_tol_atr: float = 2.0

    # bars at the start with insufficient history are never flagged
    warmup_bars: int = 60

    def __post_init__(self) -> None:
        positive = (
            "er_window", "whipsaw_window", "ema_period", "slope_lookback",
            "range_window", "atr_period", "adx_period", "chop_period",
            "hull_length", "min_duration",
        )
        for name in positive:
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be >= 1")
        if self.chop_period < 2:
            raise ValueError("chop_period must be >= 2")
        if self.structure_window < 4:
            raise ValueError("structure_window must be >= 4")
        if self.confirm_with not in ("adx", "chop", "any"):
            raise ValueError("confirm_with must be 'adx', 'chop' or 'any'")
        if self.whipsaw_source not in ("ema", "hull"):
            raise ValueError("whipsaw_source must be 'ema' or 'hull'")
        if not 1 <= self.votes_enter <= len(_VOTE_COLS):
            raise ValueError(f"votes_enter must be within 1..{len(_VOTE_COLS)}")
        if not 0 <= self.votes_exit < self.votes_enter:
            raise ValueError("votes_exit must be >= 0 and < votes_enter (hysteresis)")
        if self.cooldown_bars < 0:
            raise ValueError("cooldown_bars must be >= 0")

    @property
    def effective_warmup(self) -> int:
        """Warm-up never smaller than what the indicators physically need."""
        needed = 2 * max(
            self.er_window,
            self.whipsaw_window + (2 * self.hull_length if self.whipsaw_source == "hull" else self.ema_period),
            self.ema_period + self.slope_lookback,
            self.range_window + 1,
            self.atr_period,
            2 * self.adx_period,
            self.chop_period,
            self.structure_window,
        )
        return max(self.warmup_bars, needed)


# --------------------------------------------------------------------------- #
# Indicator building blocks (all causal)
# --------------------------------------------------------------------------- #
def _rma(s: pd.Series, n: int) -> pd.Series:
    """Wilder's smoothing (RMA)."""
    return s.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)  # first bar: prev_close NaN -> falls back to high-low
    return tr


def _wma(s: pd.Series, n: int) -> pd.Series:
    w = np.arange(1, n + 1, dtype=float)
    ws = w.sum()
    return s.rolling(n, min_periods=n).apply(lambda x: float(np.dot(x, w) / ws), raw=True)


def _hull_ma(s: pd.Series, n: int) -> pd.Series:
    half = max(int(n / 2), 1)
    root = max(int(round(np.sqrt(n))), 1)
    return _wma(2.0 * _wma(s, half) - _wma(s, n), root)


def _adx(high: pd.Series, low: pd.Series, tr: pd.Series, n: int) -> pd.Series:
    up = high.diff()
    down = -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=high.index)
    atr_n = _rma(tr, n).replace(0.0, np.nan)
    plus_di = 100.0 * _rma(plus_dm, n) / atr_n
    minus_di = 100.0 * _rma(minus_dm, n) / atr_n
    denom = (plus_di + minus_di).replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / denom
    return _rma(dx, n)


def _choppiness(high: pd.Series, low: pd.Series, tr: pd.Series, n: int) -> pd.Series:
    span = (high.rolling(n, min_periods=n).max() - low.rolling(n, min_periods=n).min())
    span = span.replace(0.0, np.nan)
    tr_sum = tr.rolling(n, min_periods=n).sum()
    return 100.0 * np.log10(tr_sum / span) / np.log10(n)


def _efficiency_ratio(close: pd.Series, n: int) -> pd.Series:
    net = (close - close.shift(n)).abs()
    path = close.diff().abs().rolling(n, min_periods=n).sum().replace(0.0, np.nan)
    return net / path


def _direction_flips(direction: pd.Series, window: int) -> pd.Series:
    """Number of sign changes of `direction` inside the trailing window."""
    d = direction.replace(0.0, np.nan).ffill()
    prev = d.shift(1)
    flips = ((d != prev) & d.notna() & prev.notna()).astype(float)
    return flips.rolling(window, min_periods=window).sum()


# --------------------------------------------------------------------------- #
# Input handling
# --------------------------------------------------------------------------- #
def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise column names / index; validate OHLC input."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):  # recent yfinance returns (Price, Ticker)
        out.columns = out.columns.get_level_values(0)
    out.columns = [str(c).strip().lower().replace(" ", "_") for c in out.columns]
    missing = [c for c in ("high", "low", "close") if c not in out.columns]
    if missing:
        raise ValueError(f"missing required column(s): {missing} (need high, low, close)")
    if not isinstance(out.index, pd.DatetimeIndex):
        try:
            out.index = pd.to_datetime(out.index)
        except Exception as exc:  # noqa: BLE001
            raise ValueError("index must be (convertible to) a DatetimeIndex") from exc
    out = out[~out.index.duplicated(keep="last")].sort_index()
    for c in ("high", "low", "close"):
        out[c] = pd.to_numeric(out[c], errors="coerce").astype(float)
    out = out.dropna(subset=["high", "low", "close"])
    if (out["high"] < out["low"]).any():
        raise ValueError("found bars with high < low; data looks corrupt")
    return out


# --------------------------------------------------------------------------- #
# Core detector
# --------------------------------------------------------------------------- #
def detect_sideways(df: pd.DataFrame, config: Optional[SidewaysConfig] = None) -> pd.DataFrame:
    """
    Detect sideways regimes on a daily OHLC DataFrame (needs high, low, close).

    Returns a DataFrame indexed like the cleaned input with columns:
      indicators : atr, er, whipsaws, range_atr, drift_atr, slope_atr, adx, chop, ema
      votes      : v_er, v_whipsaw, v_range, v_slope, v_confirm, votes
      structure  : +1 (HH+HL), -1 (LL+LH), 0
      breakouts  : breakout_up, breakout_down
      result     : sideways (causal, use for gating), sideways_backfilled (look-ahead, charts only)
    Make sure the last row is a COMPLETED bar (see fetch_nse_data).
    """
    cfg = config or SidewaysConfig()
    d = _prepare(df)
    n = len(d)
    high, low, close = d["high"], d["low"], d["close"]

    tr = _true_range(high, low, close)
    atr = _rma(tr, cfg.atr_period)
    atr_safe = atr.replace(0.0, np.nan)

    # 1. efficiency ratio
    er = _efficiency_ratio(close, cfg.er_window)

    # 2. whipsaws
    ema = close.ewm(span=cfg.ema_period, adjust=False, min_periods=cfg.ema_period).mean()
    if cfg.whipsaw_source == "ema":
        direction = np.sign(close - ema)
    else:
        hma = _hull_ma(close, cfg.hull_length)
        direction = np.sign(hma - hma.shift(2))
    whipsaws = _direction_flips(direction, cfg.whipsaw_window)

    # 3. range and drift in ATR units
    hh = high.rolling(cfg.range_window, min_periods=cfg.range_window).max()
    ll = low.rolling(cfg.range_window, min_periods=cfg.range_window).min()
    range_atr = (hh - ll) / atr_safe
    drift_atr = (close - close.shift(cfg.range_window)).abs() / atr_safe

    # 4. normalised slope
    slope_atr = (ema - ema.shift(cfg.slope_lookback)) / atr_safe

    # 5. confirming indicators
    adx = _adx(high, low, tr, cfg.adx_period)
    chop = _choppiness(high, low, tr, cfg.chop_period)

    # votes (NaN comparisons are False -> a missing input never votes "sideways")
    v_er = er <= cfg.er_max
    v_whipsaw = whipsaws >= cfg.whipsaw_min
    v_range = (range_atr <= cfg.range_atr_max) & (drift_atr <= cfg.drift_atr_max)
    v_slope = slope_atr.abs() <= cfg.slope_atr_max
    if cfg.confirm_with == "adx":
        v_confirm = adx < cfg.adx_max
    elif cfg.confirm_with == "chop":
        v_confirm = chop > cfg.chop_min
    else:
        v_confirm = (adx < cfg.adx_max) | (chop > cfg.chop_min)
    votes_df = pd.concat([v_er, v_whipsaw, v_range, v_slope, v_confirm], axis=1)
    votes_df.columns = _VOTE_COLS
    votes_df = votes_df.astype(bool)
    votes = votes_df.sum(axis=1).astype(int)

    # structure: compare the recent half-window with the one before it
    half = cfg.structure_window // 2
    r_hi = high.rolling(half, min_periods=half).max()
    r_lo = low.rolling(half, min_periods=half).min()
    p_hi = high.shift(half).rolling(half, min_periods=half).max()
    p_lo = low.shift(half).rolling(half, min_periods=half).min()
    tol = cfg.structure_tol_atr * atr_safe
    up_struct = ((r_hi - p_hi) > tol) & ((r_lo - p_lo) > tol)
    dn_struct = ((p_hi - r_hi) > tol) & ((p_lo - r_lo) > tol)
    structure = up_struct.astype(int) - dn_struct.astype(int)

    # breakouts beyond the PRIOR range (excludes current bar)
    prior_hi = high.shift(1).rolling(cfg.range_window, min_periods=cfg.range_window).max()
    prior_lo = low.shift(1).rolling(cfg.range_window, min_periods=cfg.range_window).min()
    buf = cfg.breakout_buffer_atr * atr_safe
    breakout_up = (close > prior_hi + buf)
    breakout_dn = (close < prior_lo - buf)

    # validity: enough history and every input present
    needed = pd.concat([atr_safe, er, whipsaws, range_atr, drift_atr, slope_atr, adx, chop], axis=1)
    valid = needed.notna().all(axis=1).to_numpy(dtype=bool, copy=True)
    valid = valid & (np.arange(n) >= cfg.effective_warmup)

    sideways, run_confirmed = _run_state_machine(
        valid=valid,
        votes=votes.to_numpy(),
        structure=structure.to_numpy(),
        b_up=breakout_up.to_numpy(),
        b_dn=breakout_dn.to_numpy(),
        cfg=cfg,
    )
    backfilled = _backfill_runs(sideways, run_confirmed)

    res = pd.DataFrame(
        {
            "close": close,
            "atr": atr,
            "ema": ema,
            "er": er,
            "whipsaws": whipsaws,
            "range_atr": range_atr,
            "drift_atr": drift_atr,
            "slope_atr": slope_atr,
            "adx": adx,
            "chop": chop,
        },
        index=d.index,
    )
    for c in _VOTE_COLS:
        res[c] = votes_df[c]
    res["votes"] = votes
    res["structure"] = structure
    res["breakout_up"] = breakout_up
    res["breakout_down"] = breakout_dn
    res["sideways"] = run_confirmed
    res["sideways_backfilled"] = backfilled
    return res


def _run_state_machine(*, valid, votes, structure, b_up, b_dn, cfg: SidewaysConfig):
    """
    Hysteresis state machine. Returns (raw_state, confirmed_state) boolean arrays.
    raw_state is the hysteresis state; confirmed_state additionally requires the
    state to have persisted for cfg.min_duration bars (still strictly causal).
    """
    n = len(votes)
    raw = np.zeros(n, dtype=bool)
    confirmed = np.zeros(n, dtype=bool)
    in_state = False
    run = 0
    cooldown = 0
    for i in range(n):
        blocked = cooldown > 0
        if blocked:
            cooldown -= 1
        if not valid[i]:
            in_state, run = False, 0
            continue
        trend_struct = cfg.structure_veto and structure[i] != 0
        breakout = cfg.breakout_lifts_gate and (b_up[i] or b_dn[i])
        if in_state:
            if trend_struct:
                in_state = False
            elif breakout:
                in_state = False
                cooldown = cfg.cooldown_bars
            elif votes[i] <= cfg.votes_exit:
                in_state = False
        else:
            if votes[i] >= cfg.votes_enter and not blocked and not trend_struct and not breakout:
                in_state = True
        run = run + 1 if in_state else 0
        raw[i] = in_state
        confirmed[i] = in_state and run >= cfg.min_duration
    return raw, confirmed


def _backfill_runs(raw: np.ndarray, confirmed: np.ndarray) -> np.ndarray:
    """Mark whole runs (from their first raw bar) once any bar in them is confirmed."""
    n = len(raw)
    out = np.zeros(n, dtype=bool)
    i = 0
    while i < n:
        if raw[i]:
            j = i
            while j < n and raw[j]:
                j += 1
            if confirmed[i:j].any():
                out[i:j] = True
            i = j
        else:
            i += 1
    return out


# --------------------------------------------------------------------------- #
# Reporting helpers
# --------------------------------------------------------------------------- #
def sideways_periods(result: pd.DataFrame, column: str = "sideways_backfilled") -> pd.DataFrame:
    """List contiguous sideways runs: start, end, bars, range low/high, still_active."""
    flag = result[column].to_numpy(dtype=bool)
    rows = []
    i, n = 0, len(flag)
    while i < n:
        if flag[i]:
            j = i
            while j < n and flag[j]:
                j += 1
            seg = result.iloc[i:j]
            rows.append(
                {
                    "start": seg.index[0],
                    "end": seg.index[-1],
                    "bars": j - i,
                    "close_low": float(seg["close"].min()),
                    "close_high": float(seg["close"].max()),
                    "still_active": j == n,
                }
            )
            i = j
        else:
            i += 1
    return pd.DataFrame(rows, columns=["start", "end", "bars", "close_low", "close_high", "still_active"])


def apply_gate(signal: pd.Series, sideways: pd.Series) -> pd.Series:
    """
    Block entry signals on sideways bars. `signal` is a boolean Series of entries
    generated at the close of each bar; `sideways` is res["sideways"] (causal).
    Bars missing from `sideways` are treated as NOT sideways.
    """
    flag = sideways.reindex(signal.index).fillna(False).astype(bool)
    return signal.fillna(False).astype(bool) & ~flag


def gate_report(
    close: pd.Series,
    signal: pd.Series,
    sideways: pd.Series,
    horizons: tuple = (5, 10, 20),
) -> pd.DataFrame:
    """
    Compare forward close-to-close returns of entries taken on sideways bars (the
    ones the gate would block) vs on other bars. Approximation: entry at close of
    the signal bar, no costs/slippage. Use your full backtester for real numbers.
    """
    flag = sideways.reindex(signal.index).fillna(False).astype(bool)
    sig = signal.fillna(False).astype(bool)
    rows = []
    for h in horizons:
        fwd = close.shift(-h) / close - 1.0
        for label, mask in (("blocked (sideways)", sig & flag), ("allowed", sig & ~flag)):
            r = fwd.reindex(signal.index)[mask].dropna()
            rows.append(
                {
                    "horizon": h,
                    "group": label,
                    "entries": int(mask.sum()),
                    "evaluated": int(len(r)),
                    "avg_ret_%": float(r.mean() * 100) if len(r) else np.nan,
                    "win_rate_%": float((r > 0).mean() * 100) if len(r) else np.nan,
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Data fetch (Yahoo, NSE)
# --------------------------------------------------------------------------- #
def fetch_nse_data(
    symbol: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    years: float = 3.0,
    drop_incomplete_today: bool = True,
) -> pd.DataFrame:
    """
    Daily OHLCV for an NSE symbol via yfinance (unadjusted prices, `.NS` suffix
    added automatically). If today's bar is still forming (before ~15:45 IST) it
    is dropped so signals only ever use completed bars.
    """
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("yfinance is required for fetch_nse_data: pip install yfinance") from exc

    ticker = symbol.strip().upper()
    if not (ticker.endswith(".NS") or ticker.endswith(".BO") or ticker.startswith("^")):
        ticker += ".NS"
    if start is None:
        start = (pd.Timestamp.today().normalize() - pd.DateOffset(days=int(years * 365.25))).strftime("%Y-%m-%d")
    raw = yf.download(ticker, start=start, end=end, interval="1d", auto_adjust=False, progress=False)
    if raw is None or raw.empty:
        raise ValueError(f"no data returned for {ticker}; check the symbol / dates / connection")
    df = _prepare(raw)
    if drop_incomplete_today and len(df):
        now_ist = pd.Timestamp.now(tz="Asia/Kolkata")
        if df.index[-1].date() == now_ist.date() and now_ist.time() < dtime(15, 45):
            df = df.iloc[:-1]
    return df


def analyze_symbol(
    symbol: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    years: float = 3.0,
    config: Optional[SidewaysConfig] = None,
) -> pd.DataFrame:
    """Fetch + detect in one call."""
    return detect_sideways(fetch_nse_data(symbol, start=start, end=end, years=years), config)


# --------------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------------- #
def _attach_hover(fig, axes, result: pd.DataFrame):
    """
    Interactive tooltip: while the mouse is over the chart, show a guide line on
    both panels plus a box with the bar's date, close, votes and causal flag.
    Returns the handler (also kept alive on the figure).
    """
    import matplotlib.dates as mdates

    ax_price = axes[0]
    xs = mdates.date2num(result.index.to_pydatetime())
    closes = result["close"].to_numpy(dtype=float)
    votes = result["votes"].to_numpy()
    flags = result["sideways"].to_numpy(dtype=bool)
    dates = result.index

    guides = [ax.axvline(xs[0], color="grey", lw=0.8, ls="--", visible=False) for ax in axes]
    tip = ax_price.annotate(
        "", xy=(xs[0], closes[0]), xytext=(14, 14), textcoords="offset points",
        bbox=dict(boxstyle="round", fc="white", ec="grey", alpha=0.95), fontsize=9,
        arrowprops=dict(arrowstyle="-", color="grey"), visible=False, zorder=10,
    )

    def on_move(event) -> None:
        if event.inaxes not in axes or event.xdata is None:
            if tip.get_visible():
                tip.set_visible(False)
                for g in guides:
                    g.set_visible(False)
                fig.canvas.draw_idle()
            return
        i = int(np.clip(np.searchsorted(xs, event.xdata), 0, len(xs) - 1))
        if i > 0 and abs(xs[i - 1] - event.xdata) <= abs(xs[i] - event.xdata):
            i -= 1
        for g in guides:
            g.set_xdata([xs[i], xs[i]])
            g.set_visible(True)
        lo, hi = ax_price.get_xlim()
        right_side = (xs[i] - lo) / (hi - lo) > 0.7
        tip.xyann = (-14, 14) if right_side else (14, 14)
        tip.set_horizontalalignment("right" if right_side else "left")
        tip.xy = (xs[i], closes[i])
        tip.set_text(
            f"{dates[i]:%a %d-%b-%Y}\nClose {closes[i]:.2f}\n"
            f"Votes {votes[i]}/{len(_VOTE_COLS)} | sideways: {'yes' if flags[i] else 'no'}"
        )
        tip.set_visible(True)
        fig.canvas.draw_idle()

    cid = fig.canvas.mpl_connect("motion_notify_event", on_move)
    fig._sideways_hover = (cid, on_move)  # keep a reference so it is not garbage-collected
    return on_move


def plot_result(result: pd.DataFrame, title: str = "", save: Optional[str] = None) -> None:
    """Close + EMA with sideways runs shaded (backfilled, for eyeballing) and a vote panel."""
    import matplotlib

    if save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(13, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    ax1.plot(result.index, result["close"], lw=1.2, label="Close")
    ax1.plot(result.index, result["ema"], lw=1.0, alpha=0.8, label="EMA")
    ax1.fill_between(
        result.index, 0, 1, where=result["sideways_backfilled"].to_numpy(),
        transform=ax1.get_xaxis_transform(), alpha=0.18, color="orange", label="Sideways (backfilled)",
    )
    live = result["sideways"].to_numpy()
    ax1.scatter(result.index[live], result["close"][live], s=6, color="red", zorder=3, label="Sideways (causal flag)")
    ax1.set_title(title or "Sideways detection")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(alpha=0.25)
    ax2.step(result.index, result["votes"], where="post", lw=1.0)
    ax2.set_ylabel("votes")
    ax2.set_ylim(-0.2, len(_VOTE_COLS) + 0.2)
    ax2.grid(alpha=0.25)
    _attach_hover(fig, (ax1, ax2), result)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=130)
        plt.close(fig)
    else:
        plt.show()


# --------------------------------------------------------------------------- #
# Synthetic-data self test
# --------------------------------------------------------------------------- #
def _synthetic_ohlc(close: np.ndarray, rng: np.random.Generator, wick: float = 0.5) -> pd.DataFrame:
    idx = pd.bdate_range("2023-01-02", periods=len(close))
    open_ = np.concatenate([[close[0]], close[:-1]])
    scale = np.abs(np.diff(close, prepend=close[0])).mean() + 1e-9
    high = np.maximum(open_, close) + np.abs(rng.normal(0, wick * scale, len(close)))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, wick * scale, len(close)))
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)


def _ou(n: int, rng, mean=100.0, theta=0.12, sigma=0.9, x0=None) -> np.ndarray:
    x = np.empty(n)
    x[0] = mean if x0 is None else x0
    for t in range(1, n):
        x[t] = x[t - 1] + theta * (mean - x[t - 1]) + sigma * rng.normal()
    return x


def _trend(n: int, rng, start=100.0, drift=0.6, sigma=1.0) -> np.ndarray:
    return start + np.cumsum(drift + sigma * rng.normal(size=n))


def _selftest(verbose: bool = True) -> bool:
    ok = True

    def check(name: str, cond: bool, detail: str = "") -> None:
        nonlocal ok
        ok &= bool(cond)
        if verbose:
            print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")

    cfg = SidewaysConfig()
    seeds = range(20)
    rng_frac, trend_frac, pull_frac, rw_frac = [], [], [], []
    for s in seeds:
        rng = np.random.default_rng(s)
        r = detect_sideways(_synthetic_ohlc(_ou(400, rng), rng), cfg)
        rng_frac.append(r["sideways"].iloc[cfg.effective_warmup:].mean())
        rng = np.random.default_rng(1000 + s)
        r = detect_sideways(_synthetic_ohlc(_trend(400, rng), rng), cfg)
        trend_frac.append(r["sideways"].iloc[cfg.effective_warmup:].mean())
        rng = np.random.default_rng(2000 + s)  # uptrend with a 12-bar pullback in the middle
        seg = np.concatenate([_trend(150, rng), np.zeros(0)])
        pull = seg[-1] + np.cumsum(-0.5 + 0.8 * rng.normal(size=12))
        rest = pull[-1] + np.cumsum(0.6 + 1.0 * rng.normal(size=150))
        r = detect_sideways(_synthetic_ohlc(np.concatenate([seg, pull, rest]), rng), cfg)
        pull_frac.append(r["sideways"].iloc[cfg.effective_warmup:].mean())
        rng = np.random.default_rng(3000 + s)
        r = detect_sideways(_synthetic_ohlc(100 + np.cumsum(rng.normal(size=400)), rng), cfg)
        rw_frac.append(r["sideways"].iloc[cfg.effective_warmup:].mean())

    check("range-bound series flagged most of the time", np.mean(rng_frac) > 0.55, f"(mean {np.mean(rng_frac):.2f})")
    check("steady uptrend rarely flagged", np.mean(trend_frac) < 0.08, f"(mean {np.mean(trend_frac):.2f})")
    check("uptrend with short pullback rarely flagged", np.mean(pull_frac) < 0.10, f"(mean {np.mean(pull_frac):.2f})")
    if verbose:
        print(f"      info: driftless random walk flagged {np.mean(rw_frac):.2f} of the time")

    # range then breakout: gate must lift soon after the breakout
    rng = np.random.default_rng(7)
    base = _ou(220, rng)
    brk = base[-1] + np.cumsum(1.2 + 0.8 * rng.normal(size=80))
    r = detect_sideways(_synthetic_ohlc(np.concatenate([base, brk]), rng), cfg)
    in_range = r["sideways"].iloc[100:220].mean()
    after = r["sideways"].iloc[240:].mean()
    check("range then breakout: flagged inside range", in_range > 0.5, f"({in_range:.2f})")
    check("range then breakout: lifted after breakout", after < 0.1, f"({after:.2f})")

    # causality: truncating the data must not change earlier results
    rng = np.random.default_rng(11)
    df = _synthetic_ohlc(np.concatenate([_ou(200, rng), _trend(100, rng, start=100)]), rng)
    full = detect_sideways(df, cfg)
    causal_ok = True
    for k in (120, 160, 200, 230, 280):
        part = detect_sideways(df.iloc[:k], cfg)
        cols = ["er", "whipsaws", "range_atr", "adx", "chop", "votes", "structure", "sideways"]
        a = full[cols].iloc[:k].astype(float).to_numpy()
        b = part[cols].astype(float).to_numpy()
        causal_ok &= bool(np.allclose(a, b, equal_nan=True))
    check("no look-ahead (truncation invariance)", causal_ok)

    # hull whipsaw source runs and behaves
    h = detect_sideways(df, replace(cfg, whipsaw_source="hull", hull_length=21))
    check("hull whipsaw source works", h["whipsaws"].notna().sum() > 100 and h["sideways"].dtype == bool)

    # gate helper
    sig = pd.Series(True, index=full.index)
    gated = apply_gate(sig, full["sideways"])
    check("apply_gate blocks exactly the sideways bars", int(gated.sum()) == int((~full["sideways"]).sum()))

    # warm-up never flagged; periods table consistent
    check("warm-up bars never flagged", not full["sideways"].iloc[: cfg.effective_warmup].any())
    per = sideways_periods(full)
    check("periods table matches backfilled flag", int(per["bars"].sum()) == int(full["sideways_backfilled"].sum()))
    check("causal flag is subset of backfilled flag", bool((full["sideways"] <= full["sideways_backfilled"]).all()))

    # robustness: duplicates, unsorted, NaN rows, upper-case columns, MultiIndex columns
    messy = df.copy()
    messy.columns = [c.upper() for c in messy.columns]
    messy = pd.concat([messy.iloc[::-1], messy.iloc[[5]]])
    messy.iloc[3, messy.columns.get_loc("CLOSE")] = np.nan
    try:
        detect_sideways(messy, cfg)
        check("handles unsorted / duplicate / NaN / upper-case input", True)
    except Exception as exc:  # noqa: BLE001
        check("handles unsorted / duplicate / NaN / upper-case input", False, repr(exc))
    mi = df.copy()
    mi.columns = pd.MultiIndex.from_product([[c.title() for c in df.columns], ["X.NS"]])
    try:
        detect_sideways(mi, cfg)
        check("handles yfinance MultiIndex columns", True)
    except Exception as exc:  # noqa: BLE001
        check("handles yfinance MultiIndex columns", False, repr(exc))
    try:
        detect_sideways(df.iloc[:20], cfg)
        check("short history returns all-False instead of crashing", True)
    except Exception as exc:  # noqa: BLE001
        check("short history returns all-False instead of crashing", False, repr(exc))

    # config validation
    for bad in (dict(votes_enter=9), dict(votes_exit=3, votes_enter=3), dict(confirm_with="x"), dict(er_window=0)):
        try:
            SidewaysConfig(**bad)
            check(f"config rejects {bad}", False)
        except ValueError:
            check(f"config rejects {bad}", True)

    if verbose:
        print("ALL TESTS PASSED" if ok else "SOME TESTS FAILED")
    return ok


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Sideways / range-bound detector for NSE stocks (daily bars).")
    p.add_argument("symbol", nargs="?", help="NSE symbol, e.g. KOPRAN (the .NS suffix is added automatically)")
    p.add_argument("--start", help="start date YYYY-MM-DD (default: --years back)")
    p.add_argument("--end", help="end date YYYY-MM-DD")
    p.add_argument("--years", type=float, default=3.0, help="history length if --start not given (default 3)")
    p.add_argument("--plot", action="store_true", help="show chart")
    p.add_argument("--save-plot", metavar="FILE", help="save chart to FILE instead of showing it")
    p.add_argument("--csv", metavar="FILE", help="write all indicator columns to CSV")
    p.add_argument("--selftest", action="store_true", help="run synthetic-data tests and exit")
    p.add_argument("--votes-enter", type=int)
    p.add_argument("--votes-exit", type=int)
    p.add_argument("--min-duration", type=int)
    p.add_argument("--er-max", type=float)
    p.add_argument("--confirm-with", choices=["adx", "chop", "any"])
    p.add_argument("--whipsaw-source", choices=["ema", "hull"])
    p.add_argument("--hull-length", type=int)
    return p


def main(argv: Optional[list] = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.selftest:
        return 0 if _selftest() else 1
    if not args.symbol:
        print("error: symbol required (or use --selftest)", file=sys.stderr)
        return 2

    overrides = {
        k: v
        for k, v in dict(
            votes_enter=args.votes_enter, votes_exit=args.votes_exit, min_duration=args.min_duration,
            er_max=args.er_max, confirm_with=args.confirm_with, whipsaw_source=args.whipsaw_source,
            hull_length=args.hull_length,
        ).items()
        if v is not None
    }
    cfg = SidewaysConfig(**overrides)
    res = analyze_symbol(args.symbol, start=args.start, end=args.end, years=args.years, config=cfg)

    last = res.iloc[-1]
    print(f"\n{args.symbol.upper()}  last completed bar: {res.index[-1].date()}  close: {last['close']:.2f}")
    print(f"Sideways now (causal flag): {bool(last['sideways'])}   votes: {int(last['votes'])}/{len(_VOTE_COLS)}")
    print(
        "  ER={er:.2f}  whipsaws={whipsaws:.0f}  range/ATR={range_atr:.1f}  drift/ATR={drift_atr:.1f}  "
        "slope/ATR={slope_atr:+.2f}  ADX={adx:.1f}  CHOP={chop:.1f}  structure={structure:+d}".format(**last.to_dict())
    )
    print("  votes: " + ", ".join(f"{c[2:]}={'Y' if last[c] else 'n'}" for c in _VOTE_COLS))
    per = sideways_periods(res)
    if per.empty:
        print("\nNo sideways periods found in this window.")
    else:
        print("\nSideways periods (backfilled labels - look-ahead, for eyeballing):")
        show = per.copy()
        show["start"] = show["start"].dt.date
        show["end"] = show["end"].dt.date
        print(show.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    if args.csv:
        res.to_csv(args.csv)
        print(f"\nwrote {args.csv}")
    if args.plot or args.save_plot:
        plot_result(res, title=f"{args.symbol.upper()} sideways detection", save=args.save_plot)
        if args.save_plot:
            print(f"saved chart to {args.save_plot}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
