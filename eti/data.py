"""Public OHLCV collection helpers. No authenticated exchange access is used."""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def validate_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    """Return sorted, de-duplicated UTC OHLCV data or raise on invalid candles."""
    missing = set(OHLCV_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing OHLCV columns: {sorted(missing)}")
    optional_columns = [name for name in ("funding_rate", "open_interest") if name in frame.columns]
    out = frame[OHLCV_COLUMNS + optional_columns].copy()
    # CCXT returns Unix epoch timestamps in milliseconds. Passing those
    # integers to pandas without a unit interprets them as nanoseconds,
    # silently moving candles to 1970 and causing valid history to be dropped.
    if pd.api.types.is_numeric_dtype(out["timestamp"]):
        out["timestamp"] = pd.to_datetime(
            out["timestamp"], unit="ms", utc=True, errors="coerce"
        )
    else:
        out["timestamp"] = pd.to_datetime(
            out["timestamp"], utc=True, errors="coerce"
        )
    for col in OHLCV_COLUMNS[1:]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out[OHLCV_COLUMNS] = out[OHLCV_COLUMNS].replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=OHLCV_COLUMNS)
    for name in optional_columns:
        out[name] = pd.to_numeric(out[name], errors="coerce").replace([np.inf, -np.inf], np.nan)
    out = out.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    if out.empty:
        raise ValueError("No valid OHLCV candles")
    if (out[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("OHLC prices must be positive")
    if (out["volume"] < 0).any():
        raise ValueError("Volume cannot be negative")
    if (out["high"] < out[["open", "close", "low"]].max(axis=1)).any():
        raise ValueError("Candle high is inconsistent")
    if (out["low"] > out[["open", "close", "high"]].min(axis=1)).any():
        raise ValueError("Candle low is inconsistent")
    return out


def drop_incomplete_candles(
    frame: pd.DataFrame, timeframe_ms: int, now_ms: int
) -> pd.DataFrame:
    """Keep only candles whose full timeframe has elapsed."""
    if timeframe_ms <= 0:
        raise ValueError("timeframe_ms must be positive")
    clean = validate_ohlcv(frame)
    # Pandas 3 may retain microsecond-resolution timestamps while older
    # versions commonly use nanoseconds. Normalize explicitly before
    # comparing with the millisecond timestamps used by CCXT.
    candle_open_ms = clean["timestamp"].dt.as_unit("ms").astype("int64")
    completed = candle_open_ms + timeframe_ms <= int(now_ms)
    clean = clean.loc[completed].reset_index(drop=True)
    if clean.empty:
        raise ValueError("No completed OHLCV candles returned")
    return clean


def fetch_ohlcv(
    exchange_id: str = "binance",
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    limit: int = 1500,
    since_ms: int | None = None,
    pause_seconds: float = 0.15,
    market_type: str = "spot",
) -> pd.DataFrame:
    """Fetch completed public OHLCV candles for a CCXT spot, swap or dated-future market."""
    try:
        import ccxt
    except ImportError as exc:
        raise RuntimeError("Install dependencies first: pip install -r requirements.txt") from exc
    if limit < 1:
        raise ValueError("limit must be positive")
    exchange_class: Any = getattr(ccxt, exchange_id, None)
    if exchange_class is None:
        raise ValueError(f"Unsupported CCXT exchange id: {exchange_id}")
    if market_type not in {"spot", "swap", "future"}:
        raise ValueError("market_type must be spot, swap, or future")
    exchange = exchange_class({
        "enableRateLimit": True,
        "options": {"defaultType": market_type},
    })
    if not exchange.has.get("fetchOHLCV"):
        raise ValueError(f"{exchange_id} does not support fetchOHLCV")
    markets = exchange.load_markets()
    try:
        market = markets[symbol]
    except KeyError as exc:
        raise ValueError(f"Unknown {market_type} symbol for {exchange_id}: {symbol}") from exc
    market_flag = {"spot": "spot", "swap": "swap", "future": "future"}[market_type]
    if not market.get(market_flag, False):
        raise ValueError(f"{exchange_id} {symbol} is not a {market_type} market")
    if timeframe.endswith("M"):
        raise ValueError("Calendar-month candles are not yet supported by the fixed-duration completion filter")
    timeframe_seconds = exchange.parse_timeframe(timeframe)
    if timeframe_seconds <= 0:
        raise ValueError(f"Invalid timeframe: {timeframe}")
    timeframe_ms = int(timeframe_seconds * 1000)
    rows: list[list[float]] = []
    cursor = since_ms
    if cursor is None:
        cursor = exchange.milliseconds() - timeframe_ms * limit
    page_size = min(1000, limit)
    while len(rows) < limit:
        requested = min(page_size, limit - len(rows))
        batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=requested)
        if not batch:
            break
        rows.extend(batch)
        next_cursor = int(batch[-1][0]) + 1
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        # Some exchanges silently cap page size below the requested limit.
        # A short non-empty page is not necessarily the end of available history:
        # continue from its final timestamp until the API returns an empty page
        # or stops advancing. This is essential for collecting enough history
        # for nested walk-forward folds.
        if len(rows) < limit:
            time.sleep(pause_seconds)
    if not rows:
        raise RuntimeError(f"No OHLCV data returned for {exchange_id} {symbol} {timeframe}")
    frame = pd.DataFrame(rows, columns=OHLCV_COLUMNS)
    return drop_incomplete_candles(frame, timeframe_ms, exchange.milliseconds()).tail(limit).reset_index(drop=True)
