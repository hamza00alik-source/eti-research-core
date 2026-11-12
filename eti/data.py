"""Public OHLCV collection helpers. No authenticated exchange access is used."""
from __future__ import annotations

import time
from typing import Any

import pandas as pd

OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def validate_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    """Return sorted, de-duplicated UTC OHLCV data or raise on invalid candles."""
    missing = set(OHLCV_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing OHLCV columns: {sorted(missing)}")
    out = frame[OHLCV_COLUMNS].copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    for col in OHLCV_COLUMNS[1:]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna().sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
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


def fetch_ohlcv(
    exchange_id: str = "binance",
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    limit: int = 1500,
    since_ms: int | None = None,
    pause_seconds: float = 0.15,
) -> pd.DataFrame:
    """Fetch public OHLCV candles via CCXT with pagination where supported."""
    try:
        import ccxt
    except ImportError as exc:
        raise RuntimeError("Install dependencies first: pip install -r requirements.txt") from exc
    if limit < 1:
        raise ValueError("limit must be positive")
    exchange_class: Any = getattr(ccxt, exchange_id, None)
    if exchange_class is None:
        raise ValueError(f"Unsupported CCXT exchange id: {exchange_id}")
    exchange = exchange_class({"enableRateLimit": True})
    if not exchange.has.get("fetchOHLCV"):
        raise ValueError(f"{exchange_id} does not support fetchOHLCV")
    rows: list[list[float]] = []
    cursor = since_ms
    page_size = min(1000, limit)
    while len(rows) < limit:
        batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=min(page_size, limit-len(rows)))
        if not batch:
            break
        rows.extend(batch)
        next_cursor = int(batch[-1][0]) + 1
        if cursor is not None and next_cursor <= cursor:
            break
        cursor = next_cursor
        if len(batch) < min(page_size, limit-len(rows)+len(batch)):
            break
        time.sleep(pause_seconds)
    if not rows:
        raise RuntimeError(f"No OHLCV data returned for {exchange_id} {symbol} {timeframe}")
    frame = pd.DataFrame(rows, columns=OHLCV_COLUMNS)
    return validate_ohlcv(frame).tail(limit).reset_index(drop=True)
