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
    page_count = 0
    cursor = since_ms
    if cursor is None:
        cursor = exchange.milliseconds() - timeframe_ms * limit
    page_size = min(1000, limit)
    while len(rows) < limit:
        page_request = min(page_size, limit - len(rows))
        batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=page_request)
        if not batch:
            break
        page_count += 1
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
    completed = drop_incomplete_candles(frame, timeframe_ms, exchange.milliseconds()).tail(limit).reset_index(drop=True)
    completed.attrs["source_metadata"] = {
        "provider": "ccxt-public",
        "exchange": exchange_id,
        "market_type": market_type,
        "symbol": symbol,
        "timeframe": timeframe,
        "requested_candles": limit,
        "raw_candles_received": len(rows),
        "completed_candles_returned": len(completed),
        "fetch_pages": page_count,
        "source": "public_exchange_ohlcv",
        "first_returned_candle": completed["timestamp"].iloc[0].isoformat(),
        "last_returned_candle": completed["timestamp"].iloc[-1].isoformat(),
    }
    return completed



def assess_ohlcv_quality(
    frame: pd.DataFrame, timeframe: str, now: pd.Timestamp | None = None
) -> dict[str, Any]:
    """Summarize candle continuity and freshness without assuming a 24/7 market calendar."""
    timeframe_ms = {
        "1m": 60_000, "5m": 300_000, "15m": 900_000,
        "30m": 1_800_000, "1h": 3_600_000,
        "4h": 14_400_000, "1d": 86_400_000,
    }.get(timeframe)
    if timeframe_ms is None:
        raise ValueError("Unsupported timeframe for data-quality assessment")
    clean = validate_ohlcv(frame)
    timestamps = clean["timestamp"]
    interval_ms = timeframe_ms
    deltas = timestamps.diff().dropna().dt.total_seconds() * 1000
    gap_deltas = deltas[deltas > interval_ms * 1.5]
    expected_slots = (
        int(round((timestamps.iloc[-1] - timestamps.iloc[0]).total_seconds() * 1000 / interval_ms)) + 1
    )
    observed_slots = len(clean)
    coverage = min(1.0, observed_slots / expected_slots) if expected_slots > 0 else 0.0
    current = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if current.tzinfo is None:
        current = current.tz_localize("UTC")
    else:
        current = current.tz_convert("UTC")
    latest_close_time = timestamps.iloc[-1] + pd.Timedelta(milliseconds=interval_ms)
    age_seconds = max(0.0, (current - latest_close_time).total_seconds())
    freshness_limit_seconds = max(3 * interval_ms / 1000, 48 * 60 * 60)
    return {
        "status": "stale" if age_seconds > freshness_limit_seconds else "fresh",
        "observed_candles": observed_slots,
        "first_candle": timestamps.iloc[0].isoformat(),
        "latest_completed_candle": timestamps.iloc[-1].isoformat(),
        "candle_interval_ms": interval_ms,
        "expected_slots_in_observed_span": expected_slots,
        "calendar_coverage_ratio": round(coverage, 6),
        "gap_count": int(len(gap_deltas)),
        "largest_gap_intervals": (
            round(float(gap_deltas.max() / interval_ms), 3) if len(gap_deltas) else 0.0
        ),
        "latest_candle_age_seconds": round(age_seconds, 1),
        "freshness_threshold_seconds": freshness_limit_seconds,
        "calendar_note": "Continuity and freshness use elapsed calendar time; exchange sessions, holidays and planned maintenance are not modeled.",
    }
