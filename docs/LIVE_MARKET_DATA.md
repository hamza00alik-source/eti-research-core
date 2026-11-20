# Public live market-data ingestion

## Binance spot kline stream (first slice)

`eti.live_market_data.BinanceSpotKlineFeed` opens the unauthenticated Binance
spot kline stream. It emits only completed candles (`k.x == true`) and checks
the configured BASE/QUOTE symbol and interval, event-time freshness, bounded
future clock skew, OHLC consistency, finite prices/volume, and candle sequence.

The parser deliberately ignores in-progress kline updates. Repeated closed
candles are deduplicated; older candles and missing intervals fail closed and
require the caller to resynchronize from REST history before resuming. The
socket is public and read-only; it has no credential or order methods.

Example:

```python
from eti.live_market_data import BinanceSpotKlineFeed

feed = BinanceSpotKlineFeed("BTC/USDT", "1m")
socket = feed.connect(timeout_seconds=10)
try:
    candle = feed.receive_closed_candle(socket, now_ms=current_time_ms)
finally:
    socket.close()
```

The example is intentionally a low-level building block. The current slice
does not yet provide durable raw-event storage, a reconnect supervisor,
REST/WebSocket gap-repair orchestration, multi-symbol multiplexing, clock
synchronization, or direct paper-session dispatch. Those are required before
calling this a production live-paper service. A caller must refresh `now_ms`
for each receive operation; do not reuse a stale clock value in a long-running
loop.

## Safety boundary

This feed supports Binance **spot market data only**. It does not connect to
private user streams, read balances, decrypt credentials, create orders, or
change `LIVE_ORDER_EXECUTION_ENABLED`. It is not a trading signal and does
not imply that ETI has a profitable strategy. The existing BTC research result
remains NO-GO.
