# ETI Live Public Market Feed and Paper Trading

## Current implementation

ETI now has a read-only public spot snapshot collector in `eti.live_market.capture_public_spot_snapshot`. It uses CCXT without credentials and captures:

- Venue, symbol, spot market type and observation time
- Latest completed OHLCV candle (the still-open candle is excluded)
- Bid, ask, last, mid and quoted spread in basis points
- Quote and candle ages
- A deterministic SHA-256 snapshot receipt

The collector fails closed when the market is not active spot, a quote timestamp is missing, the quote is stale, bid/ask values are invalid, the latest completed candle is stale, or the venue cannot provide the required data. The default maximum quote age is 30 seconds; candle freshness is bounded to one timeframe interval.

The paper tick now requires a fresh, eligible snapshot aligned to the exact completed candle used for model features. It marks long positions at bid, simulates entries at ask plus configured slippage, and exits at bid minus configured slippage. Configured fees are still charged separately. Each paper fill event stores the quote timestamp, bid/ask, spread and snapshot digest for later audit. This is a conservative top-of-book approximation, not a queue-position, depth, partial-fill or market-impact model.

## Run a single snapshot

```python
from eti.live_market import capture_public_spot_snapshot

snapshot = capture_public_spot_snapshot(
    "binance",
    "BTC/USDT",
    timeframe="1m",
)
print(snapshot)
```

Supported public spot venue clients are subject to CCXT and venue endpoint availability. No private API key is needed for this data path.

## Safety and limitations

- Public REST polling, not a private authenticated feed or guaranteed low-latency WebSocket.
- No real orders, exchange credentials, account balances or broker connections are used by the snapshot collector.
- The paper engine remains long-only crypto spot and requires an already approved ETI model version.
- The paper engine still processes at most one decision per completed candle; it is not an intrabar scalping simulator.
- Bid/ask top-of-book does not represent executable size. Large orders may incur additional spread, depth and impact costs.
- CCXT ticker timestamps and exchange clocks are trusted only for freshness checks; no independent clock attestation is provided.
- Quote and OHLCV are separate REST requests. The paper tick checks candle alignment and rejects mismatches, but a venue can still change between requests.
- Current paper fills remain simulated and cannot prove future execution quality or profitability.
- Live order execution remains hard-disabled.

## Next required work before a meaningful paper pilot

1. Add a continuously supervised feed process with reconnect/backoff, rate-limit budgeting, heartbeat monitoring, gap detection and durable event storage.
2. Persist quote/candle receipts alongside each paper decision and fill, not only return the latest snapshot digest.
3. Add depth-aware simulated fills, partial fills, rejects, cancel/replace lifecycle and venue-specific fee schedules.
4. Add restart/replay determinism, monitoring, alerting and a runbook.
5. Run a multi-day live paper trial and compare simulated fills against subsequent observed quotes/trades. Keep the model in candidate/review status unless independent out-of-sample evidence supports promotion.

A successful unit test or a fresh quote is not evidence of a profitable strategy.
