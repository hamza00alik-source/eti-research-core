# ETI Live Public Market Feed and Paper Trading

## Current implementation

ETI now has a read-only public spot snapshot collector in `eti.live_market.capture_public_spot_snapshot`. It uses CCXT without credentials and captures:

- Venue, symbol, spot market type and observation time
- Latest completed OHLCV candle (the still-open candle is excluded)
- Bid, ask, last, mid and quoted spread in basis points
- Quote and candle ages
- A deterministic SHA-256 snapshot receipt

The collector fails closed when the market is not active spot, a quote timestamp is missing, the quote is stale, bid/ask values are invalid, the latest completed candle is stale, recent candles contain a gap, or the venue cannot provide the required data. The default maximum quote age is 30 seconds; candle freshness is bounded to one timeframe interval.

The paper tick now requires a fresh, eligible snapshot aligned to the exact completed candle used for model features. Temporary exchange/network failures, stale or malformed quote data, stale candles and candle gaps are classified as retryable market-data failures. The durable worker retries a scheduled paper tick up to three attempts before marking that task failed. Every processed candle now persists its full public market snapshot, model prediction, confidence, risk decision and resulting event types in the paper-decision ledger; retrieve it with the authenticated `GET /v1/paper/sessions/{session_id}/decisions` endpoint. It marks long positions at bid, simulates entries at ask plus configured slippage, and exits at bid minus configured slippage. Configured fees are still charged separately. Each paper fill event stores the quote timestamp, bid/ask, spread and snapshot digest for later audit. This is a conservative top-of-book approximation, not a queue-position, depth, partial-fill or market-impact model.

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

## Feed supervision and health

The existing ETI durable worker remains the scheduler; it polls running paper sessions at their configured candle interval. It now records per-session feed health in SQLite and exposes it through the authenticated `GET /v1/paper/sessions/{session_id}/health` endpoint. Health states are `healthy`, `degraded` (one or two consecutive failures), and `unavailable` (three or more consecutive failures); a separate freshness field turns `stale` if the worker has not attempted a feed check for more than two candle intervals. A successful eligible snapshot resets the consecutive-failure counter and stores the last quote timestamp and snapshot digest. The worker emits payload-free structured health events. This is supervised REST polling with bounded task retries, not a separate always-on WebSocket stream.

## Next required work before a meaningful paper pilot

1. Run the durable worker under an external process supervisor with automatic restart, bounded backoff, heartbeat monitoring, rate-limit budgets and operational alerts.
2. Add longer-retention archival and integrity verification for the per-candle decision ledger, including periodic backups and restore tests.
3. Add depth-aware simulated fills, partial fills, rejects, cancel/replace lifecycle and venue-specific fee schedules.
4. Add restart/replay determinism, monitoring, alerting and a runbook.
5. Run a multi-day live paper trial and compare simulated fills against subsequent observed quotes/trades. Keep the model in candidate/review status unless independent out-of-sample evidence supports promotion.

A successful unit test or a fresh quote is not evidence of a profitable strategy.
