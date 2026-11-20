# ETI Live Public Market Feed and Paper Trading

## Current implementation

ETI now has a read-only public spot snapshot collector in `eti.live_market.capture_public_spot_snapshot`. It uses CCXT without credentials and captures:

- Venue, symbol, spot market type and observation time
- Latest completed OHLCV candle (the still-open candle is excluded)
- Bid, ask, last, mid and quoted spread in basis points
- Quote and candle ages
- A deterministic SHA-256 snapshot receipt

The collector fails closed when the market is not active spot, a quote timestamp is missing, the quote is stale, bid/ask values are invalid, the latest completed candle is stale, recent candles contain a gap, or the venue cannot provide the required data. The default maximum quote age is 30 seconds; candle freshness is bounded to one timeframe interval.

The paper tick now requires a fresh, eligible snapshot aligned to the exact completed candle used for model features. It also stops a session if its worker has missed a completed candle since the prior decision, rather than silently skipping the intervening market history. Temporary exchange/network failures, stale or malformed quote data, stale candles and candle gaps are classified as retryable market-data failures. The durable worker retries a scheduled paper tick up to three attempts before marking that task failed. Every processed candle now persists its full public market snapshot, model prediction, confidence, risk decision and resulting event types in the paper-decision ledger; retrieve it with the authenticated `GET /v1/paper/sessions/{session_id}/decisions` endpoint. It marks long positions at bid, simulates entries at ask plus configured slippage, and exits at bid minus configured slippage. Configured fees are still charged separately. Each paper fill event stores the quote timestamp, bid/ask, spread and snapshot digest for later audit. This is a conservative top-of-book approximation, not a queue-position, depth, partial-fill or market-impact model.

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

## Running the paper worker

The repository's Compose file already defines separate `api` and `worker` services. Both mount the same `eti-data` volume and use the same SQLite database path; the worker has `restart: unless-stopped`. On a deployment host, after configuring the required `.env` secrets, start both services with:

```sh
docker compose up -d --build api worker
docker compose ps
docker compose logs --tail=100 worker
```

The worker is not started implicitly by FastAPI. If it is stopped, paper sessions remain stored but no new ticks are scheduled. The health endpoint marks old feed status as stale after two candle intervals. Use the authenticated API/UI to create a paused session, inspect it, and explicitly start it. The UI's feed-health and decision-audit buttons provide session-level diagnostics.

Do not start a session with a candidate merely because it is registered or calibrated. ETI requires an approved model, and the earlier BTC/USDT research result remains NO-GO for profitability. A paper session can validate plumbing and collect evidence; it cannot turn that result into a proven edge.

## Next required work before a meaningful paper pilot

1. Deploy the existing Compose `worker` service (already configured with `restart: unless-stopped`) and add host-level monitoring/alerts for worker liveness, SQLite health, rate-limit budgets and prolonged stale feed status.
2. Add longer-retention archival and integrity verification for the per-candle decision ledger, including periodic backups and restore tests.
3. Add depth-aware simulated fills, partial fills, rejects, cancel/replace lifecycle and venue-specific fee schedules.
4. Add restart/replay determinism, monitoring, alerting and a runbook.
5. Run a multi-day live paper trial and compare simulated fills against subsequent observed quotes/trades. Keep the model in candidate/review status unless independent out-of-sample evidence supports promotion.

A successful unit test or a fresh quote is not evidence of a profitable strategy.
