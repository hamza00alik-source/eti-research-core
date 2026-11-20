# ETI first live-market paper pilot runbook

This runbook is for the first supervised **paper-only** trial. It uses public
market data and simulated cash. It does not require exchange API keys and must
not place real orders.

## Pilot scope

Start with one instrument and one session:

- Venue: Binance spot
- Symbol: BTC/USDT
- Timeframe: 1h (the paper-session API currently allows 1h, 4h or 1d)
- Direction: long-only
- Initial cash: a clearly labelled simulated amount, not a deposit
- Risk defaults: at most 0.5% risk per trade, 10% position fraction, 25% gross
  exposure, 2% daily loss and 8% max drawdown
- Costs: default 6 bps fee and 8 bps slippage are configuration assumptions;
  verify the venue fee tier and record the assumptions before interpreting PnL

Do not run multiple symbols or strategies until the single-session path has
completed restart, pause, recovery and audit checks.

## Preflight

1. Deploy the reviewed code only after its pull requests are merged and CI is
   green on the exact merge commit.
2. Start both API and worker services; the worker is a separate process and is
   not launched by FastAPI:

   ```sh
   docker compose up -d --build api worker
   docker compose ps
   docker compose logs --tail=100 worker
   ```

3. Confirm API and worker share the same persistent `ETI_DATABASE_PATH` volume.
   Verify a backup exists and can be restored before a multi-day run.
4. Confirm host clock synchronization, stable outbound connectivity, Binance
   public endpoint availability, rate-limit headroom and disk space.
5. Confirm `LIVE_ORDER_EXECUTION_ENABLED = False` in the deployed source.
   Do not configure exchange trading credentials for this paper-only test.
6. Confirm the model is already approved for the exact canonical model identity
   and version. A candidate, calibrated model, or model with a passing unit
   test is not sufficient. The existing BTC research result is NO-GO; do not
   represent it as profitable or promote it based on this pilot.

## Create a paused paper session

Authenticate using the API's normal bearer-token flow. Never put access tokens
in source control, issue comments or logs.

Create a session using `POST /v1/paper/sessions`:

```json
{
  "model_id": "<approved-canonical-model-id>",
  "model_version": "<approved-version>",
  "venue": "binance",
  "symbol": "BTC/USDT",
  "timeframe": "1h",
  "initial_cash": 10000,
  "max_risk_per_trade": 0.005,
  "max_position_fraction": 0.10,
  "max_gross_exposure": 0.25,
  "max_daily_loss": 0.02,
  "max_drawdown": 0.08,
  "fee_bps": 6.0,
  "slippage_bps": 8.0,
  "confidence_threshold": 0.55
}
```

The amount above is simulated USD cash only. Keep the returned `session.id`.
The create endpoint returns `mode=paper_only` and
`live_execution_enabled=false`. Creation should leave the session paused.

Before starting, inspect the returned model identity, venue, symbol, timeframe,
risk configuration and cash. If any value is unexpected, leave it paused.

## Start and supervise

Start explicitly with `POST /v1/paper/sessions/{session_id}/start`. Then
monitor:

- `GET /v1/paper/sessions/{session_id}/health`
- `GET /v1/paper/sessions/{session_id}/decisions?limit=200`
- `GET /v1/paper/sessions/{session_id}/trades?limit=200`
- `GET /v1/paper/sessions`

Each processed candle should have a decision record with the market snapshot,
prediction, confidence, risk decision and resulting event types. Check quote
age, candle age, spread, snapshot digest and fill price assumptions. Feed health
must remain healthy and fresh. A degraded/unavailable/stale feed, a candle gap,
a missing worker, unexpected session configuration or unexplained duplicate
decision is a stop condition.

Pause immediately with `POST /v1/paper/sessions/{session_id}/pause` if any
stop condition occurs. Do not resume until the cause is understood, data is
resynchronized and the audit trail is intact.

## Trial acceptance gates

Do not call the trial successful based only on a green CI run, a connected
socket, or positive simulated PnL. Before interpreting the result:

1. Complete a controlled start/pause/restart test and verify no candle is
   skipped or processed twice.
2. Confirm decisions and fills survive worker and API restarts.
3. Confirm risk limits reject oversized orders and pause/stop conditions work.
4. Reconcile selected snapshot receipts against an independent public source.
5. Compare simulated bid/ask fills with subsequent observed quotes/trades.
6. Preserve the full decision and cost assumptions for a multi-day sample.
7. Evaluate the strategy on untouched chronological data and account for
   multiple testing before any model promotion or capital allocation.

A successful paper trial validates operational plumbing and creates evidence;
it does not prove a future trading edge or live execution quality.

## Hard safety boundary

The paper engine is long-only crypto spot. It uses public data and simulated
fills; it does not read balances, sign transactions, submit exchange orders or
withdraw funds. Derivatives, leverage, FX and equity execution are outside this
pilot. Real-money execution remains globally disabled.
