# Raw derivative funding-history retrieval

`CcxtPublicProvider.fetch_funding_rate_history` retrieves historical funding-rate observations for a configured CCXT swap/future venue using paginated `fetchFundingRateHistory` calls.

It records:
- exact venue, symbol, requested UTC millisecond interval and observation timestamp;
- ordered settlement timestamps and raw signed funding rates plus basis-point conversion;
- page count and the exchange's public API URI when CCXT exposes it;
- contract base/quote/settlement, linear/inverse flags and contract size.

The retrieval fails closed when the venue lacks the endpoint, the market is not found, timestamps/rates are malformed, conflicting duplicate settlement values are returned, pagination stops advancing, or the requested interval extends into the future.

## Not an approved funding schedule

The adapter deliberately sets `history_completeness: unverified` and `evidence_eligible: false`. CCXT's paginated response does not prove that the venue returned every settlement in the requested interval; it may be truncated, retention-limited or affected by endpoint pagination semantics. The endpoint URI also does not establish the license terms for downstream use.

Therefore this output is **not directly accepted** by the transaction-cost evidence validator. An operator must independently verify endpoint semantics, retention, page boundaries, missing settlement intervals, source terms and contract-specific settlement rules before converting it into a complete expected-timestamp schedule.

No 8-hour cadence is assumed, no missing event is synthesized, and no backtest or live execution integration is enabled by this retrieval method.
