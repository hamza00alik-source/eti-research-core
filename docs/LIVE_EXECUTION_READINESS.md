# Live order readiness gate

ETI's live order path is intentionally hard-disabled in `eti.execution` by the code-level constant `LIVE_ORDER_EXECUTION_ENABLED = False`. The gate runs immediately after checking the owner's stored profile and before approved-model lookup, credential decryption, exchange initialization, balance reads or order submission.

This is not an environment variable or UI preference. A configured owner profile and a calibrated/approved model are not sufficient to bypass it. The default CI test asserts that an authorized profile cannot even initialize the exchange or decrypt credentials while the global gate is off.

The existing order-path tests deliberately monkeypatch the constant to exercise reconciliation, idempotency and fill-state behavior in a controlled fake-exchange test. That test-only override is not a production configuration mechanism.

The gate must remain off until all of the following are independently reviewed:

1. Genuine current and point-in-time provider data, instrument metadata and source provenance.
2. Fresh, independent, cost-adjusted out-of-sample results with multiple-testing controls and stable probability calibration.
3. Venue-specific fills, fee tiers, order-book depth, partial-fill and slippage/impact validation.
4. Contract-aware notional, funding, borrow, settlement currency, margin and liquidation accounting for any derivatives.
5. Deterministic risk limits, account reconciliation, kill-switch drills, idempotency, recovery and operational monitoring.
6. Security review, owner authorization, key permissions, deployment controls and audit trail.

The present live execution implementation is limited to Binance/MEXC spot, one-times leverage, long-only market orders. It is not an autonomous multi-market broker router. The current research pipeline does not approve live deployment. No profitability or live readiness is implied by passing CI.
