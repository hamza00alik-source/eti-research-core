# Transaction-cost model semantics

ETI's current economic evaluator models **market-order execution**, so the transaction-cost model must declare `fee_basis: "taker"`. A maker-only fee schedule is rejected: market orders do not qualify for maker pricing by assumption. The taker fee value is charged per unit of position turnover; a full exit and re-entry therefore incur both sides' costs.

All bps fields are relative to normalized traded notional:

- `fee_bps`: taker fee per unit turnover;
- `spread_bps`: full quoted bid/ask spread; the evaluator charges half the spread per unit turnover;
- `slippage_bps`: adverse slippage per unit turnover;
- `market_impact_bps`: additional adverse impact per unit turnover;
- `funding_schedule`: discrete signed funding events, not a per-candle rate.

The model requires field-level evidence to cover every numeric assumption and the `fee_basis` declaration. This is a structural check only: ETI does not currently retrieve or authenticate the cited fee schedule, verify the account's fee tier, or prove the spread/slippage/impact estimates from order-book and fill data.

## Still unsupported or incomplete

- Borrow costs are rejected when nonzero by the governed research runner.
- Funding currency conversion and account-currency P&L are not modeled.
- Tick size, lot size, minimum notional, price bands, partial fills, queue position, market depth, and volatility-dependent impact are not represented in the historical backtest.
- The backtest's proportional costs are a scenario model, not a fill simulator. The live execution path uses market orders and actual exchange responses, but historical simulated fills are not calibrated from that owner's realized fills.
- One generic taker fee is used for entries and exits; tier changes, discounts, rebates, and symbol-specific fee schedules require fresh evidence.

A candidate must not pass economic approval merely because these fields are present. Evidence must be independently checked and conservative assumptions must be stress-tested.
