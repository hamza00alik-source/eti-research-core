# Realized execution-cost measurement

`eti.fill_costs.measure_market_order_fill_cost` calculates side-adjusted implementation shortfall from an order's arrival mid-price and its individual fills.

- VWAP is computed from actual fill quantities and prices.
- Buy shortfall is positive when VWAP is above arrival mid; sell shortfall is positive when VWAP is below arrival mid.
- Fees are supplied already converted into quote currency; positive fees are costs and negative values are rebates.
- The all-in observed cost is shortfall plus fee, in basis points of filled quote notional.

## What this does not infer

The measurement does not split implementation shortfall into quoted spread, temporary slippage and market impact. That decomposition requires synchronized order-book snapshots and order submission/fill timestamps. It also does not estimate future costs, model partial-fill opportunity cost, convert third-currency fees, or predict impact for a different order size/liquidity regime.

The existing backtest schema stores separate taker fee, full spread, slippage and impact assumptions. Do not copy this all-in metric into one of those fields or add it to them without reconciling double-counting. Use this report to build and validate conservative empirical assumptions once real fill logs and arrival-mid snapshots exist.
