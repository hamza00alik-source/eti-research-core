# Causal market-regime diagnostics

ETI's shared regime module provides a scale-independent market context:
trailing realized volatility, a prior-history volatility quantile, normalized
directional trend strength, and one of `unknown`, `high_volatility`,
`uptrend`, `downtrend` or `range`.

The volatility cutoff is calculated from prior bars only (the current
realized-volatility observation is shifted out before the rolling quantile).
The module is diagnostic and does not generate orders or override risk
controls. It can be evaluated across asset classes once each provider supplies
validated OHLCV data. Specialist models may consume this context later, after
out-of-sample validation; the regime labels are not assumed to transfer
perfectly across markets.
