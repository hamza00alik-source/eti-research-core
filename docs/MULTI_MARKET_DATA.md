# ETI Multi-Market Data Architecture

## Objective

ETI is a market-selection and research system, not an exchange-specific trading bot. It must discover instruments across supported asset classes, query multiple independent data sources, assess source quality and coverage, and preserve exactly which source produced every observation.

## Canonical asset classes

- Crypto: spot, perpetual swaps and dated futures where a venue exposes them.
- Forex: spot currency pairs and reference feeds.
- Equities: listed shares and ETFs as data adapters/universes are added.
- Commodities: cash/reference series and exchange futures, with contract/roll metadata required before continuous-series research.
- Indices: cash/reference indices and index futures.
- Macro: timestamped economic series, distinct from tradable OHLCV instruments.

The canonical instrument catalog is venue-aware. Similar symbols on different venues remain distinct unless an explicit mapping establishes equivalence.

## Data source strategy

| Domain | Initial adapters | Future independent sources |
|---|---|---|
| Crypto | CCXT public venue adapters; Coinbase Exchange and Kraken fallback for supported spot pairs | More CCXT venues, CoinGecko reference data, exchange archives and licensed feeds |
| Forex | Yahoo Finance reference history | Stooq, broker/reference FX feeds and licensed sources |
| Equities | Yahoo Finance reference history | Stooq, exchange/vendor feeds, corporate-action-aware sources |
| Commodities | Yahoo Finance futures/reference symbols | Exchange settlement feeds, EIA, USDA and other domain-specific public datasets |
| Indices | Yahoo Finance index/futures references | Exchange/index-provider datasets |
| Macro | Separate timestamped economic-series adapters | FRED, central-bank, statistical-agency and licensed macro datasets |

This is a source roadmap, not a claim that every listed source is integrated or offers unlimited/free history. Each adapter must document access limits, history depth, licensing/terms, update cadence and missing-data behavior.

## Orchestration rules

1. Discover instruments from every registered adapter that declares support for the requested asset class and market type.
2. Preserve venue, symbol, asset class, market type, quote/settlement currency and contract metadata.
3. Validate timestamps and OHLCV values before feature engineering.
4. Enforce the experiment's minimum history requirement. A provider returning too few usable rows is ineligible for that experiment.
5. Fail over only through an explicit ordered list of economically equivalent instrument mappings. Matching ticker strings or quote conversions do not prove equivalence.
6. Use one provider's validated time series for a snapshot. Never concatenate or average exchange candles implicitly.
7. Record selected source, requested/returned rows, source failures, data window and quality diagnostics in experiment provenance.
8. Keep public market-data access separate from authenticated broker access and order execution.

## Quality controls

The first hub implementation reports row count, start/end timestamps, estimated missing intervals, maximum timestamp gap and age of the latest candle. Further gates should add:
- provider availability and latency history;
- stale-feed thresholds appropriate to each asset's trading calendar;
- cross-source price divergence checks for explicitly mapped instruments;
- corporate actions and adjusted/unadjusted equity price policy;
- futures expiry, roll methodology, multiplier, tick size and settlement;
- FX session calendars and quote convention;
- survivorship, delisting and symbol-change handling;
- licensing and redistribution constraints.

A quality score must be explainable. Missing data must never be fabricated or provider failures silently hidden.

## Training and model policy

- Discovery and ingestion are separate from training. A newly discovered instrument is not automatically trained or approved.
- Each instrument/market-type/timeframe dataset receives its own provenance and temporal validation.
- Do not pool raw price levels across asset classes. Shared models require normalized, economically meaningful features and explicit asset/venue identifiers.
- Use chronological, purged walk-forward evaluation, transaction costs and market-specific execution assumptions.
- A model trained on a reference index or Yahoo futures series is not automatically executable through a broker.
- No candidate is eligible for autonomous live trading until it passes data-quality, out-of-sample, cost-adjusted performance, calibration and risk gates.

## Scope of this implementation

The MarketDataHub provides read-only adapter orchestration, instrument discovery, validated history retrieval and explicit source failover. It does not implement orders, scrape arbitrary websites, bypass paywalls, or guarantee every public endpoint is reachable from every cloud region.


## Running a bounded data-readiness scan

The scanner can discover a small, venue-diverse sample and check that each candidate has enough validated history. It does not fit models or place orders:

```bash
python -m eti.pipeline --scan-universe \
  --asset-classes crypto,forex,equity,commodity,index \
  --market-types spot --timeframe 1h --limit 2000 \
  --min-market-rows 662 --max-per-asset-class 5 --max-per-venue 2 \
  --output artifacts/market-universe.json
```

The report records discovered counts, sources checked, provider errors, selected instruments, data readiness, provenance and timestamp-gap diagnostics. The initial default scan uses spot markets to keep the first research comparison simpler; derivatives can be included explicitly, but require separate contract, funding, expiry, margin and cost assumptions before model evaluation.
