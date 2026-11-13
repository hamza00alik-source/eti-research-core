# ETI architecture: shared intelligence, instrument-aware adapters

## Governing principle

ETI is organized around **asset classes and instrument types, not broker names**.
A broker connection is a data/execution transport adapter. It is not a separate
AI model. Adding a broker must not automatically trigger full model retraining.

## Three independent layers

1. **Shared intelligence core** — common causal feature processing, market-regime
   representation, uncertainty handling, candidate evaluation and decision
   orchestration. It learns reusable market behaviour across compatible data.
2. **Specialist market heads** — optional heads for crypto, FX, equities,
   commodities and indices, with further instrument-specific specialisation
   such as spot, margin, perpetual swap, dated future or CFD. A specialist is
   added only when out-of-sample evidence justifies it.
3. **Venue adapters** — translate broker-specific symbols, candles, contract
   metadata, trading calendars, fees, funding and margin data into canonical
   ETI contracts. Adapter code must not contain model logic or alter risk limits.

## Canonical instrument metadata

The eti.instruments module defines InstrumentSpec, the normalized instrument
contract. It records venue, symbol, asset class, market type, currencies, contract
multiplier, tick/lot sizes, minimum quantity, leverage capability, shorting,
funding and active status. InstrumentCatalog keys instruments by venue,
market type and symbol; conflicting registrations fail closed.

The eti.adapters module defines AdapterCapabilities and AdapterRegistry.
They describe exactly which asset classes and market types an adapter supports.
The MarketAdapter protocol is read-side only: it defines instrument discovery
and OHLCV retrieval, not order placement.

## Training and adding markets

- A new broker exposing an already-supported instrument type: implement its
  adapter, normalize its instrument metadata, run data-quality and out-of-sample
  validation. Do not retrain the entire shared core by default.
- A new instrument on an already-supported broker: discover and validate its
  metadata and history. Reuse compatible models if validation supports it.
- A new asset class or materially different instrument mechanics: build the
  appropriate data/feature adapter and specialist head; train that specialist
  from scratch and evaluate it independently before any shared-model change.
- Fine-tuning or shared-core retraining is a measured decision based on
  transfer performance, calibration drift and out-of-sample evidence—not a
  mandatory step for every broker addition.

## Important distinction: same symbol name is not same exposure

A tokenized gold spot pair (for example, a gold-backed token quoted in USDT),
XAU/USD CFD, and a COMEX gold future are different instruments. Their price
sources, contract multipliers, trading hours, funding/financing, custody,
margin and liquidation mechanics differ. ETI must represent and validate each
instrument separately even if all are classified under commodity.

## Safety and current implementation status

The current connected CCXT adapter is public crypto spot only. The canonical
catalog and adapter registry are extension points, not proof that MEXC or
Binance currently list a given gold, silver, FX or futures product. CSV history
can be used for research where no live adapter exists. No authenticated order
execution is exposed; research risk controls remain independent and fail closed.
InstrumentSpec.max_leverage is capability metadata, not permission to use that
leverage. The risk gate remains authoritative.
