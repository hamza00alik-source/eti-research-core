# Vintage-aware macroeconomic data

## Why this is separate from OHLCV

Economic releases are observations about reference periods, not tradable candles.
A value can be published after its reference period and revised later. A historical
backtest must use the value actually available at its decision timestamp, not the
latest revised value now shown by a data vendor.

## Current implementation

`eti.macro_data` defines an immutable `MacroObservation` record with:
- series identifier and reference/observation period;
- numeric value and unit;
- release timestamp and vintage/revision timestamp;
- retrieval timestamp, source, license status and license reference.

`MacroDataHub.fetch_as_of(provider, series, decision_time)` selects, for each
observation period, the latest vintage whose release and vintage timestamps are
both no later than the decision time. All timestamps must be timezone-aware.
Duplicate (series, period, vintage) identities, non-finite values, mismatched
series, inconsistent source metadata and any license status other than the
explicit `permitted` assertion fail closed.

The provider protocol and in-memory implementation are included for deterministic
testing. No network provider is registered, no API reachability is claimed, and
no data source is assumed to be free or licensed. The license assertion is supplied
by the provider integration and must be backed by reviewed source terms; it is not
a legal determination by ETI.

## Deliberate limitations

- No FRED, central-bank, statistical-agency or commercial API integration yet.
- No automatic release-calendar discovery or source-license verification.
- No macro features are joined into OHLCV features or training.
- No interpolation, forward-fill, or revision backfill is performed.
- Series transformations, units, seasonal adjustment, frequency alignment and
  release-calendar semantics must be explicit before macro features are used.

A future provider adapter must preserve each historical vintage and its true
public-availability timestamp. A provider that exposes only the latest revised
series cannot safely power point-in-time historical research.
