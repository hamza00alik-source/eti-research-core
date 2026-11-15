# Derivative funding accounting contract

## Timestamp and position convention

ETI's event-timed funding evaluator uses **UTC OHLCV bar-open timestamps**. The backtester delays a signal formed from bar *t* by one bar: it first holds that signal from the next bar's open and applies it to the next close-to-close return interval. A funding event at bar *t*'s open is charged to the already-shifted position at that timestamp. The signal computed using bar *t*'s close must never be charged at that same bar-open settlement.

This convention is only valid when the adapter supplies candle-open timestamps. A close-stamped or provider-specific timestamp convention must be explicitly normalized by the adapter before it reaches the backtester. The current public CCXT OHLCV path uses candle-open timestamps; this is not a universal guarantee for every future adapter.

Funding events must align exactly to a UTC bar timestamp. An event between bars is rejected rather than rounded to a neighboring candle. ETI does not synthesize a funding cadence (including the often-assumed eight-hour cadence).

## Schedule completeness

A derivative funding schedule must provide:

- the actual timestamp and signed rate for every settlement;
- source, HTTPS URI, license declaration and observed-at timestamp for each event;
- a coverage start and end;
- an explicit, chronological, unique `expected_timestamps` list in that coverage declaration.

The validator requires the supplied event timestamps to equal the enumerated expected timestamps exactly. This detects omissions relative to the supplied expected list. **It does not authenticate the list, prove that the source is authoritative, or establish that the expected list itself is complete.** The current schema validates declarations structurally; it does not fetch, archive, hash, or independently verify exchange funding-history responses. A user can still fabricate both the event list and its expected list. Derivative economic evaluation therefore remains blocked from any claim of verified funding completeness until a trusted provider adapter verifies source identity, pagination boundaries, response integrity and full settlement history.

## Economic limitations

Current funding arithmetic multiplies a normalized position in [-1, 1] by the signed funding rate in basis points. It is a simplified normalized-notional diagnostic. It does **not** model:

- contract quantity, contract multiplier in actual account currency or inverse contracts;
- mark/index price at settlement and venue-specific funding notional;
- leverage, margin mode, collateral, liquidation or account equity;
- funding settlement currency and FX conversion;
- position changes or partial fills occurring at the settlement instant;
- venue-specific funding caps, floors, premium indexes or settlement rules.

Consequently, this calculation is not an account P&L ledger and must not be used to infer leveraged returns or live-trading economics.

## Fail-closed conditions

The evaluator rejects naive timestamps, duplicate or non-chronological bars, duplicate settlements, non-finite/out-of-range positions, non-finite rates, and settlement timestamps that do not align exactly to a bar. The governed runner also requires funding coverage to span the entire OHLCV window. These checks improve internal consistency; they do not substitute for independently sourced funding history or a contract-level ledger.
