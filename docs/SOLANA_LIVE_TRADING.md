# Solana live trading integration — foundation

## Scope delivered in this change

- Adds a Jupiter Swap API v2 client for the Solana mainnet order/execute contract.
- Requests a taker-bound order, validates the returned transaction, request ID and quoted output.
- Defines a strict signer interface. ETI receives only a signing callback; private keys are not loaded, stored, or logged by this module.
- Executes a signed order through Jupiter `POST /execute` and returns a transaction signature and actual amount fields.
- Includes transport-injected tests for quote validation, amount validation, disabled execution and the end-to-end request/sign/execute contract.

## Important current limitation

`SOLANA_LIVE_EXECUTION_ENABLED` is hard-coded false. This PR cannot submit a mainnet trade. The end-to-end test temporarily overrides the constant in-process only; it uses a fake transport and cannot reach the network. This is an execution integration foundation, not a live-ready autonomous trader.

## Required work before real-money activation

1. Integrate a production signer/KMS or owner-controlled isolated wallet service. Never store a seed phrase in GitHub, chat, frontend code, or ordinary application settings.
2. Add Solana-specific token discovery and risk intelligence: mint/freeze authority, holder concentration, liquidity depth, pool age, sellability simulation, program allowlists and wallet-cluster risk.
3. Add deterministic pre-trade policy: max SOL per trade/day, max open positions, max slippage/price impact, token allow/deny rules, SOL reserve, cooldowns and daily loss kill switch.
4. Add transaction simulation, blockhash/expiry handling, confirmation/finality checks, idempotency, balance reconciliation and exit-order recovery.
5. Add a durable strategy/decision ledger with exact quote, route, fee, signed transaction hash, signature, confirmation and realized P&L.
6. Run integration tests against devnet or a controlled mainnet-fork environment, then a separately authorized tiny-capital mainnet canary.
7. Only after independent security and operational review should the hard gate be changed in a separate reviewed change.

## API contract

Jupiter Swap API v2:
- `GET /order` with `inputMint`, `outputMint`, `amount`, and `taker`.
- Sign the returned base64 transaction using the isolated signer.
- `POST /execute` with `signedTransaction` and the matching `requestId`.

The current implementation uses Jupiter's managed landing path. Quotes and execution are distinct; obtaining a quote does not trade. A successful API response must include a signature and amount fields before ETI records a receipt.

## Configuration

Set `JUPITER_API_KEY` in the backend environment for production rate limits. Keyless requests are intended only for low-rate prototyping. Do not commit the API key.


## Solana token screening foundation (added in Phase 3)

`eti.solana_token_risk` now provides a deterministic, read-only first-pass policy
for candidate tokens. It evaluates mint/freeze authority, largest-holder
concentration, reported pool liquidity, pool age, and an external sell-simulation
result. Missing or malformed evidence never defaults to safe. The result is
`PASS`, `WATCH`, or `REJECT`, with a numeric score, evidence completeness,
and explicit reason codes.

Default initial screening thresholds are intentionally conservative:
- Reject active mint authority and active freeze authority.
- Require a successful sell simulation.
- Require at least USD 25,000 reported liquidity.
- Require a pool age of at least two hours.
- Flag a largest-holder share above 20%.
- Token verification is optional by default and can be required by policy.

These are configurable engineering defaults, not empirically validated
profitability thresholds. The module does not yet fetch evidence itself; it
requires trusted upstream providers and must be integrated with live token
discovery, on-chain RPC, holder aggregation, and a real transaction simulation
before any candidate can reach execution. A PASS is not a safety guarantee.
