# Technical audit: five-agent Pump.fun / FOMO trencher article

Audit date: 2026-11-14  
Scope: the article "My Six Figure Grok Bot Trencher Setup: 5 Agents, 1 App, 1 Prompt (Full Guide)" by @savipww, dated Nov 1. This is an engineering/API audit, not investment advice. No claim is made that the author's six-figure result is independently verified.

## Executive result

The article describes a **research-and-alert desk with manual execution**, not an autonomous trading bot. Its own master prompt explicitly says FOMO has no public order API, the system must never place orders or touch funds, and the user must tap through to FOMO. Its separate copy bot reads account/session endpoints and emits a manual "Copy" link; it does not execute copies.

The broad pipeline is coherent as a prototype: discover tokens -> screen market/liquidity -> perform on-chain/token-risk checks -> score market conditions -> calculate a hypothetical ticket -> monitor paper positions -> alert on exits. The five agent names are mostly software modules and deterministic rules, not five independent AI systems. The article does not supply evidence of a profitable, reproducible strategy, a verified six-figure PnL history, or a complete live execution implementation.

ETI implementation decision: reuse the useful separation of responsibilities, but implement it as a venue/asset-specific **Solana research adapter** inside ETI's existing shared platform. Keep the current research/paper-only boundary. No FOMO session scraping, no wallet/session token collection, no live order routes, and no live key requirement.

## Claim-by-claim API audit

| Article dependency | Article usage | Verification | ETI decision |
|---|---|---|---|
| PumpPortal WebSocket | Subscribe to new-token/migration events | The documented WebSocket is `wss://pumpportal.fun/api/data?api-key=...`; documented subscriptions include `subscribeNewToken` and `subscribeMigration`. The article's migration-event approach is plausible. PumpPortal asks users to use one WebSocket connection, and some trade subscriptions are metered. | Supported optional discovery adapter. Validate actual payloads, reconnect, deduplicate by signature/mint, and record source timestamps. Do not assume every migration is immediately tradeable. |
| DexScreener | Token/pair discovery, liquidity, volume, price changes, prices | Official API reference documents pair search, pair lookup, token-pair lookup, and multi-token lookup. The `/tokens/v1/{chainId}/{tokenAddresses}` route accepts up to 30 token addresses; pair/token endpoints have different rate limits. | Supported market-data enrichment. Use the documented route and batch size; select the correct pool/pair; enforce freshness and rate-limit/backoff. The article's "~300/min" is not a universal limit for every endpoint. |
| Solana JSON-RPC | `getSlot`, `getAccountInfo`, `getTokenLargestAccounts`, `getTokenAccountsByOwner`, `getProgramAccounts` | These are standard Solana RPC methods. Public mainnet RPC is explicitly rate-limited and not intended for production infrastructure. `getProgramAccounts` can be expensive/large and provider policy varies. | Valid read-only source, but require provider capability tests, pagination/size safeguards where available, bounded concurrency, cache and redundant configured RPC for reliable research. Never interpret RPC timeout as a pass. |
| RugCheck report | `GET /v1/tokens/{mint}/report` | The report route is referenced in the article and the API family exists in public integrations. Treat response schema as versioned external data: verify live response keys and schema before relying on nested fields. | Optional risk-intelligence adapter. Store raw response, retrieval time, schema version and provenance. A missing field is unknown, not clean. |
| RugCheck insiders graph | `GET /v1/tokens/{mint}/insiders/graph` | Public API documentation describes an insider graph with `nodes` and `links`; large graphs can be truncated/capped. The article's proposed connected-component interpretation is not itself proof of common control or malicious intent. | Use as a risk feature only. Do not label a wallet cluster malicious solely because it is connected; expose graph coverage and uncertainty. |
| TypeSafe / Jev | `POST https://api.typesafe.ai/v1/systemone` with `model`, `state`, `questions`; Noul outputs | The article's host and request shape match the documented TypeSafe System One endpoint. Noul values are probabilities in [0,1]. The article's stated 1,200 requests/minute limit was not verified from the primary API material reviewed. | Optional external second opinion only. Default analyst must work without Jev. Pin a model version, handle 401/422/429/529 and timeout, log usage, and never let an LLM/decision model override deterministic risk gates. |
| FOMO app private API | `https://prod-api.fomo.family`, Privy bearer session token, `/v2/users/userHandle/{handle}`, `/v2/users/{id}/swaps`, `/v2/users/{id}/balances`, `/v2/leaderboard/{window}`, `/proxy/filterTokens` | The article itself calls this unofficial and says it may break. The bearer token is an authenticated app session, not a scoped developer key. These paths and payloads are not established by a stable public FOMO Labs developer contract in the reviewed material. Third-party FOMO data APIs are separate vendors and must not be confused with FOMO Labs. | Reject. No browser-header scraping, Privy/session token collection, token refresh automation, or private-account API polling. Use public on-chain sources or a clearly identified, separately contracted third-party data provider if ever needed. |
| FOMO trading | User taps Buy/Sell in FOMO | Article explicitly says no public order API and no automated order placement. The "LIVE" mode only tracks a manually confirmed buy/sell at an assumed price. | Do not call this automated execution. ETI paper simulator only. No FOMO dependency or execution button is required for the research equivalent. |
| Telegram Bot API / Web Push | Notifications and owner commands | These are standard delivery mechanisms, but Telegram bot tokens and Web Push private keys are secrets and require owner-only authorization, replay-safe callbacks, and secure storage. | Optional notification layer after the research pipeline is stable. It must not be an order-control channel. |

## Article agent architecture

1. **Scanner** — listens for Pump.fun token/migration events, enriches with DexScreener, maintains a waiting room, rejects obvious low-liquidity/concentration/overheated/copycat candidates.
2. **Auditor** — runs eight sequential checks: mint authority, freeze authority, creator holdings/sales, top holders, LP lock/burn, pool age, insider/bundled wallets, and token-account delegates.
3. **Analyst** — builds a compact market/on-chain state and asks Jev four Noul questions (future activity, organic buyers, flush absorption, chase risk), or computes rule-based proxies. Code combines outputs into a score and gate.
4. **Executor** — calculates ticket size from stack and pool liquidity, rechecks liquidity/price movement, then sends a manual FOMO link. PAPER mode creates a synthetic fill; the article's LIVE mode still requires the user to buy manually.
5. **Risk** — polls price, tracks peak, stop, trailing stop and timeout; PAPER closes synthetically, while the manual mode waits for the user to sell and confirm.

These are five bounded responsibilities, not necessarily five autonomous LLM agents. ETI should implement them as deterministic services/capabilities with typed messages and auditable state transitions. Only the optional Analyst judgment may call an external decision model.

## Critical technical and trading-rule defects

### Scanner thresholds are heuristics, not validated filters

Article defaults include liquidity >= $40,000, top-10 concentration <= 25%, 1h price change <= +80%, pool age >= 10 minutes, max waiting time 60 minutes and a "runner" threshold based on 24h change. These are arbitrary fixed thresholds. They are not justified by a labelled dataset, regime analysis or out-of-sample testing. A fixed $40k liquidity floor does not account for pool depth, quote asset, volatility, fee tier, price impact or exit capacity. Token/pool selection can be manipulated and data can be stale.

The copycat rule (normalized name/symbol substring matching) produces false positives and false negatives. It should be an informational flag, never a hard safety conclusion.

### Auditor checks have important semantic gaps

- **Mint/freeze authority revoked** reduces specific admin risks but does not make a token safe. Other programs, upgrade authorities, transfer hooks, taxes, liquidity ownership and market manipulation remain relevant.
- **Creator balance/sales** cannot be inferred from one current balance snapshot. "Never dropped since migration" requires complete historical token-account/transaction history and a correct creator identity. A null creator must remain unknown.
- **Top 10**: `getTokenLargestAccounts` returns up to 20 token accounts, not a guaranteed list of ten unique beneficial owners. Multiple accounts can belong to one owner; pools, bonding curves, lockers and program-owned accounts must be identified correctly. Exclusions can materially distort concentration.
- **LP lock/burn**: a percentage from an external report is only meaningful if it refers to the exact pool, correct LP token/program, lock contract and unlock schedule. "99% locked/burned" is not proof that liquidity is permanently safe or withdrawable under no circumstances.
- **Pool age**: migration/pair creation time is not a safety measure and can be confused with token mint age.
- **Insider/bundled graph**: connected wallets are not necessarily controlled by one actor. Graph data is incomplete and capped. Use uncertainty and coverage; do not hard reject based on a raw component count alone.
- **Delegations**: counting token accounts grouped by delegate is not the same as counting distinct beneficial holders or proving a drain. SPL Token and Token-2022 account layouts and semantics must both be handled. A delegate may be legitimate; delegate allowance/amount and current authority must be evaluated.
- **Fail closed** is correct for unavailable critical evidence, but the UI must distinguish "failed risk check" from "unknown due to provider outage". Otherwise outages can look like a market-wide scam wave.

Eight checks cannot guarantee a token is safe. The "8/8" badge is a checklist completion indicator, not a safety certificate.

### Analyst model is not a trained trading edge

The questions "will still be active six hours from now" and "are buyers separate real wallets" are not directly observable from a single snapshot. A model's Noul probability is not a calibrated trading probability unless ETI collects timestamped snapshots and later labels outcomes with a fixed definition. The article's score averages four probabilities with equal weights, but supplies no empirical reason for equal weights or the 70 cutoff. "Flush" and "chase" questions are subjective and may overlap with features already in the rule score. The fallback rule formulas (volume, buys, short-term price change) are hand-set thresholds, not learned coefficients.

The article's claim that Jev "decides" should not be mistaken for a market forecast guarantee. Model outputs must be evaluated against future labels and compared with simple baselines.

### Position sizing and execution assumptions

- Size = 8% of stack, capped at 2% of pool liquidity. This is not stop-distance-based risk sizing. A 6% stop means nominal stop loss is about 0.48% of stack before gaps, slippage, fees and liquidity impact; actual loss can be much larger.
- The $47.50 minimum ticket conflicts with small balances and creates a "bump" rule that can override the intended percentage allocation. The article itself calculates a minimum stack of $593.75 for 8% sizing.
- Recheck rules (liquidity drop >20%, price rise >15%) are coarse and do not estimate executable price impact or quote depth.
- PAPER fills at the price when a push is sent, not at a realistic bid/ask/AMM execution price. Exits use the next polled price, not the stop line; this can still be optimistic because polling can miss gaps and transient liquidity removal.
- The quoted FOMO fee schedule and referral discount are not independently verified in the article's technical spec. Fee tiers, discount eligibility and gas assumptions must come from current official terms and be parameterized. Do not hardcode promotional economics.
- Gas = 0 is not a universal economic cost assumption. The app may abstract gas from the user, but protocol/application fees and spread/price impact still exist.
- No market-impact model, failed-transaction model, MEV/sandwich risk, priority fee, stale quote rejection, partial fills or route failure model is provided.
- No empirical evidence supports the stop/trailing/timeout values: stop -6%, activate trailing at +20%, trail 12% from peak, close after 3h if gain <5%. A 12% trailing drawdown after +20% allows a large round-trip loss; the timeout condition is not a risk limit. Stops cannot guarantee execution at the stop price.

### Copy-bot design is unacceptable for ETI

The article asks for a second empty FOMO account but then reads its full bearer session token from browser requests. An "empty" account does not make a full-access session token safe. The token may permit account-level actions or reveal private data; storing or refreshing it creates a credential-theft and terms-of-service risk. Polling only page 1 every interval can miss trades during bursts; de-duplication does not fix missing pages or race conditions. "Mark all old swaps as seen" is a useful onboarding behavior but does not guarantee complete event capture.

Do not reproduce this mechanism.

## Paper-only ETI design decisions

- Build an optional Solana/Pump.fun discovery adapter, not a Solana-only replacement for ETI.
- Use public documented sources: PumpPortal for event discovery, DexScreener for pool/market snapshots, Solana JSON-RPC for chain state, and RugCheck as optional third-party risk intelligence.
- Treat source data as untrusted, timestamp every observation, persist raw response hashes and normalized values, and surface stale/missing/partial coverage explicitly.
- Implement Scanner, Auditor, Analyst, Sizer and Risk as deterministic, testable ETI capabilities with durable task/message events. Preserve the existing portfolio and risk engine as the final gate.
- Use ETI's own trained/calibrated model and nested walk-forward evaluation for market prediction. Jev may be an optional external comparator, never the authority.
- Paper fills must model AMM/quote-side execution, spread, price impact, fees, priority/transaction costs where applicable, latency, missed events, stale data, partial fills and stop gaps. If these are unavailable, mark the simulation assumption as incomplete and block any profitability claim.
- No FOMO app account, referral link, private session token, copy trading, wallet signing, private key, live order route or live execution.
- Keep all five agent outputs and all risk decisions attributable to code/model version, input snapshot and timestamp.

## Source references

- Article: https://teapawt.net/savipww/article/2116923821064167731
- PumpPortal real-time API: https://pumpportal.fun/data-api/real-time/
- DexScreener API reference: https://docs.dexscreener.com/api/reference
- Solana RPC overview and HTTP methods: https://solana.com/docs/rpc and https://solana.com/docs/rpc/http
- Solana public cluster limits: https://solana.com/docs/references/clusters
- RugCheck insider graph reference: https://fluxrpc.com/docs/rugcheck/getTokenInsiderGraph
- TypeSafe Jev System One endpoint reference: https://api.typesafe.ai/v1/systemone (official endpoint; current request contract described in TypeSafe developer docs)
- FOMO app/private API distinction: the article itself labels `prod-api.fomo.family` unofficial. Third-party FOMO API aggregators are not FOMO Labs' trading API and must be evaluated as separate vendors.
