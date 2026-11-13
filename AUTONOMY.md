# ETI autonomy contract

## Product decision

ETI is an owner-supervised autonomous trading system. The owner should not have to choose markets, symbols, strategies, timeframes, models, scan schedules, or individual trades during normal operation. The owner performs identity and financial-account setup; ETI performs the research and operating workflow within owner-defined, immutable limits.

## Owner-only control plane

Only the authenticated account owner may:
- Sign in, recover the account and manage account security.
- Add, verify, rotate, disable or remove exchange/broker connections.
- Grant provider permissions and choose the account/sub-account to connect.
- Set or change capital allocation, permitted venues/markets, maximum leverage, per-trade risk, exposure, daily loss and drawdown limits.
- Enable/disable paper or live execution, and operate the global kill switch.
- Approve a transition from paper trading to live trading.

Provider credentials are secrets, not agent memory. They must be sent over HTTPS, encrypted at rest with a server-side key-management boundary, masked in all responses, excluded from logs/exports/prompts and inaccessible to ordinary agents. Withdrawal permissions must never be requested. Start with read-only credentials; execution credentials are a separate explicitly enabled permission.

## AI-managed workflow

Once the owner has connected accounts and set a risk policy, ETI should autonomously:
1. Discover eligible instruments and venues from the enabled provider capabilities.
2. Collect, validate and refresh market data.
3. Classify market regimes and rank opportunities.
4. Select suitable specialist models and strategies based on out-of-sample evidence.
5. Train candidate models on approved datasets, evaluate them with purged walk-forward tests, calibrate probabilities and reject weak candidates.
6. Backtest candidates with fees, spread, slippage, funding, contract specifications and realistic execution assumptions.
7. Select a portfolio under correlation, concentration, liquidity and risk constraints.
8. Monitor positions, data freshness, model drift, venue health and risk limits.
9. In paper mode, generate, simulate and manage orders automatically.
10. In live mode, submit and manage orders only through a separately enabled execution service and only when every independent risk gate passes.
11. Explain decisions, record evidence and audit every action; alert the owner only for material exceptions, credential issues, risk events or decisions requiring owner authority.

The AI may propose a risk-policy change but must never apply it itself. The execution service, not the language model, enforces limits.

## Autonomy states

- **Setup:** owner login, provider connection and risk policy are incomplete; no trading.
- **Research:** autonomous market discovery, research, training, evaluation and paper simulation; no real orders.
- **Paper autonomous:** ETI manages a simulated portfolio continuously under the owner's policy.
- **Live armed:** owner explicitly enables live execution for a selected connected account after paper-readiness checks. The system still cannot exceed the policy.
- **Halted:** kill switch, stale data, provider failure, risk breach, model invalidation or operational uncertainty stops new orders; existing positions follow the configured safe-management policy and are surfaced to the owner.

Default state is Research/Paper, never Live Armed.

## Hard safety invariants

- The model/agent layer cannot read raw API secrets.
- The model/agent layer cannot edit owner risk policy, enable live mode, alter permissions or disable the kill switch.
- Every proposed order is checked by deterministic risk and portfolio services immediately before submission.
- Risk checks fail closed on missing, stale, malformed or contradictory inputs.
- No order is sent if market data, balance, position, instrument metadata, fees or account permissions cannot be verified.
- Enforce idempotency, client-order IDs, bounded retries, rate limits and reconciliation against the broker's actual order/position state.
- Use least privilege and separate read-only from trade-enabled credentials.
- Record a durable audit trace: data snapshot, strategy/model version, decision, policy version, risk result, order request, broker response and subsequent reconciliation.
- Owner kill switch must be available independently of the AI orchestrator.

## Implementation gates

1. **Owner control plane:** secure provider credential vault, connection verification, owner-only account/permission controls, immutable risk policy and kill switch.
2. **Autonomous research supervisor:** scheduled discovery, instrument selection, data quality checks, strategy/model orchestration, training/evaluation and exception reporting without manual task creation.
3. **Paper autonomy:** portfolio construction, simulated order lifecycle, continuous monitoring and restart-safe reconciliation.
4. **Execution adapters:** venue-specific order, cancel/replace, precision, fees, funding, margin and position reconciliation, first in sandbox/testnet.
5. **Live execution:** only after paper validation, independent security review, operational monitoring, owner opt-in and hard risk gates.

Do not describe ETI as fully autonomous until these gates are implemented and tested. A deterministic worker that executes one allowlisted market-snapshot task is not an AI autonomous trading system.
