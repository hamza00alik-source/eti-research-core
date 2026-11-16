# ETI shared intelligence and specialist departments

## Architecture

ETI uses one shared intelligence foundation coordinated by the Queen and a
portfolio of specialist systems. It does not assume one universal model can
reliably serve every task, asset class, market type or timeframe.

The architecture contract is implemented in `eti.intelligence`:

- `market_data`: deterministic instrument discovery, provenance and quality.
- `market_intelligence`: regimes, volatility, trends, correlations and derivatives context.
- `prediction`: calibrated task-specific predictive models.
- `strategy`: strategy research, comparison and portfolio optimization.
- `risk`: independent deterministic risk limits, with optional forecasting models.
- `macro`: point-in-time macro data and event/regime specialists.
- `execution`: cost/liquidity models and deterministic order controls.
- `learning`: drift monitoring, evidence, retraining and champion/challenger lifecycle.

The shared foundation consists of common data and feature contracts, the
existing immutable model-version registry, calibration/evaluation evidence,
workspace memory and audit history.

## Department-aware routing

Existing server-implemented task capabilities map explicitly to departments.
The Queen reports the department on task completion and emits a read-only
department readiness manifest at startup. Unknown capabilities remain
unmapped; no model-generated or user-provided capability is executed.

Model selection is fail-closed:

1. A model must be explicitly assigned using `input_schema.department` or
   `evaluation_report.department`.
2. Conflicting or unknown declarations are rejected.
3. Only lifecycle=`approved` versions are eligible for specialist selection.
4. The model must declare support for the requested `asset_class_market_type`
   class. Optional `input_schema.market_types` and `timeframes` further
   constrain eligibility.
5. No cross-department, cross-market or legacy-name fallback is performed.

Existing model versions without explicit department metadata are reported as
unassigned legacy models. Their department is never guessed from their model
ID. Existing model registration and human approval lifecycle remains intact.

## What is and is not active

This change adds the department contract, explicit task routing, specialist
selection rules, readiness reporting and Queen observability. It does not
pretend every department already has a trained model or autonomous agent.
Departments with no implementation are reported as `not_configured`.

The BTC deployable classifier is explicitly assigned to the prediction
department as a direction-classifier candidate. Each exported artifact receives
a content-addressed immutable version, experiment ID, and code/config/dataset
lineage hashes. Registry metadata preserves the department, role, experiment ID
and lineage in the input-schema record without changing the existing feature
contract. Legacy registry entries remain unassigned until reviewed. Separate
trained specialists for additional departments, roles, asset classes and
timeframes remain subsequent implementation steps.

## Authority boundaries

- Queen coordinates tasks and reports readiness; it does not approve models.
- Specialist model selection does not prove profitability or execution fitness.
- Risk controls remain independent and cannot be overridden by predictions.
- Research budget caps and permissions remain unchanged.
- No department or model in this architecture can enable live orders.
- A model can become approved only through the existing independent approval
  process; promotion evidence remains a separate human-reviewed gate.


## Role-level specialization

Model role is part of specialist identity and selection; department readiness alone is not sufficient for a specific prediction task.


A department may contain multiple distinct specialist roles. The registry
selector accepts an explicit `model_role` in addition to department, asset
class, market type and timeframe. A role must belong to the selected department;
models with conflicting or invalid role metadata fail closed. No cross-role
fallback is permitted. Queen's readiness manifest reports readiness for every
declared role separately, so a ready prediction department does not imply that
its return-distribution or volatility specialists are ready.


## Active deterministic specialist services

The Queen's allowlisted worker registry exposes bounded specialist handlers:

| Department | Capability | Service |
|---|---|---|
| Market intelligence | `market.regime` | Causal volatility/trend regime diagnostics |
| Strategy | `strategy.select` | Fixed strategy candidate selection on a caller-supplied validation window |
| Risk | `risk.assess` | Independent deterministic risk gate |
| Macro | `macro.as_of` | Point-in-time macro vintage selection with release/license checks |
| Execution | `execution.cost_estimate` | Evidence-validated fee/spread/slippage/impact/funding estimate; no orders |
| Learning | `learning.economic_assess` | Conservative research economics disposition |

These services reuse existing ETI components. They are intentionally not all ML
models: market-data quality, risk controls and execution cost validation remain
deterministic and auditable. Prediction is a separate calibrated model
department. Macro services process only supplied observations until a licensed,
release-time-aware external provider is integrated.

All handlers are allowlisted, accept data-only payloads, and return
`execution_enabled=false`. The cost service estimates costs only; it never
places orders. Strategy selection uses validation data only; callers must
preserve a separate untouched test window.


## Approved inference boundaries

The authenticated inference endpoint now requires an explicit model role, market
type and timeframe. It rejects a model unless its approved registry metadata
declares `department=prediction`, the exact requested role, and exact supported
market-type/timeframe values. The current probability response contract supports
only `direction_classifier`; other specialist roles cannot be routed through
that three-class output by assumption. Legacy approved records without these
metadata fields fail closed until reviewed and re-registered with explicit
scope.
