"""Explicitly authorized, budget-gated multi-market research evaluation.

This runner consumes a read-only universe scan and allocation plan, fetches each
instrument independently, preflights the total fit-call budget, and runs the
existing nested purged walk-forward evaluator. It has no order-execution path.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any
from uuid import uuid4

import pandas as pd

from .instruments import InstrumentSpec
from .market_data_hub import MarketDataHub, build_public_market_hub
from .pipeline import NESTED_CANDIDATE_NAMES, estimate_nested_model_fits_from_rows, run_nested_experiment
from .research_allocator import MAX_DAILY_USD, MAX_MONTHLY_USD
from .transaction_costs import validate_transaction_cost_model


def _number(name: str, value: Any, *, positive: bool = False, allow_negative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if (not math.isfinite(result) or (result < 0 and not allow_negative)
            or (positive and result <= 0)):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'non-negative'}")
    return result


def _validate_funding_event_alignment(events: list[dict[str, Any]],
                                       candle_timestamps: pd.Series,
                                       identity: str) -> None:
    """Preflight exact funding/bar alignment before spending any model fits."""
    bars = pd.DatetimeIndex(pd.to_datetime(candle_timestamps, utc=True))
    if not bars.is_unique or not bars.is_monotonic_increasing:
        raise ValueError(f"{identity} funding preflight requires unique chronological bars")
    available = set(bars)
    for index, event in enumerate(events):
        try:
            timestamp = pd.Timestamp(event["timestamp"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{identity} funding event {index} has an invalid timestamp") from exc
        if timestamp.tzinfo is None:
            raise ValueError(f"{identity} funding event {index} timestamp must include a timezone")
        if timestamp.tz_convert("UTC") not in available:
            raise ValueError(
                f"{identity} funding event {index} does not align to a fetched OHLCV bar; "
                "preflight stopped before model fitting"
            )


def _instrument(record: dict[str, Any]) -> InstrumentSpec:
    item = record.get("instrument")
    if not isinstance(item, dict):
        raise ValueError("Allocation candidate is missing instrument metadata")
    return InstrumentSpec(
        symbol=item["symbol"], venue=item["venue"],
        asset_class=item["asset_class"], market_type=item["market_type"],
        base_currency=item.get("base_currency"),
        quote_currency=item.get("quote_currency"),
        settlement_currency=item.get("settlement_currency"),
        contract_multiplier=(1.0 if item.get("contract_multiplier") is None else item.get("contract_multiplier")),
        supports_short=item.get("supports_short", False),
        supports_funding=item.get("supports_funding", False),
        active=item.get("active", True),
    )


def run_approved_multi_market_research(
    scan_report: dict[str, Any],
    allocation_plan: dict[str, Any],
    approval: dict[str, Any],
    *,
    hub: MarketDataHub | None = None,
    trial_ledger: Any | None = None,
    owner_id: str | None = None,
    workspace_id: str | None = None,
    trial_family_prefix: str = "eti-run",
) -> dict[str, Any]:
    """Run only explicitly approved candidates after complete budget preflight.

    Approval is a local governance artifact, not a cryptographic identity check.
    It must be reviewed by the operator. Costs are explicit assumptions, not
    automatically inferred from a provider's market data.
    """
    if (not isinstance(scan_report, dict) or scan_report.get("scan_only") is not True
            or scan_report.get("execution_enabled") is not False):
        raise ValueError("A read-only market-universe scan with execution disabled is required")
    if not isinstance(allocation_plan, dict) or allocation_plan.get("plan_only") is not True:
        raise ValueError("A plan-only research allocation is required")
    if allocation_plan.get("training_authorized") is not False:
        raise ValueError("Unexpected allocation schema: allocator must remain plan-only")
    if not isinstance(approval, dict) or approval.get("schema_version") != "eti.research_approval.v1":
        raise ValueError("A versioned research approval artifact is required")
    if approval.get("approved") is not True or approval.get("authorize_research_fits") is not True:
        raise ValueError("Research approval and fit authorization must both be explicit")
    if approval.get("execution_enabled") is not False:
        raise ValueError("Approval must explicitly keep execution_enabled=false")
    if trial_ledger is not None:
        if not isinstance(owner_id, str) or not owner_id.strip():
            raise ValueError("owner_id is required when trial_ledger is configured")
        if not isinstance(workspace_id, str) or not workspace_id.strip():
            raise ValueError("workspace_id is required when trial_ledger is configured")
        if not isinstance(trial_family_prefix, str) or not trial_family_prefix.strip():
            raise ValueError("trial_family_prefix must be non-empty")
    elif owner_id is not None or workspace_id is not None:
        raise ValueError("owner_id/workspace_id require a configured trial_ledger")

    budget = approval.get("usd_budget")
    if not isinstance(budget, dict):
        raise ValueError("Approval must include explicit USD budget inputs")
    estimated_cost = _number("estimated_run_cost_usd", budget.get("estimated_run_cost_usd"))
    daily_spend = _number("daily_spend_to_date_usd", budget.get("daily_spend_to_date_usd"))
    monthly_spend = _number("monthly_spend_to_date_usd", budget.get("monthly_spend_to_date_usd"))
    if estimated_cost > MAX_DAILY_USD or daily_spend + estimated_cost > MAX_DAILY_USD:
        raise ValueError("Daily USD research ceiling exceeded")
    if monthly_spend + estimated_cost > MAX_MONTHLY_USD:
        raise ValueError("Monthly USD research ceiling exceeded")

    config = scan_report.get("config") or {}
    timeframe = config.get("timeframe")
    limit = config.get("requested_rows")
    min_rows = config.get("minimum_valid_rows")
    if not isinstance(timeframe, str) or not timeframe:
        raise ValueError("Scan report is missing its timeframe")
    if any(isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in (limit, min_rows)):
        raise ValueError("Scan report has invalid history limits")
    if min_rows > limit:
        raise ValueError("Scan minimum history cannot exceed requested rows")

    max_fits = approval.get("max_model_fit_calls")
    if isinstance(max_fits, bool) or not isinstance(max_fits, int) or not 1 <= max_fits <= 32:
        raise ValueError("max_model_fit_calls must be an integer from 1 through 32")
    settings = approval.get("evaluation")
    if not isinstance(settings, dict):
        raise ValueError("Approval must include evaluation settings")
    horizon = settings.get("horizon", 6)
    min_train = settings.get("min_train", 500)
    validation_size = settings.get("validation_size", 100)
    test_size = settings.get("test_size", 100)
    step_size = settings.get("step_size", 100)
    for name, value in (("horizon", horizon), ("min_train", min_train),
                        ("validation_size", validation_size), ("test_size", test_size),
                        ("step_size", step_size)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"evaluation.{name} must be a positive integer")

    scan_rows = {}
    for source in scan_report.get("candidates", []):
        if not isinstance(source, dict) or not isinstance(source.get("instrument"), dict):
            continue
        source_instrument = source["instrument"]
        source_key = f"{source_instrument.get('venue')}:{source_instrument.get('market_type')}:{source_instrument.get('symbol')}"
        if source_key in scan_rows:
            raise ValueError(f"Duplicate instrument identity in scan report: {source_key}")
        scan_rows[source_key] = source

    rows_by_key = {}
    for row in allocation_plan.get("candidates", []):
        if not isinstance(row, dict) or not isinstance(row.get("instrument"), dict):
            continue
        instrument = row["instrument"]
        key = f"{instrument.get('venue')}:{instrument.get('market_type')}:{instrument.get('symbol')}"
        if key in rows_by_key:
            raise ValueError(f"Duplicate instrument identity in allocation plan: {key}")
        rows_by_key[key] = row

    approvals = approval.get("instruments")
    if not isinstance(approvals, dict) or not approvals:
        raise ValueError("Approval must list exact venue:market_type:symbol instruments")
    prepared = []
    data_hub = hub or build_public_market_hub()
    for key, cost in sorted(approvals.items()):
        allocated = rows_by_key.get(key)
        if allocated is None:
            raise ValueError(f"{key} is not present in the allocation plan")
        source = scan_rows.get(key)
        if source is None or source.get("status") != "data_ready":
            raise ValueError(f"{key} is not backed by a data_ready row in the supplied scan")
        if allocated.get("economic_evaluation_eligible") is not True:
            raise ValueError(f"{key} is not eligible: {allocated.get('eligibility_reasons')}")
        if allocated.get("transaction_cost_model_status") != "validated":
            raise ValueError(f"{key} does not have a validated transaction-cost status")
        if allocated.get("training_authorized") is not False:
            raise ValueError(f"{key} has an unexpected allocator authorization state")
        if not isinstance(cost, dict):
            raise ValueError(f"Missing explicit cost assumptions for {key}")
        evidence_model = allocated.get("transaction_cost_model")
        validated_model = validate_transaction_cost_model(evidence_model, identity=key)
        fee = _number(f"{key}.fee_bps", cost.get("fee_bps"))
        slippage_component = _number(f"{key}.slippage_bps", cost.get("slippage_bps"))
        spread = _number(f"{key}.spread_bps", cost.get("spread_bps"))
        market_impact = _number(f"{key}.market_impact_bps", cost.get("market_impact_bps"))
        funding = _number(
            f"{key}.funding_bps_per_bar", cost.get("funding_bps_per_bar"),
            allow_negative=True,
        )
        if _number(f"{key}.borrow_bps_per_bar", cost.get("borrow_bps_per_bar")) != 0:
            raise ValueError(f"{key} borrow cost is not yet supported by the evaluator")
        expected = {
            "fee_bps": fee, "slippage_bps": slippage_component,
            "spread_bps": spread, "market_impact_bps": market_impact,
            "funding_bps_per_bar": funding,
        }
        for field, value in expected.items():
            if not math.isclose(validated_model[field], value, rel_tol=0, abs_tol=1e-9):
                raise ValueError(f"{key} approval {field} does not match validated evidence")
        allow_short = cost.get("allow_short", False)
        if not isinstance(allow_short, bool):
            raise ValueError(f"{key}.allow_short must be boolean")
        instrument = _instrument(allocated)
        if instrument.market_type != "spot" or instrument.supports_funding:
            raise ValueError(
                f"{key} derivative economic evaluation is blocked until contract quantity, "
                "mark-price funding, settlement currency, margin and account-equity P&L "
                "are integrated end to end"
            )
        if allow_short:
            raise ValueError(
                f"{key} short research is blocked until borrow and financing costs are modeled"
            )
        if not math.isclose(validated_model["contract_multiplier"], instrument.contract_multiplier, rel_tol=0, abs_tol=1e-9):
            raise ValueError(f"{key} contract multiplier does not match instrument metadata")
        if not instrument.active:
            raise ValueError(f"{key} is inactive and cannot be researched")
        if source.get("instrument") != allocated.get("instrument"):
            raise ValueError(f"{key} instrument metadata differs between scan and allocation")
        if source.get("provenance") != allocated.get("provenance") or source.get("quality") != allocated.get("quality"):
            raise ValueError(f"{key} scan provenance/quality differs from allocation evidence")
        if allow_short and not instrument.supports_short:
            raise ValueError(f"{key} does not declare short capability")
        funding_schedule = validated_model.get("funding_schedule", [])
        if funding != 0 or funding_schedule:
            raise ValueError(f"{key} spot research cannot include derivative funding assumptions")
        snapshot = data_hub.fetch(instrument, timeframe, limit, min_rows=min_rows)
        from .features import build_features, feature_columns_for_frame, make_labels
        frame = snapshot.frame
        feature_frame = build_features(frame)
        columns = feature_columns_for_frame(frame)
        labels = make_labels(frame, horizon=horizon)
        usable = feature_frame.join(labels.rename("label")).dropna(subset=columns + ["label"])
        if trial_ledger is not None:
            # Match the exact feature/label/close/timestamp frame hashed by
            # run_nested_experiment so ledger lineage is reproducible.
            usable["close"] = frame.loc[usable.index, "close"].astype(float)
            usable["timestamp"] = frame.loc[usable.index, "timestamp"].to_numpy()
        fit_calls = estimate_nested_model_fits_from_rows(
            len(usable), horizon, min_train, validation_size, test_size, step_size
        )
        trial_lineage = None
        if trial_ledger is not None:
            from .evaluation import walk_forward_folds
            folds = walk_forward_folds(
                len(usable), min_train=min_train, validation_size=validation_size,
                test_size=test_size, step_size=step_size, purge_bars=horizon,
            )
            training_rows = sorted({index for fold in folds for index in range(fold.train_end)})
            evaluation_rows = sorted({
                index for fold in folds for index in range(fold.test_start, fold.test_end)
            })
            def rows_sha256(indices: list[int]) -> str:
                payload = pd.util.hash_pandas_object(usable.iloc[indices], index=True).to_numpy().tobytes()
                return hashlib.sha256(payload).hexdigest()
            trial_lineage = {
                "training_data_sha256": rows_sha256(training_rows),
                "evaluation_data_sha256": rows_sha256(evaluation_rows),
            }
        prepared.append({
            "key": key, "instrument": instrument, "frame": frame,
            "provenance": snapshot.provenance, "quality": snapshot.quality,
            "fit_calls": fit_calls, "fee_bps": fee,
            "slippage_bps": slippage_component + spread / 2.0 + market_impact,
            "funding_bps_per_bar": funding, "funding_events": funding_schedule,
            "allow_short": allow_short,
            "transaction_cost_model": validated_model,
            "trial_lineage": trial_lineage,
        })

    total_fits = sum(item["fit_calls"] for item in prepared)
    if total_fits > max_fits:
        raise ValueError(
            f"Preflight stopped before any model fitting: {total_fits} estimated fits "
            f"exceed approval cap {max_fits}"
        )

    if trial_ledger is not None and step_size < test_size:
        raise ValueError(
            "Trial-ledger recording requires non-overlapping outer-test windows: step_size >= test_size"
        )
    results = {}
    run_id = uuid4().hex[:16]
    for item in prepared:
        registered_trials: dict[str, str] = {}
        started_trials: dict[str, str] = {}
        family_id = None
        try:
            if trial_ledger is not None:
                family_id = f"{trial_family_prefix}:{run_id}:{item['key']}"
                if len(family_id) > 160:
                    raise ValueError("Generated trial family ID exceeds the storage limit")
                cost_sha256 = hashlib.sha256(json.dumps(
                    item["transaction_cost_model"], sort_keys=True, separators=(",", ":")
                ).encode("utf-8")).hexdigest()
                for candidate_name in NESTED_CANDIDATE_NAMES:
                    config_payload = {
                        "candidate": candidate_name, "instrument": item["key"],
                        "timeframe": timeframe, "horizon": horizon,
                        "min_train": min_train, "validation_size": validation_size,
                        "test_size": test_size, "step_size": step_size,
                        "fee_bps": item["fee_bps"], "slippage_bps": item["slippage_bps"],
                        "allow_short": item["allow_short"],
                    }
                    config_sha256 = hashlib.sha256(json.dumps(
                        config_payload, sort_keys=True, separators=(",", ":")
                    ).encode("utf-8")).hexdigest()
                    trial_id = hashlib.sha256(
                        f"{family_id}:{candidate_name}".encode("utf-8")
                    ).hexdigest()[:32]
                    trial_ledger.register_research_trial(
                        owner_id, workspace_id, family_id, trial_id, candidate_name,
                        config_sha256, item["trial_lineage"]["training_data_sha256"],
                        item["trial_lineage"]["evaluation_data_sha256"], cost_sha256,
                        f"no_trade:{item['key']}",
                    )
                    # Keep registration and start as separate lifecycle facts. If
                    # start fails, the registered-but-unstarted hypothesis remains
                    # visible in the family but is not counted as an attempted trial.
                    registered_trials[candidate_name] = trial_id
                    trial_ledger.start_research_trial(owner_id, workspace_id, trial_id)
                    started_trials[candidate_name] = trial_id
            report = run_nested_experiment(
                item["frame"], horizon=horizon, min_train=min_train,
                validation_size=validation_size, test_size=test_size,
                step_size=step_size, timeframe=timeframe,
                allow_short=item["allow_short"],
                funding_bps_per_bar=item["funding_bps_per_bar"],
                funding_events=item["funding_events"],
                fee_bps=item["fee_bps"], slippage_bps=item["slippage_bps"],
                max_model_fits=item["fit_calls"],
            )
            if trial_ledger is not None:
                streams = report.get("candidate_oos_returns", {})
                if set(streams) != set(NESTED_CANDIDATE_NAMES):
                    raise ValueError("Research report omitted one or more registered candidate return streams")
                for candidate_name, trial_id in started_trials.items():
                    stream = streams[candidate_name]
                    if stream.get("ledger_compatible") is not True:
                        raise ValueError(
                            "Outer-test windows overlap or are too short for a unique ledger return series; "
                            "set step_size >= test_size and use at least eight observations"
                        )
                    trial_ledger.complete_research_trial(
                        owner_id, workspace_id, trial_id,
                        stream["timestamps"], stream["net_return_differentials"],
                    )
                report["trial_ledger"] = {
                    "family_id": family_id, "trial_ids": registered_trials,
                    "registered_candidate_count": len(registered_trials),
                    "started_candidate_count": len(started_trials),
                    "completed_candidate_count": len(started_trials),
                    "benchmark": "no_trade",
                    "warning": "Candidate returns are net of configured fee/slippage assumptions and relative to no-trade. Data/cost authenticity and unrecorded external searches remain operator responsibilities.",
                }
            results[item["key"]] = {
                "status": "evaluated", "instrument": item["instrument"].__dict__,
                "provenance": item["provenance"], "quality": item["quality"],
                "transaction_cost_model": item["transaction_cost_model"],
                "fit_calls": item["fit_calls"], "report": report,
            }
        except Exception as exc:
            if trial_ledger is not None:
                for trial_id in started_trials.values():
                    try:
                        trial_ledger.fail_research_trial(
                            owner_id, workspace_id, trial_id,
                            f"{type(exc).__name__}: {str(exc)[:900]}",
                        )
                    except (LookupError, ValueError):
                        pass
            results[item["key"]] = {
                "status": "evaluation_failed", "fit_calls": item["fit_calls"],
                "trial_ledger": ({
                    "family_id": family_id,
                    "registered_trial_ids": registered_trials,
                    "started_trial_ids": started_trials,
                    "registered_candidate_count": len(registered_trials),
                    "started_candidate_count": len(started_trials),
                    "warning": "Registered-but-unstarted trials remain visible but are excluded from attempted-trial statistics.",
                } if trial_ledger is not None else None),
                "error": {"type": type(exc).__name__, "message": str(exc)[:500]},
            }

    return {
        "schema_version": "eti.multi_market_research_result.v1",
        "training_performed": any(x["status"] == "evaluated" for x in results.values()),
        "planned_model_fit_calls": total_fits,
        "model_fit_cap": max_fits,
        "usd_budget": {
            "estimated_run_cost_usd": estimated_cost,
            "daily_spend_to_date_usd": daily_spend,
            "monthly_spend_to_date_usd": monthly_spend,
            "daily_cap_usd": MAX_DAILY_USD, "monthly_cap_usd": MAX_MONTHLY_USD,
        },
        "market_count": len(results), "markets": results,
        "execution_enabled": False,
        "warning": (
            "Independent per-instrument nested walk-forward research only. "
            "No orders were sent. USD authorization is an operator-supplied "
            "estimate, not a measured Kaggle invoice. Yahoo reference data, "
            "session-based markets without calendars, futures rolls, and macro "
            "release vintages require separate validation before economic use."
        ),
    }
