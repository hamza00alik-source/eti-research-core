"""Evidence-backed transaction-cost model validation.

A status string is never sufficient. Models must identify the exact instrument,
carry time-bounded assumptions, and cite retrievable, licensed evidence.
"""
from __future__ import annotations

from datetime import datetime, timezone
import math
from typing import Any

SCHEMA_VERSION = "eti.transaction_cost_model.v1"
NUMERIC_FIELDS = (
    "fee_bps", "spread_bps", "slippage_bps", "market_impact_bps",
    "funding_bps_per_bar", "borrow_bps_per_bar", "contract_multiplier",
)


def _timestamp(name: str, value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{name} must include a timezone")
    return parsed.astimezone(timezone.utc)


def validate_transaction_cost_model(
    model: Any, *, identity: str, now: datetime | None = None,
) -> dict[str, Any]:
    """Return a normalized copy or raise ValueError; never infer missing costs."""
    if not isinstance(model, dict) or model.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported or missing transaction-cost model schema")
    if model.get("instrument_identity") != identity:
        raise ValueError("transaction-cost model instrument identity mismatch")
    for field in ("quote_currency", "settlement_currency"):
        value = model.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} must be explicitly supplied")
    numeric = {}
    for field in NUMERIC_FIELDS:
        value = model.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{field} must be finite and explicitly supplied")
        # Funding is signed: positive means longs pay shorts; negative means shorts pay longs.
        if (field != "funding_bps_per_bar" and value < 0) or (
            field == "contract_multiplier" and value <= 0
        ):
            raise ValueError(f"{field} is outside its permitted range")
        numeric[field] = float(value)
    retrieved = _timestamp("retrieved_at", model.get("retrieved_at"))
    valid_from = _timestamp("valid_from", model.get("valid_from"))
    valid_until = _timestamp("valid_until", model.get("valid_until"))
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None or current.utcoffset() is None:
        raise ValueError("validation time must be timezone-aware")
    current = current.astimezone(timezone.utc)
    if not valid_from <= current <= valid_until:
        raise ValueError("transaction-cost model is outside its validity window")
    if retrieved > current:
        raise ValueError("transaction-cost model retrieval timestamp is in the future")
    if valid_until <= valid_from:
        raise ValueError("transaction-cost validity window is empty")
    evidence = model.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("at least one evidence record is required")
    normalized_evidence = []
    covered_fields: set[str] = set()
    allowed_coverage = set(NUMERIC_FIELDS)
    for index, item in enumerate(evidence):
        if not isinstance(item, dict):
            raise ValueError(f"evidence[{index}] must be an object")
        source = item.get("source")
        uri = item.get("uri")
        license_status = item.get("license_status")
        if not isinstance(source, str) or not source.strip():
            raise ValueError(f"evidence[{index}].source is required")
        if not isinstance(uri, str) or not uri.startswith("https://"):
            raise ValueError(f"evidence[{index}].uri must be an HTTPS URL")
        if license_status not in ("open", "licensed"):
            raise ValueError(f"evidence[{index}] has an unapproved or unknown license")
        observed_at = _timestamp(f"evidence[{index}].observed_at", item.get("observed_at"))
        if observed_at > current:
            raise ValueError(f"evidence[{index}] is dated in the future")
        covers = item.get("covers")
        if not isinstance(covers, list) or not covers:
            raise ValueError(f"evidence[{index}].covers must list the assumptions supported")
        if any(not isinstance(field, str) or field not in allowed_coverage for field in covers):
            raise ValueError(f"evidence[{index}].covers contains an unsupported cost field")
        if len(set(covers)) != len(covers):
            raise ValueError(f"evidence[{index}].covers contains duplicates")
        covered_fields.update(covers)
        normalized_evidence.append({
            "source": source.strip(), "uri": uri,
            "license_status": license_status, "observed_at": observed_at.isoformat(),
            "covers": sorted(covers),
        })
    missing_coverage = sorted(allowed_coverage - covered_fields)
    if missing_coverage:
        raise ValueError("transaction-cost evidence does not cover: " + ", ".join(missing_coverage))
    schedule = model.get("funding_schedule", [])
    if not isinstance(schedule, list):
        raise ValueError("funding_schedule must be a list when supplied")
    normalized_schedule = []
    seen_events: set[str] = set()
    for index, event in enumerate(schedule):
        if not isinstance(event, dict):
            raise ValueError(f"funding_schedule[{index}] must be an object")
        event_at = _timestamp(f"funding_schedule[{index}].timestamp", event.get("timestamp"))
        event_rate = event.get("rate_bps")
        if isinstance(event_rate, bool) or not isinstance(event_rate, (int, float)) or not math.isfinite(event_rate):
            raise ValueError(f"funding_schedule[{index}].rate_bps must be finite and signed")
        event_key = event_at.isoformat()
        if event_key in seen_events:
            raise ValueError("funding_schedule contains duplicate settlement timestamps")
        seen_events.add(event_key)
        if event_at < valid_from or event_at > valid_until or event_at > current:
            raise ValueError("funding settlement is outside its evidence validity window")
        source = event.get("source")
        uri = event.get("uri")
        license_status = event.get("license_status")
        observed_at = _timestamp(f"funding_schedule[{index}].observed_at", event.get("observed_at"))
        if not isinstance(source, str) or not source.strip():
            raise ValueError(f"funding_schedule[{index}].source is required")
        if not isinstance(uri, str) or not uri.startswith("https://"):
            raise ValueError(f"funding_schedule[{index}].uri must be HTTPS")
        if license_status not in ("open", "licensed") or observed_at > current:
            raise ValueError(f"funding_schedule[{index}] has invalid evidence or license")
        normalized_schedule.append({
            "timestamp": event_at.isoformat(), "rate_bps": float(event_rate),
            "source": source.strip(), "uri": uri, "license_status": license_status,
            "observed_at": observed_at.isoformat(),
        })
    normalized_schedule.sort(key=lambda item: item["timestamp"])
    coverage = model.get("funding_schedule_coverage")
    normalized_coverage = None
    if normalized_schedule:
        if not isinstance(coverage, dict):
            raise ValueError("funding_schedule_coverage is required when funding events are supplied")
        coverage_start = _timestamp("funding_schedule_coverage.start", coverage.get("start"))
        coverage_end = _timestamp("funding_schedule_coverage.end", coverage.get("end"))
        if coverage_start > coverage_end or coverage_start < valid_from or coverage_end > valid_until or coverage_end > current:
            raise ValueError("funding schedule coverage is invalid or outside its evidence window")
        if normalized_schedule[0]["timestamp"] < coverage_start.isoformat() or normalized_schedule[-1]["timestamp"] > coverage_end.isoformat():
            raise ValueError("funding events fall outside declared schedule coverage")
        normalized_coverage = {"start": coverage_start.isoformat(), "end": coverage_end.isoformat()}
    elif coverage is not None:
        raise ValueError("funding schedule coverage cannot be supplied without funding events")
    if numeric["funding_bps_per_bar"] != 0 and normalized_schedule:
        raise ValueError("per-bar funding and event-timed funding cannot both be specified")
    return {
        "schema_version": SCHEMA_VERSION,
        "instrument_identity": identity,
        "quote_currency": model["quote_currency"].strip().upper(),
        "settlement_currency": model["settlement_currency"].strip().upper(),
        **numeric,
        "funding_schedule": normalized_schedule,
        "funding_schedule_coverage": normalized_coverage,
        "retrieved_at": retrieved.isoformat(),
        "valid_from": valid_from.isoformat(),
        "valid_until": valid_until.isoformat(),
        "evidence": normalized_evidence,
    }
