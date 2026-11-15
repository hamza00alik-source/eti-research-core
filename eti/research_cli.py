"""Command-line orchestration for ETI's governed research-only workflow.

Examples:
  python -m eti.research_cli scan --output artifacts/market-universe.json
  python -m eti.research_cli allocate artifacts/market-universe.json --cost-models artifacts/validated-cost-models.json --estimated-run-cost-usd 2 --daily-spend-to-date-usd 0 --monthly-spend-to-date-usd 10
  python -m eti.research_cli run artifacts/market-universe.json artifacts/research-allocation.json artifacts/research-approval.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .research_allocator import allocate_research_budget
from .research_runner import run_approved_multi_market_research
from .universe import scan_market_universe


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Cannot read JSON input {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"JSON input must contain an object: {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m eti.research_cli",
        description="ETI scan → plan → explicitly approved research; never executes orders.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    scan = commands.add_parser("scan", help="discover instruments and validate read-only history")
    scan.add_argument("--asset-classes", default="crypto,forex,equity,commodity,index")
    scan.add_argument("--market-types", default="spot")
    scan.add_argument("--timeframe", default="1h")
    scan.add_argument("--limit", type=int, default=2000)
    scan.add_argument("--min-rows", type=int, default=662)
    scan.add_argument("--max-per-asset-class", type=int, default=5)
    scan.add_argument("--max-per-venue", type=int, default=2)
    scan.add_argument("--output", type=Path, default=Path("artifacts/market-universe.json"))

    allocate = commands.add_parser("allocate", help="rank data readiness and produce a plan only")
    allocate.add_argument("scan_report", type=Path)
    allocate.add_argument("--output", type=Path, default=Path("artifacts/research-allocation.json"))
    allocate.add_argument("--max-model-fit-calls", type=int, default=32)
    allocate.add_argument("--fits-per-candidate", type=int, default=2)
    allocate.add_argument("--max-candidates", type=int, default=8)
    allocate.add_argument("--max-age-seconds", type=float, default=7200)
    allocate.add_argument("--estimated-run-cost-usd", type=float, required=True)
    allocate.add_argument("--daily-spend-to-date-usd", type=float, required=True)
    allocate.add_argument("--monthly-spend-to-date-usd", type=float, required=True)
    allocate.add_argument("--cost-models", type=Path, required=True)

    run = commands.add_parser("run", help="run nested research using a separately reviewed approval JSON")
    run.add_argument("scan_report", type=Path)
    run.add_argument("allocation_plan", type=Path)
    run.add_argument("approval", type=Path)
    run.add_argument("--output", type=Path, default=Path("artifacts/multi-market-research.json"))

    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "scan":
        report = scan_market_universe(
            asset_classes=[x.strip() for x in args.asset_classes.split(",") if x.strip()],
            market_types=[x.strip() for x in args.market_types.split(",") if x.strip()],
            timeframe=args.timeframe,
            limit=args.limit,
            min_rows=args.min_rows,
            max_per_asset_class=args.max_per_asset_class,
            max_per_venue=args.max_per_venue,
        )
    elif args.command == "allocate":
        cost_models = _read_json(args.cost_models)
        report = allocate_research_budget(
            _read_json(args.scan_report),
            max_model_fit_calls=args.max_model_fit_calls,
            fits_per_candidate=args.fits_per_candidate,
            max_candidates=args.max_candidates,
            max_age_seconds=args.max_age_seconds,
            estimated_run_cost_usd=args.estimated_run_cost_usd,
            daily_spend_to_date_usd=args.daily_spend_to_date_usd,
            monthly_spend_to_date_usd=args.monthly_spend_to_date_usd,
            cost_models=cost_models,
        )
    else:
        report = run_approved_multi_market_research(
            _read_json(args.scan_report),
            _read_json(args.allocation_plan),
            _read_json(args.approval),
        )
    _write_json(args.output, report)
    print(json.dumps({
        "command": args.command,
        "output": str(args.output),
        "execution_enabled": report.get("execution_enabled", False),
        "training_performed": report.get("training_performed", False),
        "summary": report.get("summary", {
            "market_count": report.get("market_count"),
            "planned_model_fit_calls": report.get("planned_model_fit_calls"),
        }),
    }, indent=2))


if __name__ == "__main__":
    main()
