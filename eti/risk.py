"""Independent deterministic risk gate. It cannot be overridden by model output."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from .config import RiskConfig


@dataclass(frozen=True)
class RiskRequest:
    equity: float
    entry_price: float
    stop_price: float
    requested_notional: float
    gross_exposure: float = 0.0
    daily_pnl_fraction: float = 0.0
    current_drawdown: float = 0.0
    leverage: float = 1.0
    side: str = "long"


@dataclass(frozen=True)
class RiskDecision:
    approved: bool
    approved_notional: float
    quantity: float
    reasons: tuple[str, ...]


class RiskEngine:
    def __init__(self, config: RiskConfig | None = None):
        self.config = config or RiskConfig()

    def assess(self, request: RiskRequest) -> RiskDecision:
        c, reasons = self.config, []
        numeric = (
            request.equity, request.entry_price, request.stop_price,
            request.requested_notional, request.gross_exposure,
            request.daily_pnl_fraction, request.current_drawdown, request.leverage,
        )
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
               for v in numeric):
            reasons.append("non_finite_or_invalid_numeric_input")
        if any(not math.isfinite(v) for v in numeric if isinstance(v, (int, float)) and not isinstance(v, bool)):
            return RiskDecision(False, 0.0, 0.0, tuple(reasons))
        if request.equity <= 0 or request.entry_price <= 0 or request.requested_notional <= 0:
            reasons.append("invalid_equity_price_or_notional")
        if request.gross_exposure < 0:
            reasons.append("negative_gross_exposure")
        if request.leverage < 1:
            reasons.append("invalid_leverage")
        if request.side not in ("long", "short"):
            reasons.append("invalid_side")
        if request.side == "short" and not c.allow_short:
            reasons.append("shorts_disabled")
        if request.leverage > c.max_leverage:
            reasons.append("leverage_limit")
        if request.daily_pnl_fraction <= -c.max_daily_loss:
            reasons.append("daily_loss_limit")
        if request.current_drawdown >= c.max_drawdown:
            reasons.append("drawdown_limit")
        stop_distance = abs(request.entry_price - request.stop_price) / request.entry_price if request.entry_price > 0 else 1
        if stop_distance <= 0 or stop_distance >= 1:
            reasons.append("invalid_stop_distance")
        elif request.side == "long" and request.stop_price >= request.entry_price:
            reasons.append("long_stop_must_be_below_entry")
        elif request.side == "short" and request.stop_price <= request.entry_price:
            reasons.append("short_stop_must_be_above_entry")
        if reasons:
            return RiskDecision(False, 0.0, 0.0, tuple(reasons))
        risk_budget = request.equity * c.max_risk_per_trade
        risk_limited = risk_budget / stop_distance
        exposure_room = max(0.0, request.equity * c.max_gross_exposure - request.gross_exposure)
        position_cap = request.equity * c.max_position_fraction
        approved = min(request.requested_notional, risk_limited, exposure_room, position_cap)
        if approved <= 0:
            reasons.append("no_exposure_capacity")
        return RiskDecision(approved > 0, approved, approved / request.entry_price if approved > 0 else 0.0, tuple(reasons))

    def snapshot(self) -> dict:
        return asdict(self.config)
