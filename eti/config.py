"""Central ETI configuration. Safe defaults are research-only and non-leveraged."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class RiskConfig:
    max_risk_per_trade: float = 0.005
    max_position_fraction: float = 0.10
    max_gross_exposure: float = 0.25
    max_leverage: float = 1.0
    max_daily_loss: float = 0.02
    max_drawdown: float = 0.08
    allow_short: bool = False
    allow_live_orders: bool = False

    def __post_init__(self):
        fractions = (self.max_risk_per_trade, self.max_position_fraction,
                     self.max_gross_exposure, self.max_daily_loss, self.max_drawdown)
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or
               not math.isfinite(v) or not 0 < v <= 1 for v in fractions):
            raise ValueError("Risk fractions must be finite numbers in (0, 1]")
        if (isinstance(self.max_leverage, bool) or
            not isinstance(self.max_leverage, (int, float)) or
            not math.isfinite(self.max_leverage) or self.max_leverage < 1):
            raise ValueError("max_leverage must be finite and >= 1")
        if not isinstance(self.allow_short, bool) or not isinstance(self.allow_live_orders, bool):
            raise ValueError("Risk feature flags must be booleans")
        if self.allow_live_orders:
            raise ValueError("Live order execution is not implemented and must remain disabled")


@dataclass(frozen=True)
class AppConfig:
    app_name: str = "ETI Research Core"
    version: str = "0.2.0"
    research_only: bool = True
