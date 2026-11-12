"""Central ETI configuration. Safe defaults are research-only and non-leveraged."""
from dataclasses import dataclass


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


@dataclass(frozen=True)
class AppConfig:
    app_name: str = "ETI Research Core"
    version: str = "0.2.0"
    research_only: bool = True
