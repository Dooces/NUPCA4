"""Budget governor helpers for v5 timing invariants."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from ..config import AgentConfig
from ..types import AgentState, EnvObs



@dataclass
class BudgetMeter:
    """Track compute budget usage and deterministic degradation."""

    B_rt: float
    B_hat_max: float
    B_max: float
    B_plan: float
    B_use: float = 0.0
    degrade_level: int = 0
    degrade_history: list[str] = field(default_factory=list)
    degradation_ladder: Sequence[str] = field(
        default_factory=lambda: (
            "reduce_planning_micro_steps",
            "shrink_support_update",
            "trim_stage2_queries",
        )
    )

    def limit(self) -> float:
        """Return the hard limit min(B_rt, B_max)."""
        return min(self.B_rt, self.B_max)

    def spend(self, cost: float, tag: str) -> bool:
        """Charge `cost` units of compute; degrade if we would exceed the limit."""
        cost = float(cost)
        if cost <= 0.0:
            return True
        limit = self.limit()
        if self.B_use + cost > limit:
            self.degrade_once(f"budget_overflow:{tag}")
            limit = self.limit()
            if self.B_use + cost > limit:
                return False
        self.B_use += cost
        return True

    def degrade_once(self, reason: str) -> None:
        """Advance the deterministic degradation ladder (once per level)."""
        if self.degrade_level >= len(self.degradation_ladder):
            self.degrade_history.append(reason)
            return
        reason_entry = f"{self.degradation_ladder[self.degrade_level]}:{reason}"
        self.degrade_history.append(reason_entry)
        self.degrade_level += 1


def create_budget_meter(state: AgentState, env_obs: EnvObs, cfg: AgentConfig) -> BudgetMeter:
    """Build a BudgetMeter using the latest wallclock and configuration."""
    B_rt = float(getattr(cfg, "B_rt", 0.0))
    units_per_ms = float(getattr(cfg, "compute_units_per_ms", 1.0))
    if units_per_ms <= 0.0:
        units_per_ms = 1.0

    prev_wall = getattr(state, "wall_ms", None)
    current_wall = getattr(env_obs, "wall_ms", None)
    delta_ms = 0.0
    if prev_wall is not None and current_wall is not None:
        try:
            delta_ms = max(0.0, float(current_wall) - float(prev_wall))
        except Exception:
            delta_ms = 0.0

    measured_units = delta_ms * units_per_ms
    tick_budget_ms = float(getattr(cfg, "tick_budget_ms", 0.0))
    if tick_budget_ms > 0.0:
        cap_units = tick_budget_ms * units_per_ms
        if measured_units <= 0.0:
            measured_units = cap_units
        else:
            measured_units = min(measured_units, cap_units)

    if measured_units <= 0.0:
        measured_units = B_rt

    measured_units = max(measured_units, 0.0)
    return BudgetMeter(
        B_rt=B_rt,
        B_hat_max=measured_units,
        B_max=measured_units,
        B_plan=B_rt,
    )
