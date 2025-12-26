"""nupca3/control/commitment.py

Commitment gate and action selection.

Axiom coverage: A8.
"""

from __future__ import annotations

from ..config import AgentConfig
from ..types import RolloutResult, Action


def commit_gate(rest: bool, h: int, c: list[float], cfg: AgentConfig) -> bool:
    """Compute commit(t).

    Skeleton: commit if not in REST, horizon >= d, and confidence at d exceeds threshold.
    Replace with your exact A8.2 definition.
    """
    if rest:
        return False
    if h < cfg.d_latency_floor:
        return False
    if len(c) < cfg.d_latency_floor:
        return False
    return bool(c[cfg.d_latency_floor - 1] >= cfg.theta_act)


def select_action(commit: bool, rollout: RolloutResult, cfg: AgentConfig) -> Action:
    """Select action.

    Skeleton: returns 1 if commit else 0. Replace with your safe policy and plan execution.
    Axiom coverage: A8.3.
    """
    return 1 if commit else 0
