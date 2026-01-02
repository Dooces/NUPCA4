"""Planning intake controller for contemplation gating."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

from ..config import AgentConfig
from ..types import Stress


@dataclass(frozen=True)
class PlanningDecision:
    """Result of the intake + compute decision."""

    g_contemplate: bool
    focus_mode: str
    O_t_target: Tuple[int, ...] | None
    planning_budget: float


def decide_intake_and_compute(
    *,
    rest: bool,
    stress: Stress,
    budget_slack: float,
    cfg: AgentConfig,
    P_haz: float = 0.0,
    P_nov: float = 0.0,
    problem_bound: Any | None = None,
    value_of_compute: float = 0.0,
) -> PlanningDecision:
    """Decide contemplation gating, intake override, and planning budget."""

    if rest:
        return PlanningDecision(
            g_contemplate=False,
            focus_mode="operate",
            O_t_target=None,
            planning_budget=0.0,
        )

    hazard = max(
        float(getattr(stress, "s_ext_th", 0.0)),
        float(getattr(stress, "s_int_need", 0.0)),
        float(P_haz),
    )
    slack = max(0.0, float(budget_slack))
    slack_frac = slack / max(float(getattr(cfg, "B_rt", 1.0)), 1.0)
    g_safe = hazard >= float(cfg.contemplate_hazard_threshold)
    g_novelty = float(P_nov) >= float(cfg.contemplate_novelty_threshold) and slack_frac >= float(
        cfg.contemplate_novelty_slack_frac
    )
    g_perm = cfg.contemplate_force or (g_safe or g_novelty)
    g_contemplate = g_perm and slack_frac >= float(cfg.contemplate_budget_slack_frac)

    if not g_contemplate:
        focus_mode = "operate"
        target: Tuple[int, ...] | None = None
    elif problem_bound is not None and isinstance(problem_bound, dict):
        focus_mode = "bound"
        blocks = problem_bound.get("blocks") or []
        target = tuple(sorted(int(b) for b in blocks if isinstance(b, (int,))))
        if not target:
            target = tuple(sorted(int(b) for b in getattr(cfg, "contemplate_anchor_blocks", (0,)) if isinstance(b, (int))))
    else:
        focus_mode = "free"
        anchor_blocks = getattr(cfg, "contemplate_anchor_blocks", (0,))
        target = tuple(sorted(int(b) for b in anchor_blocks if isinstance(b, (int))))

    planning_budget = max(0.0, slack * float(cfg.contemplate_budget_reuse_frac))
    plan_scale = float(getattr(cfg, "value_of_compute_planning_scale", 0.5))
    bonus = min(max(0.0, float(value_of_compute)), 1.0)
    planning_budget *= 1.0 + plan_scale * bonus

    return PlanningDecision(
        g_contemplate=bool(g_contemplate),
        focus_mode=str(focus_mode),
        O_t_target=target,
        planning_budget=planning_budget,
    )
