"""nupca3/state/macrostate.py

Macrostate logic for OPERATING/REST transitions (A14).

This file implements A14 as written, including the lag discipline:

  rest(t) = rest_permitted(t-1) · demand(t-1) · (1 - interrupt(t-1))

and the exact v1.5b definitions for:
- Queue evolution Q_struct(t)
- Timers T_since(t), T_rest(t)
- Rest pressure P_rest(t) and effective pressure P_rest_eff(t)
- Demand toggles for entering and exiting REST

The pipeline is expected to:
1) compute demand(t) and the permission/interrupt predicates at the end of step t
2) use these (lagged) values at the start of step t+1 to decide rest(t+1).


[AXIOM_CLARIFICATION_ADDENDUM — Representation & Naming]

- Terminology: identifiers like "Expert" in this codebase refer to NUPCA3 **abstraction/resonance nodes** (a "constellation"), not conventional Mixture-of-Experts "experts" or router-based MoE.

- Representation boundary (clarified intent of v1.5b): the completion/fusion operator (A7) is defined over an **encoded, multi-resolution abstraction vector** \(x(t)\). Raw pixels may exist only in a transient observation buffer for the current step; **raw pixel values must never be inserted into long-term storage** (library/cold storage) and must not persist across REST boundaries.

- Decomposition intuition: each node is an operator that *factors out* a predictable/resonant component on its footprint, leaving residual structure for other nodes (or for REST-time proposal) to capture. This is the intended "FFT-like" interpretation of masks/constellations.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable, List, Tuple
import math

from ..config import AgentConfig
from ..types import EditProposal, MacrostateVars, Margins, Stress


# =============================================================================
# Initialization
# =============================================================================

def init_macrostate(cfg: AgentConfig) -> MacrostateVars:
    """Initialize macrostate variables.

    This establishes the A14 state at t=0:
    - Q_struct(0) = []
    - T_since(0) = 0, T_rest(0) = 0
    - P_rest(0) = 0
    - rest(0) = 0

    Args:
      cfg: AgentConfig (unused; kept for symmetry and future extension).

    Returns:
      A MacrostateVars instance.
    """
    _ = cfg
    return MacrostateVars(rest=False, Q_struct=[], T_since=0, T_rest=0, P_rest=0.0)


# Backwards-compatible alias used by some WIP call sites.
def init_macro(cfg: AgentConfig) -> MacrostateVars:
    return init_macrostate(cfg)


def rest_permitted(
    stress: Stress,
    coverage_debt: float,
    cfg: AgentConfig,
    *,
    arousal: float = 0.0,
) -> tuple[bool, str]:
    """A14.6 rest_permitted(t) gate updated to include unsafe signals."""
    theta_safe = float(getattr(cfg, "theta_safe_th", getattr(cfg, "P_rest_theta_safe_th", 0.2)))
    theta_interrupt = float(getattr(cfg, "theta_interrupt_th", getattr(cfg, "P_rest_theta_interrupt_th", 0.6)))
    theta_ar_rest = float(getattr(cfg, "theta_ar_rest", getattr(cfg, "P_rest_theta_E_rest", 0.25)))
    debt_max = float(getattr(cfg, "coverage_debt_max", 1e6))

    s_ext_th = float(getattr(stress, "s_ext_th", 0.0))

    if s_ext_th >= theta_interrupt:
        return False, "interrupt"
    if coverage_debt > debt_max:
        return False, "coverage_debt"
    if s_ext_th >= theta_safe:
        return False, "external_threat"
    if float(arousal) >= theta_ar_rest:
        return False, "arousal_high"

    return True, "safe"


def interrupt(stress: Stress, cfg: AgentConfig) -> bool:
    """A14.6 interrupt(t) = 1{s_ext^th(t) ≥ θ_interrupt^th}."""
    theta_int = float(getattr(cfg, "theta_interrupt_th", getattr(cfg, "P_rest_theta_interrupt_th", 0.6)))
    return bool(float(stress.s_ext_th) >= theta_int)


def restored(margins: Margins, cfg: AgentConfig) -> bool:
    """A14.5 restored(t) predicate used for exiting REST."""
    theta_E = float(getattr(cfg, "theta_E_rest", getattr(cfg, "P_rest_theta_E_rest", 0.25)))
    theta_D = float(getattr(cfg, "theta_D_rest", getattr(cfg, "P_rest_theta_D_rest", 0.25)))
    theta_S = float(getattr(cfg, "theta_S_rest", getattr(cfg, "P_rest_theta_S_rest", 0.25)))
    return bool(
        float(margins.m_E) >= theta_E
        and float(margins.m_D) >= theta_D
        and float(margins.m_S) >= theta_S
    )


def update_P_rest(
    *,
    P_rest_prev: float,
    rest_t: bool,
    s_int_need_t: float,
    cfg: AgentConfig,
) -> float:
    """Update P_rest(t) per A14.3."""
    gamma_rest = float(getattr(cfg, "gamma_rest", getattr(cfg, "P_rest_gamma_rest", 0.96)))
    delta_base = float(getattr(cfg, "delta_base", getattr(cfg, "P_rest_delta_base", 0.01)))
    delta_need = float(getattr(cfg, "delta_need", getattr(cfg, "P_rest_delta_need", 0.10)))

    if bool(rest_t):
        return gamma_rest * float(P_rest_prev)
    return float(P_rest_prev) + delta_base + delta_need * float(s_int_need_t)


def P_rest_eff(*, P_rest_t: float, stress: Stress, cfg: AgentConfig) -> float:
    """A14.3 P_rest_eff(t) = P_rest(t)+α_E s_E^need+α_D s_D^need+α_S s_S^need."""
    alpha_E = float(getattr(cfg, "alpha_E", getattr(cfg, "P_rest_alpha_E", 0.10)))
    alpha_D = float(getattr(cfg, "alpha_D", getattr(cfg, "P_rest_alpha_D", 0.10)))
    alpha_S = float(getattr(cfg, "alpha_S", getattr(cfg, "P_rest_alpha_S", 0.10)))
    return (
        float(P_rest_t)
        + alpha_E * float(stress.s_E)
        + alpha_D * float(stress.s_D)
        + alpha_S * float(stress.s_S)
    )


def demand(
    *,
    rest_t: bool,
    P_rest_eff_t: float,
    coverage_debt: float,
    Q_struct_len_t: int,
    T_since_t: int,
    T_rest_t: int,
    restored_t: bool,
    actionable: bool,
    cycles_needed: int,
    rest_cooldown: int,
    cfg: AgentConfig,
) -> bool:
    """Demand toggle per A14.4–A14.5 with action availability gating."""
    debt_thresh = float(getattr(cfg, "coverage_debt_thresh", 1e6))
    rest_min_cycles = max(1, int(getattr(cfg, "rest_min_cycles", 1)))
    if rest_cooldown > 0:
        return False
    wake_pressure = float(coverage_debt)

    if not bool(rest_t):
        if not actionable or cycles_needed < rest_min_cycles:
            return False
        theta_enter = float(getattr(cfg, "theta_demand_enter", getattr(cfg, "P_rest_theta_demand_enter", 0.50)))
        Theta_Q_on = int(getattr(cfg, "Theta_Q_on", getattr(cfg, "P_rest_Theta_Q_on", 8)))
        Tmax_wake = int(getattr(cfg, "Tmax_wake", getattr(cfg, "P_rest_Tmax_wake", 500)))
        return bool(
            float(P_rest_eff_t) >= theta_enter
            or int(Q_struct_len_t) >= Theta_Q_on
            or int(T_since_t) >= Tmax_wake
            and wake_pressure <= debt_thresh
        )

    theta_exit = float(getattr(cfg, "theta_demand_exit", getattr(cfg, "P_rest_theta_demand_exit", 0.30)))
    Theta_Q_off = int(getattr(cfg, "Theta_Q_off", getattr(cfg, "P_rest_Theta_Q_off", 3)))
    Tmax_rest = int(getattr(cfg, "Tmax_rest", getattr(cfg, "P_rest_Tmax_rest", 200)))

    if int(T_rest_t) >= Tmax_rest:
        return False
    if not actionable or cycles_needed <= 0:
        return False
    return bool(
        (not bool(restored_t))
        or int(Q_struct_len_t) > Theta_Q_off
        or float(P_rest_eff_t) > theta_exit
        or wake_pressure > debt_thresh
    )


def next_rest_state(
    *,
    rest_permitted_prev: bool,
    demand_prev: bool,
    interrupt_prev: bool,
) -> bool:
    """A14.7 rest(t) update from lagged predicates."""
    return bool(rest_permitted_prev and demand_prev and (not interrupt_prev))


def update_queue(
    *,
    Q_prev: List[EditProposal],
    rest_t: bool,
    proposals_t: Iterable[EditProposal],
    edits_processed_t: int,
) -> List[EditProposal]:
    """A14.2 queue update rule."""
    Q = list(Q_prev)
    if not bool(rest_t):
        Q.extend(list(proposals_t))
        return Q

    # REST: pop oldest edits_processed_t items
    n = max(0, int(edits_processed_t))
    if n <= 0:
        return Q
    return Q[n:]


def update_timers(*, rest_t: bool, T_since_prev: int, T_rest_prev: int) -> Tuple[int, int]:
    """A14.2 timers."""
    if not bool(rest_t):
        return int(T_since_prev) + 1, 0
    return 0, int(T_rest_prev) + 1


def evolve_macrostate(
    *,
    prev: MacrostateVars,
    rest_t: bool,
    proposals_t: Iterable[EditProposal],
    edits_processed_t: int,
    stress_t: Stress,
    margins_t: Margins,
    coverage_debt: float = 0.0,
    rest_actionable: bool = False,
    cfg: AgentConfig,
) -> Tuple[MacrostateVars, bool, bool, float]:
    """Evolve MacrostateVars for the end of the step.

    Returns:
      - macro_t: updated macro vars (with Q_struct(t), timers, P_rest(t), rest(t))
      - demand_t: demand(t) for use at start of t+1
      - interrupt_t: interrupt(t) for use at start of t+1
      - P_rest_eff_t: effective pressure used for demand(t)
    """
    Q_t = update_queue(Q_prev=prev.Q_struct, rest_t=rest_t, proposals_t=proposals_t, edits_processed_t=edits_processed_t)
    T_since_t, T_rest_t = update_timers(rest_t=rest_t, T_since_prev=prev.T_since, T_rest_prev=prev.T_rest)

    P_rest_t = update_P_rest(P_rest_prev=prev.P_rest, rest_t=rest_t, s_int_need_t=stress_t.s_int_need, cfg=cfg)
    P_eff_t = P_rest_eff(P_rest_t=P_rest_t, stress=stress_t, cfg=cfg)

    restored_t = restored(margins_t, cfg)
    max_edits = max(1, int(getattr(cfg, "max_edits_per_rest_step", 32)))
    queue_len = len(Q_t)
    cycles_needed = int(math.ceil(float(queue_len) / float(max_edits))) if queue_len > 0 else 0
    prev_cooldown = int(getattr(prev, "rest_cooldown", 0))
    cooldown = max(0, prev_cooldown - 1)
    streak_prev = int(getattr(prev, "rest_zero_processed_streak", 0))
    if rest_t and edits_processed_t == 0:
        streak = streak_prev + 1
    else:
        streak = 0
    rest_cooldown_steps = max(0, int(getattr(cfg, "rest_cooldown_steps", 0)))
    if rest_t and edits_processed_t == 0 and rest_cooldown_steps > 0 and streak >= rest_cooldown_steps:
        cooldown = rest_cooldown_steps
    demand_t = demand(
        rest_t=rest_t,
        P_rest_eff_t=P_eff_t,
        coverage_debt=coverage_debt,
        Q_struct_len_t=len(Q_t),
        T_since_t=T_since_t,
        T_rest_t=T_rest_t,
        restored_t=restored_t,
        actionable=rest_actionable,
        cycles_needed=cycles_needed,
        rest_cooldown=cooldown,
        cfg=cfg,
    )
    interrupt_t = interrupt(stress_t, cfg)

    macro_t = replace(
        prev,
        rest=bool(rest_t),
        Q_struct=Q_t,
        T_since=T_since_t,
        T_rest=T_rest_t,
        P_rest=float(P_rest_t),
        rest_cooldown=cooldown,
        rest_zero_processed_streak=streak,
    )
    return macro_t, bool(demand_t), bool(interrupt_t), float(P_eff_t)
