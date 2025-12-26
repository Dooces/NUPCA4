"""nupca3/dynamics/margin_dynamics.py

Hard dynamics for (E, D, drift_P) and derived margins.

Axiom coverage
-------------
- A15.1: operating costs (energy drain, danger accumulation, probe drift)
- A15.2: REST recovery dynamics
- A15.3: noise injection terms are modeled externally (not imposed here)

This module intentionally avoids “margin-space shortcut” dynamics. It updates
the underlying observables directly, and the caller derives margins via A2.*.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import AgentConfig


@dataclass(frozen=True)
class HardState:
    """Underlying margin observables at time t."""

    E: float
    D: float
    drift_P: float


def step_hard_dynamics(
    *,
    prev: HardState,
    rest: bool,
    commit: bool,
    L_eff: float,
    cfg: AgentConfig,
) -> HardState:
    """Apply one discrete-time update of A15.1–A15.2.

    v1.5b uses continuous-time derivatives; here we use a unit-step forward
    Euler discretization consistent with the rest of the discrete agent.

    Inputs:
      - prev: HardState at time t-1
      - rest: rest(t) ∈ {0,1}
      - commit: commit(t) ∈ {0,1}
      - L_eff: L_eff(t) = Σ_{j∈A_t} a_j(t)L_j
      - cfg: dynamic parameters c_*^*, k_*^op, c_*^rest, L_max_work

    Outputs:
      - HardState at time t
    """
    L_max_work = float(getattr(cfg, "L_max_work", 1.0))
    L_ratio = float(L_eff) / max(L_max_work, 1e-12)
    u = L_ratio * (1.0 if bool(commit) else 0.0)

    # A15 parameters
    c_E_star = float(getattr(cfg, "c_E_star", 0.0))
    k_E_op = float(getattr(cfg, "k_E_op", 0.0))
    c_E_rest = float(getattr(cfg, "c_E_rest", 0.0))

    c_D_star = float(getattr(cfg, "c_D_star", 0.0))
    k_D_op = float(getattr(cfg, "k_D_op", 0.0))
    c_D_rest = float(getattr(cfg, "c_D_rest", 0.0))

    c_S_star = float(getattr(cfg, "c_S_star", 0.0))
    k_S_op = float(getattr(cfg, "k_S_op", 0.0))
    c_S_rest = float(getattr(cfg, "c_S_rest", 0.0))

    r = 1.0 if bool(rest) else 0.0

    # A15.1–A15.2 updates.
    E = float(prev.E) + (c_E_star - k_E_op * u - r * c_E_rest)
    D = float(prev.D) + (c_D_star + k_D_op * u - r * c_D_rest)
    drift_P = float(prev.drift_P) + (c_S_star + k_S_op * u - r * c_S_rest)

    # Clamp to configured bounds (A15.3 implicitly assumes bounded ranges).
    E = float(np.clip(E, float(getattr(cfg, "E_min", -np.inf)), float(getattr(cfg, "E_max", np.inf))))
    D = float(np.clip(D, float(getattr(cfg, "D_min", -np.inf)), float(getattr(cfg, "D_max", np.inf))))

    return HardState(E=E, D=D, drift_P=float(drift_P))
