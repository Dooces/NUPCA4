"""nupca3/control/edit_control.py

Freeze predicate and permission to perform parameter updates.

Axiom coverage: A10.
"""

from __future__ import annotations

from ..config import AgentConfig
from ..types import Stress


def freeze_predicate(stress_lagged: Stress, cfg: AgentConfig) -> bool:
    """Compute freeze(t) (A10.1).

    v1.5b (A10.1):
      freeze(t) = 1{ s_ext^th(t-1) >= χ^{th} }

    Inputs:
      - stress_lagged: Stress evaluated at t-1 (must include s_ext_th)
      - cfg: AgentConfig (chi_th)

    Outputs:
      - bool freeze(t)
    """
    chi_th = float(getattr(cfg, "chi_th", 0.90))
    return bool(float(getattr(stress_lagged, "s_ext_th", 0.0)) >= chi_th)


def permit_param_updates(
    *,
    rest_t: bool,
    freeze_t: bool,
    x_C_lagged: float,
    arousal_lagged: float,
    rawE_lagged: float,
    rawD_lagged: float,
    cfg: AgentConfig,
) -> bool:
    """Compute permit_param(t) (A10.2).

    v1.5b (A10.2):
      permit_param(t) = 1{rest(t)=0}·1{freeze(t)=0}·1{x_C(t-1)>τ_C^{edit}}
                        ·1{s^ar(t-1)<θ_ar^{panic}}·1{rawE(t-1)>τ_E^{edit}}·1{rawD(t-1)>τ_D^{edit}}

    Inputs:
      - rest_t: macrostate flag for time t (OPERATING vs REST)
      - freeze_t: freeze(t) computed from t-1 threat
      - x_C_lagged: compute slack x_C(t-1)
      - arousal_lagged: s^ar(t-1)
      - rawE_lagged: rawE(t-1)=E(t-1)-E_min
      - rawD_lagged: rawD(t-1)=D_max-D(t-1)
      - cfg: AgentConfig (tau_C_edit, tau_E_edit, tau_D_edit, theta_ar_panic)

    Outputs:
      - bool permit_param(t)
    """
    if not bool(getattr(cfg, "enable_learning", True)):
        return False
    if bool(rest_t):
        return False
    if bool(freeze_t):
        return False

    tau_C = float(getattr(cfg, "tau_C_edit", 0.0))
    tau_E = float(getattr(cfg, "tau_E_edit", 0.0))
    tau_D = float(getattr(cfg, "tau_D_edit", 0.0))
    theta_panic = float(getattr(cfg, "theta_ar_panic", 0.95))

    if float(x_C_lagged) <= tau_C:
        return False
    if float(arousal_lagged) >= theta_panic:
        return False
    if float(rawE_lagged) <= tau_E:
        return False
    if float(rawD_lagged) <= tau_D:
        return False
    return True
