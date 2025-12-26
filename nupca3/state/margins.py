"""nupca3/state/margins.py

Margins, stress signals, and arousal dynamics.

Axiom coverage
-------------
- A0.1: margin vector v(t) = (m_E, m_D, m_L, m_C, m_S)
- A0.2: instantaneous arousal s_inst^ar(t)
- A0.3: stress channels and internal need aggregation
- A0.4: leaky arousal dynamics
- A2.1–A2.4: margin definitions (headroom, opportunity, compute slack)

This module does *not* implement the hard dynamics of E(t), D(t), drift_P(t)
itself; that is A15 and lives in `nupca3/dynamics/margin_dynamics.py`.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ..config import AgentConfig
from ..types import Baselines, Margins, Stress


def _sigmoid(z: float) -> float:
    return float(1.0 / (1.0 + np.exp(-float(z))))


def init_stress(cfg: AgentConfig) -> Stress:
    """Initialize stress channels.

    The axioms define stress as a function of margins and baseline statistics.
    At t=0, callers often have not yet computed meaningful baselines. For
    determinism and to keep the pipeline well-typed, we initialize all channels
    to 0 and rely on the first step to compute stress from margins/baselines.

    Args:
      cfg: AgentConfig (unused; kept for future extensions).

    Returns:
      Stress(...) with all components set to 0.
    """
    _ = cfg
    return Stress(s_E=0.0, s_D=0.0, s_L=0.0, s_C=0.0, s_S=0.0, s_int_need=0.0, s_ext_th=0.0)


def init_margins(*, E: float, D: float, drift_P: float, cfg: AgentConfig) -> Margins:
    """Initialize the margin vector from underlying observables (A2.1).

    Inputs:
      - E: energy-like scalar E(0)
      - D: danger-like scalar D(0)
      - drift_P: probe drift proxy drift_P(0)
      - cfg: AgentConfig (E_min, D_max, sigma_E, sigma_D, S_max_stab)

    Outputs:
      - Margins with (m_E, m_D, m_L, m_C, m_S). m_L and m_C are 0 at init.
    """
    E_min = float(getattr(cfg, "E_min", 0.0))
    D_max = float(getattr(cfg, "D_max", 1.0))
    sigma_E = float(getattr(cfg, "sigma_E", 1.0))
    sigma_D = float(getattr(cfg, "sigma_D", 1.0))
    S_max = float(getattr(cfg, "S_max_stab", 0.0))

    rawE = float(E) - E_min
    rawD = D_max - float(D)
    rawS = S_max - float(drift_P)

    return Margins(
        m_E=rawE / max(sigma_E, 1e-12),
        m_D=rawD / max(sigma_D, 1e-12),
        m_L=0.0,
        m_C=0.0,
        m_S=rawS,
    )

def compute_margins(
    *,
    E: float,
    D: float,
    drift_P: float,
    opp: float,
    x_C: float,
    cfg: AgentConfig,
) -> tuple[Margins, float, float, float]:
    """Compute margins v(t) from observables (A2.1–A2.4).

    This is the canonical A2 implementation used by the step pipeline.

    Definitions (v1.5b):
      rawE(t) = E(t) - E_min
      m_E(t)  = rawE(t) / σ_E

      rawD(t) = D_max - D(t)
      m_D(t)  = rawD(t) / σ_D

      rawS(t) = S_max^{stab} - drift_P(t)
      x_S(t)  = rawS(t)

      x_L(t)  = opp(t)
      x_C(t)  = compute slack from A2.4 (passed in by caller)

    We embed x_L and x_C directly as the L and C components of v(t), because
    A0.1 defines v(t) = (m_E, m_D, m_L, m_C, m_S) and A2 specifies x_L, x_C.

    Args:
      E, D, drift_P: underlying observables.
      opp: world-supplied learning opportunity proxy (A2.3).
      x_C: compute slack value already computed from A2.4.
      cfg: configuration parameters.

    Returns:
      (margins, rawE, rawD, rawS) where margins is v(t) and the raw headrooms
      are returned for A10.2 lag plumbing.
    """
    E_min = float(getattr(cfg, "E_min", 0.0))
    D_max = float(getattr(cfg, "D_max", 1.0))
    sigma_E = float(getattr(cfg, "sigma_E", 1.0))
    sigma_D = float(getattr(cfg, "sigma_D", 1.0))
    S_max = float(getattr(cfg, "S_max_stab", 0.0))

    rawE = float(E) - E_min
    rawD = D_max - float(D)
    rawS = S_max - float(drift_P)

    margins = Margins(
        m_E=rawE / max(sigma_E, 1e-12),
        m_D=rawD / max(sigma_D, 1e-12),
        m_L=float(opp),
        m_C=float(x_C),
        m_S=float(rawS),
    )

    return margins, float(rawE), float(rawD), float(rawS)



def compute_stress(
    *,
    E: float,
    D: float,
    drift_P: float,
    s_ext_th: float,
    cfg: AgentConfig,
) -> Stress:
    """Compute stress channels (A0.3).

    v1.5b (A0.3):
      s_E^need(t) = σ(-(rawE(t)-τ_E)/κ_E)
      s_D^need(t) = σ(-(rawD(t)-τ_D)/κ_D)
      s_S^need(t) = σ(-(rawS(t)-τ_S)/κ_S)
      s_int^need(t) = max(s_E^need, s_D^need, s_S^need)

    Inputs:
      - E, D, drift_P: underlying observables
      - s_ext_th: world-supplied external threat s_ext^th(t)
      - cfg: AgentConfig with thresholds τ_* and slopes κ_*

    Outputs:
      - Stress dataclass.
    """
    E_min = float(getattr(cfg, "E_min", 0.0))
    D_max = float(getattr(cfg, "D_max", 1.0))
    S_max = float(getattr(cfg, "S_max_stab", 0.0))

    rawE = float(E) - E_min
    rawD = D_max - float(D)
    rawS = S_max - float(drift_P)

    tau_E = float(getattr(cfg, "tau_E_need", getattr(cfg, "tau_E", 0.0)))
    tau_D = float(getattr(cfg, "tau_D_need", getattr(cfg, "tau_D", 0.0)))
    tau_S = float(getattr(cfg, "tau_S_need", getattr(cfg, "tau_S", 0.0)))

    kappa_E = float(getattr(cfg, "kappa_E_need", getattr(cfg, "kappa_E", 1.0)))
    kappa_D = float(getattr(cfg, "kappa_D_need", getattr(cfg, "kappa_D", 1.0)))
    kappa_S = float(getattr(cfg, "kappa_S_need", getattr(cfg, "kappa_S", 1.0)))

    s_E = _sigmoid(-(rawE - tau_E) / max(kappa_E, 1e-12))
    s_D = _sigmoid(-(rawD - tau_D) / max(kappa_D, 1e-12))
    s_S = _sigmoid(-(rawS - tau_S) / max(kappa_S, 1e-12))

    s_int_need = float(max(s_E, s_D, s_S))

    # v1.5b does not define s_L or s_C as stress channels. They exist in the
    # repo type for convenience but are not used in A5/A10/A14.
    return Stress(
        s_E=float(s_E),
        s_D=float(s_D),
        s_L=0.0,
        s_C=0.0,
        s_S=float(s_S),
        s_int_need=float(s_int_need),
        s_ext_th=float(s_ext_th),
    )


def compute_arousal(
    *,
    arousal_prev: float,
    margins: Margins,
    baselines: Baselines,
    pred_error: float,
    cfg: AgentConfig,
) -> Tuple[float, float]:
    """Compute s_inst^ar(t) and s^ar(t) (A0.2–A0.4).

    A0.2 uses baseline-normalized margins \tilde{m}_k(t) and their step deltas.
    The normalization itself is handled in `nupca3/state/baselines.py`.

    Inputs:
      - arousal_prev: s^ar(t-1)
      - margins: v(t)
      - baselines: baseline state (used for normalization)
      - pred_error: E(t) in A0.2 (prediction error magnitude proxy)
      - cfg: weights and arousal dynamics params

    Outputs:
      - (s_inst_ar, s_ar)
    """
    from .baselines import normalize_margins  # local import to avoid cycles

    tilde, delta_tilde = normalize_margins(margins=margins, baselines=baselines, cfg=cfg)

    # Unpack normalized margins in A0.1 order.
    mE, mD, mL, mC, mS = [float(x) for x in tilde]
    dE, dD, dL, dC, dS = [float(x) for x in delta_tilde]

    w_L = float(getattr(cfg, "w_L", getattr(cfg, "w_L_ar", 1.0)))
    w_C = float(getattr(cfg, "w_C", getattr(cfg, "w_C_ar", 1.0)))
    w_S = float(getattr(cfg, "w_S", getattr(cfg, "w_S_ar", 1.0)))
    w_delta = float(getattr(cfg, "w_delta", getattr(cfg, "w_delta_ar", 1.0)))
    w_E = float(getattr(cfg, "w_E", getattr(cfg, "w_E_ar", 0.0)))

    theta_ar = float(getattr(cfg, "theta_ar", 0.5))
    kappa_ar = float(getattr(cfg, "kappa_ar", 0.1))

    A = (
        w_L * abs(mL)
        + w_C * abs(mC)
        + w_S * abs(mS)
        + w_delta * (abs(dE) + abs(dD) + abs(dL) + abs(dC) + abs(dS))
        + w_E * abs(float(pred_error))
    )

    s_inst_ar = _sigmoid((A - theta_ar) / max(kappa_ar, 1e-12))

    # A0.4 leaky integrator with asymmetric rise/decay.
    tau_rise = float(getattr(cfg, "tau_rise", 1.0))
    tau_decay = float(getattr(cfg, "tau_decay", 1.0))
    tau = tau_rise if s_inst_ar >= float(arousal_prev) else tau_decay
    tau = max(tau, 1e-12)

    s_ar = float(arousal_prev) + (float(s_inst_ar) - float(arousal_prev)) / tau
    return float(s_inst_ar), float(s_ar)
