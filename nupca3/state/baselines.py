"""nupca3/state/baselines.py

Baseline estimators and the stability predicate.

Axiom coverage
-------------
- A3.1: running baseline mean and fast/slow variance estimators
- A3.2: baseline-normalized signals \tilde{x}_k(t)
- A3.3: stable(t) predicate gating structural edits

This implementation is intended to be a faithful instantiation of the v1.5b
axioms. It is still necessarily *partial* because A3.3 references probe drift
variance (A11) and “feature variance”, which are model- and probe-specific.
We implement the exact control structure and accept inputs for probe metrics
when available.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ..config import AgentConfig
from ..types import Baselines, Margins


def init_baselines(cfg: AgentConfig) -> Baselines:
    """Initialize baselines per A14.8 / A3.1.

    v(t) has 5 channels (m_E,m_D,m_L,m_C,m_S). We initialize means and variances
    to 0.
    """
    mu = np.zeros(5, dtype=float)
    var_fast = np.zeros(5, dtype=float)
    var_slow = np.zeros(5, dtype=float)
    return Baselines(mu=mu, var_fast=var_fast, var_slow=var_slow, tilde_prev=np.zeros(5, dtype=float))


def update_baselines(
    *,
    baselines: Baselines,
    margins: Margins,
    cfg: AgentConfig,
) -> Baselines:
    """Update baseline mean and variances (A3.1).

    A3.1:
      μ_k(t) = (1-β) μ_k(t-1) + β x_k(t)
      σ_fast,k^2(t) = (1-β) σ_fast,k^2(t-1) + β (x_k(t)-μ_k(t))^2
      σ_slow,k^2(t) = (1-β_slow) σ_slow,k^2(t-1) + β_slow (x_k(t)-μ_k(t))^2

    Inputs:
      - baselines: previous Baselines
      - margins: current margin vector v(t)
      - cfg: β, β_slow

    Output:
      - new Baselines
    """
    beta = float(getattr(cfg, "beta", 0.01))
    beta_slow = float(getattr(cfg, "beta_slow", 0.001))

    x = np.array([margins.m_E, margins.m_D, margins.m_L, margins.m_C, margins.m_S], dtype=float)
    mu_prev = np.array(baselines.mu, dtype=float)

    mu = (1.0 - beta) * mu_prev + beta * x

    # Deviations are measured against the updated mean (A3.1 uses μ_k(t) in the variance update).
    dev = x - mu
    var_fast = (1.0 - beta) * np.array(baselines.var_fast, dtype=float) + beta * (dev * dev)
    var_slow = (1.0 - beta_slow) * np.array(baselines.var_slow, dtype=float) + beta_slow * (dev * dev)

    return Baselines(
        mu=mu,
        var_fast=var_fast,
        var_slow=var_slow,
        tilde_prev=(baselines.tilde_prev if baselines.tilde_prev is not None else np.zeros(5, dtype=float)),
        last_struct_edit_t=int(getattr(baselines, "last_struct_edit_t", -10**9)),
    )


def normalize_margins(
    *,
    margins: Margins,
    baselines: Baselines,
    cfg: AgentConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute baseline-normalized margin vector \tilde{v}(t) and its delta.

    A3.2:
      \tilde{x}_k(t) = (x_k(t) - μ_k(t)) / (sqrt(σ_fast,k^2(t)) + ε)

    A0.2 additionally uses \Delta\tilde{v}(t) = \tilde{v}(t) - \tilde{v}(t-1).

    Returns:
      (tilde, delta_tilde)
    """
    eps = float(getattr(cfg, "eps_baseline", getattr(cfg, "epsilon", 1e-6)))
    x = np.array([margins.m_E, margins.m_D, margins.m_L, margins.m_C, margins.m_S], dtype=float)
    mu = np.array(baselines.mu, dtype=float)
    var_fast = np.array(baselines.var_fast, dtype=float)
    denom = np.sqrt(np.maximum(var_fast, 0.0)) + eps
    tilde = (x - mu) / denom

    tilde_prev = baselines.tilde_prev
    if tilde_prev is None:
        tilde_prev = np.zeros_like(tilde)
    delta_tilde = tilde - np.array(tilde_prev, dtype=float)
    return tilde, delta_tilde


def stable(
    *,
    t: int,
    baselines: Baselines,
    cfg: AgentConfig,
    probe_var: Optional[float] = None,
    feature_var: Optional[float] = None,
) -> bool:
    """Stability predicate stable(t) (A3.3).

    stable(t) := 1{ edits_struct(t-W:t) == 0 } · 1{Var(probes) ≤ ν_max} · 1{Var(features) ≤ ξ_max}

    Inputs:
      - t: current step index
      - baselines: must expose last_struct_edit_t (most recent edit)
      - cfg: W, ν_max, ξ_max
      - probe_var: caller-provided probe variance over the last W steps (if available)
      - feature_var: caller-provided feature variance over the last W steps (if available)

    Output:
      - bool stable(t)
    """
    W = int(getattr(cfg, "W", 200))
    nu_max = float(getattr(cfg, "nu_max", 0.02))
    xi_max = float(getattr(cfg, "xi_max", 0.1))

    last_edit = int(getattr(baselines, "last_struct_edit_t", -10**9))
    no_recent_edits = (int(t) - last_edit) >= W

    if probe_var is None:
        # If probe variance is not supplied, treat it as unknown and do NOT
        # grant stability. This is the conservative axiom-faithful choice.
        return False
    if feature_var is None:
        # Same: A3.3 explicitly includes feature variance.
        return False

    return bool(no_recent_edits and (float(probe_var) <= nu_max) and (float(feature_var) <= xi_max))


def commit_struct_edit(baselines: Baselines, *, t: int) -> Baselines:
    """Record that a structural edit occurred at time t (A3.3)."""
    return Baselines(
        mu=np.array(baselines.mu, dtype=float),
        var_fast=np.array(baselines.var_fast, dtype=float),
        var_slow=np.array(baselines.var_slow, dtype=float),
        tilde_prev=(baselines.tilde_prev if baselines.tilde_prev is not None else np.zeros(5, dtype=float)),
        last_struct_edit_t=int(t),
    )


def commit_tilde_prev(baselines: Baselines, *, tilde: np.ndarray) -> Baselines:
    """Persist \tilde{v}(t) for A0.2 delta computation."""
    return Baselines(
        mu=np.array(baselines.mu, dtype=float),
        var_fast=np.array(baselines.var_fast, dtype=float),
        var_slow=np.array(baselines.var_slow, dtype=float),
        tilde_prev=np.array(tilde, dtype=float),
        last_struct_edit_t=int(getattr(baselines, "last_struct_edit_t", -10**9)),
    )
