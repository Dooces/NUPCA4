"""nupca3/memory/rollout.py

Multi-step rollout and confidence estimates.

Axiom coverage: A7.4.

This module turns a 1-step fused prediction (x̂_{t+1}, Σ_{t+1}) into an
h-step rollout and provides per-step uncertainty H_k and confidence c_k.

v1.5b semantics (implemented at the level this repo snapshot supports)
---------------------------------------------------------------------
- Uncertainty propagation: Σ_{k+1} = Σ_k + η_proc · diag(diag(Σ_1))
  (process noise proportional to the 1-step diagonal covariance).
- Uncertainty summary: H_k = mean(diag(Σ_k))
- Confidence (Option A): c_k = exp(-alpha * H_cov) * rho^beta

#ITOOKASHORTCUT
The full v1.5b spec allows richer (possibly block-structured) covariance and
may define H_k in terms of precision-weighted uncertainty. This implementation
is diagonal-only because the rest of the repo uses diagonal precision fusion.
"""

from __future__ import annotations

import numpy as np

from ..config import AgentConfig
from ..types import RolloutResult


def _sigmoid(z: float) -> float:
    """Numerically stable logistic."""
    if z >= 0:
        return float(1.0 / (1.0 + np.exp(-z)))
    ez = np.exp(z)
    return float(ez / (1.0 + ez))


def rollout_and_confidence(
    x0: np.ndarray,
    x_hat_1: np.ndarray,
    Sigma_1: np.ndarray,
    h: int,
    cfg: AgentConfig,
) -> RolloutResult:
    """Compute rollouts up to horizon h (A7.4).

    Inputs:
      - x0: current latent state estimate (not used directly in diagonal-only rollouts)
      - x_hat_1: fused 1-step prediction x̂_{t+1}
      - Sigma_1: fused 1-step covariance Σ_{t+1}
      - h: horizon length h(t)
      - cfg: AgentConfig with rollout_eta_proc, rollout_mu_H, rollout_sigma_H

    Outputs:
      - RolloutResult with lists x_hats[1..h], Sigma_hats[1..h], H[1..h], c[1..h]
    """
    h = int(max(0, h))
    if h <= 0:
        return RolloutResult()

    D = int(getattr(cfg, "D", int(x_hat_1.shape[0])))

    # Process noise uses the 1-step diagonal covariance as a scale.
    # Ensure Sigma_1 is at least diagonal and finite.
    diag1 = np.diag(Sigma_1).copy() if Sigma_1.ndim == 2 else np.asarray(Sigma_1, dtype=float).copy()
    diag1 = np.where(np.isfinite(diag1), diag1, np.inf)

    eta_proc = float(getattr(cfg, "rollout_eta_proc", 0.01))
    alpha = float(getattr(cfg, "rollout_c_alpha", 1.0))
    beta = float(getattr(cfg, "rollout_c_beta", 1.0))
    if alpha <= 0.0:
        alpha = 1.0
    if beta <= 0.0:
        beta = 1.0

    # Diagonal covariance for rollout.
    Sigma_diag = diag1.copy()
    proc_diag = eta_proc * np.where(np.isfinite(diag1), diag1, 0.0)

    x_hats: list[np.ndarray] = []
    Sigma_hats: list[np.ndarray] = []
    H: list[float] = []
    c: list[float] = []
    n_cov_list: list[int] = []
    rho_list: list[float] = []
    c_qual_list: list[float] = []
    c_cov_list: list[float] = []

    x = np.asarray(x_hat_1, dtype=float).copy()
    for k in range(1, h + 1):
        # Store current k-step forecast.
        x_hats.append(x.copy())
        Sigma_hats.append(np.diag(Sigma_diag.copy()))

        # Uncertainty summary and confidence.
        finite = Sigma_diag[np.isfinite(Sigma_diag)]
        n_cov = int(finite.size)
        rho = float(n_cov) / float(D) if D > 0 else 0.0
        H_k = float(np.mean(finite)) if finite.size else float("inf")
        H.append(H_k)
        n_cov_list.append(n_cov)
        rho_list.append(rho)

        if finite.size:
            c_qual_k = float(np.exp(-alpha * H_k))
        else:
            c_qual_k = 0.0
        c_cov_k = float(rho ** beta) if rho > 0.0 else 0.0
        c_qual_list.append(c_qual_k)
        c_cov_list.append(c_cov_k)
        c.append(float(c_qual_k * c_cov_k))

        # Propagate uncertainty forward.
        Sigma_diag = Sigma_diag + proc_diag

    return RolloutResult(
        x_hats=x_hats,
        Sigma_hats=Sigma_hats,
        H=H,
        c=c,
        n_cov=n_cov_list,
        rho=rho_list,
        c_qual=c_qual_list,
        c_cov=c_cov_list,
    )
