"""nupca3/memory/fusion.py

Precision-weighted fusion with coverage invariant.

Axiom coverage: A7.2 (coverage handling), A7.3 (global fusion).

This skeleton fuses only 1-step prediction; extend to multi-step via rollout module.


[AXIOM_CLARIFICATION_ADDENDUM — Representation & Naming]

- Terminology: identifiers like "Expert" in this codebase refer to NUPCA3 **abstraction/resonance nodes** (a "constellation"), not conventional Mixture-of-Experts "experts" or router-based MoE.

- Representation boundary (clarified intent of v1.5b): the completion/fusion operator (A7) is defined over an **encoded, multi-resolution abstraction vector** \(x(t)\). Raw pixels may exist only in a transient observation buffer for the current step; **raw pixel values must never be inserted into long-term storage** (library/cold storage) and must not persist across REST boundaries.

- Decomposition intuition: each node is an operator that *factors out* a predictable/resonant component on its footprint, leaving residual structure for other nodes (or for REST-time proposal) to capture. This is the intended "FFT-like" interpretation of masks/constellations.
"""

from __future__ import annotations

import numpy as np

from ..config import AgentConfig
from ..types import ExpertLibrary, WorkingSet, ObservationBuffer
from .expert import predict, precision_vector


def fuse_predictions(lib: ExpertLibrary, A_t: WorkingSet, buf: ObservationBuffer, O_t: set[int], cfg: AgentConfig) -> tuple[np.ndarray, np.ndarray]:
    """Fuse local predictions into global prediction x_hat and Sigma.

    Inputs:
      - lib, working set, observation buffer
      - O_t: observed dims set (for coverage logic)

    Outputs:
      - x_hat_1: np.ndarray shape (D,)
      - Sigma_1: np.ndarray shape (D,D) (skeleton: diagonal covariance)

    Coverage invariant (A7.2):
      - If a dim is uncovered by any expert in A_t, persist x[k] and set variance to inf.
    """
    D = cfg.D
    x = buf.x_last
    if not A_t.active:
        return x.copy(), np.diag(np.full(D, np.inf))

    # Aggregate precision and precision-weighted means (diagonal-only skeleton)
    prec_sum = np.zeros(D, dtype=float)
    mean_prec_sum = np.zeros(D, dtype=float)
    covered = np.zeros(D, dtype=bool)

    for node_id in A_t.active:
        node = lib.nodes[node_id]
        mu_j = predict(node, x)
        prec_j = precision_vector(node)
        # Covered where mask true and precision finite
        mask = node.mask.astype(bool) & np.isfinite(prec_j) & (prec_j > 0)
        covered |= mask
        prec_sum[mask] += prec_j[mask]
        mean_prec_sum[mask] += prec_j[mask] * mu_j[mask]

    x_hat = x.copy()
    Sigma_diag = np.full(D, np.inf, dtype=float)

    good = covered & (prec_sum > 0)
    x_hat[good] = mean_prec_sum[good] / prec_sum[good]
    Sigma_diag[good] = 1.0 / prec_sum[good]

    # Uncovered dims: persist x and infinite variance already set.
    Sigma = np.diag(Sigma_diag)
    return x_hat, Sigma
