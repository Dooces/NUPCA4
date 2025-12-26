"""nupca3/diagnostics/metrics.py

Diagnostics and the v1.5b “feel proxy” components.

Axiom coverage
-------------
- A17.1: q_res(t) = mean_{k∈O_t} |Λ_global(t)[k] · e(t)[k]|
- A17.2: q_maint(t) = L_eff(t)
- A17.3: q_unc(t) = H_d(t) where d is the latency floor (A7.4)

Important semantics
-------------------
A17 defines *components* (q_res, q_maint, q_unc). It does not define a single
scalar combination, and it explicitly states these values have **no control
authority** (must not be used as gates or triggers).

Implementation clarification
----------------------------
Earlier repo snapshots approximated q_unc using local precision-weighted
uncertainty on observed dims. That is not the v1.5b definition. In v1.5b,
q_unc is taken from the rollout uncertainty summary at the latency floor.
This module therefore requires the caller to provide H_d(t).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class FeelProxy:
    """A17 feel proxy components."""

    q_res: float
    q_res_raw: float
    q_maint: float
    q_unc: float


def compute_feel_proxy(
    *,
    observed_dims: Iterable[int],
    error_vec: np.ndarray,
    sigma_global_diag: np.ndarray,
    L_eff: float,
    H_d: float,
    sigma_floor: float = 1e-2,
) -> FeelProxy:
    """Compute (q_res, q_maint, q_unc) for time t.

    Inputs
    ------
    observed_dims:
        The observation set O_t (A16.5). Only these dimensions contribute to
        q_res (A17.1).
    error_vec:
        e(t) = x(t) - \hat{x}(t|t-1). Must be shape (D,).
    sigma_global_diag:
        Diagonal of Σ_global(t) associated with \hat{x}(t|t-1) (A7.3).
        Must be shape (D,), strictly positive.
    L_eff:
        Effective maintenance load L_eff(t) (A5.5).
    H_d:
        Rollout uncertainty summary at latency floor d: H_d(t) (A7.4, A17.3).
    sigma_floor:
        Lower bound on Σ_diag to prevent precision spikes in sparse settings.

    Outputs
    -------
    FeelProxy with fields (q_res, q_maint, q_unc).
    """

    O = [int(k) for k in observed_dims]
    e = np.asarray(error_vec, dtype=float)
    sig = np.asarray(sigma_global_diag, dtype=float)

    if e.ndim != 1 or sig.ndim != 1 or e.shape[0] != sig.shape[0]:
        raise ValueError("error_vec and sigma_global_diag must be 1D arrays of the same length")

    # Precision Λ_global = 1 / Σ_global (diagonal form).
    safe_sig = np.maximum(sig, float(sigma_floor))
    lam = 1.0 / safe_sig

    # A17.1
    if len(O) == 0:
        q_res = 0.0
        q_res_raw = 0.0
    else:
        q_res = float(np.mean(np.abs(lam[O] * e[O])))
        q_res_raw = float(np.mean(np.abs(e[O])))

    # A17.2
    q_maint = float(L_eff)

    # A17.3
    q_unc = float(H_d)

    return FeelProxy(q_res=q_res, q_res_raw=q_res_raw, q_maint=q_maint, q_unc=q_unc)

# -----------------------------------------------------------------------------
# Compatibility alias
# -----------------------------------------------------------------------------

def compute_feel_proxy_components(**kwargs):
    """Alias retained for older call sites.

    The canonical API is :func:`compute_feel_proxy`.
    """
    return compute_feel_proxy(**kwargs)
