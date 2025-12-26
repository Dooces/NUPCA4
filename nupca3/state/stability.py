"""nupca3/state/stability.py

Probe/feature variance plumbing for stable(t) gating (A3.3).

Axiom coverage
--------------
- A3.3: stable(t) depends on (i) no recent structural edits and (ii) low
  variance in exogenous probes and internal features over a trailing window W.

Why this module exists
----------------------
The repository's acceptance gate (nupca3/edits/acceptance.py) implements
permit_struct(t) with an explicit stable(t) conjunct. The stability predicate
requires scalar summaries:
    probe_var(t), feature_var(t)

Without explicit plumbing, these fields remain None and stable(t) becomes
permanently false (the conservative default), which suppresses structural edits
regardless of true environmental stability.

This module provides:
  - a rolling-window accumulator (length W)
  - scalar variance summaries stored on AgentState

Representation note
-------------------
"Probes" and "features" are intentionally left representation-agnostic:
  - probes: low-dimensional exogenous signals (task-dependent)
  - features: low-dimensional summaries of internal representation dynamics

The step pipeline is responsible for choosing appropriate vectors to feed.

#ITOOKASHORTCUT
In a full v1.5b instantiation, probes/features would be explicitly defined
(e.g., dedicated probe heads, canonical feature embeddings). Here we expose an
API that can accept those vectors when available, while allowing toy defaults.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..config import AgentConfig
from ..types import AgentState


def _append_window(window: list[np.ndarray], vec: np.ndarray, W: int) -> None:
    """Append vec into window and trim to at most W elements."""
    if W <= 0:
        window.clear()
        return
    window.append(np.asarray(vec, dtype=float).reshape(-1))
    if len(window) > W:
        del window[: len(window) - W]


def _window_variance(window: list[np.ndarray]) -> float:
    """Compute a scalar variance summary from a window of vectors.

    Returns:
      mean_{dims}( Var_{time}(v_t[d]) )

    If the window is too short, returns 0.
    """
    if len(window) < 2:
        return 0.0

    # Pad/trim to min dimensionality across the window for safety.
    dims = min(int(v.shape[0]) for v in window)
    if dims <= 0:
        return 0.0

    X = np.stack([v[:dims] for v in window], axis=0)  # (T, D)
    var_t = np.var(X, axis=0, ddof=0)                 # (D,)
    return float(np.mean(var_t))


def update_stability_metrics(
    state: AgentState,
    cfg: AgentConfig,
    *,
    probe_vec: Optional[np.ndarray] = None,
    feature_vec: Optional[np.ndarray] = None,
) -> None:
    """Update probe_var(t) and feature_var(t) on AgentState.

    Inputs:
      - state: AgentState (mutated in place)
      - cfg: AgentConfig (uses W)
      - probe_vec: optional probe vector at time t
      - feature_vec: optional feature vector at time t

    Outputs:
      - state.probe_window, state.feature_window appended
      - state.probe_var, state.feature_var updated

    Timing discipline
    -----------------
    A3.3 uses a trailing window of size W. This function is designed to be
    called once per step, after the step pipeline has computed the time-t probe
    and feature vectors.

    If a vector is None, the corresponding window is not updated.
    """

    W = int(getattr(cfg, "W", 50))

    if probe_vec is not None:
        _append_window(state.probe_window, probe_vec, W)
        state.probe_var = _window_variance(state.probe_window)

    if feature_vec is not None:
        _append_window(state.feature_window, feature_vec, W)
        state.feature_var = _window_variance(state.feature_window)
