"""nupca3/memory/completion.py

A13 completion formalism: perception/recall completion as the same operator as prediction.

This file defines the *completion operator* as it is used throughout the v1.5b axioms:

  - A7 provides a prior (prediction) \hat{x}(t|t-1) and a global covariance \Sigma_global(t)
  - A13 defines completion as cue-driven overwrite (clamp) into that prior:
        x(t) = Clamp( \hat{x}(t|t-1), cue(O_t) )
  - A16.5 enforces stale persistence: unobserved dims retain their prior values.

Design constraints (axiom-faithful)
----------------------------------
1) **Same operator for perception/recall/prediction (A13).**
   The only differences are the cue source and whether the cue is empty.

2) **Sparse cue is canonical.**
   The canonical cue format is Dict[int, float] (EnvObs.x_partial). No dense/legacy cues accepted.

3) **No new hidden authorities.**
   Completion does not decide attention, working sets, or storage. It only:
     - obtains / accepts a prior, and
     - overwrites observed dimensions.

Public API expected by the current snapshot
-------------------------------------------
- complete(cue, mode, state, cfg, predicted_prior_t=None, predicted_sigma_diag=None)
    -> (x_completed_t, Sigma_prior_t, prior_t)

`step_pipeline.py` imports and calls `complete(...)` directly.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, Optional, Set, Tuple, Union

import numpy as np

from ..config import AgentConfig
from ..types import AgentState, EnvObs, ExpertLibrary, ObservationBuffer, WorkingSet
from .fusion import fuse_predictions
from ..geometry.streams import apply_transport


# =============================================================================
# Cue normalization helpers
# =============================================================================

SparseCue = Dict[int, float]
DenseCue = np.ndarray
Cue = Union[SparseCue, DenseCue]


def _D_from_state(state: AgentState, cfg: AgentConfig) -> int:
    """Strict dimensionality (cfg.D required)."""
    D = int(getattr(cfg, "D", 0))
    if D <= 0:
        raise RuntimeError("cfg.D must be set for completion")
    return D


def _coerce_sparse_cue(cue: Optional[Cue], D: int) -> SparseCue:
    """Coerce cue into canonical sparse Dict[int, float] bounded to [0, D)."""
    if cue is None:
        return {}
    if not isinstance(cue, dict):
        raise RuntimeError("cue must be sparse dict")
    out: SparseCue = {}
    for k, v in cue.items():
        kk = int(k)
        if 0 <= kk < D:
            out[kk] = float(v)
    return out


def cue_from_env_obs(obs: EnvObs) -> SparseCue:
    """Extract canonical sparse cue from EnvObs (A16.5)."""
    x = getattr(obs, "x_partial", None)
    return dict(x) if isinstance(x, dict) else {}


def observed_dims(cue: SparseCue) -> Set[int]:
    """Observed dimension set O_t from sparse cue."""
    return set(int(k) for k in cue.keys())


# =============================================================================
# Clamp (A13)
# =============================================================================

def apply_cue(prior_t: np.ndarray, cue: SparseCue) -> Tuple[np.ndarray, Set[int]]:
    """A13 clamp/overwrite: overwrite observed dims into the prior.

    Returns:
      x_completed_t, O_t
    """
    prior = np.asarray(prior_t, dtype=float).reshape(-1)
    out = prior.copy()
    if cue:
        idx = np.fromiter((int(k) for k in cue.keys()), dtype=int)
        vals = np.fromiter((float(v) for v in cue.values()), dtype=float)
        # idx is already bounded by _coerce_sparse_cue, but re-guard for safety.
        good = (idx >= 0) & (idx < len(out))
        if np.any(good):
            out[idx[good]] = vals[good]
    return out, observed_dims(cue)


# =============================================================================
# Prior acquisition (A7)
# =============================================================================

def _prior_from_cache_or_fusion(
    *,
    state: AgentState,
    cfg: AgentConfig,
    predicted_prior_t: Optional[np.ndarray],
    predicted_sigma_diag: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Obtain (prior_t, Sigma_prior_t) for the current time index.

    Priority:
      1) Explicit predicted_prior_t (+ optional predicted_sigma_diag) provided by caller.
      2) State learn_cache: yhat_tp1 / sigma_tp1_diag (from last step's A7.3).
      3) Recompute via fuse_predictions using last step's A_t stored in learn_cache.
      4) Fallback: stale persistence prior = x_mem (buffer.x_last), Sigma = inf (uncovered).
    """
    D = _D_from_state(state, cfg)

    # 1) Explicitly supplied prior.
    if predicted_prior_t is not None:
        prior = np.asarray(predicted_prior_t, dtype=float).reshape(-1)
        if prior.size != D:
            raise ValueError("predicted_prior_t must be shape (D,)")
        if predicted_sigma_diag is None:
            # Unknown covariance: treat as uncovered/unknown (A7.2 default).
            Sigma = np.diag(np.full(D, np.inf, dtype=float))
        else:
            sd = np.asarray(predicted_sigma_diag, dtype=float).reshape(-1)
            if sd.size != D:
                raise ValueError("predicted_sigma_diag must be shape (D,)")
            Sigma = np.diag(sd.copy())
        return prior, Sigma

    # 2) Cached (preferred, if present).
    lc = getattr(state, "learn_cache", None)
    if lc is not None:
        yhat = getattr(lc, "yhat_tp1", None)
        sigd = getattr(lc, "sigma_tp1_diag", None)
        if yhat is not None and sigd is not None:
            prior = np.asarray(yhat, dtype=float).reshape(-1)
            sd = np.asarray(sigd, dtype=float).reshape(-1)
            if prior.size == D and sd.size == D:
                return prior.copy(), np.diag(sd.copy())

    # 3) Recompute via fusion if we have last step's working set.
    if lc is not None:
        A_prev = getattr(lc, "A_t", None)
        if A_prev is not None:
            # Buffer currently holds x(t-1) (stale persistence) and O_{t-1}.
            buf_prev: ObservationBuffer = state.buffer
            prior, Sigma = fuse_predictions(
                state.library,
                A_prev,
                buf_prev,
                set(getattr(buf_prev, "observed_dims", set())),
                cfg,
            )
            return np.asarray(prior, dtype=float).reshape(-1), np.asarray(Sigma, dtype=float)

    # 4) Fallback: stale persistence only.
    x_mem = np.asarray(getattr(state.buffer, "x_last", np.zeros(D)), dtype=float).reshape(-1)
    if x_mem.size != D:
        x_mem = np.resize(x_mem, (D,))
    Sigma = np.diag(np.full(D, np.inf, dtype=float))
    return x_mem.copy(), Sigma


# =============================================================================
# Public completion entry point (A13)
# =============================================================================

def complete(
    cue: Optional[Cue],
    *,
    mode: str,
    state: AgentState,
    cfg: AgentConfig,
    predicted_prior_t: Optional[np.ndarray] = None,
    predicted_sigma_diag: Optional[np.ndarray] = None,
    transport_shift: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unified completion operator for perception/recall/prediction (A13).

    Inputs:
      cue:
        Canonical: Dict[int, float] for observed dims. Empty/no cue is allowed.
        Legacy: dense ndarray with NaN for unobserved dims.
      mode:
        One of {'perception', 'recall', 'prediction'}.
        - 'prediction' treats the cue as empty (no overwrite).
      state, cfg:
        Used only to obtain the A7 prior (either from cache or by recomputation).

      predicted_prior_t / predicted_sigma_diag:
        Optional explicit prior plumbing. If provided, they are taken as the A7 prior.
      transport_shift:
        Optional (dx, dy) transport shift computed from coarse cues at t-1.
        When provided, the prior mean/covariance are moved into the current frame
        before clamping, which keeps content aligned even for unobserved dims.

    Returns:
      (x_completed_t, Sigma_prior_t, prior_t)
        - x_completed_t: completed x(t) after clamping cue into prior_t
        - Sigma_prior_t: the A7 global covariance for the prior at time t
        - prior_t: the A7 prior \hat{x}(t|t-1)

    Axiom notes:
      - A16.5 stale persistence is satisfied because unobserved dims retain prior values.
      - This function does not mutate state; the caller updates buffers.
    """
    D = _D_from_state(state, cfg)
    prior_t, Sigma_prior_t = _prior_from_cache_or_fusion(
        state=state,
        cfg=cfg,
        predicted_prior_t=predicted_prior_t,
        predicted_sigma_diag=predicted_sigma_diag,
    )

    prior_t = np.asarray(prior_t, dtype=float).reshape(-1)
    if prior_t.size != D:
        prior_t = np.resize(prior_t, (D,))

    Sigma_prior_arr = np.asarray(Sigma_prior_t, dtype=float)
    if Sigma_prior_arr.ndim == 2:
        diag = np.diag(Sigma_prior_arr).copy()
    else:
        diag = Sigma_prior_arr.reshape(-1).copy()
    if diag.size != D:
        diag = np.resize(diag, (D,))

    shift = (0, 0)
    if transport_shift is not None:
        try:
            shift = (int(transport_shift[0]), int(transport_shift[1]))
        except Exception:
            shift = (0, 0)
    if shift != (0, 0):
        prior_t = apply_transport(prior_t, shift, cfg)
        diag = apply_transport(diag, shift, cfg)

    Sigma_prior_t = np.diag(diag)

    mode_l = str(mode).lower().strip()
    if mode_l not in {"perception", "recall", "prediction"}:
        raise ValueError("mode must be one of {'perception','recall','prediction'}")

    cue_sparse = {} if mode_l == "prediction" else _coerce_sparse_cue(cue, D)

    x_completed_t, _O_t = apply_cue(prior_t, cue_sparse)
    return x_completed_t, Sigma_prior_t, prior_t


# =============================================================================
# Optional helper matching the A7/A13 decomposition explicitly
# =============================================================================

def a7_completion_and_prediction_step(
    *,
    buf_prev: ObservationBuffer,
    obs: EnvObs,
    library: ExpertLibrary,
    working_set_prev: WorkingSet,
    working_set_cur: WorkingSet,
    cfg: AgentConfig,
    predicted_prior_t: Optional[np.ndarray] = None,
    predicted_sigma_diag: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Set[int]]:
    """Run (prior -> clamp -> next prediction) in explicit A13 order.

    Returns:
      (x_completed_t, yhat_tp1, Sigma_tp1_diag, prior_t, O_t)
    """
    D = int(cfg.D)

    # 1) Prior for time t: x̂(t|t-1)
    if predicted_prior_t is None:
        prior_t, Sigma_prior_t = fuse_predictions(
            library,
            working_set_prev,
            buf_prev,
            set(getattr(buf_prev, "observed_dims", set())),
            cfg,
        )
        Sigma_prior_diag = np.diag(Sigma_prior_t).copy() if np.asarray(Sigma_prior_t).ndim == 2 else np.asarray(Sigma_prior_t, dtype=float).reshape(-1)
    else:
        prior_t = np.asarray(predicted_prior_t, dtype=float).reshape(-1)
        if prior_t.size != D:
            raise ValueError("predicted_prior_t must match state dimensionality")
        if predicted_sigma_diag is None:
            Sigma_prior_diag = np.full(D, np.inf, dtype=float)
        else:
            Sigma_prior_diag = np.asarray(predicted_sigma_diag, dtype=float).reshape(-1)
            if Sigma_prior_diag.size != D:
                raise ValueError("predicted_sigma_diag must match state dimensionality")

    # 2) Clamp into prior (A13 + A16.5)
    cue_t = _coerce_sparse_cue(cue_from_env_obs(obs), D)
    x_completed_t, O_t = apply_cue(prior_t, cue_t)

    # 3) Next prediction x̂(t+1|t)
    buf_t = replace(buf_prev, x_last=np.asarray(x_completed_t, dtype=float), observed_dims=set(O_t))
    yhat_tp1, Sigma_tp1 = fuse_predictions(library, working_set_cur, buf_t, set(O_t), cfg)
    Sigma_tp1_diag = np.diag(Sigma_tp1).copy() if np.asarray(Sigma_tp1).ndim == 2 else np.asarray(Sigma_tp1, dtype=float).reshape(-1)

    return np.asarray(x_completed_t, dtype=float), np.asarray(yhat_tp1, dtype=float), np.asarray(Sigma_tp1_diag, dtype=float), np.asarray(prior_t, dtype=float), set(O_t)
