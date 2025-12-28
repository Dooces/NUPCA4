"""
Observation and context helpers used across the step pipeline.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, Iterable, List, Set, Tuple

import numpy as np

from ..config import AgentConfig
from ..geometry.streams import dims_for_block, extract_coarse, periph_block_size
from ..memory.salience import infer_node_band_level
from ..types import AgentState, EnvObs


def _cfg_D(state: AgentState, cfg: AgentConfig) -> int:
    """
    Determine the working dimensionality D used for bounds checking and resizing.

    Inputs:
      state: current AgentState (may have buffer.x_last as a dense vector)
      cfg: AgentConfig (may have cfg.D)

    Output:
      D: int dimensionality used in this step
    """
    try:
        return int(getattr(cfg, "D"))
    except Exception:
        x = getattr(getattr(state, "buffer", None), "x_last", None)
        return int(len(x)) if x is not None else 0


def _filter_cue_to_Oreq(
    cue: Dict[int, float],
    O_req: Set[int],
    D: int,
) -> Dict[int, float]:
    """
    A16.5 masking contract: only dims in O_req are treated as observed, and only
    those within [0, D) are allowed through.
    """
    if not cue:
        return {}
    out: Dict[int, float] = {}
    for k, v in cue.items():
        kk = int(k)
        if 0 <= kk < D and kk in O_req:
            out[kk] = float(v)
    return out


def _ensure_node_band_levels(state: AgentState, cfg: AgentConfig) -> None:
    library = getattr(state, "library", None)
    if library is None:
        return
    nodes = getattr(library, "nodes", {})
    node_levels = getattr(state, "node_band_levels", {})
    seen: Set[int] = set()
    updated = False
    for nid, node in nodes.items():
        key = int(nid)
        seen.add(key)
        if key in node_levels:
            continue
        node_levels[key] = infer_node_band_level(node, cfg)
        updated = True
    stale = [nid for nid in node_levels if nid not in seen]
    for nid in stale:
        node_levels.pop(nid, None)
        updated = True
    if updated:
        state.node_band_levels = node_levels


def _compute_peripheral_gist(x_vec: np.ndarray, cfg: AgentConfig) -> np.ndarray:
    gist = extract_coarse(x_vec, cfg)
    if gist.size > 0:
        return gist
    data = np.asarray(x_vec, dtype=float).reshape(-1)
    if data.size == 0:
        return np.zeros(0, dtype=float)
    stats = np.array(
        [
            float(np.mean(data)),
            float(np.std(data)),
            float(np.min(data)),
            float(np.max(data)),
        ],
        dtype=float,
    )
    return np.nan_to_num(stats)


def _update_context_register(state: AgentState, gist: np.ndarray, cfg: AgentConfig) -> None:
    beta = float(cfg.beta_context)
    beta = max(0.0, min(1.0, beta))
    prev = getattr(state, "context_register", None)
    if prev is None:
        prev = np.zeros_like(gist)
    if prev.shape != gist.shape:
        prev = np.zeros_like(gist)
    new_reg = (1.0 - beta) * prev + beta * gist
    state.context_register = new_reg


def _update_context_tags(state: AgentState, cfg: AgentConfig) -> None:
    beta_tag = float(cfg.beta_context_node)
    beta_tag = max(0.0, min(1.0, beta_tag))
    if beta_tag <= 0.0:
        return
    gist = getattr(state, "context_register", np.zeros(0, dtype=float)).reshape(-1)
    if gist.size == 0:
        return
    tags = getattr(state, "node_context_tags", {})
    for nid in getattr(state, "active_set", set()):
        prev_tag = tags.get(nid)
        if prev_tag is None or prev_tag.shape != gist.shape:
            prev_tag = np.zeros_like(gist)
        tags[nid] = (1.0 - beta_tag) * prev_tag + beta_tag * gist
    state.node_context_tags = tags


def _update_coverage_debts(state: AgentState, cfg: AgentConfig) -> None:
    library = getattr(state, "library", None)
    if library is None:
        return
    nodes = getattr(library, "nodes", {})
    expert_debt = getattr(state, "coverage_expert_debt", {})
    band_debt = getattr(state, "coverage_band_debt", {})
    node_levels = getattr(state, "node_band_levels", {})
    active_set = getattr(state, "active_set", set())

    seen_nodes: Set[int] = set()
    for raw_nid, node in nodes.items():
        nid = int(raw_nid)
        seen_nodes.add(nid)
        level = node_levels.get(nid)
        if level is None:
            level = infer_node_band_level(node, cfg)
            node_levels[nid] = level
        if nid in active_set:
            expert_debt[nid] = 0
        else:
            expert_debt[nid] = expert_debt.get(nid, 0) + 1

    stale_nodes = [nid for nid in expert_debt if nid not in seen_nodes]
    for nid in stale_nodes:
        expert_debt.pop(nid, None)
    stale_levels = [nid for nid in node_levels if nid not in seen_nodes]
    for nid in stale_levels:
        node_levels.pop(nid, None)

    innovation_weight = float(getattr(cfg, "fovea_innovation_weight", 0.0))
    if innovation_weight > 0.0:
        innovation = np.asarray(
            getattr(state.fovea, "block_innovation", np.zeros(int(getattr(cfg, "B", 0)))),
            dtype=float,
        )
        incumbents = getattr(state, "incumbents_by_block", [])
        for block_id, nodes_in_block in enumerate(incumbents):
            if block_id < 0 or block_id >= innovation.size:
                continue
            if float(innovation[block_id]) <= 0.0:
                continue
            for nid in nodes_in_block:
                expert_debt[nid] = max(0, expert_debt.get(nid, 0) - 1)

    active_levels = {node_levels[nid] for nid in active_set if nid in node_levels}
    for level in set(node_levels.values()):
        band_debt.setdefault(level, 0)
    for level in list(band_debt):
        if level in active_levels:
            band_debt[level] = 0
        else:
            band_debt[level] = band_debt.get(level, 0) + 1
    valid_levels = set(node_levels.values())
    for level in list(band_debt):
        if level not in valid_levels:
            band_debt.pop(level, None)

    state.coverage_expert_debt = expert_debt
    state.coverage_band_debt = band_debt
    state.node_band_levels = node_levels


def _update_block_uncertainty(state: AgentState, sigma_diag: np.ndarray | None, cfg: AgentConfig) -> None:
    """Keep a cached uncertainty per block derived from the latest Sigma diagonal."""
    if sigma_diag is None:
        return
    B = int(getattr(cfg, "B", 0))
    if B <= 0:
        return
    diag = np.asarray(sigma_diag, dtype=float).reshape(-1)
    if not diag.size:
        diag = np.zeros(0, dtype=float)
    D = int(getattr(cfg, "D", diag.size))
    if diag.size != D:
        diag = np.resize(diag, (D,))

    default_unc = float(getattr(cfg, "fovea_uncertainty_default", 1.0))
    block_unc = np.zeros(B, dtype=float)
    for b in range(B):
        dims = list(dims_for_block(b, cfg))
        if not dims:
            block_unc[b] = default_unc
            continue
        idx = np.array(dims, dtype=int)
        vals = diag[idx]
        finite_vals = vals[np.isfinite(vals)]
        block_unc[b] = float(np.mean(finite_vals)) if finite_vals.size else default_unc

    state.fovea.block_uncertainty = block_unc


def _update_observed_history(
    state: AgentState,
    obs_idx: np.ndarray,
    cfg: AgentConfig,
    *,
    extra_dims: Iterable[int] | None = None,
) -> None:
    """Maintain the sliding support window of observed dims for multi-world merge guards."""
    window = max(1, int(getattr(cfg, "multi_world_support_window", 4)))
    history = getattr(state, "observed_history", None)
    if history is None or not isinstance(history, deque):
        history = deque()
    dims = {int(k) for k in obs_idx if np.isfinite(k)}
    if extra_dims:
        for dim in extra_dims:
            try:
                d = int(dim)
            except Exception:
                continue
            dims.add(d)
    history.append(dims)
    while len(history) > window:
        history.popleft()
    state.observed_history = history


def _support_window_union(state: AgentState) -> Set[int]:
    hist = getattr(state, "observed_history", None)
    if not hist:
        return set()
    support: Set[int] = set()
    for entry in hist:
        if entry:
            support.update(entry)
    return support


def _peripheral_dim_set(D: int, cfg: AgentConfig) -> Set[int]:
    """Return the peripheral dimensionality set (Omega_bg_t) for the config."""
    periph_size = max(0, min(periph_block_size(cfg), D))
    if periph_size <= 0:
        return set()
    start = max(0, D - periph_size)
    return set(range(start, D))


def _build_coarse_observation(
    env_obs: EnvObs,
    obs_idx: np.ndarray,
    obs_vals: np.ndarray,
    D: int,
    cfg: AgentConfig,
    periph_dims: Set[int] | None = None,
) -> np.ndarray:
    """Create a low-resolution peripheral observation vector."""
    periph_full = getattr(env_obs, "periph_full", None)
    dims = periph_dims if periph_dims is not None else set()
    obs_vec = np.zeros(max(0, D), dtype=float)
    if dims and periph_full is not None:
        full_arr = np.asarray(periph_full, dtype=float).reshape(-1)
        if full_arr.size < D:
            full_arr = np.resize(full_arr, (D,))
        for dim in dims:
            if 0 <= dim < full_arr.size:
                obs_vec[dim] = float(full_arr[int(dim)])
    if obs_idx.size and obs_vals.size:
        obs_vec[obs_idx] = obs_vals
    return extract_coarse(obs_vec, cfg)


def _compute_peripheral_metrics(
    state: AgentState,
    cfg: AgentConfig,
    prior_t: np.ndarray,
    env_obs: EnvObs,
    obs_idx: np.ndarray,
    obs_vals: np.ndarray,
    D: int,
    periph_dims: Set[int],
) -> None:
    periph_prior = extract_coarse(prior_t, cfg)
    periph_obs = _build_coarse_observation(env_obs, obs_idx, obs_vals, D, cfg, periph_dims)
    residual = float("nan")
    if periph_prior.size and periph_obs.size:
        min_len = min(periph_prior.size, periph_obs.size)
        if min_len > 0:
            diff = periph_obs[:min_len] - periph_prior[:min_len]
            finite = np.isfinite(diff)
            if finite.any():
                residual = float(np.mean(np.abs(diff[finite])))
    top = periph_prior if periph_prior.size else np.zeros(0, dtype=float)
    state.peripheral_prior = top.copy()
    state.peripheral_obs = periph_obs.copy() if periph_obs.size else np.zeros(0, dtype=float)
    periph_count = len(periph_dims)
    obs_set = {int(dim) for dim in obs_idx if np.isfinite(dim)}
    if periph_count > 0:
        if getattr(env_obs, "periph_full", None) is not None and periph_obs.size:
            observed_periph = periph_count
        else:
            observed_periph = int(sum(1 for dim in periph_dims if dim in obs_set))
        denom = float(periph_count)
        confidence = float(observed_periph) / denom if denom > 0.0 else 0.0
    else:
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    state.peripheral_confidence = confidence
    state.peripheral_residual = residual


def _prior_obs_mae(obs_idx: np.ndarray, obs_vals: np.ndarray, prior: np.ndarray) -> float:
    """Return MAE between obs_vals and prior over the observed dims."""
    if obs_idx.size == 0:
        return float("nan")
    prior = np.asarray(prior, dtype=float).reshape(-1)
    if prior.shape[0] < obs_idx.max(initial=-1) + 1:
        prior = np.resize(prior, (max(obs_idx.max(initial=-1) + 1, prior.shape[0]),))
    diff = obs_vals - prior[obs_idx]
    finite = np.isfinite(diff)
    if not finite.any():
        return float("nan")
    return float(np.mean(np.abs(diff[finite])))


def _coarse_summary(
    vec: np.ndarray | None,
    head_len: int = 8,
) -> Tuple[float, int, Tuple[float, ...]]:
    if vec is None:
        return 0.0, 0, ()
    arr = np.asarray(vec, dtype=float)
    if arr.size == 0:
        return 0.0, 0, ()
    norm = float(np.linalg.norm(arr))
    nonzero = int(np.count_nonzero(arr))
    head = tuple(float(x) for x in arr[: min(head_len, arr.size)])
    return norm, nonzero, head
