"""
Observation and context helpers used across the step pipeline.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, Iterable, List, Set, Tuple

import numpy as np

from ..config import AgentConfig
from ..geometry.fovea import dims_for_block
from ..geometry.streams import extract_coarse, periph_block_size
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


def _tracked_node_ids(state: AgentState, cfg: AgentConfig) -> Set[int]:
    """Return a bounded set of node ids for per-step bookkeeping (v5).

    v5 fixed-budget semantics forbid scanning the entire durable library each
    online step. When we need node-level metadata (band levels, coverage debt),
    we restrict updates to a bounded tracked set.

    Tracked set priority (bounded union):
      1) current active_set (already capped by working set selection)
      2) salience_candidate_ids (already capped by salience computation)
      3) optional extra ids attached by other modules
    """

    library = getattr(state, "library", None)
    nodes = getattr(library, "nodes", None) if library is not None else None

    tracked: Set[int] = set(int(n) for n in (getattr(state, "active_set", set()) or set()))
    tracked.update(int(n) for n in (getattr(state, "salience_candidate_ids", set()) or set()))
    tracked.update(int(n) for n in (getattr(state, "coverage_tracked_ids", set()) or set()))

    cap = int(
        getattr(
            cfg,
            "coverage_debt_cap",
            getattr(cfg, "salience_max_candidates", getattr(cfg, "max_candidates", 256)),
        )
    )
    cap = max(32, cap)
    if not nodes:
        return tracked

    if len(tracked) < min(cap, 2):
        need = min(cap, 2) - len(tracked)
        for nid in sorted(nodes.keys()):
            if nid in tracked:
                continue
            tracked.add(int(nid))
            need -= 1
            if need <= 0:
                break

    target_levels = min(2, len(nodes))
    tracked_levels: Set[int] = set()
    if target_levels > 0:
        for nid in tracked:
            node = nodes.get(nid)
            if node is None:
                continue
            tracked_levels.add(int(infer_node_band_level(node, cfg)))
        if len(tracked_levels) < target_levels:
            for nid in sorted(nodes.keys()):
                if len(tracked_levels) >= target_levels:
                    break
                if int(nid) in tracked:
                    continue
                node = nodes.get(nid)
                if node is None:
                    continue
                tracked.add(int(nid))
                tracked_levels.add(int(infer_node_band_level(node, cfg)))

    if len(tracked) <= cap:
        return tracked

    # Deterministic truncation: keep lowest ids (stable, not data-dependent on dict order).
    return set(sorted(tracked)[:cap])


def _ensure_node_band_levels(state: AgentState, cfg: AgentConfig) -> None:
    """Maintain a bounded cache of node->band_level (v5).

    Legacy versions scanned *all* nodes each step. This v5 version only infers
    levels for a bounded tracked set and prunes the cache to a bounded size.
    """

    library = getattr(state, "library", None)
    nodes = getattr(library, "nodes", None) if library is not None else None
    if not nodes:
        state.node_band_levels = {}
        return

    tracked = _tracked_node_ids(state, cfg)
    if not tracked:
        # No tracked ids: keep cache as-is but prune to cap.
        tracked = set()

    node_levels: Dict[int, int] = dict(getattr(state, "node_band_levels", {}) or {})
    last_seen: Dict[int, int] = dict(getattr(state, "coverage_debt_last_seen", {}) or {})
    t_now = int(getattr(state, "t_w", 0))

    # Infer for tracked ids only.
    for nid in tracked:
        if nid in node_levels:
            last_seen[nid] = t_now
            continue
        node = nodes.get(nid)
        if node is None:
            continue
        node_levels[nid] = int(infer_node_band_level(node, cfg))
        last_seen[nid] = t_now

    # Prune cache to bounded size (LRU by last_seen). This loop is safe because
    # node_levels/last_seen are bounded by cap.
    cap = int(
        getattr(
            cfg,
            "coverage_debt_cap",
            getattr(cfg, "salience_max_candidates", getattr(cfg, "max_candidates", 256)),
        )
    )
    cap = max(32, cap)
    if len(last_seen) > cap:
        overflow = len(last_seen) - cap
        for nid, _ts in sorted(last_seen.items(), key=lambda kv: kv[1])[:overflow]:
            last_seen.pop(nid, None)
            node_levels.pop(nid, None)

    # Drop ids that no longer exist (bounded scan over cache keys only).
    for nid in list(node_levels.keys()):
        if nid not in nodes:
            node_levels.pop(nid, None)
            last_seen.pop(nid, None)

    state.node_band_levels = node_levels
    state.coverage_debt_last_seen = last_seen


def _compute_peripheral_gist(x_vec: np.ndarray, cfg: AgentConfig) -> np.ndarray:
    """Compute a cheap, bounded peripheral gist.

    Priority order:
      1) If the environment already provides a peripheral coarse channel (cfg.periph_bins/blocks),
         extract it (legacy behavior).
      2) Otherwise, for grid-aware harnesses (cfg.grid_{width,height,channels}), compute an
         occupancy-based pooled gist (max-pool over bins). This avoids mean-pooling blind spots
         that can collapse distinct regimes into identical summaries.
      3) Fallback to tiny global stats when no grid metadata is available.
    """

    # 1) Legacy: use an existing peripheral coarse channel when configured.
    gist = extract_coarse(x_vec, cfg)
    if gist.size > 0:
        return gist

    # 2) Grid-aware: pooled occupancy gist (small, cheap, and more discriminative than stats).
    try:
        grid_w = int(getattr(cfg, "grid_width", 0) or getattr(cfg, "grid_side", 0) or 0)
        grid_h = int(getattr(cfg, "grid_height", 0) or getattr(cfg, "grid_side", 0) or 0)
        grid_c = int(getattr(cfg, "grid_channels", 1) or 1)
        base_dim = int(getattr(cfg, "grid_base_dim", 0) or 0)
        if base_dim <= 0 and grid_w > 0 and grid_h > 0 and grid_c > 0:
            base_dim = int(grid_w * grid_h * grid_c)
    except Exception:
        grid_w, grid_h, grid_c, base_dim = 0, 0, 1, 0

    if grid_w > 0 and grid_h > 0 and grid_c > 0 and base_dim > 0:
        arr = np.asarray(x_vec, dtype=float).reshape(-1)
        if arr.size < base_dim:
            arr = np.resize(arr, (base_dim,))
        else:
            arr = arr[:base_dim]

        # Occupancy: 1 if any channel is non-zero.
        try:
            grid = arr.reshape(grid_h, grid_w, grid_c)
        except Exception:
            grid = None

        if grid is not None:
            occ = (grid > 0).any(axis=2).astype(np.uint8)  # [H,W] in {0,1}

            bins = int(getattr(cfg, "periph_bins", 0) or 0)
            if bins <= 0:
                bins = 8
            bins_x = max(1, min(int(bins), int(grid_w)))
            bins_y = max(1, min(int(bins), int(grid_h)))
            tile_w = int(np.ceil(float(grid_w) / float(bins_x)))
            tile_h = int(np.ceil(float(grid_h) / float(bins_y)))

            pooled = np.zeros((bins_y, bins_x), dtype=float)
            for by in range(bins_y):
                y0 = by * tile_h
                y1 = min(grid_h, (by + 1) * tile_h)
                if y0 >= y1:
                    continue
                for bx in range(bins_x):
                    x0 = bx * tile_w
                    x1 = min(grid_w, (bx + 1) * tile_w)
                    if x0 >= x1:
                        continue
                    pooled[by, bx] = float(np.max(occ[y0:y1, x0:x1]))
            return pooled.reshape(-1)

    # 3) Fallback: tiny global stats.
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



def compute_sig_gist_u8(
    x_vec: np.ndarray,
    fovea_blocks: Set[int],
    cfg: AgentConfig,
) -> np.ndarray:
    """Compute NUPCA5 ephemeral periphery gist for sig64.

    Requirements:
      - blockwise max-pool occupancy computed over the periphery only
      - fovea blocks are zeroed (excluded)
      - output is small, fixed-ish size, uint8, and MUST NOT be persisted

    This is deliberately computed from the current dense vector only and uses
    only geometry helpers (dims_for_block). No dense per-dimension vectors are
    stored; callers should treat the output as ephemeral.
    """
    B = int(getattr(cfg, "B", 0) or 0)
    if B <= 0:
        return np.zeros(0, dtype=np.uint8)

    arr = np.asarray(x_vec, dtype=float).reshape(-1)
    D = int(getattr(cfg, "D", arr.size) or arr.size)
    if arr.size < D:
        arr = np.resize(arr, (D,))
    elif arr.size > D and D > 0:
        arr = arr[:D]

    # Compute per-block occupancy (1 if any dim in the block is non-zero).
    occ_blocks = np.zeros(B, dtype=np.uint8)
    fov = {int(b) for b in (fovea_blocks or set())}
    for b in range(B):
        if b in fov:
            continue
        dims = list(dims_for_block(b, cfg))
        if not dims:
            continue
        # Bound-check dims against arr length.
        any_on = False
        for d in dims:
            dd = int(d)
            if 0 <= dd < arr.size and arr[dd] != 0.0:
                any_on = True
                break
        occ_blocks[b] = 1 if any_on else 0

    # Determine a 2D grid layout if available (blocks correspond to cells).
    try:
        grid_w = int(getattr(cfg, "grid_width", 0) or getattr(cfg, "grid_side", 0) or 0)
        grid_h = int(getattr(cfg, "grid_height", 0) or getattr(cfg, "grid_side", 0) or 0)
    except Exception:
        grid_w, grid_h = 0, 0

    # Number of gist bins (prefer NUPCA5 sig_gist_bins, else fall back).
    bins = int(getattr(cfg, "sig_gist_bins", 0) or getattr(cfg, "periph_bins", 0) or 8)
    bins = max(1, bins)

    if grid_w > 0 and grid_h > 0 and (grid_w * grid_h == B):
        occ2 = occ_blocks.reshape(grid_h, grid_w)

        bins_x = max(1, min(int(bins), int(grid_w)))
        bins_y = max(1, min(int(bins), int(grid_h)))
        tile_w = int(np.ceil(float(grid_w) / float(bins_x)))
        tile_h = int(np.ceil(float(grid_h) / float(bins_y)))

        pooled = np.zeros((bins_y, bins_x), dtype=np.uint8)
        for by in range(bins_y):
            y0 = by * tile_h
            y1 = min(grid_h, (by + 1) * tile_h)
            if y0 >= y1:
                continue
            for bx in range(bins_x):
                x0 = bx * tile_w
                x1 = min(grid_w, (bx + 1) * tile_w)
                if x0 >= x1:
                    continue
                pooled[by, bx] = np.max(occ2[y0:y1, x0:x1]).astype(np.uint8)
        return pooled.reshape(-1)

    # Fallback: 1D pooling over blocks.
    tile = int(np.ceil(float(B) / float(bins)))
    out = np.zeros(bins, dtype=np.uint8)
    for i in range(bins):
        j0 = i * tile
        j1 = min(B, (i + 1) * tile)
        if j0 >= j1:
            continue
        out[i] = np.max(occ_blocks[j0:j1]).astype(np.uint8)
    return out


def _update_context_register(state: AgentState, gist: np.ndarray, cfg: AgentConfig) -> None:
    gist_vec = np.asarray(gist, dtype=float).reshape(-1)
    if bool(cfg.sig_disable_context_register):
        # NUPCA5: do not persist gist/context state.
        state.context_register = np.zeros(0, dtype=float)
        return

    beta = float(cfg.beta_context)
    beta = max(0.0, min(1.0, beta))
    prev = getattr(state, "context_register", None)
    if prev is None or prev.shape != gist_vec.shape:
        prev = np.zeros_like(gist_vec)
    new_reg = (1.0 - beta) * prev + beta * gist_vec
    state.context_register = new_reg


def _update_context_tags(state: AgentState, cfg: AgentConfig) -> None:
    if bool(cfg.sig_disable_context_register):
        # NUPCA5: do not persist per-node context tags.
        state.node_context_tags = {}
        return

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
    """Fixed-cost coverage debt maintenance (v5).

    Legacy versions scanned ``library.nodes`` each step. Under v5 fixed-budget
    semantics, coverage bookkeeping is maintained over a bounded tracked set.

    Updated fields:
      - state.coverage_expert_debt[nid]
      - state.coverage_band_debt[level]
      - state.node_band_levels[nid] (lazily inferred for tracked nodes)
      - state.coverage_debt_last_seen[nid] (LRU pruning)
    """

    library = getattr(state, "library", None)
    nodes = getattr(library, "nodes", None) if library is not None else None
    if not nodes:
        state.coverage_expert_debt = {}
        state.coverage_band_debt = {}
        state.node_band_levels = {}
        state.coverage_debt_last_seen = {}
        return

    # Ensure band levels exist for tracked ids (bounded).
    _ensure_node_band_levels(state, cfg)

    tracked = _tracked_node_ids(state, cfg)
    active_set = {int(n) for n in (getattr(state, "active_set", set()) or set())}

    expert_debt: Dict[int, int] = dict(getattr(state, "coverage_expert_debt", {}) or {})
    band_debt: Dict[int, int] = dict(getattr(state, "coverage_band_debt", {}) or {})
    node_levels: Dict[int, int] = dict(getattr(state, "node_band_levels", {}) or {})
    last_seen: Dict[int, int] = dict(getattr(state, "coverage_debt_last_seen", {}) or {})

    t_now = int(getattr(state, "t_w", 0))
    debt_max = int(getattr(cfg, "coverage_debt_max", 10_000))
    debt_max = max(1, debt_max)

    # Update expert debts over tracked set only.
    for nid in tracked:
        if nid not in nodes:
            continue
        if nid in active_set:
            expert_debt[nid] = 0
        else:
            expert_debt[nid] = min(debt_max, int(expert_debt.get(nid, 0)) + 1)
        last_seen[nid] = t_now

    # Optional innovation discount (v5): apply only over the bounded tracked set.
    # NOTE: iterating ``incumbents_by_block`` can be an implicit full-library scan
    # if incumbents lists grow with memory. We therefore apply the discount by
    # looking up the tracked node's footprint block directly.
    innovation_weight = float(getattr(cfg, "fovea_innovation_weight", 0.0))
    if innovation_weight > 0.0:
        innovation = np.asarray(
            getattr(state.fovea, "block_innovation", np.zeros(int(getattr(cfg, "B", 0)) or 0)),
            dtype=float,
        ).reshape(-1)
        if innovation.size:
            for nid in tracked:
                if nid in active_set:
                    continue
                node = nodes.get(nid)
                if node is None:
                    continue
                # Footprint block id (preferred) or explicit block id.
                block_id = int(getattr(node, "footprint", getattr(node, "block_id", -1)))
                if 0 <= block_id < innovation.size and float(innovation[block_id]) > 0.0:
                    expert_debt[nid] = max(0, int(expert_debt.get(nid, 0)) - 1)

    # Band debts: maintain only for band levels present in the tracked cache.
    tracked_levels = {int(node_levels[nid]) for nid in tracked if nid in node_levels}
    active_levels = {int(node_levels[nid]) for nid in active_set if nid in node_levels}
    for level in tracked_levels:
        if level in active_levels:
            band_debt[level] = 0
        else:
            band_debt[level] = min(debt_max, int(band_debt.get(level, 0)) + 1)
    # Drop stale levels.
    for level in list(band_debt.keys()):
        if level not in tracked_levels:
            band_debt.pop(level, None)

    # Prune dictionaries to bounded size (LRU by last_seen).
    cap = int(
        getattr(
            cfg,
            "coverage_debt_cap",
            getattr(cfg, "salience_max_candidates", getattr(cfg, "max_candidates", 256)),
        )
    )
    cap = max(32, cap)
    if len(last_seen) > cap:
        overflow = len(last_seen) - cap
        for nid, _ts in sorted(last_seen.items(), key=lambda kv: kv[1])[:overflow]:
            last_seen.pop(nid, None)
            expert_debt.pop(nid, None)
            node_levels.pop(nid, None)

    # Remove ids that no longer exist (bounded scan over cache keys only).
    for nid in list(expert_debt.keys()):
        if nid not in nodes:
            expert_debt.pop(nid, None)
            node_levels.pop(nid, None)
            last_seen.pop(nid, None)

    state.coverage_expert_debt = expert_debt
    state.coverage_band_debt = band_debt
    state.node_band_levels = node_levels
    state.coverage_debt_last_seen = last_seen


def compute_block_uncertainty(sigma_diag: np.ndarray | None, cfg: AgentConfig) -> np.ndarray | None:
    """Return per-block uncertainties derived from the latest Sigma diagonal."""
    if sigma_diag is None:
        return None
    B = int(getattr(cfg, "B", 0))
    if B <= 0:
        return None
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

    return block_unc


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
