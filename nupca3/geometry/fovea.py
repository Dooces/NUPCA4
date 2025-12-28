"""nupca3/geometry/fovea.py

DoF-aligned block partition and greedy_cov foveation.

Axiom coverage: A16.

Required semantics implemented
-----------------------------
- A16.1: Deterministic DoF-aligned block partition with remainder distribution.
- A16.2: Per-block residual tracking from prediction error (observed dims only).
- A16.3: greedy_cov block selection: score = residual + alpha_cov * log(1+age)
- A16.4: Hard coverage cap (deterministic bound): include any blocks with age >= G (oldest-first).

Notes
-----
This module computes and updates fovea statistics; the environment/harness is
responsible for honoring the selected blocks when generating the next EnvObs.
Implementation note
-------------------
`sticky_k` is retained in config as a legacy/debug knob, but it is **not** part of v1.5b A16 and is not used by the greedy_cov selector in this module.



[AXIOM_CLARIFICATION_ADDENDUM — Representation & Naming]

- Terminology: identifiers like "Expert" in this codebase refer to NUPCA3 **abstraction/resonance nodes** (a "constellation"), not conventional Mixture-of-Experts "experts" or router-based MoE.

- Representation boundary (clarified intent of v1.5b): the completion/fusion operator (A7) is defined over an **encoded, multi-resolution abstraction vector** \(x(t)\). Raw pixels may exist only in a transient observation buffer for the current step; **raw pixel values must never be inserted into long-term storage** (library/cold storage) and must not persist across REST boundaries.

- Decomposition intuition: each node is an operator that *factors out* a predictable/resonant component on its footprint, leaving residual structure for other nodes (or for REST-time proposal) to capture. This is the intended "FFT-like" interpretation of masks/constellations.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Set, Tuple, List, Sequence

import json
import logging
import math
import time
import numpy as np

from ..config import AgentConfig
from ..types import FoveaState, ObservationBuffer

_BLOCK_CACHE: Dict[Tuple[int, int, int, int, int, int, int, int], List[List[int]]] = {}
_BLOCK_LOOKUP_CACHE: Dict[Tuple[int, int, int, int, int, int, int, int], Dict[int, int]] = {}

_FOVEA_GEOM_LOGGER = logging.getLogger("nupca3_grid_harness")
_FOVEA_GEOM_LOGGER_START = time.perf_counter()

def _log_geom_event(event: str, details: dict[str, object]) -> None:
    if not _FOVEA_GEOM_LOGGER.handlers:
        return
    payload = {"event": event}
    payload["timestamp"] = float(time.perf_counter() - _FOVEA_GEOM_LOGGER_START)
    payload.update(details)
    try:
        _FOVEA_GEOM_LOGGER.info(json.dumps(payload, sort_keys=True))
    except Exception:
        _FOVEA_GEOM_LOGGER.info(f"{event} {details}")


def _grid_block_dims(
    *,
    side: int,
    color_channels: int,
    shape_channels: int,
    block_cells_y: int,
    block_cells_x: int,
    block_y: int,
    block_x: int,
) -> List[int]:
    grid_cells = side * side
    color_offset = 0
    shape_offset = grid_cells * color_channels
    dims: List[int] = []
    y_start = block_y * block_cells_y
    x_start = block_x * block_cells_x
    for y in range(y_start, y_start + block_cells_y):
        for x in range(x_start, x_start + block_cells_x):
            cell = y * side + x
            if color_channels > 0:
                base = color_offset + cell * color_channels
                dims.extend(range(base, base + color_channels))
            if shape_channels > 0:
                base = shape_offset + cell * shape_channels
                dims.extend(range(base, base + shape_channels))
    return dims


def build_blocks_from_cfg(cfg: AgentConfig) -> List[List[int]]:
    """Build block partition aligned to grid when metadata is available."""
    D = int(getattr(cfg, "D", 0))
    B = max(1, int(getattr(cfg, "B", getattr(cfg, "n_blocks", 1))))
    side = int(getattr(cfg, "grid_side", 0))
    grid_channels = int(getattr(cfg, "grid_channels", 0))
    color_channels = int(getattr(cfg, "grid_color_channels", 0))
    shape_channels = int(getattr(cfg, "grid_shape_channels", 0))
    base_dim = int(getattr(cfg, "grid_base_dim", 0))
    periph_blocks = max(0, min(int(getattr(cfg, "periph_blocks", 0)), B))
    key = (D, B, side, grid_channels, color_channels, shape_channels, base_dim, periph_blocks)
    cached = _BLOCK_CACHE.get(key)
    if cached is not None:
        return [dims.copy() for dims in cached]

    if D <= 0:
        return [[]]

    if base_dim <= 0:
        base_dim = D

    spatial_blocks = B - periph_blocks
    grid_cells = side * side
    can_use_grid = (
        side > 0
        and grid_channels > 0
        and base_dim == grid_cells * grid_channels
        and spatial_blocks > 0
    )
    if can_use_grid:
        if color_channels <= 0 and shape_channels <= 0:
            color_channels = grid_channels
            shape_channels = 0
        if color_channels + shape_channels != grid_channels:
            can_use_grid = False

    blocks: List[List[int]] = []
    if can_use_grid:
        block_grid_side = int(round(math.sqrt(spatial_blocks)))
        if block_grid_side * block_grid_side != spatial_blocks:
            can_use_grid = False
        elif side % block_grid_side != 0:
            can_use_grid = False
        else:
            block_cells = side // block_grid_side
            for by in range(block_grid_side):
                for bx in range(block_grid_side):
                    dims = _grid_block_dims(
                        side=side,
                        color_channels=color_channels,
                        shape_channels=shape_channels,
                        block_cells_y=block_cells,
                        block_cells_x=block_cells,
                        block_y=by,
                        block_x=bx,
                    )
                    blocks.append(dims)

    if not can_use_grid:
        base = D // B
        rem = D % B
        if B > D:
            B = D
            base = 1
            rem = 0
        start = 0
        for b in range(B):
            size = base + (1 if b < rem else 0)
            end = start + size
            blocks.append(list(range(start, end)))
            start = end

    if can_use_grid and periph_blocks > 0 and base_dim < D:
        periph_dim = D - base_dim
        periph_base = periph_dim // periph_blocks
        periph_rem = periph_dim % periph_blocks
        start = base_dim
        for idx in range(periph_blocks):
            size = periph_base + (1 if idx < periph_rem else 0)
            end = start + size
            blocks.append(list(range(start, end)))
            start = end

    _BLOCK_CACHE[key] = [dims.copy() for dims in blocks]
    return [dims.copy() for dims in blocks]


def init_fovea_state(cfg: AgentConfig, *, block_costs: Optional[Sequence[float]] = None) -> FoveaState:
    """Initialize fovea state arrays."""
    initial_age = int(getattr(cfg, "initial_block_age", 0))
    initial_age = max(0, initial_age)
    B = int(cfg.B)
    if B <= 0:
        cost_arr = np.zeros(0, dtype=float)
    else:
        if block_costs is None:
            cost_arr = np.ones(B, dtype=float)
        else:
            cost_arr = np.asarray(block_costs, dtype=float)
            if cost_arr.size != B:
                cost_arr = np.resize(cost_arr, (B,))
        cost_arr = np.maximum(cost_arr, 1e-6)

    return FoveaState(
        block_residual=np.zeros(B, dtype=float),
        block_age=np.full(B, initial_age, dtype=int),
        block_uncertainty=np.zeros(B, dtype=float),
        block_costs=cost_arr,
        routing_scores=np.zeros(B, dtype=float),
        block_disagreement=np.zeros(B, dtype=float),
        block_innovation=np.zeros(B, dtype=float),
        block_periph_demand=np.zeros(B, dtype=float),
        block_confidence=np.ones(B, dtype=float),
        routing_last_t=-1,
        current_blocks=set(),
        coverage_cursor=0,
    )


# =============================================================================
# Block geometry (A16.1)
# =============================================================================


def block_slices(cfg: AgentConfig) -> List[Tuple[int, int]]:
    """Return per-block [start, end) slices matching DoF-aligned partition.

    The partition distributes the remainder so that the first (D % B) blocks
    have size ceil(D/B), and the remainder have size floor(D/B).
    """
    blocks = build_blocks_from_cfg(cfg)
    slices: List[Tuple[int, int]] = []
    for dims in blocks:
        if not dims:
            slices.append((0, 0))
            continue
        slices.append((min(dims), max(dims) + 1))
    return slices


def block_of_dim(k: int, cfg: AgentConfig) -> int:
    """Map dimension index k to its block id under the A16.1 partition."""
    D = int(getattr(cfg, "D", 0))
    B = max(1, int(getattr(cfg, "B", getattr(cfg, "n_blocks", 1))))
    side = int(getattr(cfg, "grid_side", 0))
    grid_channels = int(getattr(cfg, "grid_channels", 0))
    color_channels = int(getattr(cfg, "grid_color_channels", 0))
    shape_channels = int(getattr(cfg, "grid_shape_channels", 0))
    base_dim = int(getattr(cfg, "grid_base_dim", 0))
    periph_blocks = max(0, min(int(getattr(cfg, "periph_blocks", 0)), B))
    key = (D, B, side, grid_channels, color_channels, shape_channels, base_dim, periph_blocks)
    lookup = _BLOCK_LOOKUP_CACHE.get(key)
    if lookup is None:
        lookup = {}
        blocks = build_blocks_from_cfg(cfg)
        for block_id, dims in enumerate(blocks):
            for dim in dims:
                lookup[int(dim)] = int(block_id)
        _BLOCK_LOOKUP_CACHE[key] = lookup
    return int(lookup.get(int(k), max(0, len(build_blocks_from_cfg(cfg)) - 1)))


def dims_for_block(block_id: int, cfg: AgentConfig) -> List[int]:
    """Return dimension range for a block id."""
    blocks = build_blocks_from_cfg(cfg)
    b = int(block_id)
    if b < 0 or b >= len(blocks):
        return []
    return list(blocks[b])


# =============================================================================
# Tracking updates (A16.2)
# =============================================================================


def update_fovea_tracking(
    fovea: FoveaState,
    buf: ObservationBuffer,
    cfg: AgentConfig,
    *,
    abs_error: Optional[np.ndarray] = None,
    observed_dims: Optional[Set[int]] = None,
) -> None:
    """Update per-block age/residual statistics.

    Args:
        fovea: FoveaState mutated in place.
        buf: ObservationBuffer (only used for shape safety here).
        cfg: AgentConfig.
        abs_error: Optional |x(t) - x̂(t|t-1)| vector (shape D). If provided,
            residual updates are computed for blocks with observed dims.
        observed_dims: Optional set of observed dimension indices O_t.

    Semantics:
        - Ages increment for all blocks, then reset for blocks observed at t.
        - Residuals update via EMA from the mean absolute prediction error over
          observed dims within each block.
    """
    B = int(cfg.B)
    if B <= 0:
        return

    ages = np.asarray(fovea.block_age, dtype=float)
    ages += 1.0

    D = int(cfg.D)
    err = np.asarray(abs_error, dtype=float) if abs_error is not None else np.zeros(D, dtype=float)
    if err.shape[0] != D:
        err = np.resize(err, (D,))

    obs_blocks: dict[int, list[int]] = {}
    if observed_dims:
        for k in observed_dims:
            kk = int(k)
            if 0 <= kk < D:
                b = block_of_dim(kk, cfg)
                obs_blocks.setdefault(b, []).append(kk)
    observed_block_ids = {int(b) for b in obs_blocks.keys()}

    beta = float(getattr(cfg, "fovea_residual_ema", 0.10))
    beta = max(0.0, min(1.0, beta))

    # Update per-block confidence
    conf = np.asarray(getattr(fovea, "block_confidence", np.ones(B, dtype=float)), dtype=float)
    if conf.shape[0] != B:
        conf = np.resize(conf, (B,))
    conf = np.clip(conf, 0.0, 1.0)
    beta_up = max(0.0, min(1.0, float(getattr(cfg, "fovea_confidence_beta_up", 0.50))))
    beta_down = max(0.0, min(1.0, float(getattr(cfg, "fovea_confidence_beta_down", 0.01))))
    observed_mask = np.zeros(B, dtype=bool)
    for b in observed_block_ids:
        if 0 <= b < B:
            observed_mask[b] = True
    not_observed_mask = ~observed_mask
    conf[observed_mask] = (1.0 - beta_up) * conf[observed_mask] + beta_up
    conf[not_observed_mask] = (1.0 - beta_down) * conf[not_observed_mask]
    fovea.block_confidence = np.clip(conf, 0.0, 1.0)

    ages[observed_mask] = 0.0
    fovea.block_age = ages

    if abs_error is None or observed_dims is None or not observed_dims:
        return

    for b, ks in obs_blocks.items():
        if not ks:
            continue
        r = float(np.mean(np.abs(err[ks])))
        old = float(fovea.block_residual[b])
        fovea.block_residual[b] = (1.0 - beta) * old + beta * r
        fovea.block_age[b] = 0.0


# =============================================================================
# Selection (A16.3–A16.4)
# =============================================================================


def select_fovea(fovea: FoveaState, cfg: AgentConfig) -> list[int]:
    """Select blocks via budgeted greedy_cov scoring with hard coverage cap."""
    B = int(cfg.B)
    if B <= 0:
        fovea.current_blocks = set()
        return []

    budget_units = float(getattr(cfg, "fovea_blocks_per_step", 0))
    budget_units = max(0.0, budget_units)

    alpha_cov = float(getattr(cfg, "alpha_cov", 0.10))
    residual_only = bool(getattr(cfg, "fovea_residual_only", False))
    ages = np.asarray(fovea.block_age, dtype=float)
    residuals = np.asarray(fovea.block_residual, dtype=float)
    conf = np.asarray(getattr(fovea, "block_confidence", np.ones(B)), dtype=float)
    if conf.shape[0] != B:
        conf = np.resize(conf, (B,))
    conf = np.clip(conf, 0.0, 1.0)

    use_age = bool(getattr(cfg, "fovea_use_age", True))
    if use_age and budget_units <= 1.0 and ages.size:
        # Deterministic sweep for single-block budgets to avoid getting stuck.
        if B > 1 and getattr(fovea, "current_blocks", None):
            seed = min(int(b) for b in fovea.current_blocks)
            chosen_single = [int((seed + 1) % B)]
        else:
            best_age = int(np.argmax(ages))
            chosen_single = [best_age]
        _log_geom_event(
            "select_fovea_age_deterministic",
            {
                "routing_last_t": getattr(fovea, "routing_last_t", -1),
                "budget_units": budget_units,
                "use_age": use_age,
                "chosen": chosen_single,
            },
        )
        return chosen_single
    if residual_only or not use_age:
        scores = residuals.copy()
    else:
        scores = residuals + alpha_cov * np.log1p(np.maximum(0.0, ages))

    w_conf = float(getattr(cfg, "fovea_confidence_weight", 0.0))
    if w_conf != 0.0:
        scores = scores + w_conf * (1.0 - conf)

    uncertainty_weight = float(getattr(cfg, "fovea_uncertainty_weight", 0.0))
    if uncertainty_weight != 0.0:
        uncertainties = np.asarray(getattr(fovea, "block_uncertainty", np.zeros(B)), dtype=float)
        if uncertainties.shape[0] != B:
            uncertainties = np.resize(uncertainties, (B,))
        scores = scores + uncertainty_weight * uncertainties

    disagreement_weight = float(getattr(cfg, "fovea_disagreement_weight", 0.0))
    if disagreement_weight != 0.0:
        disagreements = np.asarray(getattr(fovea, "block_disagreement", np.zeros(B)), dtype=float)
        if disagreements.shape[0] != B:
            disagreements = np.resize(disagreements, (B,))
        scores = scores + disagreement_weight * disagreements

    innovation_weight = float(getattr(cfg, "fovea_innovation_weight", 0.0))
    if innovation_weight != 0.0:
        innovations = np.asarray(getattr(fovea, "block_innovation", np.zeros(B)), dtype=float)
        if innovations.shape[0] != B:
            innovations = np.resize(innovations, (B,))
        scores = scores + innovation_weight * innovations

    periph_weight = float(getattr(cfg, "fovea_periph_demand_weight", 0.0))
    if periph_weight != 0.0:
        periph_demand = np.asarray(getattr(fovea, "block_periph_demand", np.zeros(B)), dtype=float)
        if periph_demand.shape[0] != B:
            periph_demand = np.resize(periph_demand, (B,))
        scores = scores + periph_weight * periph_demand

    routing_weight = float(getattr(cfg, "fovea_routing_weight", 0.0))
    if routing_weight > 0.0:
        routing = np.asarray(getattr(fovea, "routing_scores", np.zeros(B)), dtype=float)
        if routing.shape[0] != B:
            routing = np.resize(routing, (B,))
        scores = scores + routing_weight * routing

    costs = np.asarray(getattr(fovea, "block_costs", np.ones(B)), dtype=float)
    if costs.shape[0] != B:
        costs = np.resize(costs, (B,))
    costs = np.maximum(costs, 1e-6)
    mean_cost = float(np.mean(costs)) if costs.size else 1.0
    mean_cost = max(mean_cost, 1e-6)
    norm_costs = costs / mean_cost

    periph_blocks = max(0, min(int(getattr(cfg, "periph_blocks", 0)), B))
    periph_ids = [int(b) for b in range(max(0, B - periph_blocks), B)]
    coverage_score_tol = float(getattr(cfg, "coverage_score_tol", 0.0))
    coverage_score_threshold = float(getattr(cfg, "coverage_score_threshold", float("-inf")))
    coverage_step = max(1, int(getattr(cfg, "coverage_cursor_step", 1)))
    coverage_cursor = int(getattr(fovea, "coverage_cursor", 0)) if B > 0 else 0

    G = int(getattr(cfg, "coverage_cap_G", 0))
    mandatory: list[int] = []
    if G > 0:
        mandatory = [b for b in range(B) if float(ages[b]) >= float(G)]
        mandatory = sorted(mandatory, key=lambda b: float(ages[b]), reverse=True)

    chosen: list[int] = []
    used: Set[int] = set()
    budget_remaining = budget_units
    for b in periph_ids:
        if b in used:
            continue
        chosen.append(int(b))
        used.add(int(b))
        budget_remaining -= float(norm_costs[b])
        if budget_remaining <= 0.0:
            budget_remaining = 0.0
            break
    budget_remaining = max(0.0, budget_remaining)
    for b in mandatory:
        if b in used:
            continue
        chosen.append(int(b))
        used.add(int(b))
        budget_remaining -= float(norm_costs[b])
        if budget_remaining <= 0.0:
            budget_remaining = 0.0
            break
    budget_remaining = max(0.0, budget_remaining)

    remaining = [b for b in range(B) if b not in used]
    scores_arr = np.asarray(scores, dtype=float) if scores is not None else np.zeros(0, dtype=float)
    detail_base = {
        "routing_last_t": getattr(fovea, "routing_last_t", -1),
        "scores_max": float(np.max(scores_arr)) if scores_arr.size else None,
        "scores_min": float(np.min(scores_arr)) if scores_arr.size else None,
        "periph_blocks": periph_blocks,
        "periph_ids_sample": periph_ids[: min(len(periph_ids), 5)],
        "coverage_score_tol": coverage_score_tol,
        "coverage_score_threshold": coverage_score_threshold,
        "coverage_step": coverage_step,
        "coverage_cursor": coverage_cursor,
        "periph_weight": periph_weight,
        "routing_weight": routing_weight,
        "alpha_cov": alpha_cov,
        "use_age": use_age,
        "budget_units": budget_units,
        "budget_remaining_start": float(budget_units),
        "remaining_len": len(remaining),
    }

    def _log_selection_result(
        final_chosen: list[int],
        *,
        coverage_trigger_flag: bool,
        final_budget_remaining: float,
    ) -> list[int]:
        info = dict(detail_base)
        info.update(
            {
                "coverage_triggered": coverage_trigger_flag,
                "chosen_len": len(final_chosen),
                "chosen_sample": final_chosen[: min(len(final_chosen), 10)],
                "budget_remaining": float(final_budget_remaining),
                "budget_used": float(max(0.0, budget_units - final_budget_remaining)),
            }
        )
        _log_geom_event("select_fovea", info)
        return final_chosen

    if not remaining or budget_remaining <= 0.0:
        return _log_selection_result(
            chosen,
            coverage_trigger_flag=False,
            final_budget_remaining=budget_remaining,
        )

    def _coverage_select_blocks(
        remaining_set: Set[int],
        budget_remain: float,
        cursor: int,
    ) -> tuple[list[int], float]:
        selection: list[int] = []
        idx = cursor % B if B > 0 else 0
        forced_pick_done = False
        attempts = 0
        while remaining_set and (budget_remain > 0.0 or not forced_pick_done) and attempts < 2 * B:
            if idx in remaining_set:
                cost_unit = float(norm_costs[idx])
                pick = False
                if cost_unit <= budget_remain:
                    pick = True
                    budget_remain = max(0.0, budget_remain - cost_unit)
                elif not forced_pick_done:
                    pick = True
                    forced_pick_done = True
                    budget_remain = 0.0
                if pick:
                    selection.append(int(idx))
                    remaining_set.remove(idx)
            idx = (idx + coverage_step) % B
            attempts += 1
        fovea.coverage_cursor = idx
        return selection, budget_remain

    coverage_trigger = False
    if remaining and budget_remaining > 0.0:
        rem_scores = [float(scores[b]) for b in remaining]
        if rem_scores:
            max_score = max(rem_scores)
            min_score = min(rem_scores)
            if max_score <= coverage_score_threshold or (max_score - min_score) <= coverage_score_tol:
                coverage_trigger = True
    if coverage_trigger:
        remaining_set = set(remaining)
        coverage_selected, budget_remaining = _coverage_select_blocks(
            remaining_set,
            budget_remaining,
            coverage_cursor,
        )
        for b in coverage_selected:
            if b in used:
                continue
            chosen.append(int(b))
            used.add(int(b))
        return _log_selection_result(
            chosen,
            coverage_trigger_flag=True,
            final_budget_remaining=budget_remaining,
        )

    ratio_order = sorted(
        remaining,
        key=lambda b: float(scores[b]) / float(costs[b]),
        reverse=True,
    )
    min_norm_cost = float(np.min([norm_costs[b] for b in remaining])) if remaining else 0.0
    force_pick_allowed = bool(remaining and budget_remaining > 0.0 and min_norm_cost > budget_remaining)
    forced_pick_done = False

    for b in ratio_order:
        if b in used:
            continue
        cost_unit = float(norm_costs[b])
        if cost_unit <= budget_remaining:
            chosen.append(int(b))
            used.add(int(b))
            budget_remaining = max(0.0, budget_remaining - cost_unit)
        elif force_pick_allowed and not forced_pick_done:
            chosen.append(int(b))
            used.add(int(b))
            forced_pick_done = True
            budget_remaining = 0.0
        if budget_remaining <= 0.0:
            break

    return _log_selection_result(
        chosen,
        coverage_trigger_flag=False,
        final_budget_remaining=budget_remaining,
    )


def make_observation_set(blocks: Iterable[int], cfg: AgentConfig) -> set[int]:
    """Convert selected blocks into an observation set O_t (A16.5)."""
    O: set[int] = set()
    for b in blocks:
        for k in dims_for_block(int(b), cfg):
            O.add(int(k))
    return O




def update_fovea_routing_scores(
    fovea: FoveaState,
    x_prev: np.ndarray,
    cfg: AgentConfig,
    *,
    t: int | None = None,
) -> None:
    """Update routing scores from peripheral bins (optional, off by default)."""
    if t is not None and int(getattr(fovea, "routing_last_t", -1)) == int(t):
        return
    periph_blocks = int(getattr(cfg, "periph_blocks", 0))
    periph_bins = int(getattr(cfg, "periph_bins", 0))
    routing_weight = float(getattr(cfg, "fovea_routing_weight", 0.0))
    if periph_blocks <= 0 or periph_bins <= 0 or routing_weight <= 0.0:
        return

    D = int(getattr(cfg, "D", len(x_prev)))
    B = int(getattr(cfg, "B", 0))
    if B <= 0:
        return
    block_size = D // B if B > 0 else 0
    if block_size <= 0:
        return
    periph_dim = periph_blocks * block_size
    base_dim = D - periph_dim
    if periph_dim <= 0 or base_dim <= 0:
        return
    if x_prev.shape[0] < base_dim + periph_dim:
        return

    periph_channels = int(getattr(cfg, "periph_channels", getattr(cfg, "grid_channels", 1)))
    periph_channels = max(1, periph_channels)
    n_bins = periph_bins * periph_bins
    needed = n_bins * periph_channels
    if needed <= 0:
        return

    periph_vec = np.asarray(x_prev[base_dim : base_dim + periph_dim], dtype=float).reshape(-1)
    if periph_vec.size < needed:
        return
    periph_vec = periph_vec[:needed]
    if periph_channels > 1:
        bin_scores = periph_vec.reshape(n_bins, periph_channels).sum(axis=1)
    else:
        bin_scores = periph_vec.reshape(-1)[:n_bins]
    if not np.any(bin_scores > 0.0):
        return

    bin_idx = int(np.argmax(bin_scores))
    side = int(getattr(cfg, "grid_side", 0))
    if side <= 0:
        return
    bin_x = bin_idx % periph_bins
    bin_y = bin_idx // periph_bins
    cell_x = int((bin_x + 0.5) * side / periph_bins)
    cell_y = int((bin_y + 0.5) * side / periph_bins)
    cell_x = max(0, min(side - 1, cell_x))
    cell_y = max(0, min(side - 1, cell_y))
    base_channels = int(getattr(cfg, "grid_channels", 1))
    base_channels = max(1, base_channels)
    cell_index = cell_y * side + cell_x
    dim_idx = cell_index * base_channels
    if dim_idx < 0 or dim_idx >= base_dim:
        return
    target_block = int(dim_idx // block_size)
    max_block = int(B - periph_blocks)
    if target_block < 0 or target_block >= max_block:
        return

    routing = np.zeros(B, dtype=float)
    routing[target_block] = 1.0
    ema = float(getattr(cfg, "fovea_routing_ema", 0.0))
    ema = max(0.0, min(1.0, ema))
    prev = np.asarray(getattr(fovea, "routing_scores", np.zeros(B)), dtype=float)
    if prev.shape[0] != B:
        prev = np.resize(prev, (B,))
    fovea.routing_scores = ema * prev + (1.0 - ema) * routing
    if t is not None:
        fovea.routing_last_t = int(t)
