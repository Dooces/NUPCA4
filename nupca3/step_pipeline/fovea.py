"""
Fovea-focused helpers for block selection and budget enforcement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Set, Tuple

import numpy as np

from ..config import AgentConfig
from ..geometry.fovea import block_of_dim, select_fovea, update_fovea_routing_scores
from ..types import AgentState, FoveaState

from .logging import _log_fovea_event
from .observations import _cfg_D, _peripheral_dim_set
from .transport import _apply_pending_transport_disagreement


@dataclass
class FoveaSignals:
    """Container for per-step block signals that the fovea consumes at the next selection."""

    block_disagreement: np.ndarray | None = None
    block_innovation: np.ndarray | None = None
    block_periph_demand: np.ndarray | None = None
    block_uncertainty: np.ndarray | None = None


def _coerce_block_array(values: np.ndarray | Iterable[float] | None, B: int) -> np.ndarray | None:
    if values is None:
        return None
    arr = np.asarray(values, dtype=float).reshape(-1)
    if B != arr.shape[0]:
        arr = np.resize(arr, (max(0, B),))
    return arr


def _apply_signals_to_fovea(fovea: AgentState | FoveaState, cfg: AgentConfig, signals: FoveaSignals) -> None:
    B = max(0, int(getattr(cfg, "B", 0)))
    if B <= 0:
        return
    arrays = {
        "block_disagreement": signals.block_disagreement,
        "block_innovation": signals.block_innovation,
        "block_periph_demand": signals.block_periph_demand,
        "block_uncertainty": signals.block_uncertainty,
    }
    for attr, value in arrays.items():
        if value is None:
            continue
        coerced = _coerce_block_array(value, B)
        if coerced is None:
            continue
        setattr(fovea, attr, coerced)


def _peripheral_block_ids(cfg: AgentConfig) -> List[int]:
    """Return the configured peripheral block IDs (low-priority tail of the budget)."""
    periph_blocks = max(0, int(getattr(cfg, "periph_blocks", 0)))
    B = max(0, int(getattr(cfg, "B", 0)))
    if periph_blocks <= 0 or B <= 0:
        return []
    periph_start = max(0, B - periph_blocks)
    return list(range(periph_start, B))


def _enforce_peripheral_blocks(
    blocks: Iterable[int],
    cfg: AgentConfig,
    periph_candidates: List[int] | None = None,
) -> Tuple[List[int], List[int]]:
    """Ensure the peripheral summaries stay in the fovea budget."""
    periph_candidates = periph_candidates if periph_candidates is not None else _peripheral_block_ids(cfg)
    budget = len(blocks)
    if budget <= 0 or not periph_candidates:
        return list(blocks), []
    reserved = min(len(periph_candidates), budget)
    periph_ids = periph_candidates[-reserved:]
    seen: Set[int] = set()
    non_periph: List[int] = []
    for b in blocks:
        b_int = int(b)
        if b_int in periph_ids or b_int in seen:
            continue
        non_periph.append(b_int)
        seen.add(b_int)
    kept = max(0, budget - reserved)
    non_periph = non_periph[:kept]
    return periph_ids + non_periph, periph_ids


def _select_motion_probe_blocks(
    observed_dims: Set[int],
    cfg: AgentConfig,
    budget: int,
) -> List[int]:
    """Pick up to `budget` blocks that cover the previous observation footprint."""
    if budget <= 0:
        return []
    B = max(0, int(getattr(cfg, "B", 0)))
    if B <= 0:
        return []

    probe_blocks: List[int] = []
    seen: Set[int] = set()
    periph_block_count = max(0, int(getattr(cfg, "periph_blocks", 0)))
    fine_block_limit = max(0, B - periph_block_count)

    def _add_block(block_id: int) -> None:
        if len(probe_blocks) >= budget:
            return
        if 0 <= block_id < B and block_id not in seen:
            probe_blocks.append(block_id)
            seen.add(block_id)

    if observed_dims:
        for dim in sorted(int(k) for k in observed_dims):
            if len(probe_blocks) >= budget:
                break
            block_id = block_of_dim(int(dim), cfg)
            if fine_block_limit and block_id >= fine_block_limit:
                continue
            _add_block(block_id)

    fallback_blocks = list(range(fine_block_limit if fine_block_limit > 0 else 0))
    if len(fallback_blocks) < budget:
        fallback_blocks += [b for b in range(fine_block_limit, B) if b not in fallback_blocks]

    for block_id in fallback_blocks:
        if len(probe_blocks) >= budget:
            break
        _add_block(block_id)

    if len(probe_blocks) < budget:
        extra = [b for b in range(B) if b not in seen]
        for block_id in extra:
            if len(probe_blocks) >= budget:
                break
            _add_block(block_id)

    return probe_blocks


def _enforce_motion_probe_blocks(
    blocks: Iterable[int],
    cfg: AgentConfig,
    probe_blocks: List[int],
) -> Tuple[List[int], int]:
    """Ensure the requested motion probe blocks appear early in the selection, inserting them if needed."""
    block_list = [int(b) for b in blocks]
    if not block_list or not probe_blocks:
        return block_list, 0

    B = max(0, int(getattr(cfg, "B", 0)))
    if B <= 0:
        return block_list, 0

    periph_set = set(_peripheral_block_ids(cfg))
    periph_prefix: List[int] = []
    non_periph: List[int] = []
    for block_id in block_list:
        if block_id in periph_set:
            periph_prefix.append(block_id)
        else:
            non_periph.append(block_id)

    valid_probe_blocks: List[int] = []
    seen_probe: Set[int] = set()
    for block_id in probe_blocks:
        bid = int(block_id)
        if 0 <= bid < B and bid not in seen_probe:
            valid_probe_blocks.append(bid)
            seen_probe.add(bid)
    if not valid_probe_blocks:
        return block_list, 0

    fine_probe_blocks = [b for b in valid_probe_blocks if b not in periph_set]
    capacity = len(non_periph)
    ordered_non_periph: List[int] = []
    added: Set[int] = set()

    for block_id in fine_probe_blocks:
        if len(ordered_non_periph) >= capacity:
            break
        if block_id in added:
            continue
        ordered_non_periph.append(block_id)
        added.add(block_id)

    for block_id in non_periph:
        if len(ordered_non_periph) >= capacity:
            break
        if block_id in added:
            continue
        ordered_non_periph.append(block_id)
        added.add(block_id)

    final_blocks = periph_prefix + ordered_non_periph
    if len(final_blocks) < len(block_list):
        for block_id in block_list:
            if len(final_blocks) >= len(block_list):
                break
            if block_id in final_blocks:
                continue
            final_blocks.append(block_id)
    final_blocks = final_blocks[: len(block_list)]
    final_set = set(final_blocks)
    used_count = sum(1 for block_id in valid_probe_blocks if block_id in final_set)
    return final_blocks, used_count




def _update_grid_routing_from_full(
    state: AgentState,
    cfg: AgentConfig,
    *,
    periph_full: Iterable[float] | np.ndarray,
    budget_units: float,
) -> int | None:
    """Update routing_scores from a full grid frame (agent-side periphery).

    This is used only when we are in grid mode (grid_width/grid_height provided) and the
    fovea shape is a disk. It produces a *single connected* disk bias around the
    center-of-mass of non-zero cells (semantic occupancy), then applies EMA smoothing.

    Returns the chosen center block id (spatial), or None if no non-zero mass.
    """
    grid_w = int(getattr(cfg, "grid_width", 0) or 0)
    grid_h = int(getattr(cfg, "grid_height", 0) or 0)
    B = int(getattr(cfg, "B", 0) or 0)
    if grid_w <= 0 or grid_h <= 0 or B <= 0:
        return None
    spatial_blocks = B - int(getattr(cfg, "periph_blocks", 0) or 0)
    if spatial_blocks != grid_w * grid_h:
        return None

    full_arr = np.asarray(periph_full, dtype=float).reshape(-1)
    need = grid_w * grid_h
    if full_arr.size < need:
        return None

    grid = full_arr[:need].reshape((grid_h, grid_w))
    nz = grid != 0.0
    if not np.any(nz):
        return None

    ys, xs = np.where(nz)
    cx = int(np.clip(np.round(xs.mean()), 0, grid_w - 1))
    cy = int(np.clip(np.round(ys.mean()), 0, grid_h - 1))
    center = int(cy * grid_w + cx)

    # Build a disk bias (radius derived from budget_units; clamp to at least 1).
    radius = max(1, int(np.round(np.sqrt(max(1.0, float(budget_units)) / np.pi))))
    bias = np.zeros(B, dtype=float)
    for dy in range(-radius, radius + 1):
        yy = cy + dy
        if yy < 0 or yy >= grid_h:
            continue
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy > radius * radius:
                continue
            xx = cx + dx
            if xx < 0 or xx >= grid_w:
                continue
            bid = int(yy * grid_w + xx)
            bias[bid] = 1.0

    ema = float(getattr(cfg, "fovea_routing_ema", 0.5))
    ema = float(np.clip(ema, 0.0, 1.0))
    prev = np.asarray(getattr(state.fovea, "routing_scores", np.zeros(B)), dtype=float)
    if prev.size != B:
        prev = np.zeros(B, dtype=float)
    state.fovea.routing_scores = (1.0 - ema) * prev + ema * bias
    state.fovea.routing_last_t = int(getattr(state, "t", 0))
    return center
def _plan_fovea_selection(
    state: AgentState,
    cfg: AgentConfig,
    *,
    periph_full: Iterable[float] | np.ndarray | None = None,
    prev_observed_dims: Set[int] | None = None,
    value_of_compute: float = 0.0,
) -> dict[str, Any]:
    """Compute the upcoming fovea blocks using A16.3 and cache the result."""
    D = _cfg_D(state, cfg)
    x_prev = np.asarray(getattr(state.buffer, "x_last", np.zeros(D)), dtype=float).reshape(-1)
    periph_dims = _peripheral_dim_set(D, cfg)
    budget_units = float(getattr(cfg, "fovea_blocks_per_step", 0))
    fovea_shape = str(getattr(cfg, "fovea_shape", "") or "").lower()
    grid_w = int(getattr(cfg, "grid_width", 0) or 0)
    grid_h = int(getattr(cfg, "grid_height", 0) or 0)
    circle_mode = (fovea_shape == "circle" and grid_w > 0 and grid_h > 0)
    routing_vec = x_prev
    if periph_dims and periph_full is not None:
        routing_vec = x_prev.copy()
        full_arr = np.asarray(periph_full, dtype=float).reshape(-1)
        if full_arr.size < D:
            full_arr = np.resize(full_arr, (D,))
        for dim in periph_dims:
            if 0 <= dim < full_arr.size:
                routing_vec[int(dim)] = float(full_arr[int(dim)])
    update_fovea_routing_scores(state.fovea, routing_vec, cfg, t=int(getattr(state, "t", 0)))
    value_of_compute_scaled = min(max(0.0, float(value_of_compute)), 1.0)
    budget_boost = float(getattr(cfg, "value_of_compute_budget_scale", 0.5))
    budget_units *= 1.0 + budget_boost * value_of_compute_scaled
    grid_center = None
    if circle_mode and periph_full is not None and float(getattr(cfg, "fovea_routing_weight", 0.0)) > 0.0:
        grid_center = _update_grid_routing_from_full(state, cfg, periph_full=periph_full, budget_units=budget_units)
    B = max(0, int(getattr(cfg, "B", 0)))
    routing_scores = np.asarray(
        getattr(state.fovea, "routing_scores", np.zeros(B)), dtype=float
    )
    routing_scores = _apply_pending_transport_disagreement(state, cfg, routing=routing_scores)
    state.fovea.routing_scores = routing_scores
    blocks_t = select_fovea(state.fovea, cfg) or []
    G = int(getattr(cfg, "coverage_cap_G", 0))
    ages_now = np.asarray(
        getattr(state.fovea, "block_age", np.zeros(int(getattr(cfg, "B", 0)))), dtype=float
    )
    budget = max(1, int(getattr(cfg, "fovea_blocks_per_step", 0)))
    if (not circle_mode) and G > 0 and ages_now.size:
        mandatory = [int(b) for b in range(int(getattr(cfg, "B", 0))) if float(ages_now[b]) >= float(G)]
        if mandatory and not set(mandatory).intersection(set(blocks_t or [])):
            mandatory = sorted(mandatory, key=lambda b: float(ages_now[b]), reverse=True)
            blocks_t = mandatory[: min(len(mandatory), budget)]
    periph_candidates = _peripheral_block_ids(cfg)
    blocks_t, forced_periph_blocks = _enforce_peripheral_blocks(blocks_t or [], cfg, periph_candidates)
    motion_probe_budget = max(0, int(getattr(cfg, "motion_probe_blocks", 0)))
    if circle_mode:
        motion_probe_budget = 0
    if budget <= 1:
        motion_probe_budget = 0
    prev_dims = prev_observed_dims if prev_observed_dims is not None else set(
        getattr(state.buffer, "observed_dims", set()) or set()
    )
    motion_probe_blocks = _select_motion_probe_blocks(prev_dims, cfg, motion_probe_budget)
    blocks_t, motion_probe_blocks_used = _enforce_motion_probe_blocks(blocks_t or [], cfg, motion_probe_blocks)
    pending = {
        "blocks": [int(b) for b in blocks_t],
        "forced_periph_blocks": forced_periph_blocks,
        "motion_probe_blocks": motion_probe_blocks_used,
    }
    state.pending_fovea_selection = pending
    state.fovea.current_blocks = set(int(b) for b in pending["blocks"])
    residuals = np.asarray(getattr(state.fovea, "block_residual", np.zeros(0)), dtype=float)
    log_details = {
        "t": int(getattr(state, "t", -1)),
        "budget_units": budget_units,
        "coverage_cap_G": G,
        "periph_block_request": len(periph_candidates),
        "periph_blocks": int(getattr(cfg, "periph_blocks", 0)),
        "motion_probe_budget": motion_probe_budget,
        "motion_probe_selected": motion_probe_blocks_used,
        "prev_observed_dims": len(prev_dims),
        "blocks_selected": len(blocks_t),
        "blocks_sample": pending["blocks"][:10],
        "forced_periph_blocks": forced_periph_blocks[:10],
        "ages_max": float(np.max(ages_now)) if ages_now.size else None,
        "residuals_max": float(np.max(residuals)) if residuals.size else None,
        "periph_full_provided": periph_full is not None,
        "grid_routing_center": int(grid_center) if grid_center is not None else None,
    }
    _log_fovea_event("plan_fovea_selection", log_details)
    return pending


def apply_signals_and_select(
    state: AgentState,
    cfg: AgentConfig,
    *,
    signals: FoveaSignals | None = None,
    periph_full: Iterable[float] | np.ndarray | None = None,
    prev_observed_dims: Set[int] | None = None,
    value_of_compute: float = 0.0,
) -> dict[str, Any]:
    """Merge pending block signals, refresh routing bias, and plan the next fovea selection."""
    available_signals = signals if signals is not None else getattr(state, "pending_fovea_signals", None)
    if available_signals is None:
        available_signals = FoveaSignals()
    else:
        state.pending_fovea_signals = None
    _apply_signals_to_fovea(state.fovea, cfg, available_signals)
    return _plan_fovea_selection(
        state,
        cfg,
        periph_full=periph_full,
        prev_observed_dims=prev_observed_dims,
        value_of_compute=value_of_compute,
    )
