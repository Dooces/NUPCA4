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

from typing import Iterable, Optional, Set, Tuple, List, Sequence

import numpy as np

from ..config import AgentConfig
from ..types import FoveaState, ObservationBuffer


def init_fovea_state(cfg: AgentConfig, *, block_costs: Optional[Sequence[float]] = None) -> FoveaState:
    """Initialize fovea state arrays."""
    cap = int(getattr(cfg, "coverage_cap_G", 0))
    initial_age = int(getattr(cfg, "initial_block_age", cap + 1))
    if initial_age <= 0:
        initial_age = 1
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
        routing_last_t=-1,
        current_blocks=set(),
    )


# =============================================================================
# Block geometry (A16.1)
# =============================================================================


def block_slices(cfg: AgentConfig) -> List[Tuple[int, int]]:
    """Return per-block [start, end) slices matching DoF-aligned partition.

    The partition distributes the remainder so that the first (D % B) blocks
    have size ceil(D/B), and the remainder have size floor(D/B).
    """
    D = int(cfg.D)
    B = int(cfg.B)
    if B <= 0:
        return []

    base = D // B
    rem = D % B
    slices: List[Tuple[int, int]] = []
    start = 0
    for b in range(B):
        size = base + (1 if b < rem else 0)
        end = min(D, start + max(1, size))
        slices.append((start, end))
        start = end
    # Ensure last slice ends at D.
    if slices:
        s0, _ = slices[-1]
        slices[-1] = (s0, D)
    return slices


def block_of_dim(k: int, cfg: AgentConfig) -> int:
    """Map dimension index k to its block id under the A16.1 partition."""
    k = int(k)
    if k < 0:
        return 0
    slices = block_slices(cfg)
    for b, (s, e) in enumerate(slices):
        if s <= k < e:
            return int(b)
    return int(max(0, len(slices) - 1))


def dims_for_block(block_id: int, cfg: AgentConfig) -> range:
    """Return dimension range for a block id."""
    slices = block_slices(cfg)
    b = int(block_id)
    b = max(0, min(len(slices) - 1, b))
    s, e = slices[b]
    return range(s, e)


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

    # Age all blocks (relevance-weighted: low-residual blocks age slowly).
    ages = np.asarray(fovea.block_age, dtype=float)
    resid = np.asarray(getattr(fovea, "block_residual", np.zeros(B)), dtype=float)
    min_inc = float(getattr(cfg, "fovea_age_min_inc", 0.10))
    resid_scale = float(getattr(cfg, "fovea_age_resid_scale", 0.05))
    resid_thresh = float(getattr(cfg, "fovea_age_resid_thresh", 0.0))
    min_inc = max(0.0, min(1.0, min_inc))
    resid_scale = max(resid_scale, 1e-9)
    resid_norm = resid / (resid + resid_scale)
    gate = (resid >= resid_thresh).astype(float)
    ages += (min_inc + (1.0 - min_inc) * resid_norm) * gate
    fovea.block_age = ages

    if abs_error is None or observed_dims is None or not observed_dims:
        return

    D = int(cfg.D)
    err = np.asarray(abs_error, dtype=float)
    if err.shape[0] != D:
        # Best-effort: pad/trim.
        err = np.resize(err, (D,))

    beta = float(getattr(cfg, "fovea_residual_ema", 0.10))
    beta = max(0.0, min(1.0, beta))

    # Collect observed dims per block.
    obs_blocks: dict[int, list[int]] = {}
    for k in observed_dims:
        kk = int(k)
        if 0 <= kk < D:
            b = block_of_dim(kk, cfg)
            obs_blocks.setdefault(b, []).append(kk)

    for b, ks in obs_blocks.items():
        if not ks:
            continue
        # Mean absolute error over observed dims in this block.
        r = float(np.mean(np.abs(err[ks])))
        old = float(fovea.block_residual[b])
        fovea.block_residual[b] = (1.0 - beta) * old + beta * r
        # Reset age for observed blocks.
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

    use_age = bool(getattr(cfg, "fovea_use_age", True))
    if residual_only or not use_age:
        scores = residuals.copy()
    else:
        scores = residuals + alpha_cov * np.log1p(np.maximum(0.0, ages))

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
    if not remaining or budget_remaining <= 0.0:
        return chosen
    if not remaining:
        return chosen

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

    return chosen


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
