"""
nupca3/step_pipeline.py

Axiom-faithful step orchestrator for the 2025-12-21 snapshot.

This implementation is written to be safe to drop into the snapshot:
- No imports of symbols that do not exist in-tree.
- commit_gate() is called with the correct contract (c: list[float]).
- rest_permitted_prev is computed from the actual A14.6 predicate.
- Exports `step_pipeline` as expected by nupca3/agent.py.

Running/compiling is secondary for this repo, but this file avoids the
hard integration failures that otherwise prevent import or immediate execution.
"""

from __future__ import annotations

# =============================================================================
# Debugging
# =============================================================================
# Single switch for step-by-step tracing. When False, the file behaves identically
# except for added comments/docstrings.
DEBUG = True
_LOG_START_TIME = None
_LAST_LOG_TIME = None
def _dbg(msg: str, *, state=None) -> None:
    """Print a debug line when DEBUG is enabled. Does not affect control flow."""
    if not DEBUG:
        return
    bracket = msg.find("]")
    tail = msg[bracket + 1 :] if bracket >= 0 else msg
    if "=" not in tail:
        return
    global _LAST_LOG_TIME, _LOG_START_TIME
    try:
        t = getattr(state, 't', None) if state is not None else None
    except Exception:
        t = None
    now = time.monotonic()
    if _LOG_START_TIME is None:
        _LOG_START_TIME = now
    if _LAST_LOG_TIME is None:
        delta = 0.0
    else:
        delta = now - _LAST_LOG_TIME
    _LAST_LOG_TIME = now
    elapsed = now - _LOG_START_TIME
    prefix = (
        f"[step_pipeline t={int(t):6d} dt={delta:7.3f}s elapsed={elapsed:7.3f}s] "
        if t is not None
        else f"[step_pipeline dt={delta:7.3f}s elapsed={elapsed:7.3f}s] "
    )
    print(prefix + str(msg))

from collections import deque, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Set, Tuple

import math
import numpy as np
import time

from .config import AgentConfig
from .types import (
    AgentState,
    EnvObs,
    LearningCache,
    StepTrace,
    Margins,
    PersistentResidualState,
    FootprintResidualStats,
    TransitionRecord,
    WorldHypothesis,
)

from .control.budget import compute_budget_and_horizon
from .control.commitment import commit_gate, select_action
from .control.edit_control import freeze_predicate, permit_param_updates

from .diagnostics.metrics import compute_feel_proxy

from .dynamics.margin_dynamics import HardState, step_hard_dynamics

from .edits.rest_processor import process_struct_queue, RestProcessingResult
from .edits.proposals import propose_structural_edits

from .geometry.fovea import (
    block_of_dim,
    block_slices,
    dims_for_block,
    make_observation_set,
    select_fovea,
    update_fovea_tracking,
    update_fovea_routing_scores,
)
from .geometry.streams import (
    apply_transport,
    coarse_bin_count,
    compute_grid_shift,
    compute_transport_shift,
    extract_coarse,
    grid_cell_mass,
    periph_block_size,
)
from .geometry.binding import build_binding_maps, select_best_binding, select_best_binding_by_fit

from .memory.completion import complete
from .memory.expert import sgd_update
from .memory.fusion import fuse_predictions
from .memory.rollout import rollout_and_confidence
from .memory.salience import compute_salience, get_stress_signals, infer_node_band_level
from .memory.working_set import select_working_set

from .state.baselines import (
    commit_tilde_prev,
    normalize_margins,
    update_baselines,
)
from .state.macrostate import evolve_macrostate, rest_permitted
from .state.margins import compute_arousal, compute_margins, compute_stress, init_margins
from .state.stability import update_stability_metrics


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

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

    Inputs:
      cue: sparse observation map {dim -> value}
      O_req: requested observation dims (from fovea blocks)
      D: global dimensionality for bounds

    Output:
      filtered cue dict (subset of cue)
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
        innovation = np.asarray(getattr(state.fovea, "block_innovation", np.zeros(int(getattr(cfg, "B", 0)))), dtype=float)
        incumbents = getattr(state, "incumbents", {})
        for block_id, nodes_in_block in incumbents.items():
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


def _update_block_uncertainty(state: AgentState, sigma_diag: np.ndarray, cfg: AgentConfig) -> None:
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


@dataclass
class TransportCandidate:
    """Transport candidate summary produced during evidence scoring."""

    delta: Tuple[int, int]
    shifted: np.ndarray
    mae: float
    overlap: int
    score: float
    ascii_mismatch: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TransportTransform:
    """Specification of a transform to evaluate (delta + rotation)."""

    delta: Tuple[int, int]
    rotation: int = 0
    label: str = "grid"


def _normalize_rotation_steps(cfg: AgentConfig) -> List[int]:
    steps = tuple(getattr(cfg, "transport_rotation_steps", (0,)))
    normalized: List[int] = []
    for step in steps:
        rot = int(step) % 4
        if rot not in normalized:
            normalized.append(rot)
    if 0 not in normalized:
        normalized.insert(0, 0)
    return normalized


def _build_transport_candidate_set(
    cfg: AgentConfig,
    env_shift: Tuple[int, int] | None,
    coarse_shift: Tuple[int, int] | None,
    state: AgentState | None,
) -> List[TransportTransform]:
    """Return a deterministic list of transport transformations to consider."""
    radius = max(1, int(getattr(cfg, "transport_search_radius", 1)))
    span = radius
    deltas: Set[Tuple[int, int]] = set()
    label_map: Dict[Tuple[int, int], str] = {}
    for dy in range(-span, span + 1):
        for dx in range(-span, span + 1):
            key = (dx, dy)
            deltas.add(key)
            label_map.setdefault(key, "grid")
    deltas.add((0, 0))
    label_map.setdefault((0, 0), "grid")

    if env_shift is not None:
        key = (int(env_shift[0]), int(env_shift[1]))
        deltas.add(key)
        label_map[key] = "env_shift"
    if coarse_shift is not None:
        key = (int(coarse_shift[0]), int(coarse_shift[1]))
        deltas.add(key)
        label_map.setdefault(key, "coarse_shift")

    offset_size = max(0, int(getattr(cfg, "transport_offset_history_size", 0)))
    offsets = []
    if state is not None and offset_size > 0:
        offsets = list(getattr(state, "transport_offsets", []))
    if offsets:
        limit = radius + max(0, int(getattr(cfg, "transport_offset_radius", 0)))
        base_deltas = list(deltas)
        for offset in offsets:
            offset_key = (int(offset[0]), int(offset[1]))
            deltas.add(offset_key)
            label_map[offset_key] = "offset"
            for base in base_deltas:
                combined = (base[0] + offset_key[0], base[1] + offset_key[1])
                if abs(combined[0]) <= limit and abs(combined[1]) <= limit:
                    deltas.add(combined)
                    label_map.setdefault(combined, "offset_combo")
            base_deltas = list(deltas)

    rotations = _normalize_rotation_steps(cfg) if bool(getattr(cfg, "transport_rotation_enabled", False)) else [0]
    transforms: List[TransportTransform] = []
    sorted_deltas = sorted(deltas, key=_transport_delta_priority)
    for delta in sorted_deltas:
        label = label_map.get(delta, "grid")
        for rot in rotations:
            transforms.append(TransportTransform(delta=delta, rotation=rot, label=label))
    return transforms


_TRANSPORT_UNINFORMATIVE_SCORE = -1e6


def _transport_candidate_score(
    mae: float,
    overlap: int,
    min_overlap: int,
    overlap_penalty: float,
    overlap_bonus: float,
) -> float:
    """Convert MAE/overlap into an evidence score (higher is better)."""
    if overlap < min_overlap or math.isinf(mae):
        return _TRANSPORT_UNINFORMATIVE_SCORE
    overlap_scale = math.log1p(float(overlap))
    normalized_mae = mae / max(overlap_scale, 1e-6)
    score = -normalized_mae
    if overlap_penalty > 0.0:
        score -= overlap_penalty / float(max(overlap, 1))
    if overlap_bonus != 0.0:
        score += overlap_bonus * math.log1p(float(overlap))
    return score


def _transport_delta_priority(delta: Tuple[int, int]) -> Tuple[int, int, int, int]:
    """Zero-bias ordering for deltas: prefer (0,0), then cardinal, then diagonals."""
    dx, dy = delta
    if dx == 0 and dy == 0:
        tier = 0
    elif dx == 0 or dy == 0:
        tier = 1
    else:
        tier = 2
    span = abs(dx) + abs(dy)
    return (tier, span, abs(dx), abs(dy))


def _logsumexp(values: Iterable[float]) -> float:
    """Numerically stable log-sum-exp for transport beliefs."""
    vals = list(values)
    if not vals:
        return float("-inf")
    max_val = max(vals)
    if math.isinf(max_val):
        return max_val
    total = 0.0
    for v in vals:
        total += math.exp(v - max_val)
    return max_val + math.log(total)


def _grid_occupancy_mask(vec: np.ndarray, cfg: AgentConfig) -> np.ndarray:
    """Return boolean occupancy mask per cell derived from a fine-grained state vector."""
    side = max(0, int(getattr(cfg, "grid_side", 0)))
    if side <= 0:
        return np.zeros(0, dtype=bool)
    cell_count = side * side
    if cell_count <= 0:
        return np.zeros(0, dtype=bool)
    channels = max(1, int(getattr(cfg, "grid_channels", 1)))
    arr = np.asarray(vec, dtype=float).reshape(-1)
    base_dim = max(0, int(getattr(cfg, "grid_base_dim", 0)))
    if base_dim <= 0:
        base_dim = int(getattr(cfg, "D", 0))
    needed = cell_count * channels
    data = np.zeros(needed, dtype=float)
    copy_len = min(needed, arr.size, base_dim)
    if copy_len > 0:
        data[:copy_len] = arr[:copy_len]
    data = data.reshape(cell_count, channels)
    return np.any(np.abs(data) > 1e-6, axis=1)
def _select_transport_delta(
    x_prev: np.ndarray,
    obs_idx: np.ndarray,
    obs_vals: np.ndarray,
    cfg: AgentConfig,
    state: AgentState,
    env_shift: Tuple[int, int] | None,
    coarse_shift: Tuple[int, int] | None,
    true_env_vec: np.ndarray | None = None,
    true_delta: Tuple[int, int] | None = None,
    force_true_delta: bool = False,
) -> Tuple[
    Tuple[int, int],
    np.ndarray,
    TransportCandidate | None,
    TransportCandidate | None,
    float,
    float,
    float,
    str,
    List[TransportCandidate],
    bool,
    float,
    float,
    bool,
    int,
]:
    """Evaluate candidate shifts, maintain belief, and return the chosen delta plus diagnostics."""
    D = _cfg_D(state, cfg)
    base_dim = max(0, int(D) - periph_block_size(cfg))
    candidate_transforms = _build_transport_candidate_set(cfg, env_shift, coarse_shift, state)

    prev_obs_dims = getattr(state.buffer, "observed_dims", set()) or set()
    prev_obs_mask = np.zeros(D, dtype=float)
    prev_buffer = np.asarray(getattr(state.buffer, "x_last", np.zeros(D)), dtype=float).reshape(-1)
    if prev_buffer.shape[0] != D:
        prev_buffer = np.resize(prev_buffer, (D,))
    prev_obs_values = np.zeros(D, dtype=float)
    for dim in prev_obs_dims:
        idx = int(dim)
        if 0 <= idx < D:
            prev_obs_mask[idx] = 1.0
            prev_obs_values[idx] = prev_buffer[idx]

    obs_mask = np.zeros(D, dtype=bool)
    obs_values = np.zeros(D, dtype=float)
    if obs_idx.size:
        obs_mask[obs_idx] = True
        obs_values[obs_idx] = obs_vals

    min_overlap = max(0, int(getattr(cfg, "transport_min_overlap", 1)))
    overlap_penalty = float(getattr(cfg, "transport_overlap_penalty", 0.0))
    overlap_bonus = float(getattr(cfg, "transport_overlap_bonus", 0.0))
    ascii_penalty = float(getattr(cfg, "transport_ascii_penalty", 0.0))
    env_occ_mask: np.ndarray | None = None
    if true_env_vec is not None:
        env_occ_mask = _grid_occupancy_mask(true_env_vec, cfg)
    candidate_infos: List[TransportCandidate] = []

    for transform in candidate_transforms:
        delta = tuple(int(v) for v in transform.delta)
        rotation = int(transform.rotation) % 4
        shifted = apply_transport(x_prev, delta, cfg, rotation=rotation)
        shifted_prev_obs = apply_transport(prev_obs_values, delta, cfg, rotation=rotation)
        mask_shifted = apply_transport(prev_obs_mask, delta, cfg, rotation=rotation)
        overlap_mask = obs_mask & (mask_shifted > 0.5)
        overlap_idx = np.nonzero(overlap_mask)[0]
        overlap = int(overlap_idx.size)
        if overlap:
            diff = obs_values[overlap_idx] - shifted_prev_obs[overlap_idx]
            mae = float(np.mean(np.abs(diff)))
        else:
            mae = float("inf")
        score = _transport_candidate_score(
            mae,
            overlap,
            min_overlap,
            overlap_penalty,
            overlap_bonus,
        )
        ascii_mismatch = 0
        if env_occ_mask is not None and env_occ_mask.size:
            cand_occ = _grid_occupancy_mask(shifted, cfg)
            if cand_occ.size > 0:
                min_len = min(env_occ_mask.size, cand_occ.size)
                mismatch = int(np.count_nonzero(env_occ_mask[:min_len] != cand_occ[:min_len]))
                mismatch += abs(env_occ_mask.size - cand_occ.size)
            else:
                mismatch = int(env_occ_mask.size)
            ascii_mismatch = mismatch
            if ascii_penalty > 0.0 and mismatch > 0:
                score -= ascii_penalty * float(mismatch)
        bias_key = (delta[0], delta[1], rotation)
        bias_bonus = float(getattr(state, "transport_biases", {}).get(bias_key, 0.0))
        weight = float(getattr(cfg, "transport_bias_weight", 0.0))
        score += bias_bonus * weight
        metadata = {
            "rotation": rotation,
            "transform_source": transform.label,
        }
        candidate_infos.append(
            TransportCandidate(
                delta=delta,
                shifted=shifted,
                mae=mae,
                overlap=overlap,
                score=score,
                ascii_mismatch=ascii_mismatch,
                metadata=metadata,
            )
        )

    beliefs = getattr(state, "transport_beliefs", {}) or {}
    decay = float(getattr(cfg, "transport_belief_decay", 0.5))
    inertia = float(getattr(cfg, "transport_inertia_weight", 0.0))
    last_delta = tuple(getattr(state, "transport_last_delta", (0, 0)))
    updated_beliefs: Dict[Tuple[int, int], float] = {}
    for cand in candidate_infos:
        prev = float(beliefs.get(cand.delta, 0.0))
        inertia_bonus = inertia if cand.delta == last_delta else 0.0
        updated_beliefs[cand.delta] = decay * prev + (1.0 - decay) * cand.score + inertia_bonus

    scores = [cand.score for cand in candidate_infos]
    if scores:
        best_score = max(scores)
        worst_score = min(scores)
    else:
        best_score = 0.0
        worst_score = 0.0
    score_spread = float(best_score - worst_score)
    sorted_by_score = sorted(candidate_infos, key=lambda c: c.score, reverse=True)
    best_raw_candidate = sorted_by_score[0] if sorted_by_score else None
    runner_raw_candidate = sorted_by_score[1] if len(sorted_by_score) > 1 else None
    runner_score = runner_raw_candidate.score if runner_raw_candidate is not None else best_score
    score_margin = float(best_score - runner_score)
    tie_threshold = float(getattr(cfg, "transport_tie_threshold", 1e-4))
    tie_flag = score_margin < tie_threshold
    uninformative_threshold = float(
        getattr(cfg, "transport_uninformative_score", _TRANSPORT_UNINFORMATIVE_SCORE)
    )
    best_informative = best_raw_candidate is not None and best_raw_candidate.score > uninformative_threshold
    evidence_margin = float(getattr(cfg, "transport_evidence_margin", 0.02))
    null_evidence = (not best_informative) or (score_margin < evidence_margin)

    if null_evidence:
        updated_beliefs = {delta: 0.0 for delta in updated_beliefs}
    state.transport_beliefs = updated_beliefs

    if not updated_beliefs:
        default = TransportCandidate(delta=(0, 0), shifted=x_prev.copy(), mae=float("inf"), overlap=0, score=_TRANSPORT_UNINFORMATIVE_SCORE)
        return (
            (0, 0),
            x_prev.copy(),
            default,
            None,
            float("inf"),
            0.0,
            0.0,
            "fallback",
            [default],
            bool(null_evidence),
            0.0,
            float(score_spread),
            bool(tie_flag),
            int(default.overlap),
        )

    log_z = _logsumexp(updated_beliefs.values())
    probs: Dict[Tuple[int, int], float] = {}
    if math.isinf(log_z):
        uniform = 1.0 / float(len(updated_beliefs))
        probs = {delta: uniform for delta in updated_beliefs}
    else:
        for delta, score in updated_beliefs.items():
            probs[delta] = math.exp(score - log_z)

    if not probs:
        probs = {(0, 0): 1.0}

    posterior_entropy = 0.0
    for p in probs.values():
        if p > 0.0:
            posterior_entropy -= p * math.log(p)

    delta_to_candidate = {cand.delta: cand for cand in candidate_infos}
    tie_prob_threshold = float(getattr(cfg, "transport_tie_probability_threshold", 1e-4))
    max_prob = max(probs.values()) if probs else 0.0
    tie_candidates = [delta for delta, prob in probs.items() if prob >= max_prob - tie_prob_threshold]
    if not tie_candidates:
        tie_candidates = list(probs.keys())

    best_delta = (0, 0) if null_evidence else min(tie_candidates, key=_transport_delta_priority)

    zero_candidate = delta_to_candidate.get((0, 0))
    if zero_candidate is None:
        zero_candidate = TransportCandidate(
            delta=(0, 0),
            shifted=x_prev.copy(),
            mae=float("inf"),
            overlap=0,
            score=_TRANSPORT_UNINFORMATIVE_SCORE,
        )

    best_candidate = delta_to_candidate.get(best_delta, zero_candidate)
    if force_true_delta and true_delta is not None:
        forced_delta = tuple(int(v) for v in true_delta)
        forced_candidate = delta_to_candidate.get(forced_delta)
        if forced_candidate is not None:
            best_delta = forced_delta
            best_candidate = forced_candidate
    x_prev_post = best_candidate.shifted.copy() if best_candidate is not None else x_prev.copy()

    sorted_probs = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    runner_delta = next((delta for delta, _ in sorted_probs if delta != best_delta), None)
    runner_candidate = delta_to_candidate.get(runner_delta)

    best_prob = probs.get(best_delta, 0.0)
    runner_prob = probs.get(runner_delta, 0.0)
    best_score_log = updated_beliefs.get(best_delta, float("-inf"))
    runner_score_log = updated_beliefs.get(runner_delta, best_score_log)
    log_margin = best_score_log - runner_score_log
    prob_diff = best_prob - runner_prob

    source = "buffer_infer"
    if env_shift is not None:
        source = "grid"

    best_overlap = int(best_candidate.overlap) if best_candidate is not None else 0
    return (
        best_delta,
        x_prev_post,
        best_candidate,
        runner_candidate,
        log_margin,
        best_prob,
        prob_diff,
        source,
        candidate_infos,
        bool(null_evidence),
        float(posterior_entropy),
        float(score_spread),
        bool(tie_flag),
        best_overlap,
    )


def _compute_transport_disagreement_blocks(
    best: TransportCandidate | None,
    runner: TransportCandidate | None,
    cfg: AgentConfig,
) -> Dict[int, float]:
    """Compute normalized block disagreement weights for ambiguous candidates."""
    if best is None or runner is None:
        return {}
    D = int(getattr(cfg, "D", 0))
    base_dim = max(0, D - periph_block_size(cfg))
    if base_dim <= 0:
        return {}
    diff = np.abs(best.shifted[:base_dim] - runner.shifted[:base_dim])
    if not np.any(diff):
        return {}
    block_scores: Dict[int, float] = defaultdict(float)
    for idx, val in enumerate(diff):
        if val <= 0.0:
            continue
        block_id = block_of_dim(idx, cfg)
        block_scores[block_id] += float(val)
    total = sum(block_scores.values())
    if total <= 0.0:
        return {}
    normalized = {int(b): float(v / total) for b, v in block_scores.items() if 0 <= b < int(getattr(cfg, "B", 0))}
    return normalized


def _apply_pending_transport_disagreement(state: AgentState, cfg: AgentConfig) -> None:
    """Apply stored block disagreement scores (from previous step) to routing scores."""
    weight = float(getattr(cfg, "transport_disambiguation_weight", 1.0))
    if weight <= 0.0:
        state.transport_disagreement_scores = {}
        state.transport_disagreement_margin = float("inf")
        return
    margin = float(getattr(state, "transport_disagreement_margin", float("inf")))
    threshold = float(getattr(cfg, "transport_confidence_margin", 0.25))
    scores = getattr(state, "transport_disagreement_scores", {})
    if not scores or margin >= threshold:
        state.transport_disagreement_scores = {}
        state.transport_disagreement_margin = float("inf")
        return
    B = int(getattr(cfg, "B", 0))
    if B <= 0:
        state.transport_disagreement_scores = {}
        state.transport_disagreement_margin = float("inf")
        return
    routing = np.asarray(getattr(state.fovea, "routing_scores", np.zeros(B)), dtype=float)
    if routing.shape[0] != B:
        routing = np.resize(routing, (B,))
    total = sum(scores.values())
    if total <= 0.0:
        state.transport_disagreement_scores = {}
        state.transport_disagreement_margin = float("inf")
        return
    for block_id, val in scores.items():
        if 0 <= block_id < routing.shape[0]:
            routing[block_id] += weight * float(val)
    state.fovea.routing_scores = routing
    state.transport_disagreement_scores = {}
    state.transport_disagreement_margin = float("inf")


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
    full = getattr(env_obs, "x_full", None)
    dims = periph_dims if periph_dims is not None else set()
    obs_vec = np.zeros(max(0, D), dtype=float)
    if dims and full is not None:
        full_arr = np.asarray(full, dtype=float).reshape(-1)
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
        if getattr(env_obs, "x_full", None) is not None and periph_obs.size:
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


def _normalize_world_weights(raw_weights: List[float]) -> List[float]:
    """Normalize a list of raw world scores into a probability simplex."""
    if not raw_weights:
        return []
    arr = np.asarray(raw_weights, dtype=float)
    finite_mask = np.isfinite(arr)
    total = float(np.sum(arr[finite_mask])) if finite_mask.any() else 0.0
    if not np.isfinite(total) or total <= 0.0:
        fallback = 1.0 / float(len(arr))
        return [fallback] * len(arr)
    normalized: List[float] = []
    for val in arr:
        if not np.isfinite(val):
            normalized.append(0.0)
        else:
            normalized.append(float(val / total))
    return normalized


def _support_window_mae(prior_a: np.ndarray, prior_b: np.ndarray, support_dims: Set[int]) -> float:
    if not support_dims:
        return float("inf")
    dims = sorted(int(d) for d in support_dims if isinstance(d, int) and d >= 0)
    if not dims:
        return float("inf")
    max_dim = dims[-1]
    arr_a = np.asarray(prior_a, dtype=float).reshape(-1)
    arr_b = np.asarray(prior_b, dtype=float).reshape(-1)
    if arr_a.size <= max_dim:
        arr_a = np.resize(arr_a, (max_dim + 1,))
    if arr_b.size <= max_dim:
        arr_b = np.resize(arr_b, (max_dim + 1,))
    idxs = np.array(dims, dtype=int)
    diff = arr_a[idxs] - arr_b[idxs]
    finite = np.isfinite(diff)
    if not finite.any():
        return float("inf")
    return float(np.mean(np.abs(diff[finite])))


def _merge_world_group(group: List[WorldHypothesis], D: int) -> WorldHypothesis:
    weights = [max(0.0, float(w.weight)) for w in group]
    total = float(sum(weights))
    if total <= 0.0:
        weights = [1.0] * len(weights)
        total = float(len(weights))
    normalized = [float(w) / total for w in weights]
    prior_accum = np.zeros(D, dtype=float)
    post_accum = np.zeros(D, dtype=float)
    sigma_accum = np.zeros(D, dtype=float)
    best_idx = int(np.argmax(normalized))
    best_world = group[best_idx]
    for share, world in zip(normalized, group):
        prior = np.asarray(world.x_prior, dtype=float).reshape(-1)
        post = np.asarray(world.x_post, dtype=float).reshape(-1)
        sigma = np.asarray(world.sigma_prior_diag, dtype=float).reshape(-1)
        if prior.size != D:
            prior = np.resize(prior, (D,))
        if post.size != D:
            post = np.resize(post, (D,))
        if sigma.size != D:
            sigma = np.resize(sigma, (D,))
        prior_accum += share * prior
        post_accum += share * post
        sigma_accum += share * sigma
    metadata = dict(getattr(best_world, "metadata", {}))
    metadata["merged_count"] = len(group)
    metadata["merged_from"] = [tuple(w.delta) for w in group]
    merged = WorldHypothesis(
        delta=tuple(best_world.delta),
        x_prior=prior_accum,
        x_post=post_accum,
        sigma_prior_diag=sigma_accum,
        weight=total,
        prior_mae=float(best_world.prior_mae),
        likelihood=float(best_world.likelihood),
        metadata=metadata,
    )
    return merged


def _clone_world(world: WorldHypothesis, D: int) -> WorldHypothesis:
    prior = np.asarray(world.x_prior, dtype=float).reshape(-1)
    post = np.asarray(world.x_post, dtype=float).reshape(-1)
    sigma = np.asarray(world.sigma_prior_diag, dtype=float).reshape(-1)
    if prior.size != D:
        prior = np.resize(prior, (D,))
    if post.size != D:
        post = np.resize(post, (D,))
    if sigma.size != D:
        sigma = np.resize(sigma, (D,))
    metadata = dict(getattr(world, "metadata", {}))
    metadata["cloned_from"] = tuple(world.delta)
    return WorldHypothesis(
        delta=tuple(world.delta),
        x_prior=prior.copy(),
        x_post=post.copy(),
        sigma_prior_diag=sigma.copy(),
        weight=float(world.weight),
        prior_mae=float(world.prior_mae),
        likelihood=float(world.likelihood),
        metadata=metadata,
    )


def _consolidate_world_hypotheses(state: AgentState, cfg: AgentConfig, worlds: List[WorldHypothesis], D: int) -> List[WorldHypothesis]:
    if not worlds:
        return []
    support_dims = _support_window_union(state)
    eps = max(0.0, float(getattr(cfg, "multi_world_merge_eps", 1e-3)))
    grouped: List[WorldHypothesis] = []
    used = [False] * len(worlds)
    for i, world in enumerate(worlds):
        if used[i]:
            continue
        duplicates = [world]
        used[i] = True
        for j in range(i + 1, len(worlds)):
            if used[j]:
                continue
            distance = _support_window_mae(world.x_prior, worlds[j].x_prior, support_dims)
            if distance <= eps:
                duplicates.append(worlds[j])
                used[j] = True
        merged = _merge_world_group(duplicates, D)
        grouped.append(merged)
    k = max(1, int(getattr(cfg, "multi_world_K", 1)))
    grouped.sort(key=lambda w: float(w.weight), reverse=True)
    while grouped and len(grouped) < k:
        grouped.append(_clone_world(grouped[0], D))
    if len(grouped) > k:
        grouped = grouped[:k]
    raw_weights = [float(w.weight) for w in grouped]
    normalized = _normalize_world_weights(raw_weights)
    for world, wgt in zip(grouped, normalized):
        world.weight = wgt
    return grouped


def _update_block_signals(state: AgentState, cfg: AgentConfig, worlds: List[WorldHypothesis], D: int) -> None:
    B = int(getattr(cfg, "B", 0))
    fovea = state.fovea
    if B <= 0:
        zeros = np.zeros(0, dtype=float)
        fovea.block_disagreement = zeros
        fovea.block_innovation = zeros
        fovea.block_periph_demand = zeros
        return
    ranges = block_slices(cfg)
    values = np.zeros((len(worlds), B), dtype=float) if worlds else np.zeros((0, B), dtype=float)
    deltas = np.zeros((len(worlds), B), dtype=float) if worlds else np.zeros((0, B), dtype=float)
    for i, world in enumerate(worlds):
        prior = np.asarray(world.x_prior, dtype=float).reshape(-1)
        post = np.asarray(world.x_post, dtype=float).reshape(-1)
        if prior.size != D:
            prior = np.resize(prior, (D,))
        if post.size != D:
            post = np.resize(post, (D,))
        delta = post - prior
        for b, (start, end) in enumerate(ranges):
            if start >= end:
                continue
            slice_obj = slice(start, min(end, D))
            block_vals = prior[slice_obj]
            if block_vals.size:
                values[i, b] = float(np.mean(block_vals))
            block_delta = np.abs(delta[slice_obj])
            if block_delta.size:
                deltas[i, b] = float(np.mean(block_delta))
    if worlds:
        weights = np.array([max(0.0, float(w.weight)) for w in worlds], dtype=float)
        if not np.isfinite(weights).any():
            weights = np.ones_like(weights)
        total_weight = float(np.sum(weights))
        if total_weight <= 0.0:
            weights = np.ones_like(weights)
            total_weight = float(np.sum(weights))
        normalized = weights / total_weight
        mean_vals = np.sum(normalized[:, None] * values, axis=0)
        disagreement = np.sum(normalized[:, None] * (values - mean_vals) ** 2, axis=0)
        innovation = np.sum(normalized[:, None] * deltas, axis=0)
    else:
        disagreement = np.zeros(B, dtype=float)
        innovation = np.zeros(B, dtype=float)
    age = np.asarray(getattr(fovea, "block_age", np.zeros(B)), dtype=float)
    if age.size != B:
        age = np.resize(age, (B,))
    periph_value = float(np.nan_to_num(state.peripheral_residual, nan=0.0))
    weight = 1.0 / (1.0 + np.maximum(age, 0.0))
    weight_sum = float(np.sum(weight))
    if weight_sum <= 0.0:
        normalized_weight = np.ones_like(weight) / float(weight.size) if weight.size else weight
    else:
        normalized_weight = weight / weight_sum
    periph_demand = periph_value * normalized_weight
    fovea.block_disagreement = np.asarray(disagreement, dtype=float)
    fovea.block_innovation = np.asarray(innovation, dtype=float)
    fovea.block_periph_demand = np.asarray(periph_demand, dtype=float)
def _select_multi_world_candidates(
    candidate_infos: List[TransportCandidate],
    best_candidate: TransportCandidate | None,
    k: int,
) -> List[TransportCandidate]:
    """Return up to k distinct candidate deltas, keeping the chosen best first."""
    k = max(1, int(k))
    selection: List[TransportCandidate] = []
    seen_deltas: Set[Tuple[int, int]] = set()
    if best_candidate is not None:
        key = tuple(best_candidate.delta)
        selection.append(best_candidate)
        seen_deltas.add(key)
    sorted_by_score = sorted(candidate_infos, key=lambda cand: cand.score, reverse=True)
    for cand in sorted_by_score:
        if len(selection) >= k:
            break
        key = tuple(cand.delta)
        if key in seen_deltas:
            continue
        selection.append(cand)
        seen_deltas.add(key)
    if not selection and best_candidate is not None:
        selection.append(best_candidate)
    return selection[:k]


def _build_world_hypotheses(
    state: AgentState,
    cfg: AgentConfig,
    D: int,
    cue: Dict[int, float],
    obs_idx: np.ndarray,
    obs_vals: np.ndarray,
    candidate_infos: List[TransportCandidate],
    best_candidate: TransportCandidate | None,
    main_prior: np.ndarray,
    main_post: np.ndarray,
    sigma_prior_diag: np.ndarray,
) -> List[WorldHypothesis]:
    """Create/update the multi-world list keyed by transport candidates."""
    k = max(1, int(getattr(cfg, "multi_world_K", 1)))
    lambda_param = float(getattr(cfg, "multi_world_lambda", 1.0))
    selected = _select_multi_world_candidates(candidate_infos, best_candidate, k)
    if not selected and best_candidate is not None:
        selected = [best_candidate]
    prev_weights = {
        tuple(h.delta): float(h.weight)
        for h in getattr(state, "world_hypotheses", []) or []
        if hasattr(h, "delta")
    }
    worlds: List[WorldHypothesis] = []
    for cand in selected:
        delta = tuple(cand.delta)
        prior_candidate = np.asarray(cand.shifted, dtype=float).reshape(-1)
        if prior_candidate.shape[0] != D:
            prior_candidate = np.resize(prior_candidate, (D,))
        if cand is best_candidate:
            post = np.asarray(main_post, dtype=float).reshape(-1)
            prior_full = np.asarray(main_prior, dtype=float).reshape(-1)
            sigma_diag = np.asarray(sigma_prior_diag, dtype=float).reshape(-1)
            if post.shape[0] != D:
                post = np.resize(post, (D,))
            if prior_full.shape[0] != D:
                prior_full = np.resize(prior_full, (D,))
            if sigma_diag.shape[0] != D:
                sigma_diag = np.resize(sigma_diag, (D,))
        else:
            prior_full = prior_candidate.copy()
            post, sigma_arr, prior_full = complete(
                cue,
                mode="perception",
                state=state,
                cfg=cfg,
                predicted_prior_t=prior_full,
                predicted_sigma_diag=np.full(D, np.inf, dtype=float),
            )
            post = np.asarray(post, dtype=float).reshape(-1)
            if post.shape[0] != D:
                post = np.resize(post, (D,))
            sigma_arr = np.asarray(sigma_arr, dtype=float)
            if sigma_arr.ndim == 2 and sigma_arr.shape[0] == sigma_arr.shape[1]:
                diag = np.diag(sigma_arr).copy()
            else:
                diag = sigma_arr.reshape(-1).copy()
            if diag.shape[0] != D:
                diag = np.resize(diag, (D,))
            sigma_diag = diag
            prior_full = np.asarray(prior_full, dtype=float).reshape(-1)
            if prior_full.shape[0] != D:
                prior_full = np.resize(prior_full, (D,))
        prior_mae = _prior_obs_mae(obs_idx, obs_vals, prior_full)
        likelihood = 1.0
        if obs_idx.size and np.isfinite(prior_mae):
            likelihood = math.exp(-lambda_param * prior_mae)
        prev_weight = prev_weights.get(delta, 1.0)
        raw_weight = prev_weight * likelihood
        metadata = {
            "score": float(getattr(cand, "score", 0.0)),
            "prev_weight": prev_weight,
        }
        worlds.append(
            WorldHypothesis(
                delta=delta,
                x_prior=prior_full,
                x_post=post,
                sigma_prior_diag=sigma_diag,
                weight=raw_weight,
                prior_mae=prior_mae,
                likelihood=likelihood,
                metadata=metadata,
            )
        )
    consolidated = _consolidate_world_hypotheses(state, cfg, worlds, D)
    state.world_hypotheses = consolidated
    return consolidated


def _update_transport_learning_state(
    state: AgentState,
    cfg: AgentConfig,
    best_candidate: TransportCandidate | None,
    chosen_shift: Tuple[int, int],
    true_delta: Tuple[int, int] | None,
) -> None:
    """Decay transport biases and insert new wins/offsets when available."""
    biases = dict(getattr(state, "transport_biases", {}) or {})
    decay = float(getattr(cfg, "transport_bias_decay", 1.0))
    if decay != 1.0:
        for key, value in list(biases.items()):
            decayed = float(value) * decay
            if decayed > 0.0:
                biases[key] = decayed
            else:
                biases.pop(key, None)

    if best_candidate is not None and true_delta is not None:
        target = (int(true_delta[0]), int(true_delta[1]))
        shift_vals = (int(chosen_shift[0]), int(chosen_shift[1]))
        rot = int(best_candidate.metadata.get("rotation", 0)) % 4
        bias_key = (int(best_candidate.delta[0]), int(best_candidate.delta[1]), rot)
        if target == shift_vals:
            increment = float(getattr(cfg, "transport_bias_increment", 0.0))
            if increment > 0.0:
                biases[bias_key] = biases.get(bias_key, 0.0) + increment
        offset_history = max(0, int(getattr(cfg, "transport_offset_history_size", 0)))
        radius = max(0, int(getattr(cfg, "transport_offset_radius", 0)))
        if offset_history > 0:
            offset = (target[0] - shift_vals[0], target[1] - shift_vals[1])
            if offset != (0, 0) and abs(offset[0]) <= radius and abs(offset[1]) <= radius:
                offsets = list(getattr(state, "transport_offsets", []))
                offsets = [o for o in offsets if o != offset]
                offsets.insert(0, offset)
                state.transport_offsets = offsets[:offset_history]

    max_entries = max(1, int(getattr(cfg, "transport_bias_max_entries", 1)))
    if len(biases) > max_entries:
        sorted_items = sorted(biases.items(), key=lambda item: float(item[1]), reverse=True)
        keep_keys = {key for key, _ in sorted_items[:max_entries]}
        biases = {key: value for key, value in biases.items() if key in keep_keys}

    state.transport_biases = biases


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
        # Allow peripheral blocks once fine blocks are exhausted.
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


def _derive_margins(
    *,
    E: float,
    D: float,
    drift_P: float,
    opp: float,
    x_C: float,
    cfg: AgentConfig,
) -> Tuple[Margins, float, float, float]:
    """
    Build v(t) = (m_E, m_D, m_L, m_C, m_S) using A2.1–A2.4.

    m_L uses the external opportunity signal opp(t), not a proxy.
    """
    margins, rawE, rawD, rawS = compute_margins(
        E=E,
        D=D,
        drift_P=drift_P,
        opp=opp,
        x_C=x_C,
        cfg=cfg,
    )
    return margins, rawE, rawD, rawS


def _feature_probe_vectors(
    *,
    state: AgentState,
    obs: EnvObs,
    abs_error: np.ndarray,
    observed_dims: Set[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    A3.3 stability metrics plumbing: feed LOW-DIMENSIONAL probe/features into rolling windows.

    Inputs:
      state: AgentState (for fovea summaries)
      obs: EnvObs (for exogenous probes)
      abs_error: |x_t - prior_t| vector
      observed_dims: observed dims at t (subset of [0,D))

    Outputs:
      (probe_vec, feature_vec) low-dimensional vectors.

    IMPORTANT: This function intentionally does NOT store full x(t) vectors (which may be pixel-like).
    """
    # Probes: exogenous low-dim signals.
    probe_vec = np.asarray([float(getattr(obs, "opp", 0.0)), float(getattr(obs, "danger", 0.0))], dtype=float)

    # Features: low-dimensional internal summaries.
    fovea = getattr(state, "fovea", None)
    if fovea is None:
        feature_vec = np.zeros(4, dtype=float)
        return probe_vec, feature_vec

    br = np.asarray(getattr(fovea, "block_residual", np.zeros(0)), dtype=float)
    ba = np.asarray(getattr(fovea, "block_age", np.zeros(0)), dtype=float)

    if observed_dims:
        od = sorted(int(k) for k in observed_dims)
        mean_abs_err = float(np.mean(abs_error[od]))
    else:
        mean_abs_err = 0.0

    feature_vec = np.asarray(
        [
            float(np.mean(br)) if br.size else 0.0,
            float(np.std(br)) if br.size else 0.0,
            float(np.mean(ba)) if ba.size else 0.0,
            float(mean_abs_err),
        ],
        dtype=float,
    )
    return probe_vec, feature_vec


def _build_training_mask(*, obs_mask: np.ndarray, x_obs: np.ndarray, cfg: AgentConfig) -> np.ndarray:
    """Objective shaping: focus updates on active, observed signal."""
    mask = obs_mask.astype(float)
    if not np.any(mask):
        return mask
    if bool(getattr(cfg, "train_active_only", False)):
        thresh = float(getattr(cfg, "train_active_threshold", 0.0))
        active = np.abs(x_obs) > thresh
        mask = mask * active.astype(float)
    if bool(getattr(cfg, "train_weight_by_value", False)):
        power = float(getattr(cfg, "train_value_power", 1.0))
        power = max(power, 0.0)
        weights = np.abs(x_obs) ** power
        mask = mask * weights
    return mask


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def step_pipeline(state: AgentState, env_obs: EnvObs, cfg: AgentConfig) -> Tuple[int, AgentState, Dict[str, Any]]:
    """
    Advance the agent by one timestep.

    Inputs:
      state: current AgentState
      env_obs: environment observation (sparse cue in env_obs.x_partial)
      cfg: AgentConfig

    Outputs:
      action: int action selected for this timestep
      next_state: updated AgentState (mutated in place in this snapshot style)
      trace: dict diagnostics (runner expects a dict)
    """
    D = _cfg_D(state, cfg)
    _dbg('enter', state=state)
    _dbg(f'cfg.D={D}', state=state)
    x_prev = np.asarray(getattr(state.buffer, "x_last", np.zeros(D)), dtype=float).reshape(-1)
    if x_prev.shape[0] != D:
        raise AssertionError(
            f"Invariant violation: buffer.x_last has shape {x_prev.shape}, expected D={D}."
        )
    x_prev_pre = x_prev.copy()
    prev_observed_dims = set(getattr(state.buffer, "observed_dims", set()) or set())

    env_full = getattr(env_obs, "x_full", None)
    env_grid_mass = np.zeros(0, dtype=float)
    transport_source = "buffer_infer"
    force_true_delta = bool(getattr(cfg, "transport_force_true_delta", False))
    env_shift: Tuple[int, int] | None = None
    use_env_grid = bool(getattr(cfg, "transport_debug_env_grid", False))
    if env_full is not None and use_env_grid:
        full_arr = np.asarray(env_full, dtype=float).reshape(-1)
        env_grid_mass = grid_cell_mass(full_arr, cfg)
        prev_grid_mass = getattr(state, "grid_prev_mass", None)
        if prev_grid_mass is not None and env_grid_mass.shape == prev_grid_mass.shape and env_grid_mass.size > 0:
            grid_shift = compute_grid_shift(prev_grid_mass, env_grid_mass, cfg)
            env_shift = grid_shift
            transport_source = "grid"
        state.grid_prev_mass = env_grid_mass.copy() if env_grid_mass.size else np.zeros(0, dtype=float)
    else:
        state.grid_prev_mass = np.zeros(0, dtype=float)

    true_full_vec: np.ndarray | None = None
    if env_full is not None:
        temp_vec = np.asarray(env_full, dtype=float).reshape(-1)
        if temp_vec.shape[0] != D:
            raise AssertionError(
                f"Invariant violation: env_obs.x_full has shape {temp_vec.shape}, expected D={D}."
            )
        true_full_vec = temp_vec

    coarse_prev_snapshot = getattr(state, "coarse_prev", None)
    use_true_transport = bool(getattr(cfg, "transport_use_true_full", False))
    coarse_true = np.zeros(0, dtype=float)
    coarse_true_size = 0
    if use_true_transport and true_full_vec is not None:
        coarse_true = extract_coarse(true_full_vec, cfg)
        coarse_true_size = coarse_true.size
    coarse_shift_hint = tuple(getattr(state, "coarse_shift", (0, 0)))
    if (
        use_true_transport
        and coarse_true_size > 0
        and coarse_prev_snapshot is not None
        and coarse_prev_snapshot.shape == coarse_true.shape
    ):
        coarse_shift_hint = compute_transport_shift(coarse_prev_snapshot, coarse_true, cfg)
    env_true_delta_hint = getattr(env_obs, "true_delta", None)
    if force_true_delta and env_true_delta_hint is not None:
        env_shift = tuple(int(v) for v in env_true_delta_hint)

    # -------------------------------------------------------------------------
    # A14.7: rest(t) from lagged predicates
    _dbg('A14.7 compute rest_t from lagged predicates', state=state)
    # -------------------------------------------------------------------------
    rest_t = bool(getattr(state, "rest_permitted_prev", True)
                  and getattr(state, "demand_prev", False)
                  and (not getattr(state, "interrupt_prev", False)))
    _dbg(
        f'rest_t={rest_t} rest_permitted_prev={getattr(state,"rest_permitted_prev",None)} '
        f'demand_prev={getattr(state,"demand_prev",None)} interrupt_prev={getattr(state,"interrupt_prev",None)}',
        state=state,
    )

    # -------------------------------------------------------------------------
    # A16.3: select fovea blocks for this step (uses t-1 tracking)
    # -------------------------------------------------------------------------
    update_fovea_routing_scores(state.fovea, x_prev, cfg, t=int(getattr(state, "t", 0)))
    _apply_pending_transport_disagreement(state, cfg)
    blocks_t = select_fovea(state.fovea, cfg)
    G = int(getattr(cfg, "coverage_cap_G", 0))
    ages_now = np.asarray(
        getattr(state.fovea, "block_age", np.zeros(int(getattr(cfg, "B", 0)))), dtype=float
    )
    budget = max(1, int(getattr(cfg, "fovea_blocks_per_step", 0)))
    if G > 0 and ages_now.size:
        mandatory = [int(b) for b in range(int(getattr(cfg, "B", 0))) if float(ages_now[b]) >= float(G)]
        if mandatory and not set(mandatory).intersection(set(blocks_t or [])):
            mandatory = sorted(mandatory, key=lambda b: float(ages_now[b]), reverse=True)
            blocks_t = mandatory[: min(len(mandatory), budget)]
    use_age = bool(getattr(cfg, "fovea_use_age", True))
    grid_world = int(getattr(cfg, "grid_side", 0)) > 0 and int(getattr(cfg, "grid_channels", 0)) > 0
    if ages_now.size and grid_world:
        residuals = np.asarray(getattr(state.fovea, "block_residual", np.zeros_like(ages_now)), dtype=float)
        alpha_cov = float(getattr(cfg, "alpha_cov", 0.10))
        if use_age:
            scores = residuals + alpha_cov * np.log1p(np.maximum(0.0, ages_now))
        else:
            scores = residuals
        top_blocks = np.argsort(-scores)[: min(budget, scores.size)].tolist()
        if top_blocks:
            blocks_t = [int(b) for b in top_blocks]
    periph_candidates = _peripheral_block_ids(cfg)
    blocks_t, forced_periph_blocks = _enforce_peripheral_blocks(blocks_t or [], cfg, periph_candidates)
    motion_probe_budget = max(0, int(getattr(cfg, "motion_probe_blocks", 0)))
    motion_probe_blocks = _select_motion_probe_blocks(prev_observed_dims, cfg, motion_probe_budget)
    blocks_t, motion_probe_blocks_used = _enforce_motion_probe_blocks(blocks_t or [], cfg, motion_probe_blocks)
    selected_blocks = tuple(getattr(env_obs, "selected_blocks", ()) or ())
    if selected_blocks:
        blocks_t = [int(b) for b in selected_blocks]
    if ages_now.size and grid_world:
        residuals = np.asarray(getattr(state.fovea, "block_residual", np.zeros_like(ages_now)), dtype=float)
        alpha_cov = float(getattr(cfg, "alpha_cov", 0.10))
        if use_age:
            scores = residuals + alpha_cov * np.log1p(np.maximum(0.0, ages_now))
        else:
            scores = residuals
        top_blocks = np.argsort(-scores)[: min(budget, scores.size)].tolist()
        if top_blocks:
            blocks_t = [int(b) for b in top_blocks]
    _dbg(f'A16.3 select_fovea -> n_blocks={len(blocks_t) if blocks_t is not None else 0}', state=state)
    state.fovea.current_blocks = set(int(b) for b in blocks_t)
    _dbg(f'A16.3 current_blocks={len(state.fovea.current_blocks)}', state=state)
    log_every = int(getattr(cfg, "fovea_log_every", 0))
    if log_every > 0 and (int(getattr(state, "t", 0)) % log_every == 0):
        _dbg(f'A16.3 blocks={list(blocks_t)}', state=state)
        _dbg(f'current_blocks={sorted(list(state.fovea.current_blocks))}', state=state)

    # Rolling visit counts per block over a sliding window.
    window = int(getattr(cfg, "fovea_visit_window", 256))
    if window > 0:
        visit_queue = getattr(state, "fovea_visit_window", None)
        visit_counts = getattr(state, "fovea_visit_counts", None)
        if visit_queue is None or visit_counts is None or len(getattr(visit_counts, "shape", [])) != 1:
            visit_queue = deque()
            visit_counts = np.zeros(int(cfg.B), dtype=int)
        if int(visit_counts.shape[0]) != int(cfg.B):
            visit_counts = np.resize(visit_counts, (int(cfg.B),))
            visit_counts = np.asarray(visit_counts, dtype=int)
        if len(visit_queue) >= window:
            oldest = visit_queue.popleft()
            for b in oldest:
                if 0 <= int(b) < int(cfg.B):
                    visit_counts[int(b)] -= 1
        current = [int(b) for b in blocks_t if 0 <= int(b) < int(cfg.B)]
        visit_queue.append(current)
        for b in current:
            visit_counts[int(b)] += 1
        state.fovea_visit_window = visit_queue
        state.fovea_visit_counts = visit_counts
        if log_every > 0 and (int(getattr(state, "t", 0)) % log_every == 0):
            if visit_counts.size:
                v_min = int(np.min(visit_counts))
                v_med = float(np.median(visit_counts))
                v_max = int(np.max(visit_counts))
                _dbg(
                    f'A16.2 visit_counts(window={window}): min={v_min} median={v_med:.1f} max={v_max}',
                    state=state,
                )

    # A16.5 requested observation set
    O_req = make_observation_set(blocks_t, cfg)
    _dbg(f'A16.5 make_observation_set -> |O_req|={len(O_req)}', state=state)
    forced_periph_dims: Set[int] = set()
    missing_periph_dims: List[int] = []
    periph_dims_present = 0
    if forced_periph_blocks:
        for b in forced_periph_blocks:
            forced_periph_dims.update(dims_for_block(b, cfg))
        if forced_periph_dims:
            missing_periph_dims = sorted(
                int(dim) for dim in forced_periph_dims if dim not in O_req
            )
            if missing_periph_dims:
                missing_head = missing_periph_dims[: min(8, len(missing_periph_dims))]
                _dbg(
                    f'A16.5 periph dims missing from O_req: count={len(missing_periph_dims)} '
                    f'head={missing_head}',
                    state=state,
                )
        periph_dims_present = int(len(forced_periph_dims & O_req))

    # Mask incoming sparse cue to O_req and bounds
    env_obs_dims = {
        int(k) for k in (getattr(env_obs, "x_partial", {}) or {}).keys() if 0 <= int(k) < D
    }
    cue_t = _filter_cue_to_Oreq(getattr(env_obs, "x_partial", {}) or {}, O_req, D)
    _dbg(f'cue_in|x_partial|={len(getattr(env_obs,"x_partial",{}) or {})}', state=state)
    O_t = set(cue_t.keys())
    _dbg(f'A16.5 cue_t filtered -> |O_t|={len(O_t)}', state=state)
    if env_obs_dims:
        env_min = min(env_obs_dims)
        env_max = max(env_obs_dims)
    else:
        env_min = None
        env_max = None
    if O_req:
        req_min = min(O_req)
        req_max = max(O_req)
    else:
        req_min = None
        req_max = None
    if O_t:
        used_min = min(O_t)
        used_max = max(O_t)
    else:
        used_min = None
        used_max = None
    _dbg(
        f'A16.5 obs_sets env_size={len(env_obs_dims)} env_min={env_min} env_max={env_max} '
        f'req_size={len(O_req)} req_min={req_min} req_max={req_max} '
        f'used_size={len(O_t)} used_min={used_min} used_max={used_max}',
        state=state,
    )

    if O_t:
        obs_idx = np.array(sorted(O_t), dtype=int)
        obs_vals = np.array([float(cue_t[int(i)]) for i in obs_idx], dtype=float)
    else:
        obs_idx = np.zeros(0, dtype=int)
        obs_vals = np.zeros(0, dtype=float)

    periph_dims = _peripheral_dim_set(D, cfg)
    _update_observed_history(state, obs_idx, cfg, extra_dims=periph_dims)


    (
        shift,
        x_prev_post,
        transport_best_candidate,
        transport_runner_candidate,
        transport_margin_val,
        transport_confidence_prob,
        transport_prob_diff,
        transport_source_hint,
        transport_candidates_info,
        transport_null_evidence,
        transport_posterior_entropy,
        transport_score_spread,
        transport_tie_flag,
        transport_best_overlap,
    ) = _select_transport_delta(
        x_prev_pre,
        obs_idx,
        obs_vals,
        cfg,
        state,
        env_shift,
        coarse_shift_hint,
        true_full_vec,
        env_true_delta_hint,
        force_true_delta,
    )
    x_prev = x_prev_post
    shift = tuple(int(v) for v in shift)
    _update_transport_learning_state(state, cfg, transport_best_candidate, shift, env_true_delta_hint)
    state.transport_last_delta = shift
    state.coarse_shift = shift
    if transport_null_evidence:
        transport_confidence = 0.0
        transport_prob_diff = 0.0
        transport_margin_val = 0.0
    else:
        transport_confidence = transport_confidence_prob
    state.transport_confidence = float(transport_confidence)
    state.transport_margin = float(transport_margin_val)
    transport_source = transport_source_hint
    transport_effect = float(np.mean(np.abs(x_prev_post - x_prev_pre))) if x_prev_pre.size else 0.0
    transport_applied_norm = float(transport_effect)
    confidence_margin_threshold = float(getattr(cfg, "transport_confidence_margin", 0.25))
    if transport_prob_diff < confidence_margin_threshold:
        state.transport_disagreement_scores = _compute_transport_disagreement_blocks(
            transport_best_candidate,
            transport_runner_candidate,
            cfg,
        )
    else:
        state.transport_disagreement_scores = {}
    state.transport_disagreement_margin = float(transport_prob_diff)
    transport_score_margin = 0.0
    if transport_best_candidate is not None and transport_runner_candidate is not None:
        transport_score_margin = float(transport_best_candidate.score - transport_runner_candidate.score)
    hc_margin = float(getattr(cfg, "transport_high_confidence_margin", 0.05))
    hc_overlap = max(1, int(getattr(cfg, "transport_high_confidence_overlap", 2)))
    runner_exists = transport_runner_candidate is not None
    margin_ok = (not runner_exists) or (transport_score_margin >= hc_margin)
    first_step = int(getattr(state, "t", 0)) == 0
    base_confidence = (not transport_null_evidence) and margin_ok
    if first_step and obs_idx.size:
        base_confidence = True
    if first_step:
        transport_high_confidence = base_confidence
    else:
        transport_high_confidence = base_confidence and transport_best_overlap >= hc_overlap
    if not prev_observed_dims and int(getattr(state, "t", 0)) > 0:
        transport_high_confidence = False
    if not transport_high_confidence and first_step and bool(state.buffer.observed_dims):
        transport_high_confidence = True
    if obs_idx.size:
        mae_pos_pre_transport = float(np.mean(np.abs(obs_vals - x_prev_pre[obs_idx])))
        mae_pos_post_transport = float("nan")
    else:
        mae_pos_pre_transport = float("nan")
        mae_pos_post_transport = float("nan")

    selected_blocks = tuple(getattr(env_obs, "selected_blocks", ()) or ())
    periph_blocks_cfg = max(0, int(getattr(cfg, "periph_blocks", 0)))
    B_cfg = max(0, int(getattr(cfg, "B", 0)))
    if periph_blocks_cfg > 0:
        periph_block_ids = tuple(range(max(0, B_cfg - periph_blocks_cfg), B_cfg))
    else:
        periph_block_ids = tuple()
    n_periph_blocks_selected = sum(1 for block in selected_blocks if block in periph_block_ids)
    n_fine_blocks_selected = max(0, len(selected_blocks) - n_periph_blocks_selected)
    periph_included = bool(n_periph_blocks_selected)

    pos_dims = getattr(env_obs, "pos_dims", None) or set()
    pos_idx = np.array(sorted({int(dim) for dim in pos_dims if 0 <= int(dim) < D}), dtype=int)
    pos_unobs_idx = np.zeros(0, dtype=int)
    pos_obs_mask = np.zeros(0, dtype=bool)
    if pos_idx.size:
        pos_obs_mask = np.isin(pos_idx, obs_idx) if obs_idx.size else np.zeros(pos_idx.shape, dtype=bool)
        pos_unobs_idx = pos_idx[~pos_obs_mask]
    true_vals = None
    if true_full_vec is not None and pos_idx.size:
        true_vals = true_full_vec[pos_idx]

    mae_pos_unobs_pre_transport = 0.0
    mae_pos_unobs_post_transport = 0.0
    true_vals_unobs = None
    if pos_unobs_idx.size and true_vals is not None:
        true_vals_unobs = true_vals[~pos_obs_mask]
        mae_pos_unobs_pre_transport = float(np.mean(np.abs(x_prev_pre[pos_unobs_idx] - true_vals_unobs)))
        mae_pos_unobs_post_transport = float(np.mean(np.abs(x_prev_post[pos_unobs_idx] - true_vals_unobs)))

    periph_selected = periph_included

    # -------------------------------------------------------------------------
    # A13 (perception): complete/clamp observed dims into prior
    # -------------------------------------------------------------------------
    x_t, Sigma_prior, prior_t = complete(
        cue_t,
        mode="perception",
        state=state,
        cfg=cfg,
        transport_shift=shift,
    )
    _dbg('A13 complete(perception) begin', state=state)

    x_t = np.asarray(x_t, dtype=float).reshape(-1)
    prior_t = np.asarray(prior_t, dtype=float).reshape(-1)
    if x_t.shape[0] != D:
        raise AssertionError(
            f"Invariant violation: posterior x_t has shape {x_t.shape}, expected D={D}."
        )
    if prior_t.shape[0] != D:
        raise AssertionError(
            f"Invariant violation: prior_t has shape {prior_t.shape}, expected D={D}."
        )
    if obs_idx.size:
        mae_pos_post_transport = float(np.mean(np.abs(prior_t[obs_idx] - obs_vals)))
    else:
        mae_pos_post_transport = float("nan")
    _dbg('A13 complete(perception) end', state=state)
    _compute_peripheral_metrics(state, cfg, prior_t, env_obs, obs_idx, obs_vals, D, periph_dims)
    clamp_delta = x_t - prior_t
    not_obs_mask = np.ones(D, dtype=bool)
    if obs_idx.size:
        not_obs_mask[obs_idx] = False
    outside_idx = np.nonzero(not_obs_mask)[0]
    delta_outside_vals = clamp_delta[outside_idx] if outside_idx.size else np.zeros(0, dtype=float)
    delta_outside_O = float(np.mean(np.abs(delta_outside_vals))) if outside_idx.size else 0.0
    if O_t:
        obs_idx = np.array(sorted(O_t), dtype=int)
        mean_abs_clamp = float(np.mean(np.abs(clamp_delta[obs_idx]))) if obs_idx.size else 0.0
    else:
        mean_abs_clamp = 0.0
    posterior_obs_mae = _prior_obs_mae(obs_idx, obs_vals, x_t)
    innov_energy = float(np.mean(np.abs(clamp_delta))) if clamp_delta.size else 0.0
    innovation_mean_abs = innov_energy

    mae_pos_prior = float("nan")
    mae_pos_prior_unobs = 0.0
    if pos_idx.size and true_vals is not None:
        prior_vals = np.asarray(prior_t[pos_idx], dtype=float)
        diff_prior = prior_vals - true_vals
        finite_prior = np.isfinite(diff_prior)
        if finite_prior.any():
            mae_pos_prior = float(np.mean(np.abs(diff_prior[finite_prior])))
        if pos_unobs_idx.size and true_vals_unobs is not None:
            prior_unobs_vals = np.asarray(prior_t[pos_unobs_idx], dtype=float)
            diff_prior_unobs = prior_unobs_vals - true_vals_unobs
            finite_unobs = np.isfinite(diff_prior_unobs)
            if finite_unobs.any():
                mae_pos_prior_unobs = float(np.mean(np.abs(diff_prior_unobs[finite_unobs])))

    periph_missing_count = int(len(missing_periph_dims))
    _dbg(
        f'A13 transport_diag delta={shift} transport_mae_pre={mae_pos_pre_transport:.6f} '
        f'transport_mae_post={mae_pos_post_transport:.6f} mae_pos_prior={mae_pos_prior:.6f} '
        f'mae_pos_prior_unobs={mae_pos_prior_unobs:.6f} mae_pos_unobs_pre={mae_pos_unobs_pre_transport:.6f} '
        f'mae_pos_unobs_post={mae_pos_unobs_post_transport:.6f} trans_norm={transport_applied_norm:.6f} '
        f'transport_effect={transport_effect:.6f} transport_confidence={state.transport_confidence:.6f} '
        f'transport_margin={state.transport_margin:.6f} '
        f'transport_source={transport_source} periph_dims_missing_count={periph_missing_count} '
        f'periph_selected={periph_selected} transport_candidates={len(transport_candidates_info)}',
        state=state,
    )

    _dbg(
        f'A13 clamp_stats: obs_dims={len(O_t)} mean_abs_delta={mean_abs_clamp:.6f} '
        f'delta_outside={delta_outside_O:.6f} mae_pos_prior={mae_pos_prior:.6f} '
        f'mae_pos_prior_unobs={mae_pos_prior_unobs:.6f}',
        state=state,
    )
    Sigp_diag = np.diag(Sigma_prior).copy() if np.asarray(Sigma_prior).ndim == 2 else np.asarray(Sigma_prior, dtype=float).reshape(-1)
    if Sigp_diag.shape[0] != D:
        Sigp_diag = np.resize(Sigp_diag, (D,))
    finite_mask = np.isfinite(Sigp_diag)
    if np.any(finite_mask):
        _dbg(
            f'A7 prior_sigma_diag: mean={float(np.mean(Sigp_diag[finite_mask])):.6f} '
            f'min={float(np.min(Sigp_diag[finite_mask])):.6f} max={float(np.max(Sigp_diag[finite_mask])):.6f}',
            state=state,
        )

    # -------------------------------------------------------------------------
    # Prediction error on observed dims for A16.2 tracking and A17 diagnostics
    # -------------------------------------------------------------------------
    error_vec = np.zeros(D, dtype=float)
    if obs_idx.size:
        error_vec[obs_idx] = obs_vals - prior_t[obs_idx]
    abs_error = np.abs(error_vec)
    if obs_idx.size:
        active_thresh = float(getattr(cfg, "train_active_threshold", 0.0))
        active_obs_mask = np.abs(obs_vals) > active_thresh
        active_obs_count = int(np.sum(active_obs_mask))
        active_obs_err = float(np.mean(abs_error[obs_idx][active_obs_mask])) if active_obs_count else 0.0
        _dbg(
            f'A16.2 active_obs: count={active_obs_count} mean_abs_err={active_obs_err:.6f}',
            state=state,
        )
        prior_obs_mae = float(np.mean(abs_error[obs_idx]))
    else:
        prior_obs_mae = float("nan")

    worlds = _build_world_hypotheses(
        state,
        cfg,
        D,
        cue_t,
        obs_idx,
        obs_vals,
        transport_candidates_info,
        transport_best_candidate,
        prior_t,
        x_t,
        Sigp_diag,
    )
    _update_block_signals(state, cfg, worlds, D)
    finite_world_maes = [float(w.prior_mae) for w in worlds if np.isfinite(w.prior_mae)]
    if finite_world_maes:
        best_world_mae = float(min(finite_world_maes))
        expected_world_mae = float(
            sum(w.weight * float(w.prior_mae) for w in worlds if np.isfinite(w.prior_mae))
        )
    else:
        best_world_mae = float("nan")
        expected_world_mae = float("nan")
    weight_entropy = 0.0
    for world in worlds:
        w = float(world.weight)
        if w > 0.0 and np.isfinite(w):
            weight_entropy -= w * math.log(w)
    multi_world_summary = [
        {
            "delta": tuple(world.delta),
            "weight": float(world.weight),
            "prior_mae": float(world.prior_mae),
            "likelihood": float(world.likelihood),
            "score": float(world.metadata.get("score", 0.0)),
        }
        for world in worlds
    ]
    multi_world_count = len(worlds)
    support_window = _support_window_union(state)

    # Update observation buffer (dense estimate and observed dims)
    state.buffer.x_prior = prior_t.copy()
    state.buffer.x_last = x_t.copy()
    _dbg(f'BUFFER update x_last; D={D}', state=state)
    coarse_buffer = extract_coarse(state.buffer.x_last, cfg)
    state.buffer.observed_dims = set(int(k) for k in O_t if 0 <= int(k) < D)
    _dbg(f'BUFFER observed_dims={len(state.buffer.observed_dims)}', state=state)

    coarse_prev = getattr(state, "coarse_prev", None)
    coarse_prev_norm, coarse_prev_nonzero, coarse_prev_head = _coarse_summary(coarse_prev)
    use_true_source = use_true_transport and coarse_true_size > 0
    coarse_curr = coarse_true if use_true_source else coarse_buffer
    if use_true_source:
        transport_source = "debug_env"
    coarse_curr_norm, coarse_curr_nonzero, coarse_curr_head = _coarse_summary(coarse_curr)

    if "coarse_prev" in state.__dataclass_fields__:
        if (
            coarse_curr.size > 0
            and coarse_prev is not None
            and coarse_prev.shape == coarse_curr.shape
        ):
            coarse_shift = compute_transport_shift(coarse_prev, coarse_curr, cfg)
            if coarse_shift == (0, 0) and not np.allclose(coarse_prev, coarse_curr):
                delta = float(np.linalg.norm(coarse_curr - coarse_prev))
                nonzero_prev = int(np.count_nonzero(coarse_prev))
                nonzero_curr = int(np.count_nonzero(coarse_curr))
                _dbg(
                    f'A13 transport_no_shift delta={delta:.6f} prev_nz={nonzero_prev} curr_nz={nonzero_curr}',
                    state=state,
                )
        state.coarse_prev = coarse_curr.copy() if coarse_curr.size else np.zeros(0, dtype=float)


    _ensure_node_band_levels(state, cfg)
    gist_vec = _compute_peripheral_gist(x_prev, cfg)
    _update_context_register(state, gist_vec, cfg)

    # -------------------------------------------------------------------------
    # A5 salience (uses lagged stress + lagged scores)
    # -------------------------------------------------------------------------
    stress_signals_lagged = get_stress_signals(state)
    sal = compute_salience(
        state=state,
        stress=stress_signals_lagged,
        scores_prev=getattr(state, "scores_prev", None),
        cfg=cfg,
        observed_dims=state.buffer.observed_dims,
    )
    _dbg('A5 compute_salience', state=state)

    # -------------------------------------------------------------------------
    # A4/A5 working set selection
    # -------------------------------------------------------------------------
    A_t = select_working_set(state, salience=sal.activations, cfg=cfg)
    _dbg('A4/A5 select_working_set', state=state)
    state.active_set = set(int(nid) for nid in getattr(A_t, "active", []) or [])

    _update_context_tags(state, cfg)
    _update_coverage_debts(state, cfg)

    L_eff = float(getattr(A_t, "effective_load", 0.0))
    _dbg(f'L_eff={L_eff:.3f} active_set={len(getattr(state,"active_set",[]) or [])}', state=state)

    # Optional binding/equivariance: select a transform per active node.
    if bool(getattr(cfg, "binding_enabled", False)):
        side = int(getattr(cfg, "grid_side", 0))
        channels = int(getattr(cfg, "grid_channels", 0))
        base_dim = int(getattr(cfg, "grid_base_dim", 0) or D)
        if side > 0 and channels > 0:
            cache = getattr(state, "_binding_cache", {})
            cache_key = (side, channels, base_dim, int(getattr(cfg, "binding_shift_radius", 1)), bool(getattr(cfg, "binding_rotations", True)))
            maps = cache.get(cache_key)
            if maps is None:
                rots = [0, 90, 180, 270] if bool(getattr(cfg, "binding_rotations", True)) else [0]
                maps = build_binding_maps(
                    D=D,
                    side=side,
                    channels=channels,
                    base_dim=base_dim,
                    shift_radius=int(getattr(cfg, "binding_shift_radius", 1)),
                    rotations=rots,
                )
                cache[cache_key] = maps
                state._binding_cache = cache
            observed_dims = set(state.buffer.observed_dims)
            for nid in state.active_set:
                node = state.library.nodes.get(int(nid))
                if node is None:
                    continue
                binding = select_best_binding_by_fit(
                    mask=getattr(node, "mask", None),
                    W=getattr(node, "W", np.zeros((D, D))),
                    b=getattr(node, "b", np.zeros(D)),
                    input_mask=getattr(node, "input_mask", None),
                    x_prev=x_prev,
                    cue_t=cue_t,
                    maps=maps,
                )
                if binding is None:
                    binding = select_best_binding(mask=getattr(node, "mask", None), observed_dims=observed_dims, maps=maps)
                if binding is not None:
                    setattr(node, "binding_map", binding)

    # -------------------------------------------------------------------------
    # A5/A12 activity logging for structural proposals
    # -------------------------------------------------------------------------
    activation_log = getattr(state, "activation_log", {})
    activation_max = int(getattr(cfg, "activation_log_max", 200))
    for nid, a_j in (getattr(A_t, "weights", {}) or {}).items():
        log = list(activation_log.get(int(nid), []))
        log.append((int(getattr(state, "t", 0)), float(a_j)))
        if len(log) > activation_max:
            log = log[-activation_max:]
        activation_log[int(nid)] = log
    state.activation_log = activation_log

    # Track last active step for incumbents (A12.3 PRUNE)
    for nid in state.active_set:
        node = state.library.nodes.get(int(nid))
        if node is not None:
            node.last_active_step = int(getattr(state, "t", 0))

    # -------------------------------------------------------------------------
    # A16.2: update fovea tracking after applying observation at t
    # (kept after A4.3 retrieval so retrieval keys to t-1 greedy_cov stats)
    # -------------------------------------------------------------------------
    ages_before = np.asarray(getattr(state.fovea, "block_age", []), dtype=float).copy()
    update_fovea_tracking(
        state.fovea,
        state.buffer,
        cfg,
        abs_error=abs_error,
        observed_dims=state.buffer.observed_dims,
    )
    ages_after = np.asarray(getattr(state.fovea, "block_age", []), dtype=float)
    if ages_before.size and ages_after.size:
        delta_age_mean = float(np.mean(ages_after) - np.mean(ages_before))
        _dbg(f'A16.2 age_delta_mean={delta_age_mean:.3f} rest_t={rest_t}', state=state)
        resids = np.asarray(getattr(state.fovea, "block_residual", []), dtype=float)
        if resids.size:
            age_min = float(np.min(ages_after))
            age_max = float(np.max(ages_after))
            age_mean = float(np.mean(ages_after))
            resid_min = float(np.min(resids))
            resid_max = float(np.max(resids))
            resid_mean = float(np.mean(resids))
            top_age = np.argsort(-ages_after)[: min(5, ages_after.size)]
            top_resid = np.argsort(-resids)[: min(5, resids.size)]
            _dbg(
                'A16.2 coverage_stats: '
                f'age_mean={age_mean:.3f} age_min={age_min:.3f} age_max={age_max:.3f} '
                f'resid_mean={resid_mean:.3f} resid_min={resid_min:.3f} resid_max={resid_max:.3f} '
                f'top_age={list(top_age)} top_resid={list(top_resid)}',
                state=state,
            )
            alpha_cov = float(getattr(cfg, "alpha_cov", 0.10))
            score = resids + alpha_cov * np.log1p(np.maximum(0.0, ages_after))
            top_score = np.argsort(-score)[: min(3, score.size)]
            top_terms = [
                (
                    int(b),
                    float(resids[b]),
                    float(ages_after[b]),
                    float(alpha_cov * np.log1p(max(0.0, ages_after[b]))),
                    float(score[b]),
                )
                for b in top_score
            ]
            _dbg(
                f'A16.2 score_terms top3=(b,resid,age,age_term,score)={top_terms}',
                state=state,
            )
    _dbg('A16.2 update_fovea_tracking', state=state)

    # -------------------------------------------------------------------------
    # A12.4 persistent residuals + residual stats + transition logging
    # -------------------------------------------------------------------------
    persistent = getattr(state, "persistent_residuals", {})
    residual_stats = getattr(state, "residual_stats", {})
    observed_transitions = getattr(state, "observed_transitions", {})

    beta_R = float(getattr(cfg, "beta_R", 0.10))
    split_beta = float(getattr(cfg, "split_stats_beta", 0.10))
    trans_max = int(getattr(cfg, "transition_log_max", 128))

    for block_id, dims in enumerate(getattr(state, "blocks", []) or []):
        block_dims = set(int(d) for d in dims)
        obs_in_block = block_dims & set(state.buffer.observed_dims)
        if not obs_in_block:
            continue

        idx = np.array(sorted(obs_in_block), dtype=int)
        resid_block = float(np.mean(np.abs(error_vec[idx]))) if idx.size else 0.0

        rstate = persistent.get(int(block_id))
        if rstate is None:
            rstate = PersistentResidualState()
        rstate.value = (1.0 - beta_R) * float(getattr(rstate, "value", 0.0)) + beta_R * resid_block
        rstate.coverage_visits = int(getattr(rstate, "coverage_visits", 0)) + 1
        rstate.last_update_step = int(getattr(state, "t", 0))
        persistent[int(block_id)] = rstate

        stats = residual_stats.get(int(block_id))
        if stats is None:
            stats = FootprintResidualStats(dims=sorted(list(block_dims)))
        stats.update(error_vec, beta=split_beta)
        residual_stats[int(block_id)] = stats

        log = list(observed_transitions.get(int(block_id), []))
        dims_tuple = tuple(int(k) for k in idx)
        x_tau_vals = x_prev[idx].copy()
        x_tau_plus_1_vals = x_t[idx].copy()
        log.append(
            TransitionRecord(
                tau=int(getattr(state, "t", 0)),
                dims=dims_tuple,
                x_tau_block=x_tau_vals,
                x_tau_plus_1_block=x_tau_plus_1_vals,
                observed_dims_tau_plus_1=set(state.buffer.observed_dims),
            )
        )
        if len(log) > trans_max:
            log = log[-trans_max:]
        observed_transitions[int(block_id)] = log

    state.persistent_residuals = persistent
    state.residual_stats = residual_stats
    state.observed_transitions = observed_transitions

    # -------------------------------------------------------------------------
    # A3.3 stability metrics plumbing (low-dimensional only)
    # -------------------------------------------------------------------------
    probe_vec, feature_vec = _feature_probe_vectors(
        state=state,
        obs=env_obs,
        abs_error=abs_error,
        observed_dims=state.buffer.observed_dims,
    )
    _dbg('A3.3 feature/probe vectors', state=state)
    update_stability_metrics(state, cfg, probe_vec=probe_vec, feature_vec=feature_vec)
    _dbg('A3.3 update_stability_metrics', state=state)

    # -------------------------------------------------------------------------
    # A7.3: one-step fusion prediction x̂(t+1|t)
    # -------------------------------------------------------------------------
    yhat_tp1, Sigma_tp1 = fuse_predictions(
        state.library,
        A_t,
        state.buffer,
        set(state.buffer.observed_dims),
        cfg,
    )
    _dbg('A7.3 fuse_predictions', state=state)

    yhat_tp1 = np.asarray(yhat_tp1, dtype=float).reshape(-1)
    if yhat_tp1.shape[0] != D:
        yhat_tp1 = np.resize(yhat_tp1, (D,))

    Sigma_tp1 = np.asarray(Sigma_tp1, dtype=float)
    if Sigma_tp1.ndim == 2 and Sigma_tp1.shape[0] == Sigma_tp1.shape[1]:
        sigma_tp1_diag = np.diag(Sigma_tp1).copy()
    else:
        sigma_tp1_diag = np.asarray(Sigma_tp1, dtype=float).reshape(-1)
        if sigma_tp1_diag.shape[0] != D:
            sigma_tp1_diag = np.resize(sigma_tp1_diag, (D,))

    # A13 prediction uses the same completion operator (no cue overwrite).
    yhat_tp1, Sigma_tp1_pred, _prior_pred = complete(
        None,
        mode="prediction",
        state=state,
        cfg=cfg,
        predicted_prior_t=yhat_tp1,
        predicted_sigma_diag=sigma_tp1_diag,
    )
    yhat_tp1 = np.asarray(yhat_tp1, dtype=float).reshape(-1)
    Sigma_tp1_pred = np.asarray(Sigma_tp1_pred, dtype=float)
    if Sigma_tp1_pred.ndim == 2 and Sigma_tp1_pred.shape[0] == Sigma_tp1_pred.shape[1]:
        sigma_tp1_diag = np.diag(Sigma_tp1_pred).copy()

    _update_block_uncertainty(state, sigma_tp1_diag, cfg)

    # -------------------------------------------------------------------------
    # REST structural processing (A12/A14): REST-only, queue ownership preserved
    # -------------------------------------------------------------------------
    edits_processed_t = 0
    b_cons_t = 0.0
    rest_res = RestProcessingResult()

    if rest_t:
        _dbg('REST branch', state=state)
        # Some modules gate on state.is_rest (macro.rest). Make best-effort to
        # reflect rest(t) before calling REST processors.
        try:
            if hasattr(state, "macro") and hasattr(state.macro, "rest"):
                state.macro.rest = True  # type: ignore[attr-defined]
        except Exception:
            pass

        rest_res = process_struct_queue(
            state,
            cfg,
            queue=list(state.q_struct),
            max_edits=int(getattr(cfg, "max_edits_per_rest_step", 32)),
        )
        _dbg('REST process_struct_queue begin', state=state)
        edits_processed_t = int(getattr(rest_res, "proposals_processed", 0))
        b_cons_t = float(getattr(rest_res, "total_consolidation_cost", 0.0))
        _dbg(f'REST processed={edits_processed_t} b_cons_t={b_cons_t:.3f}', state=state)

    queue_len = int(len(getattr(state, "q_struct", []) or []))
    rest_permit_struct = bool(getattr(rest_res, "permit_struct", False))
    rest_actionable = bool(queue_len > 0)
    rest_actionable_reason = ""
    if rest_t and not rest_actionable:
        rest_actionable_reason = "queue_empty" if queue_len == 0 else "no_work"

    # -------------------------------------------------------------------------
    # A6 budget and horizon
    # -------------------------------------------------------------------------
    budget = compute_budget_and_horizon(
        rest=rest_t,
        cfg=cfg,
        L_eff=L_eff,
        L_eff_roll=float(getattr(A_t, "rollout_load", L_eff)),
        L_eff_anc=float(getattr(A_t, "anchor_load", 0.0)),
        b_cons=b_cons_t,
    )
    _dbg('A6 compute_budget_and_horizon', state=state)

    # -------------------------------------------------------------------------
    # A7.4 rollout + confidence (provides c list for A8.2)
    # -------------------------------------------------------------------------
    rollout = rollout_and_confidence(
        x0=x_t,
        x_hat_1=yhat_tp1,
        Sigma_1=np.diag(sigma_tp1_diag),
        h=int(budget.h),
        cfg=cfg,
    )
    _dbg('A7.4 rollout_and_confidence', state=state)
    c_vals = np.asarray(list(getattr(rollout, "c", []) or []), dtype=float)
    if c_vals.size:
        _dbg(
            f'A7.4 c_stats: mean={float(np.mean(c_vals)):.6f} '
            f'min={float(np.min(c_vals)):.6f} max={float(np.max(c_vals)):.6f}',
            state=state,
        )
    rho_vals = np.asarray(list(getattr(rollout, "rho", []) or []), dtype=float)
    H_vals = np.asarray(list(getattr(rollout, "H", []) or []), dtype=float)
    c_qual_vals = np.asarray(list(getattr(rollout, "c_qual", []) or []), dtype=float)
    c_cov_vals = np.asarray(list(getattr(rollout, "c_cov", []) or []), dtype=float)
    if rho_vals.size and H_vals.size:
        _dbg(
            f'A7.4 cov_stats: rho_mean={float(np.mean(rho_vals)):.6f} '
            f'H_cov_mean={float(np.mean(H_vals)):.6f} '
            f'c_qual_mean={float(np.mean(c_qual_vals)):.6f} '
            f'c_cov_mean={float(np.mean(c_cov_vals)):.6f}',
            state=state,
        )

    # -------------------------------------------------------------------------
    # A8 commitment gate and action selection
    # -------------------------------------------------------------------------
    commit_t = commit_gate(rest=rest_t, h=int(budget.h), c=list(getattr(rollout, "c", [])), cfg=cfg)
    _dbg(f'A8 commit_gate with h={int(budget.h)} c_len={len(list(getattr(rollout,"c",[]) or []))}', state=state)
    action = int(select_action(commit=commit_t, rollout=rollout, cfg=cfg))
    _dbg(f'A8 select_action -> action={action} commit={bool(commit_t)}', state=state)

    # -------------------------------------------------------------------------
    # A15 hard dynamics update of (E, D, drift_P)
    # -------------------------------------------------------------------------
    hard_prev = HardState(E=float(state.E), D=float(state.D), drift_P=float(state.drift_P))
    hard_t = step_hard_dynamics(prev=hard_prev, rest=rest_t, commit=commit_t, L_eff=L_eff, cfg=cfg)
    _dbg('A15 step_hard_dynamics', state=state)

    state.E = float(hard_t.E)
    state.D = float(hard_t.D)
    state.drift_P = float(hard_t.drift_P)

    # -------------------------------------------------------------------------
    # Margins (A0.1 / A2) + baselines (A3.1) + arousal (A0.2–A0.4)
    # -------------------------------------------------------------------------
    margins_t, rawE_t, rawD_t, _rawS = _derive_margins(
        E=state.E,
        D=state.D,
        drift_P=state.drift_P,
        opp=float(getattr(env_obs, "opp", 0.0)),
        x_C=float(budget.x_C),
        cfg=cfg,
    )
    _dbg('A0/A2 derive margins', state=state)
    _dbg(
        f'A0/A2 raw_headrooms: E={state.E:.3f} D={state.D:.3f} rawE={rawE_t:.3f} rawD={rawD_t:.3f}',
        state=state,
    )

    baselines_t = update_baselines(baselines=state.baselines, margins=margins_t, cfg=cfg)
    _dbg('A3.1 update_baselines', state=state)

    # Feel proxy (A17) uses sigma prior (Σ_global(t) diag) and H_d at latency floor.
    if Sigma_prior is None:
        sigma_prior_diag = np.ones(D, dtype=float)
    else:
        Sigp = np.asarray(Sigma_prior, dtype=float)
        if Sigp.ndim == 2 and Sigp.shape[0] == Sigp.shape[1]:
            sigma_prior_diag = np.diag(Sigp).copy()
        else:
            sigma_prior_diag = np.asarray(Sigp, dtype=float).reshape(-1)
        if sigma_prior_diag.shape[0] != D:
            sigma_prior_diag = np.resize(sigma_prior_diag, (D,))

    d_floor = int(getattr(cfg, "d_latency_floor", 1))
    if d_floor <= 0:
        d_floor = 1
    H_d = 0.0
    if len(getattr(rollout, "H", [])) >= d_floor:
        H_d = float(rollout.H[d_floor - 1])

    feel = compute_feel_proxy(
        observed_dims=state.buffer.observed_dims,
        error_vec=error_vec,
        sigma_global_diag=sigma_prior_diag,
        L_eff=L_eff,
        H_d=H_d,
        sigma_floor=float(getattr(cfg, "sigma_floor_diag", 1e-2)),
    )
    _dbg('A17 compute_feel_proxy', state=state)

    # Arousal uses pred_error magnitude proxy; use q_res (A17.1) as a scalar proxy.
    arousal_prev = float(getattr(state, "arousal_prev", getattr(state, "arousal", 0.0)))
    s_inst, s_ar = compute_arousal(
        arousal_prev=arousal_prev,
        margins=margins_t,
        baselines=baselines_t,
        pred_error=float(getattr(feel, "q_res_raw", 0.0)),
        cfg=cfg,
    )
    _dbg('A0.2 compute_arousal', state=state)

    # Persist tilde_prev for A0.2 delta computation (no other module does this yet)
    tilde, delta_tilde = normalize_margins(margins=margins_t, baselines=baselines_t, cfg=cfg)
    baselines_t = commit_tilde_prev(baselines_t, tilde=tilde)
    mE, mD, mL, mC, mS = [float(x) for x in tilde]
    dE, dD, dL, dC, dS = [float(x) for x in delta_tilde]
    w_L = float(getattr(cfg, "w_L", getattr(cfg, "w_L_ar", 1.0)))
    w_C = float(getattr(cfg, "w_C", getattr(cfg, "w_C_ar", 1.0)))
    w_S = float(getattr(cfg, "w_S", getattr(cfg, "w_S_ar", 1.0)))
    w_delta = float(getattr(cfg, "w_delta", getattr(cfg, "w_delta_ar", 1.0)))
    w_E = float(getattr(cfg, "w_E", getattr(cfg, "w_E_ar", 0.0)))
    term_L = w_L * abs(mL)
    term_C = w_C * abs(mC)
    term_S = w_S * abs(mS)
    term_delta = w_delta * (abs(dE) + abs(dD) + abs(dL) + abs(dC) + abs(dS))
    term_E = w_E * abs(float(getattr(feel, "q_res", 0.0)))
    A_raw = term_L + term_C + term_S + term_delta + term_E
    _dbg(
        'A0.2 arousal_terms: '
        f'L={term_L:.3f} C={term_C:.3f} S={term_S:.3f} '
        f'delta={term_delta:.3f} E={term_E:.3f} A_raw={A_raw:.3f}',
        state=state,
    )

    # -------------------------------------------------------------------------
    # Stress (A0.3) from hard observables + exogenous threat
    # -------------------------------------------------------------------------
    s_ext_th_t = float(getattr(env_obs, "danger", 0.0))
    stress_t = compute_stress(E=state.E, D=state.D, drift_P=state.drift_P, s_ext_th=s_ext_th_t, cfg=cfg)
    _dbg('A0.3 compute_stress', state=state)

    # -------------------------------------------------------------------------
    # A10 learning gates (freeze uses lagged stress; permit_param uses lagged slack/headrooms)
    # -------------------------------------------------------------------------
    freeze_t = freeze_predicate(stress_lagged=state.stress, cfg=cfg)
    _dbg('A10 freeze_predicate (uses lagged stress)', state=state)
    chi_th = float(getattr(cfg, "chi_th", 0.90))
    s_ext_th_lagged = float(getattr(state.stress, "s_ext_th", 0.0))
    _dbg(
        f'A10 freeze_lagged: s_ext_th={s_ext_th_lagged:.3f} chi_th={chi_th:.3f} freeze={freeze_t}',
        state=state,
    )

    use_current = bool(getattr(cfg, "gates_use_current", False))
    if use_current:
        x_C_lagged = float(budget.x_C)
        arousal_lagged = float(s_ar)
        rawE_lagged = float(rawE_t)
        rawD_lagged = float(rawD_t)
    else:
        x_C_lagged = float(getattr(state, "x_C_prev", 0.0))
        arousal_lagged = float(getattr(state, "arousal_prev", arousal_prev))
        rawE_lagged = float(getattr(state, "rawE_prev", rawE_t))
        rawD_lagged = float(getattr(state, "rawD_prev", rawD_t))

    theta_learn = float(getattr(cfg, "theta_learn", 0.10))
    permit_param_t = permit_param_updates(
        rest_t=rest_t,
        freeze_t=freeze_t,
        x_C_lagged=x_C_lagged,
        arousal_lagged=arousal_lagged,
        rawE_lagged=rawE_lagged,
        rawD_lagged=rawD_lagged,
        cfg=cfg,
    )
    _dbg('A10 permit_param_updates', state=state)
    tau_C = float(getattr(cfg, "tau_C_edit", 0.0))
    tau_E = float(getattr(cfg, "tau_E_edit", 0.0))
    tau_D = float(getattr(cfg, "tau_D_edit", 0.0))
    theta_panic = float(getattr(cfg, "theta_ar_panic", 0.95))
    _dbg(
        'A10 permit_lagged: '
        f'rest_t={rest_t} freeze={freeze_t} '
        f'x_C={x_C_lagged:.3f} tau_C={tau_C:.3f} '
        f'arousal={arousal_lagged:.3f} theta_panic={theta_panic:.3f} '
        f'rawE={rawE_lagged:.3f} tau_E={tau_E:.3f} '
        f'rawD={rawD_lagged:.3f} tau_D={tau_D:.3f} '
        f'permit={permit_param_t}',
        state=state,
    )
    _dbg(
        f'A10 gates lagged: freeze={freeze_t} x_C={x_C_lagged:.3f} arousal={arousal_lagged:.3f} rawE={rawE_lagged:.3f} rawD={rawD_lagged:.3f}',
        state=state,
    )

    # -------------------------------------------------------------------------
    # A10.3 responsibility-gated parameter learning (observed footprint only)
    # -------------------------------------------------------------------------
    learning_candidates_info = None
    permit_param_info = {
        "theta_learn": theta_learn,
        "permit": bool(permit_param_t),
        "candidate_count": 0,
        "clamped": 0,
        "updated": 0,
        "transport_high_confidence": bool(transport_high_confidence),
    }
    if permit_param_t and transport_high_confidence:
        lr = float(getattr(cfg, "lr_expert", 0.0))
        sigma_ema = float(getattr(cfg, "sigma_ema", 0.01))
        observed_dims = set(state.buffer.observed_dims)
        if not observed_dims:
            observed_dims = set()

        updated_nodes: list[int] = []
        candidate_nodes = 0
        clamped_candidates = 0
        err_j_vals: list[float] = []
        candidate_samples: list[Dict[str, Any]] = []
        sample_cap = int(min(8, max(1, getattr(cfg, "fovea_blocks_per_step", 16))))
        for node_id in getattr(A_t, "active", []) or []:
            node = state.library.nodes.get(int(node_id))
            if node is None:
                continue

            mask = np.asarray(getattr(node, "mask", np.zeros(D)), dtype=float).reshape(-1)
            if mask.shape[0] != D:
                mask = np.resize(mask, (D,))

            if not observed_dims:
                continue

            obs_mask = (mask > 0.5)
            obs_mask &= np.isin(np.arange(D), list(observed_dims))
            if not np.any(obs_mask):
                continue

            candidate_nodes += 1
            obs_idx = np.where(obs_mask)[0]
            err_j = float(np.mean(np.abs(error_vec[obs_idx]))) if obs_idx.size else float("inf")
            err_j_vals.append(err_j)
            clamped = err_j > theta_learn
            if clamped:
                clamped_candidates += 1
            if len(candidate_samples) < sample_cap:
                candidate_samples.append(
                    {
                        "node": int(node_id),
                        "footprint": int(getattr(node, "footprint", -1)),
                        "err": err_j,
                        "obs_dims": int(obs_idx.size),
                        "clamped": clamped,
                    }
                )

            if not clamped:
                sgd_update(
                    node,
                    x_t=x_prev,
                    y_target=x_t,
                    out_mask=_build_training_mask(
                        obs_mask=obs_mask,
                        x_obs=x_t,
                        cfg=cfg,
                    ),
                    lr=lr,
                    sigma_ema=sigma_ema,
                )
                updated_nodes.append(int(node_id))
        if err_j_vals:
            err_min = float(np.min(err_j_vals))
            err_mean = float(np.mean(err_j_vals))
            err_max = float(np.max(err_j_vals))
        else:
            err_min = err_mean = err_max = float("nan")
        learning_candidates_info = {
            "candidates": candidate_nodes,
            "clamped": clamped_candidates,
            "err_min": err_min,
            "err_mean": err_mean,
            "err_max": err_max,
            "samples": candidate_samples,
        }
        permit_param_info.update(
            {
                "candidate_count": candidate_nodes,
                "clamped": clamped_candidates,
                "updated": len(updated_nodes),
            }
        )
        _dbg(
            f'A10.3 learn_gate: candidates={candidate_nodes} clamped={clamped_candidates} '
            f'err_j[min/mean/max]={err_min:.6f}/{err_mean:.6f}/{err_max:.6f} '
            f'theta_learn={theta_learn:.6f}',
            state=state,
        )
        _dbg(f'A10.3 sgd_updates={len(updated_nodes)} nodes={updated_nodes}', state=state)
    elif permit_param_t and not transport_high_confidence:
        _dbg('A10 learning skipped: transport evidence not high confidence', state=state)

    _dbg(
        f'A10 permit_param_stats candidate_count={permit_param_info["candidate_count"]} '
        f'clamped_count={permit_param_info["clamped"]} updated={permit_param_info["updated"]} '
        f'permit={permit_param_info["permit"]}',
        state=state,
    )

    # -------------------------------------------------------------------------
    # A14 macrostate evolution (queue ownership in macrostate.py)
    # -------------------------------------------------------------------------
    # Generate structural proposals during OPERATING only (A14.2).
    proposals_t: List[Any] = []
    if not rest_t:
        proposals_t = list(propose_structural_edits(state, cfg))

    # Coverage debt from A16 block ages (use current ages at t).
    ages = np.asarray(getattr(state.fovea, "block_age", []), dtype=float)
    log1p_ages = np.log1p(np.maximum(0.0, ages)) if ages.size else np.zeros(0, dtype=float)
    coverage_debt = float(np.sum(log1p_ages)) if ages.size else 0.0
    coverage_debt_prev = float(getattr(state, "coverage_debt_prev", coverage_debt))
    _dbg(f'A16.2 coverage_debt_delta={coverage_debt - coverage_debt_prev:.3f}', state=state)
    if ages.size:
        _dbg(
            'A16.2 coverage_debt_terms: '
            f'sum_log1p={coverage_debt:.3f} max_log1p={float(np.max(log1p_ages)):.3f} '
            f'max_age={float(np.max(ages)):.3f}',
            state=state,
        )

    _dbg(
        'A14 inputs: '
        f's_int_need={float(getattr(stress_t, "s_int_need", 0.0)):.3f} '
        f's_ext_th={float(getattr(stress_t, "s_ext_th", 0.0)):.3f} '
        f'mE={float(getattr(margins_t, "m_E", 0.0)):.3f} '
        f'mD={float(getattr(margins_t, "m_D", 0.0)):.3f} '
        f'mL={float(getattr(margins_t, "m_L", 0.0)):.3f} '
        f'mC={float(getattr(margins_t, "m_C", 0.0)):.3f} '
        f'mS={float(getattr(margins_t, "m_S", 0.0)):.3f} '
        f'coverage_debt={coverage_debt:.3f} '
        f'proposals={len(proposals_t)} edits_processed={int(edits_processed_t)}',
        state=state,
    )
    macro_t, demand_t, interrupt_t, P_eff_t = evolve_macrostate(
        prev=state.macro,
        rest_t=rest_t,
        proposals_t=proposals_t,
        edits_processed_t=edits_processed_t,
        stress_t=stress_t,
        margins_t=margins_t,
        coverage_debt=coverage_debt,
        rest_actionable=rest_actionable,
        cfg=cfg,
    )
    _dbg('A14 evolve_macrostate', state=state)
    P_rest_t = float(getattr(macro_t, "P_rest", 0.0))
    P_wake = float(coverage_debt)
    _dbg(f'A14 pressures: coverage_debt={coverage_debt:.3f} P_wake={P_wake:.3f} P_rest={P_rest_t:.3f} P_rest_eff={P_eff_t:.3f}', state=state)
    # A14.6: rest_permitted(t) from actual predicate (not from a missing field)
    rest_perm_t, rest_perm_reason = rest_permitted(stress_t, coverage_debt, cfg, arousal=s_ar)
    # Require stability windows before REST permission (A3.3-driven guard).
    W = int(getattr(cfg, "W", 50))
    if len(getattr(state, "probe_window", [])) < W or len(getattr(state, "feature_window", [])) < W:
        rest_perm_t = False
    # REST requires work: gate entry using same condition as continuation.
    if len(getattr(state, "q_struct", []) or []) == 0 and float(b_cons_t) == 0.0:
        rest_perm_t = False
        _dbg('A14 REST entry gated: no work in queue and no maintenance debt', state=state)
    if rest_t and not rest_actionable:
        rest_perm_t = False
        _dbg(
            f'A14 rest_actionable guard: reason={rest_actionable_reason} '
            f'queue_len={queue_len} permit_struct={rest_permit_struct}',
            state=state,
        )
    _dbg(f'A14.6 rest_permitted -> {rest_perm_t}', state=state)
    _dbg(
        'A14 macro_vars: '
        f'rest={bool(getattr(macro_t, "rest", False))} '
        f'T_since={int(getattr(macro_t, "T_since", 0))} '
        f'T_rest={int(getattr(macro_t, "T_rest", 0))} '
        f'Q_struct_len={int(len(getattr(macro_t, "Q_struct", []) or []))} '
        f'rest_permitted={bool(rest_perm_t)} '
        f'rest_reason={rest_perm_reason} '
        f'demand={bool(demand_t)} interrupt={bool(interrupt_t)} '
        f'rest_actionable={rest_actionable} rest_actionable_reason={rest_actionable_reason} '
        f'rest_zero_streak={int(getattr(macro_t, "rest_zero_processed_streak", 0))} '
        f'rest_cooldown={int(getattr(macro_t, "rest_cooldown", 0))}',
        state=state,
    )

    # -------------------------------------------------------------------------
    # Learning cache for next step’s completion prior (A13) and A17 diag use
    # -------------------------------------------------------------------------
    state.learn_cache = LearningCache(
        x_t=x_t.copy(),
        yhat_tp1=yhat_tp1.copy(),
        sigma_tp1_diag=np.asarray(sigma_tp1_diag, dtype=float).copy(),
        A_t=A_t,
        permit_param_t=bool(permit_param_t),
        rest_t=bool(rest_t),
    )

    # -------------------------------------------------------------------------
    # Commit updated state fields and lagged values
    # -------------------------------------------------------------------------
    state.t = int(getattr(state, "t", 0)) + 1
    _dbg(f'commit state.t -> {state.t}', state=state)

    state.macro = macro_t
    state.margins = margins_t
    state.baselines = baselines_t
    state.stress = stress_t
    state.arousal = float(s_ar)

    # Lagged predicates/signals for t+1
    state.rest_permitted_prev = bool(rest_perm_t)
    # If REST has no work (no queue, no maintenance debt), force exit next step.
    maint_debt = float(b_cons_t)
    if rest_t and len(getattr(macro_t, "Q_struct", []) or []) == 0 and maint_debt == 0.0:
        _dbg(
            'A14 REST requires work: forcing exit next step '
            f'(Q_struct_len=0 maint_debt={maint_debt:.3f})',
            state=state,
        )
        demand_t = False
        state.rest_permitted_prev = False
    if rest_t and not rest_actionable:
        _dbg(
            'A14 REST actionable guard: forcing exit next step '
            f'reason={rest_actionable_reason} queue_len={queue_len} permit_struct={rest_permit_struct}',
            state=state,
        )
        demand_t = False
        state.rest_permitted_prev = False
    state.demand_prev = bool(demand_t)
    state.interrupt_prev = bool(interrupt_t)

    state.s_int_need_prev = float(getattr(stress_t, "s_int_need", 0.0))
    state.s_ext_th_prev = float(getattr(stress_t, "s_ext_th", 0.0))

    state.arousal_prev = float(s_ar)
    state.scores_prev = dict(getattr(sal, "scores", {}) or {})

    state.x_C_prev = float(budget.x_C)
    state.rawE_prev = float(rawE_t)
    state.rawD_prev = float(rawD_t)
    state.coverage_debt_prev = float(coverage_debt)

    # Lagged rollout confidence at latency floor (A8.2 timing discipline)
    c_list = list(getattr(rollout, "c", []) or [])
    if len(c_list) >= d_floor:
        state.c_d_prev = float(c_list[d_floor - 1])
    elif c_list:
        state.c_d_prev = float(c_list[-1])
    else:
        state.c_d_prev = 0.0

    # Consolidation cost channel (A6.2) (store for visibility)
    state.b_cons = float(b_cons_t)

    rest_queue_len_next = int(len(getattr(macro_t, "Q_struct", []) or []))
    max_edits_per_rest = max(1, int(getattr(cfg, "max_edits_per_rest_step", 32)))
    if rest_queue_len_next > 0:
        rest_cycles_needed = int((rest_queue_len_next + max_edits_per_rest - 1) // max_edits_per_rest)
    else:
        rest_cycles_needed = 0

    # -------------------------------------------------------------------------
    # Trace (runner expects dict)
    # -------------------------------------------------------------------------
    trace = StepTrace(
        t=int(state.t),
        rest=bool(rest_t),
        h=int(budget.h),
        commit=bool(commit_t),
        x_C=float(budget.x_C),
        b_enc=float(budget.b_enc),
        b_roll=float(budget.b_roll),
        b_cons=float(budget.b_cons),
        L_eff=float(L_eff),
        arousal=float(s_ar),
        feel={
            "q_res": float(getattr(feel, "q_res", 0.0)),
            "q_maint": float(getattr(feel, "q_maint", 0.0)),
            "q_unc": float(getattr(feel, "q_unc", 0.0)),
        },
        permit_param=bool(permit_param_t),
        freeze=bool(freeze_t),
    )

    trace_dict = asdict(trace)
    # Extra diagnostics (dict-only; does not affect StepTrace typing)
    trace_dict.update(
        {
            "P_rest_eff": float(P_eff_t),
            "P_rest": float(P_rest_t),
            "P_wake": float(P_wake),
            "coverage_debt": float(coverage_debt),
            "coverage_debt_delta": float(coverage_debt - coverage_debt_prev),
            "maint_debt": float(maint_debt),
            "Q_struct_len": int(len(getattr(macro_t, "Q_struct", []) or [])),
            "observed_dims": int(len(state.buffer.observed_dims)),
            "obs_env_size": int(len(env_obs_dims)),
            "obs_env_min": int(env_min) if env_min is not None else None,
            "obs_env_max": int(env_max) if env_max is not None else None,
            "obs_req_size": int(len(O_req)),
            "obs_req_min": int(req_min) if req_min is not None else None,
            "obs_req_max": int(req_max) if req_max is not None else None,
            "obs_used_size": int(len(O_t)),
            "obs_used_min": int(used_min) if used_min is not None else None,
            "obs_used_max": int(used_max) if used_max is not None else None,
            "obs_filtered_count": int(len(O_req) - len(O_t)),
            "env_full_provided": bool(env_full is not None),
            "use_true_transport": bool(use_true_transport),
            "transport_debug_env_grid": bool(use_env_grid),
            "edits_processed": int(edits_processed_t),
            "rest_permitted_t": bool(rest_perm_t),
            "rest_unsafe_reason": str(rest_perm_reason),
            "demand_t": bool(demand_t),
            "interrupt_t": bool(interrupt_t),
            "rest_actionable": bool(rest_actionable),
            "rest_queue_len": queue_len,
            "rest_permit_struct": rest_permit_struct,
            "rest_cooldown": int(getattr(macro_t, "rest_cooldown", 0)),
            "rest_cycles_needed": int(rest_cycles_needed),
            "rest_zero_processed_streak": int(getattr(macro_t, "rest_zero_processed_streak", 0)),
            "s_int_need": float(getattr(stress_t, "s_int_need", 0.0)),
            "s_ext_th": float(getattr(stress_t, "s_ext_th", 0.0)),
            "mE": float(getattr(margins_t, "m_E", 0.0)),
            "mD": float(getattr(margins_t, "m_D", 0.0)),
            "mL": float(getattr(margins_t, "m_L", 0.0)),
            "mC": float(getattr(margins_t, "m_C", 0.0)),
            "mS": float(getattr(margins_t, "m_S", 0.0)),
            "permit_struct": bool(getattr(rest_res, "permit_struct", False)),
            "permit_struct_reason": str(getattr(rest_res, "permit_struct_reason", "")),
            "transport_delta": tuple(shift),
            "transport_mae_pre": float(mae_pos_pre_transport),
            "transport_mae_post": float(mae_pos_post_transport),
            "transport_applied_norm": float(transport_applied_norm),
            "transport_source": transport_source,
            "transport_effect": float(transport_effect),
            "transport_confidence": float(state.transport_confidence),
            "transport_margin": float(state.transport_margin),
            "transport_candidate_count": int(len(transport_candidates_info)),
            "transport_score_margin": float(transport_score_margin),
            "candidate_score_spread": float(transport_score_spread),
            "overlap_count_best": int(transport_best_overlap),
            "posterior_entropy": float(transport_posterior_entropy),
            "tie_flag": bool(transport_tie_flag),
            "null_chosen_due_to_low_evidence": bool(transport_null_evidence),
            "motion_probe_blocks_used": int(motion_probe_blocks_used),
            "transport_high_confidence": bool(transport_high_confidence),
            "transport_ascii_mismatch": int(transport_best_candidate.ascii_mismatch) if transport_best_candidate is not None else 0,
            "delta_outside_O": float(delta_outside_O),
            "innovation_mean_abs": float(innovation_mean_abs),
            "innov_energy": float(innov_energy),
            "prior_obs_mae": float(prior_obs_mae),
            "posterior_obs_mae": float(posterior_obs_mae),
            "multi_world_count": int(multi_world_count),
            "multi_world_best_prior_mae": float(best_world_mae),
            "multi_world_expected_prior_mae": float(expected_world_mae),
            "multi_world_weight_entropy": float(weight_entropy),
            "multi_world_summary": multi_world_summary,
            "support_window_size": int(len(getattr(state, "observed_history", []))),
            "support_window_union_size": int(len(support_window)),
            "peripheral_confidence": float(np.clip(getattr(state, "peripheral_confidence", 0.0), 0.0, 1.0)),
            "peripheral_residual": float(np.nan_to_num(getattr(state, "peripheral_residual", float("nan")))),
            "peripheral_prior_size": int(getattr(state, "peripheral_prior", np.zeros(0)).size),
            "peripheral_obs_size": int(getattr(state, "peripheral_obs", np.zeros(0)).size),
            "peripheral_bg_dim_count": int(len(periph_dims)),
            "peripheral_bg_active": bool(periph_dims),
            "block_disagreement_mean": float(
                np.nanmean(getattr(state.fovea, "block_disagreement", np.zeros(0))) if getattr(state.fovea, "block_disagreement", np.zeros(0)).size else 0.0
            ),
            "block_innovation_mean": float(
                np.nanmean(getattr(state.fovea, "block_innovation", np.zeros(0))) if getattr(state.fovea, "block_innovation", np.zeros(0)).size else 0.0
            ),
            "block_periph_demand_mean": float(
                np.nanmean(getattr(state.fovea, "block_periph_demand", np.zeros(0))) if getattr(state.fovea, "block_periph_demand", np.zeros(0)).size else 0.0
            ),
            "mean_abs_clamp": float(mean_abs_clamp),
            "mae_pos_prior": float(mae_pos_prior),
            "mae_pos_prior_unobs": float(mae_pos_prior_unobs),
            "mae_pos_unobs_pre": float(mae_pos_unobs_pre_transport),
            "mae_pos_unobs_post": float(mae_pos_unobs_post_transport),
            "coarse_prev_norm": float(coarse_prev_norm),
            "coarse_curr_norm": float(coarse_curr_norm),
            "coarse_prev_nonzero": int(coarse_prev_nonzero),
            "coarse_curr_nonzero": int(coarse_curr_nonzero),
            "coarse_prev_head": tuple(coarse_prev_head),
            "coarse_curr_head": tuple(coarse_curr_head),
            "periph_block_ids": tuple(int(b) for b in forced_periph_blocks),
            "periph_dims_forced": int(len(forced_periph_dims)),
            "periph_dims_in_req": int(periph_dims_present),
            "periph_dims_missing_count": int(len(missing_periph_dims)),
            "periph_dims_missing_head": tuple(
                missing_periph_dims[: min(8, len(missing_periph_dims))]
            ),
            "n_fine_blocks_selected": int(n_fine_blocks_selected),
            "n_periph_blocks_selected": int(n_periph_blocks_selected),
            "periph_included": bool(periph_included),
            "probe_var": float(state.probe_var) if getattr(state, "probe_var", None) is not None else float("nan"),
            "feature_var": float(state.feature_var) if getattr(state, "feature_var", None) is not None else float("nan"),
            "arousal": float(getattr(state, "arousal", 0.0)),
            "arousal_prev": float(getattr(state, "arousal_prev", 0.0)),
            "last_struct_edit_t": int(getattr(state.baselines, "last_struct_edit_t", -10**9)),
            "W_window": int(getattr(cfg, "W", 50)),
            "learning_candidates": learning_candidates_info if learning_candidates_info is not None else {},
            "permit_param_info": permit_param_info,
        }
    )

    _dbg('returning (action, state, trace_dict)', state=state)
    return action, state, trace_dict


# Backwards-compatible alias (some branches used a different name)
__all__ = ["step_pipeline"]
