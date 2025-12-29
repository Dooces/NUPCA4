"""
Transport inference helpers for selecting/factoring delta candidates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Set, Tuple

import math
import numpy as np

from ..config import AgentConfig
from ..geometry.fovea import block_of_dim
from ..geometry.streams import apply_transport, periph_block_size
from ..types import AgentState, Action, CurriculumCommand

from .observations import _cfg_D


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
    block_scores: Dict[int, float] = {}
    for idx, val in enumerate(diff):
        if val <= 0.0:
            continue
        block_id = block_of_dim(idx, cfg)
        block_scores[block_id] = block_scores.get(block_id, 0.0) + float(val)
    total = sum(block_scores.values())
    if total <= 0.0:
        return {}
    normalized = {int(b): float(v / total) for b, v in block_scores.items() if 0 <= b < int(getattr(cfg, "B", 0))}
    return normalized


def _apply_pending_transport_disagreement(
    state: AgentState,
    cfg: AgentConfig,
    routing: np.ndarray | None = None,
) -> np.ndarray:
    """Return routing scores updated with stored block disagreement scores."""
    weight = float(getattr(cfg, "transport_disambiguation_weight", 1.0))
    if weight <= 0.0:
        state.transport_disagreement_scores = {}
        state.transport_disagreement_margin = float("inf")
        return np.asarray(
            getattr(state.fovea, "routing_scores", np.zeros(int(getattr(cfg, "B", 0)))),
            dtype=float,
        )

    margin = float(getattr(state, "transport_disagreement_margin", float("inf")))
    threshold = float(getattr(cfg, "transport_confidence_margin", 0.25))
    scores = getattr(state, "transport_disagreement_scores", {})
    if not scores or margin >= threshold:
        state.transport_disagreement_scores = {}
        state.transport_disagreement_margin = float("inf")
        return np.asarray(
            getattr(state.fovea, "routing_scores", np.zeros(int(getattr(cfg, "B", 0)))),
            dtype=float,
        )

    B = int(getattr(cfg, "B", 0))
    if B <= 0:
        state.transport_disagreement_scores = {}
        state.transport_disagreement_margin = float("inf")
        return np.zeros(0, dtype=float)

    if routing is None:
        routing = np.asarray(getattr(state.fovea, "routing_scores", np.zeros(B)), dtype=float)
    else:
        routing = np.asarray(routing, dtype=float).reshape(-1)
    if routing.shape[0] != B:
        routing = np.resize(routing, (B,))

    total = sum(scores.values())
    if total <= 0.0:
        state.transport_disagreement_scores = {}
        state.transport_disagreement_margin = float("inf")
        return routing

    for block_id, val in scores.items():
        if 0 <= block_id < routing.shape[0]:
            routing[block_id] += weight * float(val)

    state.transport_disagreement_scores = {}
    state.transport_disagreement_margin = float("inf")
    return routing


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


def synthesize_action(state: AgentState) -> Action:
    STREAK_STEPS = 25
    low_streak = getattr(state, 'low_streak', 0)
    high_streak = getattr(state, 'high_streak', 0)

    action = Action()
    if low_streak >= STREAK_STEPS:
        action.command = CurriculumCommand.ADD_SHAPE
        state.low_streak = 0
        state.high_streak = 0
    elif high_streak >= STREAK_STEPS:
        action.command = CurriculumCommand.REMOVE_SHAPE
        state.low_streak = 0
        state.high_streak = 0
    return action
