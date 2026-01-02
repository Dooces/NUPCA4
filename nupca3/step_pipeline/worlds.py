"""
Helpers for managing multi-world hypotheses.
"""

from __future__ import annotations

import math
from typing import Dict, List, Set, Tuple

import numpy as np

from ..config import AgentConfig
from ..geometry.fovea import block_slices
from ..memory.completion import complete
from ..types import AgentState, WorldHypothesis

from .observations import _prior_obs_mae, _support_window_union
from .transport import TransportCandidate


def _normalize_world_weights(raw_weights: List[float]) -> List[float]:
    """Normalize a list of raw world scores into a probability simplex."""
    if not raw_weights:
        raise RuntimeError("world weights empty")
    arr = np.asarray(raw_weights, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise RuntimeError("non-finite world weights")
    total = float(np.sum(arr))
    if total <= 0.0:
        raise RuntimeError("world weights non-positive")
    return [float(val / total) for val in arr]


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
    # Keep merge stable even if upstream likelihood underflows.
    weights = []
    for w in group:
        val = float(getattr(w, "weight", 0.0))
        if not math.isfinite(val) or val < 0.0:
            val = 0.0
        weights.append(val)
    total = float(sum(weights))
    if total <= 0.0:
        weights = [1.0 for _ in group]
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


def _compute_block_signals(
    state: AgentState,
    cfg: AgentConfig,
    worlds: List[WorldHypothesis],
    D: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    B = int(getattr(cfg, "B", 0))
    if B <= 0:
        zeros = np.zeros(0, dtype=float)
        return zeros, zeros, zeros
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
    age = np.asarray(getattr(state.fovea, "block_age", np.zeros(B)), dtype=float)
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
    return (
        np.asarray(disagreement, dtype=float),
        np.asarray(innovation, dtype=float),
        np.asarray(periph_demand, dtype=float),
    )


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
        selection.append(best_candidate)
        seen_deltas.add(tuple(best_candidate.delta))
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
        if not math.isfinite(raw_weight) or raw_weight <= 0.0:
            raw_weight = 1.0
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
