"""nupca3/edits/proposals.py

Structural edit proposal generation (A12/A14).

This module ONLY generates canonical `nupca3.types.EditProposal` objects.
It must NOT define shadow copies of EditKind/EditProposal/Evidence bundles.

REST-only invariants:
  - No library mutation here.
  - No incumbent set mutation here.
  - No queue popping here.

Axiom coverage:
  - MERGE (A12.3): high activation correlation among incumbents within same footprint.
  - PRUNE (A12.3): low reliability or inactive experts.
  - SPAWN (A12.4): persistent residual exceeds threshold for K coverage visits.
  - SPLIT (A12.4): residual covariance indicates separable subspaces.

v5 requirements enforced here
-----------------------------
- No getattr-based config fallbacks.
- Proposal generation is scan-safe in the online step loop:
  * No iteration over all nodes for SPLIT/PRUNE. Candidates come from
    incumbents buckets + bounded salience/active sets.
- Every EditProposal records `proposal_sig64` (the scan-proof signature snapshot
  at propose time) so REST-time creation can deterministically set unit addresses.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from ..config import AgentConfig
from ..incumbents import get_incumbent_bucket, iter_incumbent_buckets
from ..types import (
    AgentState,
    EditKind,
    EditProposal,
    FootprintResidualStats,
    MergeEvidence,
    Node,
    PruneEvidence,
    SpawnEvidence,
    SplitEvidence,
    infer_footprint,
)

# =============================================================================
# Proposal hyperparameters
# =============================================================================
# These are proposal-side thresholds. They are intentionally explicit (no silent
# config fallbacks). If you want them configurable, add them to AgentConfig.
_SPLIT_MIN_SAMPLES: int = 30
_SPLIT_INDEPENDENCE_THRESHOLD: float = 0.30

_THETA_MERGE: float = 0.85
_MERGE_MIN_HISTORY: int = 20

_THETA_CULL: float = 0.01
_T_INACTIVE: int = 100

_INCUMBENT_ACTIVITY_WINDOW: int = 50


# =============================================================================
# Helpers
# =============================================================================

def _require_proposal_sig64(state: AgentState) -> int:
    """Return the current scan-proof signature snapshot.

    In v5, structural proposals must carry a signature snapshot (`proposal_sig64`)
    so REST-time creation can set `unit_sig64` deterministically.
    """
    sig = state.last_sig64
    if sig is None:
        raise RuntimeError(
            "v5: state.last_sig64 is None during structural proposal generation; "
            "sig64 must be computed during the observation/decision stage before proposing edits."
        )
    return int(sig)


def _block_dims(state: AgentState, footprint: int) -> Set[int]:
    """Return block dimensions B_φ (A16.1), or empty set if undefined."""
    blocks = state.blocks
    if 0 <= footprint < len(blocks):
        return set(blocks[footprint])
    return set()


def _node_footprint(state: AgentState, node: Node) -> Optional[int]:
    """Resolve φ(node) from node.footprint or by inferring from its mask."""
    blocks = state.blocks
    if not blocks:
        return None

    fp = int(node.footprint)
    if fp >= 0 and fp < len(blocks):
        return fp

    mask = node.mask
    try:
        resolved = infer_footprint(mask, blocks)
    except Exception:
        return None

    node.footprint = int(resolved)
    return int(resolved)


def _compute_activation_correlation(
    activation_log: Dict[int, List[Tuple[int, float]]],
    node_a_id: int,
    node_b_id: int,
    min_overlap: int,
) -> Tuple[float, List[Tuple[int, float, float]]]:
    """Compute Pearson correlation between two experts' activation histories."""
    log_a = {t: a for (t, a) in activation_log.get(node_a_id, [])}
    log_b = {t: a for (t, a) in activation_log.get(node_b_id, [])}

    common = sorted(set(log_a.keys()) & set(log_b.keys()))
    if len(common) < min_overlap:
        return 0.0, []

    pairs: List[Tuple[int, float, float]] = [(int(t), float(log_a[t]), float(log_b[t])) for t in common]
    a_vals = np.array([p[1] for p in pairs], dtype=np.float64)
    b_vals = np.array([p[2] for p in pairs], dtype=np.float64)

    std_a = float(np.std(a_vals))
    std_b = float(np.std(b_vals))
    if std_a < 1e-12 or std_b < 1e-12:
        corr = 1.0 if (std_a < 1e-12 and std_b < 1e-12) else 0.0
        return corr, pairs

    corr_matrix = np.corrcoef(a_vals, b_vals)
    corr = float(corr_matrix[0, 1])
    if not np.isfinite(corr):
        corr = 0.0
    return corr, pairs


def _corr_from_cov(cov: np.ndarray) -> Optional[np.ndarray]:
    """Convert covariance matrix to correlation matrix; return None on failure."""
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        return None
    d = np.sqrt(np.maximum(np.diag(cov), 0.0))
    denom = np.outer(d, d)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.where(denom > 1e-12, cov / denom, 0.0)
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    if np.any(~np.isfinite(corr)):
        return None
    return corr


def _split_partition_from_stats(
    stats: FootprintResidualStats,
    dims_subset: List[int],
    independence_threshold: float,
) -> Optional[Tuple[Set[int], Set[int], float]]:
    """Derive a two-way partition from residual covariance stats."""
    if len(dims_subset) < 2:
        return None

    stats_dims = stats.dims
    cov_ema = stats.cov_ema
    if cov_ema is None or len(stats_dims) == 0:
        return None

    dim_to_idx = {int(d): i for i, d in enumerate(stats_dims)}
    idxs = [dim_to_idx[d] for d in dims_subset if d in dim_to_idx]
    if len(idxs) < 2:
        return None

    cov_sub = cov_ema[np.ix_(idxs, idxs)]
    corr_sub = _corr_from_cov(cov_sub)
    if corr_sub is None:
        return None

    n = corr_sub.shape[0]
    if n < 2:
        return None

    try:
        _, evecs = np.linalg.eigh(corr_sub)
    except np.linalg.LinAlgError:
        return None

    v = evecs[:, -2] if n > 1 else evecs[:, 0]
    g1_local = set(int(i) for i in np.where(v >= 0.0)[0].tolist())
    g2_local = set(int(i) for i in np.where(v < 0.0)[0].tolist())

    if not g1_local or not g2_local:
        return None

    cross_sum = 0.0
    cross_cnt = 0
    for i in g1_local:
        for j in g2_local:
            cross_sum += abs(float(corr_sub[i, j]))
            cross_cnt += 1
    avg_cross = cross_sum / float(max(1, cross_cnt))

    if avg_cross >= independence_threshold:
        return None

    group1 = {dims_subset[i] for i in g1_local}
    group2 = {dims_subset[i] for i in g2_local}
    if not group1 or not group2:
        return None

    return group1, group2, float(avg_cross)


def _estimate_mdl_cost(mask: np.ndarray, cfg: AgentConfig) -> float:
    """Estimate MDL cost from a mask, consistent with acceptance.compute_mdl_cost()."""
    return float(cfg.expert_base_cost) + float(cfg.expert_dim_cost) * float(np.sum(np.asarray(mask) > 0.5))


def _node_cost(node: Node, cfg: AgentConfig) -> float:
    """Return node.cost if valid, else derive from mask."""
    c = float(node.cost)
    if c > 0.0:
        return c
    return _estimate_mdl_cost(np.asarray(node.mask, dtype=np.float64), cfg)


# =============================================================================
# Proposal Generators
# =============================================================================

def propose_split_by_residual(state: AgentState, cfg: AgentConfig) -> List[EditProposal]:
    """Propose SPLIT edits based on residual covariance structure (A12.4)."""
    proposals: List[EditProposal] = []
    proposal_sig64 = _require_proposal_sig64(state)

    nodes = state.library.nodes
    residual_stats = state.residual_stats
    timestep = int(state.t)

    # Scan-safe: iterate incumbents per footprint (bounded), not all nodes.
    for footprint, incumbent_ids in iter_incumbent_buckets(state):
        if not incumbent_ids:
            continue

        stats = residual_stats.get(int(footprint))
        if stats is None or int(stats.n_updates) < _SPLIT_MIN_SAMPLES:
            continue

        block = _block_dims(state, int(footprint))
        if not block:
            continue

        for node_id in incumbent_ids:
            node = nodes.get(int(node_id))
            if node is None or bool(node.is_anchor):
                continue

            # Ensure footprint is consistent with bucket; infer only if missing.
            if int(node.footprint) < 0:
                node.footprint = int(footprint)
            elif int(node.footprint) != int(footprint):
                continue

            active = set(int(i) for i in np.where(np.asarray(node.mask) > 0.5)[0].tolist())
            dims_subset = sorted(list(active & block))
            if len(dims_subset) < 2:
                continue

            partition = _split_partition_from_stats(stats, dims_subset, _SPLIT_INDEPENDENCE_THRESHOLD)
            if partition is None:
                continue

            g1, g2, avg_cross = partition
            evidence = SplitEvidence(
                source_node_id=int(node_id),
                footprint=int(footprint),
                dims_group_1=set(g1),
                dims_group_2=set(g2),
                cross_correlation=float(avg_cross),
            )

            proposals.append(
                EditProposal(
                    kind=EditKind.SPLIT,
                    footprint=int(footprint),
                    proposal_sig64=proposal_sig64,
                    split_evidence=evidence,
                    priority=float(1.0 - avg_cross),
                    source_node_ids=[int(node_id)],
                    proposal_step=timestep,
                )
            )

    return proposals


def propose_merge_by_redundancy(state: AgentState, cfg: AgentConfig) -> List[EditProposal]:
    """Propose MERGE and PRUNE edits (A12.3)."""
    proposals: List[EditProposal] = []
    proposal_sig64 = _require_proposal_sig64(state)

    nodes = state.library.nodes
    activation_log = state.activation_log
    timestep = int(state.t)

    proposed_pairs: Set[Tuple[int, int]] = set()

    # =========================================================================
    # MERGE proposals: correlated incumbents within same footprint
    # =========================================================================
    for footprint, incumbent_ids in iter_incumbent_buckets(state):
        if not incumbent_ids:
            continue

        ids = [
            int(nid)
            for nid in incumbent_ids
            if int(nid) in nodes and not bool(nodes[int(nid)].is_anchor)
        ]
        if len(ids) < 2:
            continue

        for i, a_id in enumerate(ids):
            for b_id in ids[i + 1 :]:
                pair = (min(a_id, b_id), max(a_id, b_id))
                if pair in proposed_pairs:
                    continue

                corr, activation_pairs = _compute_activation_correlation(
                    activation_log, a_id, b_id, min_overlap=_MERGE_MIN_HISTORY
                )
                if corr <= _THETA_MERGE or not activation_pairs:
                    continue

                proposed_pairs.add(pair)
                node_a = nodes[a_id]
                node_b = nodes[b_id]

                cost_a = _node_cost(node_a, cfg)
                cost_b = _node_cost(node_b, cfg)

                mask_a = np.asarray(node_a.mask, dtype=np.float64)
                mask_b = np.asarray(node_b.mask, dtype=np.float64)
                if mask_a.shape == mask_b.shape:
                    merged_mask = np.maximum(mask_a, mask_b)
                    estimated_merged_cost = _estimate_mdl_cost(merged_mask, cfg)
                else:
                    estimated_merged_cost = max(cost_a, cost_b)

                evidence = MergeEvidence(
                    expert_a_id=int(a_id),
                    expert_b_id=int(b_id),
                    footprint=int(footprint),
                    correlation=float(corr),
                    evaluation_window_start=int(activation_pairs[0][0]),
                    evaluation_window_end=int(activation_pairs[-1][0]),
                    activation_pairs=activation_pairs,
                    cost_a=float(cost_a),
                    cost_b=float(cost_b),
                    estimated_merged_cost=float(estimated_merged_cost),
                )

                proposals.append(
                    EditProposal(
                        kind=EditKind.MERGE,
                        footprint=int(footprint),
                        proposal_sig64=proposal_sig64,
                        merge_evidence=evidence,
                        priority=float(corr),
                        source_node_ids=[int(a_id), int(b_id)],
                        proposal_step=timestep,
                    )
                )

    # =========================================================================
    # PRUNE proposals: scan-safe candidate set
    # =========================================================================
    candidate_ids: Set[int] = set()
    for _, inc_ids in iter_incumbent_buckets(state):
        candidate_ids |= {int(i) for i in inc_ids}
    candidate_ids |= {int(i) for i in state.salience_candidate_ids}
    candidate_ids |= {int(i) for i in state.active_set}

    for node_id in candidate_ids:
        node = nodes.get(int(node_id))
        if node is None or bool(node.is_anchor):
            continue

        reliability = float(node.reliability)
        last_active = int(node.last_active_step)
        time_since_active = int(timestep - last_active)

        reason: Optional[str] = None
        if reliability < _THETA_CULL:
            reason = "low_reliability"
        elif time_since_active > _T_INACTIVE:
            reason = "inactive"

        if reason is None:
            continue

        footprint = _node_footprint(state, node)
        if footprint is None:
            continue

        evidence = PruneEvidence(
            node_id=int(node_id),
            footprint=int(footprint),
            reason=reason,
            reliability=float(reliability),
            time_since_active=int(time_since_active),
        )

        priority = (
            (1.0 - reliability)
            if reason == "low_reliability"
            else float(time_since_active) / float(max(1, _T_INACTIVE))
        )

        proposals.append(
            EditProposal(
                kind=EditKind.PRUNE,
                footprint=int(footprint),
                proposal_sig64=proposal_sig64,
                prune_evidence=evidence,
                priority=float(priority),
                source_node_ids=[int(node_id)],
                proposal_step=timestep,
            )
        )

    return proposals


def propose_spawn_from_residual(state: AgentState, cfg: AgentConfig) -> List[EditProposal]:
    """Propose SPAWN edits when persistent residual exceeds threshold (A12.4)."""
    proposals: List[EditProposal] = []
    proposal_sig64 = _require_proposal_sig64(state)

    theta_spawn = float(cfg.theta_spawn)
    k_spawn_visits = int(cfg.K)
    if k_spawn_visits < 1:
        raise ValueError(f"cfg.K must be >= 1 (got {cfg.K!r})")

    nodes = state.library.nodes
    persistent_residuals = state.persistent_residuals
    timestep = int(state.t)

    for footprint, rstate in persistent_residuals.items():
        footprint = int(footprint)

        coverage_visits = int(rstate.coverage_visits)
        if coverage_visits < k_spawn_visits:
            continue

        residual_value = float(rstate.value)
        if residual_value <= theta_spawn:
            continue

        # A12.4: do not SPAWN if there exists an untried high-reliability incumbent.
        incumbent_ids = get_incumbent_bucket(state, footprint) or set()
        has_untried_incumbent = False

        last_obs_step = int(rstate.last_update_step)
        if last_obs_step <= 0:
            block_transitions = state.observed_transitions.get(footprint, [])
            if block_transitions:
                last_obs_step = int(block_transitions[-1].tau)

        if timestep - last_obs_step > _INCUMBENT_ACTIVITY_WINDOW:
            for inc_id in incumbent_ids:
                node = nodes.get(int(inc_id))
                if node is None or bool(node.is_anchor):
                    continue

                if float(node.reliability) < 0.3:
                    continue

                if timestep - int(node.last_active_step) > _INCUMBENT_ACTIVITY_WINDOW:
                    has_untried_incumbent = True
                    break

        if has_untried_incumbent:
            continue

        block = _block_dims(state, footprint)
        if not block:
            continue

        evidence = SpawnEvidence(
            footprint=footprint,
            persistent_residual=residual_value,
            coverage_visits=coverage_visits,
            block_dims=set(block),
            recent_transitions=[],  # populated by REST processor before fitting
        )

        proposals.append(
            EditProposal(
                kind=EditKind.SPAWN,
                footprint=footprint,
                proposal_sig64=proposal_sig64,
                spawn_evidence=evidence,
                priority=float(residual_value),
                source_node_ids=[],
                proposal_step=timestep,
            )
        )

    return proposals


# =============================================================================
# Main Entry Point
# =============================================================================

def propose_structural_edits(state: AgentState, cfg: AgentConfig) -> List[EditProposal]:
    """Generate structural edit proposals (A14.2), scan-safe in the online loop."""
    # During REST, edits are processed; proposals are queued during OPERATING.
    if bool(state.macro.rest):
        return []

    # Queue gating (A14): only generate proposals when the structural queue is on.
    queue_on = (float(state.macro.P_rest) >= float(cfg.P_rest_Theta_Q_on)) or bool(state.macro.Q_struct)
    if not queue_on:
        return []

    props: List[EditProposal] = []
    props.extend(propose_split_by_residual(state, cfg))
    props.extend(propose_merge_by_redundancy(state, cfg))
    props.extend(propose_spawn_from_residual(state, cfg))

    # Sort by priority (highest first) for queue insertion.
    props.sort(key=lambda p: float(p.priority), reverse=True)

    # Use REST processing budget as an upper bound for per-step queue insertion.
    max_props = int(cfg.max_edits_per_rest_step)
    if max_props < 1:
        return []
    return props[:max_props]
