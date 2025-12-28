"""nupca3/edits/proposals.py

VERSION: v1.5b-perf.10 (2025-01-XX)

WARNING (DO NOT IGNORE)
-----------------------
This file implements the *proposal* side of A12 (structural edits) and A14 (queue dynamics).
It must remain consistent with the NUPCA3 v1.5b axiom list.

Do NOT modify behavior here without understanding the axioms and obtaining explicit
permission from the project owner. Seemingly small changes (e.g., adding a global scan)
can silently violate the intended computational scaling.

Edit proposal generation (A12/A14).

This module ONLY generates canonical `nupca3.types.EditProposal` objects.
It must NOT define shadow copies of EditKind/EditProposal/Evidence bundles.

REST-only invariants (enforced by not having the capability here):
  - No library mutation here.
  - No incumbent set mutation here.
  - No queue popping here.

Axiom coverage:
  - MERGE (A12.3): high activation correlation among incumbents within same footprint,
                   with L_C < L_A + L_B (MDL benefit) checked at acceptance.
  - PRUNE (A12.3): low reliability π_j < θ_cull OR TimeSinceActive(j) > T_inactive.
  - SPAWN (A12.4): persistent residual R_φ(t) > θ_spawn over K coverage visits,
                   and no incumbent in I_φ has reduced R_φ below threshold.
  - SPLIT (A12.4): residual covariance structure within footprint shows separable
                   subspaces, derived from state.residual_stats[footprint].

Important integration rule:
  - All evidence MUST be stored in the typed fields on EditProposal
    (merge_evidence/spawn_evidence/split_evidence/prune_evidence), not in a payload dict.

This file is intentionally conservative: it will never require additional state fields
beyond those defined in `nupca3.types.AgentState`.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from ..types import (
    AgentState,
    Node,
    EditProposal,
    EditKind,
    MergeEvidence,
    SpawnEvidence,
    SplitEvidence,
    PruneEvidence,
    infer_footprint,
    FootprintResidualStats,
)
from ..incumbents import get_incumbent_bucket, iter_incumbent_buckets

# Type hint only; implementation uses getattr() for compatibility.
try:  # pragma: no cover
    from ..config import AgentConfig  # noqa: F401
except Exception:  # pragma: no cover
    AgentConfig = object  # type: ignore


# =============================================================================
# Helpers
# =============================================================================


def _block_dims(state: AgentState, footprint: int) -> Set[int]:
    """Return block dimensions B_φ (A16.1), or empty set if undefined."""
    blocks = getattr(state, "blocks", [])
    if 0 <= footprint < len(blocks):
        return set(blocks[footprint])
    return set()


def _node_footprint(state: AgentState, node: Node) -> Optional[int]:
    """Resolve φ(node) from stored block_id or by inferring from mask."""
    blocks = getattr(state, "blocks", [])
    if not blocks:
        return None

    block_id = getattr(node, "block_id", getattr(node, "footprint", -1))
    if block_id is not None and 0 <= block_id < len(blocks):
        return int(block_id)

    mask = getattr(node, "mask", None)
    if mask is None:
        return None

    try:
        resolved = infer_footprint(mask, blocks)
    except Exception:
        return None

    node.block_id = resolved
    return resolved


def _compute_activation_correlation(
    activation_log: Dict[int, List[Tuple[int, float]]],
    node_a_id: int,
    node_b_id: int,
    min_overlap: int,
) -> Tuple[float, List[Tuple[int, float, float]]]:
    """Compute Pearson correlation between two experts' activation histories.

    Returns:
        (correlation, activation_pairs) where activation_pairs = [(τ, a_A(τ), a_B(τ)), ...]
        on timesteps τ where both have entries.

    Used for MERGE detection (A12.3): Correlation(a_A, a_B) > θ_merge.
    """
    log_a = {t: a for (t, a) in activation_log.get(node_a_id, [])}
    log_b = {t: a for (t, a) in activation_log.get(node_b_id, [])}

    common = sorted(set(log_a.keys()) & set(log_b.keys()))
    if len(common) < min_overlap:
        return 0.0, []

    pairs: List[Tuple[int, float, float]] = [
        (int(t), float(log_a[t]), float(log_b[t])) for t in common
    ]
    a_vals = np.array([p[1] for p in pairs], dtype=np.float64)
    b_vals = np.array([p[2] for p in pairs], dtype=np.float64)

    std_a = float(np.std(a_vals))
    std_b = float(np.std(b_vals))
    if std_a < 1e-12 or std_b < 1e-12:
        # Degenerate: constant histories are perfectly correlated if both constant.
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
    """Derive a two-way partition from residual covariance stats.

    Uses spectral bipartition on the correlation matrix over dims_subset.
    Returns (group1, group2, avg_cross_corr) if partition found, else None.

    Per A12.4: SPLIT produces experts whose masks partition φ's dimensions,
    still DoF-block-aligned.
    """
    if len(dims_subset) < 2:
        return None

    stats_dims = getattr(stats, "dims", [])
    cov_ema = getattr(stats, "cov_ema", None)
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

    # Fiedler vector for spectral bipartition
    v = evecs[:, -2] if n > 1 else evecs[:, 0]
    g1_local = set(int(i) for i in np.where(v >= 0.0)[0].tolist())
    g2_local = set(int(i) for i in np.where(v < 0.0)[0].tolist())

    if not g1_local or not g2_local:
        return None

    # Compute average cross-partition correlation
    cross_sum = 0.0
    cross_cnt = 0
    for i in g1_local:
        for j in g2_local:
            cross_sum += abs(float(corr_sub[i, j]))
            cross_cnt += 1
    avg_cross = cross_sum / float(max(1, cross_cnt))

    if avg_cross >= independence_threshold:
        return None  # Not sufficiently independent

    # Map back to global dimension indices
    group1 = {dims_subset[i] for i in g1_local}
    group2 = {dims_subset[i] for i in g2_local}
    if not group1 or not group2:
        return None

    return group1, group2, float(avg_cross)


def _estimate_mdl_cost(mask: np.ndarray, cfg: object) -> float:
    """Estimate MDL cost from mask, matching acceptance.compute_mdl_cost()."""
    base_cost = float(getattr(cfg, "expert_base_cost", 1.0))
    dim_cost = float(getattr(cfg, "expert_dim_cost", 0.1))
    return base_cost + dim_cost * float(np.sum(np.asarray(mask) > 0.5))


def _node_cost(node: Node, cfg: object) -> float:
    """Return node.cost if valid, else derive from mask."""
    c = float(getattr(node, "cost", 0.0))
    if c > 0.0:
        return c
    mask = getattr(node, "mask", None)
    if mask is None:
        return float(getattr(cfg, "expert_base_cost", 1.0))
    return _estimate_mdl_cost(np.asarray(mask, dtype=np.float64), cfg)


# =============================================================================
# Proposal Generators
# =============================================================================


def propose_split_by_residual(state: AgentState, cfg: object) -> List[EditProposal]:
    """Propose SPLIT edits based on residual covariance structure (A12.4).

    SPLIT is a special case of SPAWN that partitions φ's dimensions into two
    experts, both still DoF-block-aligned within the same footprint.

    Detection: If an expert's residual covariance shows approximately independent
    subspaces (low cross-correlation), splitting may reduce prediction error.
    """
    proposals: List[EditProposal] = []

    split_min_samples = int(getattr(cfg, "split_min_samples", 30))
    independence_threshold = float(getattr(cfg, "split_independence_threshold", 0.3))

    library = getattr(state, "library", None)
    if library is None:
        return proposals
    nodes = getattr(library, "nodes", {})

    residual_stats = getattr(state, "residual_stats", {})
    timestep = int(getattr(state, "timestep", 0))

    for node_id, node in nodes.items():
        if getattr(node, "is_anchor", False):
            continue

        footprint = _node_footprint(state, node)
        if footprint is None:
            continue

        stats = residual_stats.get(footprint)
        if stats is None:
            continue
        n_updates = int(getattr(stats, "n_updates", 0))
        if n_updates < split_min_samples:
            continue

        # Get dimensions within this footprint's block
        block = _block_dims(state, footprint)
        mask = getattr(node, "mask", None)
        if mask is None:
            continue
        active = set(int(i) for i in np.where(np.asarray(mask) > 0.5)[0].tolist())
        dims_subset = sorted(list(active & block))

        if len(dims_subset) < 2:
            continue

        partition = _split_partition_from_stats(stats, dims_subset, independence_threshold)
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
                split_evidence=evidence,
                priority=float(1.0 - avg_cross),  # Lower cross-corr = better split
                source_node_ids=[int(node_id)],
                proposal_step=timestep,
            )
        )

    return proposals


def propose_merge_by_redundancy(state: AgentState, cfg: object) -> List[EditProposal]:
    """Propose MERGE and PRUNE edits (A12.3).

    MERGE(A, B → C) proposal generated if:
      - φ(A) = φ(B) (same footprint, guaranteed by iterating within I_φ)
      - Correlation(a_A, a_B) > θ_merge
      - Estimated L_C < L_A + L_B (MDL benefit; verified exactly at acceptance)

    PRUNE(j) proposal generated if:
      - π_j < θ_cull OR TimeSinceActive(j) > T_inactive
      - j is not an anchor
    """
    proposals: List[EditProposal] = []

    theta_merge = float(getattr(cfg, "theta_merge", 0.85))
    min_overlap = int(getattr(cfg, "merge_min_history", 20))
    theta_cull = float(getattr(cfg, "theta_cull", 0.01))
    t_inactive = int(getattr(cfg, "t_inactive", 100))

    library = getattr(state, "library", None)
    if library is None:
        return proposals
    nodes = getattr(library, "nodes", {})

    activation_log = getattr(state, "activation_log", {})
    timestep = int(getattr(state, "timestep", 0))

    proposed_pairs: Set[Tuple[int, int]] = set()

    # =========================================================================
    # MERGE proposals: find correlated incumbent pairs within same footprint
    # =========================================================================

    for footprint, incumbent_ids in iter_incumbent_buckets(state):
        if not incumbent_ids:
            continue
        # Filter to non-anchor incumbents that exist in library
        ids = [
            int(nid)
            for nid in incumbent_ids
            if nid in nodes and not getattr(nodes[nid], "is_anchor", False)
        ]
        if len(ids) < 2:
            continue

        for i, a_id in enumerate(ids):
            for b_id in ids[i + 1 :]:
                pair = (min(a_id, b_id), max(a_id, b_id))
                if pair in proposed_pairs:
                    continue

                corr, activation_pairs = _compute_activation_correlation(
                    activation_log, a_id, b_id, min_overlap=min_overlap
                )

                if corr <= theta_merge or not activation_pairs:
                    continue

                proposed_pairs.add(pair)

                node_a = nodes[a_id]
                node_b = nodes[b_id]

                cost_a = _node_cost(node_a, cfg)
                cost_b = _node_cost(node_b, cfg)

                # Estimate merged cost: for same-footprint experts with overlapping
                # masks, merged cost ≈ max(cost_a, cost_b)
                mask_a = np.asarray(getattr(node_a, "mask", []), dtype=np.float64)
                mask_b = np.asarray(getattr(node_b, "mask", []), dtype=np.float64)
                if mask_a.shape == mask_b.shape:
                    merged_mask = np.maximum(mask_a, mask_b)
                    estimated_merged_cost = _estimate_mdl_cost(merged_mask, cfg)
                else:
                    estimated_merged_cost = max(cost_a, cost_b)

                # A12.3 requires L_C < L_A + L_B. If estimate doesn't satisfy,
                # still propose but acceptance will reject.
                # (No artificial adjustment here - let acceptance be authoritative)

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
                        merge_evidence=evidence,
                        priority=float(corr),  # Higher correlation = more redundant
                        source_node_ids=[int(a_id), int(b_id)],
                        proposal_step=timestep,
                    )
                )

    # =========================================================================
    # PRUNE proposals: low reliability or inactive experts
    # =========================================================================

    for node_id, node in nodes.items():
        if getattr(node, "is_anchor", False):
            continue

        reliability = float(getattr(node, "reliability", 1.0))
        last_active = int(getattr(node, "last_active_step", timestep))
        time_since_active = timestep - last_active

        reason: Optional[str] = None
        if reliability < theta_cull:
            reason = "low_reliability"
        elif time_since_active > t_inactive:
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
            else float(time_since_active) / float(max(1, t_inactive))
        )

        proposals.append(
            EditProposal(
                kind=EditKind.PRUNE,
                footprint=int(footprint),
                prune_evidence=evidence,
                priority=float(priority),
                source_node_ids=[int(node_id)],
                proposal_step=timestep,
            )
        )

    return proposals


def propose_spawn_from_residual(state: AgentState, cfg: object) -> List[EditProposal]:
    """Propose SPAWN edits when persistent residual exceeds threshold (A12.4).

    SPAWN for footprint φ proposed when:
      - R_φ(t) > θ_spawn over K distinct coverage visits
      - No existing incumbent in I_φ has reduced R_φ below θ_spawn

    The second condition is checked by verifying that high-reliability incumbents
    have been recently active (had opportunity to contribute to predictions).
    If an incumbent hasn't been active, we can't conclude SPAWN is needed.

    A SPAWN creates a new expert with mask aligned to block φ.
    Anti-aliasing (A4.4) is checked at acceptance time in REST.
    """
    proposals: List[EditProposal] = []

    theta_spawn = float(getattr(cfg, "theta_spawn", 0.25))
    k_spawn_visits = int(getattr(cfg, "k_spawn_visits", getattr(cfg, "K", 3)))
    incumbent_activity_window = int(getattr(cfg, "incumbent_activity_window", 50))

    library = getattr(state, "library", None)
    if library is None:
        return proposals
    nodes = getattr(library, "nodes", {})

    persistent_residuals = getattr(state, "persistent_residuals", {})
    timestep = int(getattr(state, "timestep", 0))

    for footprint, rstate in persistent_residuals.items():
        footprint = int(footprint)

        # Check coverage visit threshold
        coverage_visits = int(getattr(rstate, "coverage_visits", 0))
        if coverage_visits < k_spawn_visits:
            continue

        # Check residual threshold
        residual_value = float(getattr(rstate, "value", 0.0))
        if residual_value <= theta_spawn:
            continue

        # A12.4: "no existing incumbent in I_φ reduces R_φ below θ_spawn"
        # Check that incumbents have been given a fair chance.
        # If any high-reliability incumbent hasn't been active recently,
        # we can't conclude it wouldn't help if activated.
        incumbent_ids = get_incumbent_bucket(state, footprint) or set()
        has_untried_incumbent = False
        last_obs_step = int(getattr(rstate, "last_update_step", 0))
        if last_obs_step <= 0:
            observed_transitions = getattr(state, "observed_transitions", {})
            block_transitions = observed_transitions.get(footprint, [])
            if block_transitions:
                last_obs_step = int(getattr(block_transitions[-1], "tau", 0))

        if timestep - last_obs_step > incumbent_activity_window:
            for inc_id in incumbent_ids:
                node = nodes.get(inc_id)
                if node is None:
                    continue
                if getattr(node, "is_anchor", False):
                    continue  # Anchors always active, already contributing

                reliability = float(getattr(node, "reliability", 0.0))
                if reliability < 0.3:
                    continue  # Low-reliability incumbents don't block SPAWN

                last_active = int(getattr(node, "last_active_step", 0))
                if timestep - last_active > incumbent_activity_window:
                    # This incumbent hasn't been tried recently
                    has_untried_incumbent = True
                    break

        if has_untried_incumbent:
            continue  # Don't spawn until incumbents have been tried

        # Get block dimensions for mask construction
        block = _block_dims(state, footprint)
        if not block:
            continue

        evidence = SpawnEvidence(
            footprint=footprint,
            persistent_residual=residual_value,
            coverage_visits=coverage_visits,
            block_dims=set(block),
            recent_transitions=[],  # Populated by REST processor before fitting
        )

        proposals.append(
            EditProposal(
                kind=EditKind.SPAWN,
                footprint=footprint,
                spawn_evidence=evidence,
                priority=float(residual_value),  # Higher residual = more urgent
                source_node_ids=[],  # SPAWN creates new, doesn't modify existing
                proposal_step=timestep,
            )
        )

    return proposals


# =============================================================================
# Main Entry Point
# =============================================================================


def propose_structural_edits(state: AgentState, cfg: object) -> List[EditProposal]:
    """Generate structural edit proposals (A14.2).

    Collects proposals from:
      - propose_split_by_residual: SPLIT from independent subspaces (A12.4)
      - propose_merge_by_redundancy: MERGE from correlation, PRUNE from disuse (A12.3)
      - propose_spawn_from_residual: SPAWN from persistent residual (A12.4)

    Per A14.2, proposals accumulate in Q_struct during OPERATING:
        Q_struct(t) = Q_struct(t-1) + proposals(t)    if OPERATING
        Q_struct(t) = max(0, Q_struct(t-1) - edits_processed(t))  if REST

    Returns a priority-sorted list, truncated to cfg.max_proposals_per_step.
    """
    props: List[EditProposal] = []
    props.extend(propose_split_by_residual(state, cfg))
    props.extend(propose_merge_by_redundancy(state, cfg))
    props.extend(propose_spawn_from_residual(state, cfg))

    # Sort by priority (highest first) for queue insertion
    props.sort(key=lambda p: float(p.priority), reverse=True)

    max_props = int(getattr(cfg, "max_proposals_per_step", 10))
    return props[:max_props]
