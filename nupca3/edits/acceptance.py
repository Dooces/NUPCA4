"""nupca3/edits/acceptance.py

VERSION: v1.5b-perf.9 (2025-12-20)

WARNING (DO NOT IGNORE)
----------------------
This file implements acceptance math for A12 (ΔJ and feasibility checks).
It must remain aligned with the axiom list.

Do NOT change acceptance criteria or cost terms without understanding the axioms
and obtaining explicit permission from the project owner.

Acceptance evaluation for structural edits (A12).

Evaluates whether proposed edits should be accepted:
- A12.1: Net value ΔJ(e) = ΔF(e) - β·ΔL^MDL(e)
- A12.3: MERGE/PRUNE structural acceptance (REST-only)
- A12.4: SPAWN/SPLIT structural acceptance (REST-only)
- A4.4: Anti-aliasing check for new experts

Evaluation occurs during REST processing. This module contains only evaluation logic;
state mutation happens in rest_processor.py.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Set
import numpy as np

from ..types import (
    AgentState, Node, EditProposal, EditKind,
    MergeEvidence, SpawnEvidence, SplitEvidence, PruneEvidence,
    AcceptanceResult, TransitionRecord, infer_footprint
)
from ..config import AgentConfig
from ..memory.expert import predict
from ..incumbents import get_incumbent_bucket

# =============================================================================
# Permission Checks (A12.3)
# =============================================================================

def check_permit_struct(state: AgentState, cfg: AgentConfig, *, explain: bool = False) -> bool | tuple[bool, str]:
    """permit_struct(t) predicate for REST-time structural edits."""
    permitted = bool(state.is_rest)
    reason = "in_rest" if permitted else "not_rest"
    return (permitted, reason) if explain else permitted


# =============================================================================
# Delta Computations (A12.1)
# =============================================================================

def compute_mdl_cost(mask: np.ndarray, cfg: AgentConfig) -> float:
    """Simple MDL cost: base + per-active-dimension."""
    base_cost = getattr(cfg, "expert_base_cost", 1.0)
    dim_cost = getattr(cfg, "expert_dim_cost", 0.1)
    return float(base_cost + dim_cost * np.sum(mask > 0.5))


def estimate_delta_f_merge(state: AgentState, evidence: MergeEvidence, cfg: AgentConfig) -> float:
    """Heuristic: MERGE ΔF ~ 0; benefit is MDL reduction."""
    return 0.0


def estimate_delta_f_spawn(state: AgentState, evidence: SpawnEvidence, cfg: AgentConfig) -> float:
    """Heuristic: SPAWN captures fraction of persistent residual."""
    capture_fraction = 0.6
    return float(evidence.persistent_residual) * capture_fraction


def estimate_delta_f_split(state: AgentState, evidence: SplitEvidence, cfg: AgentConfig) -> float:
    """Heuristic: SPLIT improvement proportional to subspace independence."""
    independence = 1.0 - float(evidence.cross_correlation)
    base_improvement = 0.05
    return base_improvement * independence


def estimate_delta_f_prune(state: AgentState, evidence: PruneEvidence, cfg: AgentConfig) -> float:
    """Heuristic: PRUNE has negligible prediction impact."""
    return 0.0


def compute_delta_j(delta_f: float, delta_mdl: float, cfg: AgentConfig) -> float:
    """ΔJ(e) = ΔF(e) - β·ΔL^MDL(e) (A12.1)."""
    beta = getattr(cfg, "mdl_beta", 0.01)
    return float(delta_f - beta * delta_mdl)


def compute_delta_s(proposal: EditProposal, state: AgentState, cfg: AgentConfig) -> float:
    """Semantic shift bound ΔS(e). Kept minimal here."""
    return 0.0


def compute_delta_c(proposal: EditProposal, state: AgentState, cfg: AgentConfig) -> float:
    """Compute-cost change ΔC(e). Minimal sign-based heuristic."""
    expert_cost = getattr(cfg, "expert_base_cost", 1.0)

    if proposal.kind == EditKind.MERGE:
        return -float(expert_cost)
    if proposal.kind == EditKind.PRUNE:
        return -float(expert_cost)
    if proposal.kind == EditKind.SPAWN:
        return float(expert_cost)
    if proposal.kind == EditKind.SPLIT:
        return float(expert_cost)

    return 0.0


# =============================================================================
# MERGE Acceptance (A12.3)
# =============================================================================

def compute_merge_mse(
    node: Node,
    transitions: List[TransitionRecord],
    timesteps: Set[int],
    state_dim: int,
) -> float:
    """Compute MSE for an expert on specified timesteps."""
    if not timesteps:
        return float("inf")

    mask = node.mask
    se_sum = 0.0
    count = 0

    active_dims = set(np.where(mask > 0.5)[0].tolist())

    for trans in transitions:
        if trans.tau not in timesteps:
            continue

        x_tau_full, x_tau_plus_1_full = trans.full_vectors(state_dim)
        mu_pred = predict(node, x_tau_full)
        eval_dims = active_dims & trans.observed_dims_tau_plus_1
        if not eval_dims:
            continue

        se = 0.0
        for k in eval_dims:
            err = float(x_tau_plus_1_full[k] - mu_pred[k])
            se += err * err
        se /= float(len(eval_dims))

        se_sum += se
        count += 1

    if count == 0:
        return float("inf")
    return se_sum / float(count)


def evaluate_merge_replacement_consistent(
    state: AgentState,
    evidence: MergeEvidence,
    merged_node: Node,
    cfg: AgentConfig
) -> Tuple[bool, bool, bool]:
    """Replacement-consistent MERGE test (A12.3).

    Uses evidence.taus as authoritative T_AB if provided (compat with (3)).
    """
    epsilon_merge = getattr(cfg, "epsilon_merge", 0.05)

    # Authoritative T_AB
    T_AB: Set[int] = set(evidence.taus) if evidence.taus else set(tau for (tau, _, _) in evidence.activation_pairs)

    # Partition by pre-merge activations, but only on T_AB
    T_A: Set[int] = set()
    T_B: Set[int] = set()
    for tau, a_A, a_B in evidence.activation_pairs:
        if tau not in T_AB:
            continue
        if a_A >= a_B:
            T_A.add(tau)
        else:
            T_B.add(tau)

    if not T_A or not T_B:
        return False, False, False

    library = state.library
    node_a = library.nodes.get(evidence.expert_a_id)
    node_b = library.nodes.get(evidence.expert_b_id)
    if node_a is None or node_b is None:
        return False, False, False

    transitions = state.observed_transitions.get(evidence.footprint, [])

    state_dim = state.state_dim
    mse_a_on_T_A = compute_merge_mse(node_a, transitions, T_A, state_dim)
    mse_b_on_T_B = compute_merge_mse(node_b, transitions, T_B, state_dim)
    mse_c_on_T_A = compute_merge_mse(merged_node, transitions, T_A, state_dim)
    mse_c_on_T_B = compute_merge_mse(merged_node, transitions, T_B, state_dim)

    domain_a_ok = mse_c_on_T_A <= mse_a_on_T_A + epsilon_merge
    domain_b_ok = mse_c_on_T_B <= mse_b_on_T_B + epsilon_merge

    return (domain_a_ok and domain_b_ok), domain_a_ok, domain_b_ok


def _target_shape(*shapes: tuple[int, ...]) -> tuple[int, ...]:
    """Return per-axis maximum of the provided shapes, padding shorter dims with zeros."""
    if not shapes:
        return ()
    max_len = max(len(shape) for shape in shapes)
    result: list[int] = []
    for axis in range(max_len):
        values = [shape[axis] if axis < len(shape) else 0 for shape in shapes]
        result.append(max(values))
    return tuple(result)


def _pad_to_shape(arr: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    if not shape:
        return np.asarray(arr, dtype=float)
    out = np.zeros(shape, dtype=arr.dtype)
    slices = tuple(slice(0, min(dim, out_dim)) for dim, out_dim in zip(arr.shape, shape))
    out[slices] = arr[slices]
    return out


def create_merged_node(state: AgentState, evidence: MergeEvidence, cfg: AgentConfig) -> Optional[Node]:
    """Create candidate merged node for evaluation (no Node.footprint field)."""
    library = state.library
    node_a = library.nodes.get(evidence.expert_a_id)
    node_b = library.nodes.get(evidence.expert_b_id)
    if node_a is None or node_b is None:
        return None

    merged_mask = np.maximum(node_a.mask, node_b.mask)

    # Validate that merged mask stays inside evidence.footprint (block-aligned)
    try:
        phi = infer_footprint(merged_mask, state.blocks)
    except ValueError:
        return None
    if phi != evidence.footprint:
        return None

    total_pi = float(node_a.reliability + node_b.reliability)
    if total_pi <= 0.0:
        w_a = 0.5
        w_b = 0.5
    else:
        w_a = float(node_a.reliability) / total_pi
        w_b = float(node_b.reliability) / total_pi

    W_shape = _target_shape(node_a.W.shape, node_b.W.shape)
    b_shape = _target_shape(node_a.b.shape, node_b.b.shape)
    Sigma_shape = _target_shape(node_a.Sigma.shape, node_b.Sigma.shape)

    Wa = _pad_to_shape(node_a.W, W_shape)
    Wb = _pad_to_shape(node_b.W, W_shape)
    ba = _pad_to_shape(node_a.b, b_shape)
    bb = _pad_to_shape(node_b.b, b_shape)
    Sigma_a = _pad_to_shape(node_a.Sigma, Sigma_shape)
    Sigma_b = _pad_to_shape(node_b.Sigma, Sigma_shape)

    merged_W = w_a * Wa + w_b * Wb
    merged_b = w_a * ba + w_b * bb
    merged_Sigma = np.maximum(Sigma_a, Sigma_b)
    merged_reliability = max(float(node_a.reliability), float(node_b.reliability))
    merged_cost = max(float(node_a.cost), float(node_b.cost))

    return Node(
        node_id=-1,
        mask=merged_mask,
        W=merged_W,
        b=merged_b,
        Sigma=merged_Sigma,
        reliability=merged_reliability,
        cost=merged_cost,
        is_anchor=False,
        footprint=evidence.footprint,
        last_active_step=state.timestep,
        created_step=state.timestep
    )


# =============================================================================
# SPAWN/SPLIT Anti-Aliasing (A4.4)
# =============================================================================

def compute_distinguishability(
    state: AgentState,
    footprint: int,
    new_node: Node,
    existing_node: Node,
    cfg: AgentConfig
) -> float:
    """Δ_φ(p,q) = E_{t∈T_φ}[||μ_p(t+1|t) - μ_q(t+1|t)||_{1,φ}]"""
    transitions = state.observed_transitions.get(footprint, [])
    if not transitions:
        return float("inf")

    active_dims = set(np.where(new_node.mask > 0.5)[0].tolist())
    state_dim = state.state_dim

    total_diff = 0.0
    count = 0
    for trans in transitions:
        x_tau_full, _ = trans.full_vectors(state_dim)
        mu_new = predict(new_node, x_tau_full)
        mu_existing = predict(existing_node, x_tau_full)

        if not active_dims:
            continue

        diff = 0.0
        for k in active_dims:
            diff += abs(float(mu_new[k] - mu_existing[k]))
        diff /= float(len(active_dims))

        total_diff += diff
        count += 1

    if count == 0:
        return float("inf")
    return total_diff / float(count)


def check_anti_aliasing(
    state: AgentState,
    footprint: int,
    new_node: Node,
    cfg: AgentConfig
) -> Tuple[bool, Optional[int], bool]:
    """Anti-aliasing constraint (A4.4) keyed by footprint I_φ."""
    theta_alias = getattr(cfg, "theta_alias", 0.04)

    incumbent_ids = get_incumbent_bucket(state, footprint) or set()
    library = state.library

    for inc_id in incumbent_ids:
        incumbent = library.nodes.get(inc_id)
        if incumbent is None:
            continue

        # Only compare if incumbent is actually in this footprint (mask-aligned)
        try:
            inc_phi = infer_footprint(incumbent.mask, state.blocks)
        except ValueError:
            continue
        if inc_phi != footprint:
            continue

        delta = compute_distinguishability(state, footprint, new_node, incumbent, cfg)
        if delta < theta_alias:
            if float(new_node.reliability) > float(incumbent.reliability):
                return True, inc_id, True
            return False, inc_id, False

    return True, None, False


# =============================================================================
# Main Evaluation Functions
# =============================================================================

def evaluate_merge(proposal: EditProposal, state: AgentState, cfg: AgentConfig) -> AcceptanceResult:
    result = AcceptanceResult(accepted=False)

    evidence = proposal.merge_evidence
    if evidence is None:
        result.rejection_reason = "missing_merge_evidence"
        return result

    result.permit_struct = check_permit_struct(state, cfg)
    if not result.permit_struct:
        result.rejection_reason = "permit_struct_false"
        return result

    if evidence.estimated_merged_cost >= evidence.cost_a + evidence.cost_b:
        result.rejection_reason = "no_mdl_benefit"
        return result

    merged_node = create_merged_node(state, evidence, cfg)
    if merged_node is None:
        result.rejection_reason = "cannot_create_merged_node"
        return result

    overall_ok, domain_a_ok, domain_b_ok = evaluate_merge_replacement_consistent(
        state, evidence, merged_node, cfg
    )
    result.merge_domain_a_ok = domain_a_ok
    result.merge_domain_b_ok = domain_b_ok

    if not overall_ok:
        result.rejection_reason = f"replacement_consistent_failed:a={domain_a_ok},b={domain_b_ok}"
        return result

    delta_f = estimate_delta_f_merge(state, evidence, cfg)
    delta_mdl = float(evidence.estimated_merged_cost - (evidence.cost_a + evidence.cost_b))
    result.delta_j = compute_delta_j(delta_f, delta_mdl, cfg)
    result.delta_s = compute_delta_s(proposal, state, cfg)
    result.delta_c = compute_delta_c(proposal, state, cfg)

    epsilon = getattr(cfg, "acceptance_epsilon", 0.0)
    s_max = getattr(cfg, "s_max_semantic", 1.0)
    c_max = getattr(cfg, "c_max_edit", 10.0)

    result.delta_j_ok = result.delta_j >= epsilon
    result.delta_s_ok = result.delta_s <= s_max
    result.delta_c_ok = result.delta_c <= c_max
    result.quality_ok = True

    result.accepted = all([result.permit_struct, result.delta_j_ok, result.delta_s_ok, result.delta_c_ok, result.quality_ok])
    if not result.accepted:
        result.rejection_reason = "acceptance_thresholds_failed"
    return result


def evaluate_spawn(proposal: EditProposal, state: AgentState, cfg: AgentConfig) -> AcceptanceResult:
    result = AcceptanceResult(accepted=False)

    evidence = proposal.spawn_evidence
    if evidence is None:
        result.rejection_reason = "missing_spawn_evidence"
        return result

    result.permit_struct = check_permit_struct(state, cfg)
    if not result.permit_struct:
        result.rejection_reason = "permit_struct_false"
        return result

    candidate_mask = np.zeros(state.state_dim, dtype=float)
    for dim in evidence.block_dims:
        if 0 <= dim < state.state_dim:
            candidate_mask[dim] = 1.0

    # Validate mask aligns to the intended footprint (block)
    try:
        phi = infer_footprint(candidate_mask, state.blocks)
    except ValueError:
        result.rejection_reason = "spawn_mask_not_block_aligned"
        return result
    if phi != evidence.footprint:
        result.rejection_reason = "spawn_mask_wrong_footprint"
        return result

    candidate_node = Node(
        node_id=-1,
        mask=candidate_mask,
        W=np.eye(state.state_dim),
        b=np.zeros(state.state_dim),
        Sigma=np.ones(state.state_dim),
        reliability=0.5,
        cost=compute_mdl_cost(candidate_mask, cfg),
        is_anchor=False,
        last_active_step=state.timestep,
        created_step=state.timestep
    )

    ok_to_add, aliased_with, should_replace = check_anti_aliasing(state, evidence.footprint, candidate_node, cfg)
    result.anti_alias_ok = ok_to_add
    result.aliased_with = aliased_with
    result.replace_incumbent = should_replace

    if not ok_to_add and not should_replace:
        result.rejection_reason = f"aliased_with_{aliased_with}_no_improvement"
        return result

    delta_f = estimate_delta_f_spawn(state, evidence, cfg)
    delta_mdl = compute_mdl_cost(candidate_mask, cfg)
    result.delta_j = compute_delta_j(delta_f, delta_mdl, cfg)
    result.delta_s = compute_delta_s(proposal, state, cfg)
    result.delta_c = compute_delta_c(proposal, state, cfg)

    epsilon = getattr(cfg, "acceptance_epsilon", 0.0)
    s_max = getattr(cfg, "s_max_semantic", 1.0)
    c_max = getattr(cfg, "c_max_edit", 10.0)

    result.delta_j_ok = result.delta_j >= epsilon
    result.delta_s_ok = result.delta_s <= s_max
    result.delta_c_ok = result.delta_c <= c_max
    result.quality_ok = True

    result.accepted = all([
        result.permit_struct,
        result.delta_j_ok,
        result.delta_s_ok,
        result.delta_c_ok,
        (result.anti_alias_ok or result.replace_incumbent)
    ])

    if not result.accepted:
        result.rejection_reason = "acceptance_thresholds_failed"
    return result


def evaluate_split(proposal: EditProposal, state: AgentState, cfg: AgentConfig) -> AcceptanceResult:
    result = AcceptanceResult(accepted=False)

    evidence = proposal.split_evidence
    if evidence is None:
        result.rejection_reason = "missing_split_evidence"
        return result

    result.permit_struct = check_permit_struct(state, cfg)
    if not result.permit_struct:
        result.rejection_reason = "permit_struct_false"
        return result

    if not evidence.dims_group_1 or not evidence.dims_group_2:
        result.rejection_reason = "empty_partition_group"
        return result

    source_node = state.library.nodes.get(evidence.source_node_id)
    if source_node is None:
        result.rejection_reason = "source_node_not_found"
        return result

    mask_1 = np.zeros(state.state_dim, dtype=float)
    mask_2 = np.zeros(state.state_dim, dtype=float)
    for d in evidence.dims_group_1:
        if 0 <= d < state.state_dim:
            mask_1[d] = 1.0
    for d in evidence.dims_group_2:
        if 0 <= d < state.state_dim:
            mask_2[d] = 1.0

    # Validate both masks remain within the same footprint block
    try:
        phi1 = infer_footprint(mask_1, state.blocks)
        phi2 = infer_footprint(mask_2, state.blocks)
    except ValueError:
        result.rejection_reason = "split_masks_not_block_aligned"
        return result
    if phi1 != evidence.footprint or phi2 != evidence.footprint:
        result.rejection_reason = "split_masks_wrong_footprint"
        return result

    delta_f = estimate_delta_f_split(state, evidence, cfg)

    original_cost = float(source_node.cost)
    new_cost = compute_mdl_cost(mask_1, cfg) + compute_mdl_cost(mask_2, cfg)
    delta_mdl = float(new_cost - original_cost)

    result.delta_j = compute_delta_j(delta_f, delta_mdl, cfg)
    result.delta_s = compute_delta_s(proposal, state, cfg)
    result.delta_c = compute_delta_c(proposal, state, cfg)

    epsilon = getattr(cfg, "acceptance_epsilon", 0.0)
    s_max = getattr(cfg, "s_max_semantic", 1.0)
    c_max = getattr(cfg, "c_max_edit", 10.0)

    result.delta_j_ok = result.delta_j >= epsilon
    result.delta_s_ok = result.delta_s <= s_max
    result.delta_c_ok = result.delta_c <= c_max
    result.quality_ok = True
    result.anti_alias_ok = True

    result.accepted = all([result.permit_struct, result.delta_j_ok, result.delta_s_ok, result.delta_c_ok])
    if not result.accepted:
        result.rejection_reason = "acceptance_thresholds_failed"
    return result


def evaluate_prune(proposal: EditProposal, state: AgentState, cfg: AgentConfig) -> AcceptanceResult:
    result = AcceptanceResult(accepted=False)

    evidence = proposal.prune_evidence
    if evidence is None:
        result.rejection_reason = "missing_prune_evidence"
        return result

    result.permit_struct = check_permit_struct(state, cfg)
    if not result.permit_struct:
        result.rejection_reason = "permit_struct_false"
        return result

    theta_cull = getattr(cfg, "theta_cull", 0.01)
    t_inactive = getattr(cfg, "t_inactive", 100)

    node = state.library.nodes.get(evidence.node_id)
    if node is None:
        result.rejection_reason = "node_not_found"
        return result

    if node.is_anchor:
        result.rejection_reason = "cannot_prune_anchor"
        return result

    current_reliability = float(node.reliability)
    current_time_since_active = int(state.timestep - node.last_active_step)

    prune_ok = (current_reliability < theta_cull) or (current_time_since_active > t_inactive)
    if not prune_ok:
        result.rejection_reason = "prune_conditions_no_longer_met"
        return result

    delta_f = estimate_delta_f_prune(state, evidence, cfg)
    delta_mdl = -compute_mdl_cost(node.mask, cfg)

    result.delta_j = compute_delta_j(delta_f, delta_mdl, cfg)
    result.delta_s = 0.0
    result.delta_c = -float(node.cost)

    result.delta_j_ok = True
    result.delta_s_ok = True
    result.delta_c_ok = True
    result.quality_ok = True

    result.accepted = True
    return result


def evaluate_proposal(proposal: EditProposal, state: AgentState, cfg: AgentConfig) -> AcceptanceResult:
    if proposal.kind == EditKind.MERGE:
        return evaluate_merge(proposal, state, cfg)
    if proposal.kind == EditKind.SPAWN:
        return evaluate_spawn(proposal, state, cfg)
    if proposal.kind == EditKind.SPLIT:
        return evaluate_split(proposal, state, cfg)
    if proposal.kind == EditKind.PRUNE:
        return evaluate_prune(proposal, state, cfg)

    result = AcceptanceResult(accepted=False)
    result.rejection_reason = f"unknown_edit_kind:{proposal.kind}"
    return result
