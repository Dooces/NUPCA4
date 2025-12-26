"""nupca3/edits/rest_processor.py

VERSION: v1.5b-perf.9 (2025-12-20)

WARNING (DO NOT IGNORE)
----------------------
This file is the ONLY place where library DAG structure may be mutated.
It operationalizes REST-only structural edits per A12, with queue handling per A14.

Do NOT modify mutation rules, queue semantics, or cost accounting here without
first reconciling with the axiom list and obtaining explicit permission from the
project owner.

REST-phase structural edit processor (A12.3, A12.4, A14).

This module is the ONLY location where library DAG structure is mutated:
- Add/remove nodes
- Modify masks
- Update incumbent sets I_φ

Processing occurs during REST (rest(t) = 1) and respects:
- permit_struct(t) = 1{rest(t)=1} · 1{s^ar(t-1) < θ_ar^rest}
- Queue dynamics Q_struct per A14.2
- Consolidation cost b_cons(t) per A6.2

Architecture:
1. Read a *prefix* of Q_struct (A14.2) without mutating it
2. Evaluate acceptance via acceptance.py
3. If accepted, apply mutation to library
4. Report proposals_processed so macrostate can pop the processed prefix

Queue authority note (A14.2):
- Macrostate (nupca3/state/macrostate.py) is the sole authoritative owner of
  Q_struct appends/pops. This module must not change queue length.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np

from ..types import (
    AgentState, Node, EditProposal, EditKind,
    AcceptanceResult, TransitionRecord, infer_footprint
)
from ..config import AgentConfig
from .acceptance import (
    evaluate_proposal, check_permit_struct, create_merged_node,
    compute_mdl_cost
)


# =============================================================================
# Edit Application Results
# =============================================================================

@dataclass
class EditApplicationResult:
    """Result of applying an accepted edit."""
    success: bool
    nodes_added: List[int] = None
    nodes_removed: List[int] = None
    consolidation_cost: float = 0.0
    error_message: str = ""

    def __post_init__(self):
        if self.nodes_added is None:
            self.nodes_added = []
        if self.nodes_removed is None:
            self.nodes_removed = []


@dataclass
class RestProcessingResult:
    """Summary of REST processing step."""
    proposals_processed: int = 0
    edits_accepted: int = 0
    edits_rejected: int = 0
    total_consolidation_cost: float = 0.0
    permit_struct: bool = False
    permit_struct_reason: str = ""
    application_results: List[Tuple[EditProposal, AcceptanceResult, EditApplicationResult]] = None

    def __post_init__(self):
        if self.application_results is None:
            self.application_results = []


# =============================================================================
# DAG hygiene (required if Node.parents/children exist)
# =============================================================================

def _detach_node_edges(state: AgentState, node_id: int) -> None:
    """Detach node_id from neighbors before deletion (prevents dangling parent/child refs)."""
    lib = state.library
    node = lib.nodes.get(node_id)
    if node is None:
        return
    for p in list(node.parents):
        pnode = lib.nodes.get(p)
        if pnode is not None:
            pnode.children.discard(node_id)
    for c in list(node.children):
        cnode = lib.nodes.get(c)
        if cnode is not None:
            cnode.parents.discard(node_id)
    node.parents.clear()
    node.children.clear()


# =============================================================================
# Edit Application Functions
# =============================================================================

def fit_expert_from_transitions(
    transitions: List[TransitionRecord],
    mask: np.ndarray,
    state_dim: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit linear dynamics W, b, Σ from observed transitions.

    Uses least-squares on masked dimensions.
    """
    active_dims = list(np.where(mask > 0.5)[0])

    if not transitions or not active_dims:
        return np.eye(state_dim), np.zeros(state_dim), np.ones(state_dim)

    X_list = []
    Y_list = []

    for trans in transitions:
        # Only use transitions where all active dims were observed at τ+1 (targets exist).
        if not (set(active_dims) <= trans.observed_dims_tau_plus_1):
            continue
        x_tau_full, x_tau_plus_1_full = trans.full_vectors(state_dim)
        X_list.append(x_tau_full)
        Y_list.append(x_tau_plus_1_full)

    if len(X_list) < 2:
        return np.eye(state_dim), np.zeros(state_dim), np.ones(state_dim)

    X = np.array(X_list)  # (n_samples, state_dim)
    Y = np.array(Y_list)  # (n_samples, state_dim)

    W = np.eye(state_dim)
    b = np.zeros(state_dim)
    Sigma = np.ones(state_dim)

    X_aug = np.hstack([X, np.ones((X.shape[0], 1))])

    for k in active_dims:
        y_k = Y[:, k]
        try:
            coeffs, residuals, _, _ = np.linalg.lstsq(X_aug, y_k, rcond=None)
            W[k, :] = coeffs[:-1]
            b[k] = coeffs[-1]

            if len(residuals) > 0:
                Sigma[k] = max(0.01, float(residuals[0]) / max(1, len(y_k)))
            else:
                pred = X_aug @ coeffs
                Sigma[k] = max(0.01, float(np.mean((y_k - pred) ** 2)))
        except np.linalg.LinAlgError:
            pass

    return W, b, Sigma


def apply_merge(
    state: AgentState,
    proposal: EditProposal,
    cfg: AgentConfig
) -> EditApplicationResult:
    """Apply accepted MERGE edit to library."""
    result = EditApplicationResult(success=False)

    evidence = proposal.merge_evidence
    if evidence is None:
        result.error_message = "missing_merge_evidence"
        return result

    library = state.library

    merged_node = create_merged_node(state, evidence, cfg)
    if merged_node is None:
        result.error_message = "failed_to_create_merged_node"
        return result

    # (5) Footprint consistency: infer footprint from mask and validate
    try:
        phi = infer_footprint(merged_node.mask, state.blocks)
    except ValueError:
        result.error_message = "merged_mask_not_block_aligned"
        return result
    if phi != evidence.footprint:
        result.error_message = "merged_mask_wrong_footprint"
        return result

    new_id = library.add_node(merged_node)

    footprint = evidence.footprint
    state.incumbents.setdefault(footprint, set()).add(new_id)
    state.incumbents[footprint].discard(evidence.expert_a_id)
    state.incumbents[footprint].discard(evidence.expert_b_id)

    # Remove old nodes (detach edges first)
    _detach_node_edges(state, evidence.expert_a_id)
    _detach_node_edges(state, evidence.expert_b_id)
    library.remove_node(evidence.expert_a_id)
    library.remove_node(evidence.expert_b_id)

    state.active_set.discard(evidence.expert_a_id)
    state.active_set.discard(evidence.expert_b_id)

    state.activation_log.pop(evidence.expert_a_id, None)
    state.activation_log.pop(evidence.expert_b_id, None)

    result.success = True
    result.nodes_added = [new_id]
    result.nodes_removed = [evidence.expert_a_id, evidence.expert_b_id]
    result.consolidation_cost = getattr(cfg, "merge_consolidation_cost", 1.0)
    return result


def apply_spawn(
    state: AgentState,
    proposal: EditProposal,
    acceptance: AcceptanceResult,
    cfg: AgentConfig
) -> EditApplicationResult:
    """Apply accepted SPAWN edit to library."""
    result = EditApplicationResult(success=False)

    evidence = proposal.spawn_evidence
    if evidence is None:
        result.error_message = "missing_spawn_evidence"
        return result

    library = state.library
    footprint = evidence.footprint

    mask = np.zeros(state.state_dim, dtype=float)
    for dim in evidence.block_dims:
        if 0 <= dim < state.state_dim:
            mask[dim] = 1.0

    # (5) Footprint consistency: infer footprint from mask and validate
    try:
        phi = infer_footprint(mask, state.blocks)
    except ValueError:
        result.error_message = "spawn_mask_not_block_aligned"
        return result
    if phi != footprint:
        result.error_message = "spawn_mask_wrong_footprint"
        return result

    transitions = state.observed_transitions.get(footprint, [])
    recent_transitions = transitions[-50:]

    W, b, Sigma = fit_expert_from_transitions(recent_transitions, mask, state.state_dim)

    new_node = Node(
        node_id=-1,
        mask=mask,
        W=W,
        b=b,
        Sigma=Sigma,
        reliability=0.5,
        cost=compute_mdl_cost(mask, cfg),
        is_anchor=False,
        last_active_step=state.timestep,
        created_step=state.timestep
    )

    # Handle anti-aliasing replacement if requested
    if acceptance.replace_incumbent and acceptance.aliased_with is not None:
        old_id = acceptance.aliased_with
        old_node = library.nodes.get(old_id)
        if old_node is not None:
            new_node.reliability = max(new_node.reliability, old_node.reliability)

        _detach_node_edges(state, old_id)
        library.remove_node(old_id)
        state.incumbents.get(footprint, set()).discard(old_id)
        state.active_set.discard(old_id)
        state.activation_log.pop(old_id, None)

        result.nodes_removed = [old_id]

    new_id = library.add_node(new_node)

    state.incumbents.setdefault(footprint, set()).add(new_id)

    if footprint in state.persistent_residuals:
        state.persistent_residuals[footprint].value = 0.0
        state.persistent_residuals[footprint].coverage_visits = 0

    result.success = True
    result.nodes_added = [new_id]
    result.consolidation_cost = getattr(cfg, "spawn_consolidation_cost", 2.0)
    return result


def apply_split(
    state: AgentState,
    proposal: EditProposal,
    cfg: AgentConfig
) -> EditApplicationResult:
    """Apply accepted SPLIT edit to library."""
    result = EditApplicationResult(success=False)

    evidence = proposal.split_evidence
    if evidence is None:
        result.error_message = "missing_split_evidence"
        return result

    library = state.library
    source_node = library.nodes.get(evidence.source_node_id)
    if source_node is None:
        result.error_message = "source_node_not_found"
        return result

    footprint = evidence.footprint

    mask_1 = np.zeros(state.state_dim, dtype=float)
    for dim in evidence.dims_group_1:
        if 0 <= dim < state.state_dim:
            mask_1[dim] = 1.0

    mask_2 = np.zeros(state.state_dim, dtype=float)
    for dim in evidence.dims_group_2:
        if 0 <= dim < state.state_dim:
            mask_2[dim] = 1.0

    # (5) Footprint consistency: both masks must infer to the same footprint
    try:
        phi1 = infer_footprint(mask_1, state.blocks)
        phi2 = infer_footprint(mask_2, state.blocks)
    except ValueError:
        result.error_message = "split_masks_not_block_aligned"
        return result
    if phi1 != footprint or phi2 != footprint:
        result.error_message = "split_masks_wrong_footprint"
        return result

    transitions = state.observed_transitions.get(footprint, [])
    recent_transitions = transitions[-50:]

    W_1, b_1, Sigma_1 = fit_expert_from_transitions(recent_transitions, mask_1, state.state_dim)
    W_2, b_2, Sigma_2 = fit_expert_from_transitions(recent_transitions, mask_2, state.state_dim)

    node_1 = Node(
        node_id=-1,
        mask=mask_1,
        W=W_1,
        b=b_1,
        Sigma=Sigma_1,
        reliability=source_node.reliability,
        cost=compute_mdl_cost(mask_1, cfg),
        is_anchor=False,
        last_active_step=state.timestep,
        created_step=state.timestep
    )
    node_2 = Node(
        node_id=-1,
        mask=mask_2,
        W=W_2,
        b=b_2,
        Sigma=Sigma_2,
        reliability=source_node.reliability,
        cost=compute_mdl_cost(mask_2, cfg),
        is_anchor=False,
        last_active_step=state.timestep,
        created_step=state.timestep
    )

    new_id_1 = library.add_node(node_1)
    new_id_2 = library.add_node(node_2)

    state.incumbents.setdefault(footprint, set()).add(new_id_1)
    state.incumbents[footprint].add(new_id_2)
    state.incumbents[footprint].discard(evidence.source_node_id)

    _detach_node_edges(state, evidence.source_node_id)
    library.remove_node(evidence.source_node_id)
    state.active_set.discard(evidence.source_node_id)
    state.activation_log.pop(evidence.source_node_id, None)

    result.success = True
    result.nodes_added = [new_id_1, new_id_2]
    result.nodes_removed = [evidence.source_node_id]
    result.consolidation_cost = getattr(cfg, "split_consolidation_cost", 2.0)
    return result


def apply_prune(
    state: AgentState,
    proposal: EditProposal,
    cfg: AgentConfig
) -> EditApplicationResult:
    """Apply accepted PRUNE edit to library."""
    result = EditApplicationResult(success=False)

    evidence = proposal.prune_evidence
    if evidence is None:
        result.error_message = "missing_prune_evidence"
        return result

    library = state.library
    node_id = evidence.node_id

    if node_id not in library.nodes:
        result.error_message = "node_not_found"
        return result

    footprint = evidence.footprint

    _detach_node_edges(state, node_id)
    library.remove_node(node_id)

    if footprint in state.incumbents:
        state.incumbents[footprint].discard(node_id)

    state.active_set.discard(node_id)
    state.activation_log.pop(node_id, None)

    result.success = True
    result.nodes_removed = [node_id]
    result.consolidation_cost = getattr(cfg, "prune_consolidation_cost", 0.5)
    return result


def apply_edit(
    state: AgentState,
    proposal: EditProposal,
    acceptance: AcceptanceResult,
    cfg: AgentConfig
) -> EditApplicationResult:
    """Dispatch edit application based on kind."""
    if proposal.kind == EditKind.MERGE:
        return apply_merge(state, proposal, cfg)
    if proposal.kind == EditKind.SPAWN:
        return apply_spawn(state, proposal, acceptance, cfg)
    if proposal.kind == EditKind.SPLIT:
        return apply_split(state, proposal, cfg)
    if proposal.kind == EditKind.PRUNE:
        return apply_prune(state, proposal, cfg)
    return EditApplicationResult(success=False, error_message=f"unknown_edit_kind:{proposal.kind}")


# =============================================================================
# Queue Management (A14.2)
# =============================================================================

# A14.2 Queue ownership purity
# ----------------------------
#
# Per v1.5b, the Macrostate module is authoritative for Q_struct updates:
#   - OPERATING: append proposals_t into Q_struct(t)
#   - REST: pop `edits_processed_t` items from the *front* of Q_struct
#
# Therefore, THIS module must not mutate the queue (no pop/append/sort). We only
# *read a prefix* of the current queue and report `proposals_processed` so the
# Macrostate queue update can perform the authoritative pop at step end.


def peek_queue_prefix(queue: List[EditProposal], max_count: int = 1) -> List[EditProposal]:
    """Return up to max_count proposals from the *front* of queue, without mutation."""
    n = max(0, int(max_count))
    if n <= 0:
        return []
    return list((queue or [])[:n])


def get_queue_size(state: AgentState) -> int:
    """Return |Q_struct| (compat helper)."""
    return len(state.q_struct)


# =============================================================================
# Main Processing Function
# =============================================================================

def process_struct_queue(
    state: AgentState,
    cfg: AgentConfig,
    *,
    queue: Optional[List[EditProposal]] = None,
    max_edits: Optional[int] = None,
) -> RestProcessingResult:
    """Process a prefix of the structural edit queue during REST.

    Axiom coverage
    -------------
    - A12.3/A12.4: structural edits are REST-only
    - A14.2: Q_struct mutation is owned by Macrostate.update_queue

    Inputs:
      - state: AgentState (library/baselines may be mutated if edits accepted)
      - cfg: AgentConfig
      - queue: read-only view of current Q_struct (defaults to state.q_struct)
      - max_edits: maximum number of proposals to *process* this REST step

    Outputs:
      - RestProcessingResult where proposals_processed is the number of proposals
        consumed conceptually this step. The step pipeline must pass this count
        to Macrostate.evolve_macrostate so the authoritative pop occurs.

    Notes
    -----
    This function MUST NOT mutate the queue (no pop/append/sort). It only reads
    a prefix and reports how many were processed.
    """
    result = RestProcessingResult()

    if not state.is_rest:
        result.permit_struct = False
        result.permit_struct_reason = "not_rest"
        return result

    permit_res = check_permit_struct(state, cfg, explain=True)
    if isinstance(permit_res, tuple):
        permit_struct, permit_reason = permit_res
    else:
        permit_struct = bool(permit_res)
        permit_reason = "unknown"
    result.permit_struct = permit_struct
    result.permit_struct_reason = permit_reason
    if not permit_struct:
        return result

    if queue is None:
        queue = list(state.q_struct)

    if max_edits is None:
        max_edits = int(getattr(cfg, "max_edits_per_rest_step", 3))

    proposals = peek_queue_prefix(queue, max_edits)
    result.proposals_processed = len(proposals)

    total_consolidation_cost = 0.0

    for proposal in proposals:
        acceptance = evaluate_proposal(proposal, state, cfg)

        if acceptance.accepted:
            app_result = apply_edit(state, proposal, acceptance, cfg)

            if app_result.success:
                result.edits_accepted += 1
                total_consolidation_cost += float(app_result.consolidation_cost)

                # A3.3: record that a structural edit occurred at time t.
                try:
                    from ..state.baselines import commit_struct_edit
                    state.baselines = commit_struct_edit(state.baselines, t=int(state.t))
                except Exception:
                    # Best-effort; keep edit application authoritative.
                    pass
            else:
                result.edits_rejected += 1

            result.application_results.append((proposal, acceptance, app_result))
        else:
            result.edits_rejected += 1
            result.application_results.append(
                (proposal, acceptance, EditApplicationResult(success=False))
            )

    # A6.2 consolidation cost channel.
    state.b_cons = float(total_consolidation_cost)
    result.total_consolidation_cost = float(total_consolidation_cost)

    return result


def add_proposals_to_queue(state: AgentState, proposals: List[EditProposal]) -> List[EditProposal]:
    """Deprecated queue helper (A14.2 ownership purity).

    v1.5b A14.2 makes Q_struct(t) a macrostate variable updated only by the
    macrostate evolution rule. REST processors must not mutate Q_struct.

    This helper is kept only for older call sites. It returns the proposals
    unchanged so the *caller* (typically the step pipeline) can pass them to
    `evolve_macrostate(..., proposals_t=...)`.

    Args:
        state: AgentState (unused; retained for signature compatibility)
        proposals: proposed structural edits to enqueue

    Returns:
        A list copy of proposals, suitable for macrostate-managed enqueue.
    """
    _ = state
    return list(proposals)
