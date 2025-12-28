"""nupca3/memory/working_set.py

Working set selection A_t under cardinality and load constraints.

Axiom coverage:
- A4.2: Bounded working set |A_t| ≤ N_max, Σ L_j ≤ L_max^work
- A4.3: Cold storage retrieval integration (candidate universe)
- A5.4: GreedySelect with score = (a_j · π_j) / L_j
- A5.5: Effective complexity L^eff(t) = Σ a_j(t) · L_j
- A5.6: Anchor force-inclusion semantics

This module selects the active working set each step. It does NOT:
- Compute salience a_j(t) (that's in salience.py per A5.1-A5.3)
- Perform prediction/fusion (that's in completion.py per A7)
- Mutate library structure (REST-only per A12)
"""

from __future__ import annotations

from typing import Dict, List, Set, Optional

from ..config import AgentConfig
from ..types import AgentState, Node, Library, WorkingSet
from ..geometry.fovea import select_fovea
from ..incumbents import get_incumbent_bucket


# =============================================================================
# Helpers
# =============================================================================


def _get_node_cost(node: Node) -> float:
    """Get node cost L_j, handling attribute name variations."""
    # Try 'cost' first (v1.5b standard), fall back to 'L'
    cost = getattr(node, "cost", None)
    if cost is not None and cost > 0:
        return float(cost)
    cost = getattr(node, "L", None)
    if cost is not None and cost > 0:
        return float(cost)
    return 1.0  # Default cost


def _get_node_reliability(node: Node) -> float:
    """Get node reliability π_j ∈ (0,1]."""
    pi = getattr(node, "reliability", None)
    if pi is not None:
        return float(max(0.0, min(1.0, pi)))
    pi = getattr(node, "pi_j", None)
    if pi is not None:
        return float(max(0.0, min(1.0, pi)))
    return 1.0  # Default reliability


def _greedy_select(
    candidates: List[int],
    scores: Dict[int, float],
    costs: Dict[int, float],
    budget_load: float,
    budget_count: int,
) -> List[int]:
    """GreedySelect implementation (A5.4).

    Selects candidates by descending score, respecting load and count constraints.
    Score should be pre-computed as (a_j · π_j) / L_j.

    Args:
        candidates: Node IDs to consider (non-anchor)
        scores: Pre-computed score per candidate
        costs: L_j per candidate
        budget_load: Remaining load budget
        budget_count: Remaining count budget

    Returns:
        List of selected node IDs in selection order
    """
    if budget_count <= 0 or budget_load <= 0:
        return []

    # Sort by score descending
    ranked = sorted(
        [(nid, scores.get(nid, 0.0)) for nid in candidates],
        key=lambda x: x[1],
        reverse=True,
    )

    selected: List[int] = []
    used_load = 0.0

    for node_id, _score in ranked:
        if len(selected) >= budget_count:
            break

        cost = costs.get(node_id, 1.0)
        if used_load + cost > budget_load:
            continue  # Skip, try next (knapsack-style)

        selected.append(node_id)
        used_load += cost

    return selected


# =============================================================================
# Cold Storage Retrieval (A4.3)
# =============================================================================


def get_retrieval_candidates(
    state: AgentState,
    cfg: AgentConfig,
) -> Set[int]:
    """Get candidates from cold storage via block-keyed retrieval (A4.3).

    v1.5b definition:
      C^ret_t := ∪_{b∈F_t} (I_b ∩ (V \ A_{t-1}))

    Critical clarification (A4.3 keyed to A16.3 / A17.1):
    - Retrieval is *explicitly* keyed to the greedy_cov signal used to select
      the fovea blocks (A16.3): block_residual(b,t-1) + α_cov·log(1+age(b,t-1)).
    - This prevents retrieval from degenerating into "some candidates" when
      call sites forget to pass a fresh fovea.

    Implementation note:
    - v1.5b does not mandate a hard cap on |C^ret_t|. We apply an optional
      implementation cap (cfg.max_retrieval_candidates / cfg.max_candidates)
      purely to keep worst-case CPU bounded in toy harnesses.
      #ITOOKASHORTCUT: cap is an implementation bound, not an axiom.

    Args:
        state: AgentState (must include fovea residual/age and incumbents map)
        cfg: Agent configuration

    Returns:
        A (possibly capped) set of node IDs retrieved from cold storage.
    """
    library = getattr(state, "library", None)
    if library is None:
        return set()

    nodes = getattr(library, "nodes", {})

    prev_active = set(getattr(state, "active_set", set()))

    # Use the fovea blocks already chosen by the step pipeline.
    fovea_blocks = set(getattr(state, "current_fovea", set()))

    # Defensive fallback: recompute F_t from stored greedy_cov stats.
    if not fovea_blocks:
        try:
            fovea_blocks = set(select_fovea(state.fovea, cfg))
        except Exception:
            fovea_blocks = set()

    # Compute greedy_cov scores per block (explicit keying).
    import math

    alpha_cov = float(getattr(cfg, "alpha_cov", 0.10))
    resid = getattr(state.fovea, "block_residual", None)
    age = getattr(state.fovea, "block_age", None)

    block_scores: Dict[int, float] = {}
    for b in fovea_blocks:
        r_b = float(resid[b]) if resid is not None and int(b) < len(resid) else 0.0
        a_b = float(age[b]) if age is not None and int(b) < len(age) else 0.0
        block_scores[int(b)] = r_b + alpha_cov * math.log1p(max(0.0, a_b))

    # Score each retrieved node by its footprint block score.
    scored: Dict[int, float] = {}
    for b in fovea_blocks:
        score_b = float(block_scores.get(int(b), 0.0))
        bucket = get_incumbent_bucket(state, int(b))
        for node_id in bucket or set():
            nid = int(node_id)
            if nid in prev_active:
                continue
            if nid not in nodes:
                continue
            scored[nid] = max(float(scored.get(nid, -1e18)), score_b)

    if not scored:
        return set()

    # Optional implementation cap.
    max_ret = int(getattr(cfg, "max_retrieval_candidates", getattr(cfg, "max_candidates", 256)))
    max_ret = max(1, max_ret)

    ranked = sorted(scored.items(), key=lambda kv: (kv[1], -kv[0]), reverse=True)
    ranked = ranked[:max_ret]

    return set(nid for nid, _ in ranked)


# =============================================================================
# Main Selection Function
# =============================================================================


def select_working_set(
    state: AgentState,
    salience: Dict[int, float],
    cfg: AgentConfig,
) -> WorkingSet:
    """Select active experts A_t (A4.2, A5.4, A5.6).

    Implements the full working set selection per v1.5b axioms:

    1. Force-include anchors (A5.6):
       A_t ← P_anchor ∩ {j : alive(j)}
       Verify Σ L_j ≤ L_max^work (anchors must fit)

    2. Build candidate universe (A4.3):
       U_t := A_{t-1} ∪ C^ret_t ∪ A^anchor
       (Non-anchors from previous active + cold storage retrieval)

    3. GreedySelect non-anchors (A5.4):
       score_j = (a_j(t) · π_j(t)) / L_j
       Fill remaining capacity after anchors

    4. Compute load metrics (A5.5, A6.2):
       L^eff(t), L^eff_anc(t), L^eff_roll(t)

    Args:
        state: Agent state (provides library, incumbents, active_set, fovea)
        salience: Pre-computed salience a_j(t) from A5.3
        cfg: Agent configuration (N_max, L_work_max)

    Returns:
        WorkingSet with active nodes, weights, and load decomposition
    """
    library = getattr(state, "library", None)
    if library is None:
        return WorkingSet()

    nodes = getattr(library, "nodes", {})
    if not nodes:
        return WorkingSet()

    # Configuration bounds (A4.2)
    N_max = int(getattr(cfg, "N_max", getattr(cfg, "n_max", 64)))
    L_max = float(getattr(cfg, "L_work_max", getattr(cfg, "l_max_work", 48.0)))

    # =========================================================================
    # Step 1: Force-include anchors (A5.6)
    # =========================================================================

    anchor_ids: List[int] = []
    anchor_load_raw = 0.0

    for node_id, node in nodes.items():
        if not getattr(node, "is_anchor", False):
            continue
        cost = _get_node_cost(node)
        anchor_ids.append(int(node_id))
        anchor_load_raw += cost

    # A5.6 step 2: Verify anchors fit within budget
    if anchor_load_raw > L_max:
        # Configuration error - anchors must fit. Truncate by cost (best effort).
        anchor_ids_sorted = sorted(anchor_ids, key=lambda nid: _get_node_cost(nodes[nid]))
        anchor_ids = []
        anchor_load_raw = 0.0
        for nid in anchor_ids_sorted:
            cost = _get_node_cost(nodes[nid])
            if anchor_load_raw + cost <= L_max:
                anchor_ids.append(nid)
                anchor_load_raw += cost

    anchor_set = set(anchor_ids)

    # =========================================================================
    # Step 2: Build candidate universe (A4.3)
    # =========================================================================

    prev_active = set(getattr(state, "active_set", set()))
    retrieved = get_retrieval_candidates(state, cfg)

    # U_t = A_{t-1} ∪ C^ret_t (non-anchors only; anchors handled separately)
    candidate_pool: Set[int] = set()

    # Previous active (non-anchor)
    for nid in prev_active:
        if nid in nodes and nid not in anchor_set:
            candidate_pool.add(int(nid))

    # Retrieved from cold storage
    for nid in retrieved:
        if nid in nodes and nid not in anchor_set:
            candidate_pool.add(int(nid))

    # Optional linger: recently active nodes remain candidates for a short TTL.
    linger_steps = int(getattr(cfg, "working_set_linger_steps", 0))
    if linger_steps > 0:
        t_now = int(getattr(state, "t", 0))
        for nid, node in nodes.items():
            if nid in anchor_set:
                continue
            last_active = int(getattr(node, "last_active_step", -10**9))
            if t_now - last_active <= linger_steps:
                candidate_pool.add(int(nid))

    # Ensure incumbents for currently observed footprints are tried.
    observed_dims = set(getattr(state, "observed_dims", set()))
    blocks = getattr(state, "blocks", []) or []
    observed_blocks: Set[int] = set()
    if observed_dims and blocks:
        for b, dims in enumerate(blocks):
            if observed_dims.intersection(dims):
                observed_blocks.add(int(b))
    must_include: Set[int] = set()
    if observed_blocks:
        for b in observed_blocks:
            bucket = get_incumbent_bucket(state, int(b))
            for nid in bucket or set():
                if nid in nodes and nid not in anchor_set:
                    must_include.add(int(nid))
                    candidate_pool.add(int(nid))

    # =========================================================================
    # Step 3: GreedySelect non-anchors (A5.4)
    # =========================================================================

    # Compute selection scores: (a_j · π_j) / L_j
    scores: Dict[int, float] = {}
    costs: Dict[int, float] = {}

    for nid in candidate_pool:
        node = nodes[nid]
        a_j = float(salience.get(nid, 0.0))
        pi_j = _get_node_reliability(node)
        L_j = _get_node_cost(node)

        if L_j > 0:
            score = (a_j * pi_j) / L_j
        else:
            # Zero-cost nodes are maximally efficient
            score = a_j * pi_j * 1e6

        scores[nid] = score

        costs[nid] = L_j

    # Remaining budget after anchors
    remaining_load = L_max - anchor_load_raw
    remaining_count = N_max - len(anchor_ids)

    # Force-include observed incumbents (subject to budget), highest score first.
    must_include_list = sorted(
        list(must_include),
        key=lambda nid: (scores.get(nid, 0.0), -nid),
        reverse=True,
    )
    forced_ids: List[int] = []
    forced_load = 0.0
    for nid in must_include_list:
        cost = float(costs.get(nid, _get_node_cost(nodes[nid])))
        if remaining_count <= 0 or remaining_load - cost < -1e-9:
            continue
        forced_ids.append(int(nid))
        forced_load += cost
        remaining_load -= cost
        remaining_count -= 1

    selected_non_anchors = forced_ids + _greedy_select(
        candidates=list(candidate_pool),
        scores=scores,
        costs=costs,
        budget_load=remaining_load,
        budget_count=remaining_count,
    )

    # =========================================================================
    # Step 4: Assemble working set and compute metrics
    # =========================================================================

    # Anchors first, then non-anchors (order may matter for downstream)
    active = anchor_ids + selected_non_anchors

    weights: Dict[int, float] = {}
    total_load = 0.0
    effective_load = 0.0
    eff_anchor_load = 0.0
    eff_rollout_load = 0.0

    for nid in active:
        node = nodes[nid]
        is_anchor = getattr(node, "is_anchor", False)

        # Anchors get weight 1.0 if not in salience dict
        a_j = float(salience.get(nid, 1.0 if is_anchor else 0.0))
        L_j = _get_node_cost(node)

        weights[nid] = a_j
        total_load += L_j

        # Effective load (A5.5): L^eff = Σ a_j · L_j
        eff_contribution = a_j * L_j
        effective_load += eff_contribution

        # Decomposition (A6.2)
        if is_anchor:
            eff_anchor_load += eff_contribution
        else:
            eff_rollout_load += eff_contribution

    return WorkingSet(
        active=active,
        weights=weights,
        load=total_load,
        effective_load=effective_load,
        anchor_load=eff_anchor_load,
        rollout_load=eff_rollout_load,
        anchor_ids=anchor_ids,
        non_anchor_ids=selected_non_anchors,
    )


# =============================================================================
# Convenience Functions
# =============================================================================


def compute_effective_complexity(working_set: WorkingSet) -> float:
    """Return L^eff(t) = Σ_{j∈A_t} a_j(t) · L_j (A5.5)."""
    return working_set.effective_load


def get_load_decomposition(working_set: WorkingSet) -> tuple[float, float]:
    """Return (L^eff_anc, L^eff_roll) for horizon computation (A6.2)."""
    return working_set.anchor_load, working_set.rollout_load


# =============================================================================
# Legacy Compatibility Wrapper
# =============================================================================


def select_working_set_legacy(
    lib: Library,
    a_raw: Dict[int, float],
    cfg: AgentConfig,
) -> WorkingSet:
    """Legacy interface for backward compatibility.

    WARNING: This wrapper cannot implement full A4.3 retrieval or A5.6 anchors
    correctly because it lacks AgentState. Use select_working_set() instead.

    This provides a best-effort selection using only the library and raw weights.
    """
    nodes = getattr(lib, "nodes", {})
    if not nodes:
        return WorkingSet()

    N_max = int(getattr(cfg, "N_max", getattr(cfg, "n_max", 64)))
    L_max = float(getattr(cfg, "L_work_max", getattr(cfg, "l_max_work", 48.0)))

    # Separate anchors
    anchor_ids: List[int] = []
    anchor_load = 0.0
    non_anchor_candidates: List[int] = []

    for node_id, node in nodes.items():
        if getattr(node, "is_anchor", False):
            cost = _get_node_cost(node)
            if anchor_load + cost <= L_max:
                anchor_ids.append(int(node_id))
                anchor_load += cost
        else:
            if node_id in a_raw:
                non_anchor_candidates.append(int(node_id))

    # Compute scores for non-anchors
    scores: Dict[int, float] = {}
    costs: Dict[int, float] = {}

    for nid in non_anchor_candidates:
        node = nodes[nid]
        a_j = float(a_raw.get(nid, 0.0))
        pi_j = _get_node_reliability(node)
        L_j = _get_node_cost(node)

        if L_j > 0:
            scores[nid] = (a_j * pi_j) / L_j
        else:
            scores[nid] = a_j * pi_j * 1e6

        costs[nid] = L_j

    # Select non-anchors
    remaining_load = L_max - anchor_load
    remaining_count = N_max - len(anchor_ids)

    selected_non_anchors = _greedy_select(
        candidates=non_anchor_candidates,
        scores=scores,
        costs=costs,
        budget_load=remaining_load,
        budget_count=remaining_count,
    )

    # Assemble result
    active = anchor_ids + selected_non_anchors
    weights: Dict[int, float] = {}
    total_load = 0.0
    effective_load = 0.0
    eff_anchor_load = 0.0
    eff_rollout_load = 0.0

    for nid in active:
        node = nodes[nid]
        is_anchor = getattr(node, "is_anchor", False)
        a_j = float(a_raw.get(nid, 1.0 if is_anchor else 0.0))
        L_j = _get_node_cost(node)

        weights[nid] = a_j
        total_load += L_j
        eff = a_j * L_j
        effective_load += eff

        if is_anchor:
            eff_anchor_load += eff
        else:
            eff_rollout_load += eff

    return WorkingSet(
        active=active,
        weights=weights,
        load=total_load,
        effective_load=effective_load,
        anchor_load=eff_anchor_load,
        rollout_load=eff_rollout_load,
        anchor_ids=anchor_ids,
        non_anchor_ids=selected_non_anchors,
    )
