# /mnt/data/NUPCA4/NUPCA4/nupca3/memory/working_set.py

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

import heapq
from typing import Dict, List, Set, Optional

from ..config import AgentConfig
from ..control.governor import BudgetMeter
from ..types import AgentState, ExpertLibrary, Node, WorkingSet
from ..geometry.fovea import select_fovea
from .primitives import compute_primitive_tokens
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


def get_retrieval_candidates(
    state: AgentState,
    cfg: AgentConfig,
    *,
    budget_meter: BudgetMeter | None = None,
) -> Set[int]:
    """Scan-proof retrieval (A4.3′). Fails loudly if any prerequisite is missing."""
    library = getattr(state, "library", None)
    if library is None:
        raise RuntimeError("v5 retrieval requires state.library")

    nodes = getattr(library, "nodes", {})
    sig_index = getattr(library, "sig_index", None)
    if sig_index is None:
        raise RuntimeError("v5 retrieval requires library.sig_index")

    _sig64_t = getattr(state, "last_sig64", None)

    fovea_blocks = set(getattr(state, "current_fovea", set()))
    if not fovea_blocks:
        raise RuntimeError("v5 retrieval requires current_fovea to be set")

    prev_active = set(getattr(state, "active_set", set()))

    value_of_compute = min(max(0.0, float(getattr(state, "value_of_compute", 0.0))), 1.0)
    candidate_scale = float(getattr(cfg, "value_of_compute_candidate_scale", 0.5))
    stage2_scale = float(getattr(cfg, "value_of_compute_stage2_scale", 0.5))
    candidate_boost = 1.0 + candidate_scale * value_of_compute
    stage2_boost = 1.0 + stage2_scale * value_of_compute
    C_cand_max = int(getattr(cfg, "C_cand_max", 0)) or int(getattr(cfg, "sig_query_cand_cap", 64))
    C_cand_max = max(1, C_cand_max)
    base_cap = C_cand_max
    degrade_level = 0
    if budget_meter is not None:
        degrade_level = max(0, int(getattr(budget_meter, "degrade_level", 0)))
    else:
        degrade_level = max(0, int(getattr(state, "budget_degradation_level", 0)))
    cap_divisor = 1 << min(degrade_level, 4)
    cand_cap = max(1, int(base_cap * candidate_boost))
    if degrade_level > 0:
        cand_cap = max(1, cand_cap // cap_divisor)
    cand_cap = min(cand_cap, C_cand_max)
    K_max = int(getattr(cfg, "K_max", 0)) or int(getattr(cfg, "max_retrieval_candidates", 1))
    K_max = max(1, K_max)
    N_max = max(1, int(getattr(cfg, "N_max", K_max)))
    K_cap = min(K_max, N_max)
    if degrade_level > 0:
        K_cap = max(1, K_cap // cap_divisor)

    query_tokens = compute_primitive_tokens(state, cfg)
    raw = list(sig_index.query(query_tokens, cand_cap=cand_cap))
    # raw: List[(node_id, evidence)]
    raw = sorted(raw, key=lambda x: (x[1], -int(x[0])), reverse=True)[:C_cand_max]

    alpha_err = float(cfg.sig_stage2_alpha_err)
    stage2_limit = max(1, int(len(raw) * stage2_boost))
    if degrade_level > 0 and stage2_limit > 0:
        stage2_limit = max(1, stage2_limit // cap_divisor)
    stage2_limit = min(stage2_limit, C_cand_max)
    top_heap: List[tuple[float, int]] = []
    processed_stage2 = 0
    for node_id, evidence in raw:
        if processed_stage2 >= stage2_limit:
            break
        processed_stage2 += 1
        nid = int(node_id)
        if nid < 0 or nid in prev_active:
            continue
        node = nodes.get(nid, None)
        if node is None:
            continue

        if not hasattr(sig_index, "get_error"):
            raise RuntimeError("PackedSigIndex must expose get_error() under v5")
        err = float(sig_index.get_error(int(nid), 2))

        score = float(evidence) - alpha_err * float(err)
        entry = (score, -nid)
        if len(top_heap) < K_cap:
            heapq.heappush(top_heap, entry)
        elif entry > top_heap[0]:
            heapq.heapreplace(top_heap, entry)

    if top_heap:
        top_sorted = sorted(top_heap, key=lambda x: (x[0], x[1]), reverse=True)
        return {int(-nid_neg) for _, nid_neg in top_sorted}

    return set()


# =============================================================================
# Main Selection Function
# =============================================================================


def select_working_set(
    state: AgentState,
    salience: Dict[int, float],
    cfg: AgentConfig,
    *,
    budget_meter: BudgetMeter | None = None,
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
    N_max = int(cfg.N_max)
    L_max = float(cfg.L_work_max)

    # =========================================================================
    # Step 1: Force-include anchors (A5.6)
    # =========================================================================
    anchor_ids: List[int] = []
    anchor_load_raw = 0.0

    # v5 requirement: anchor ids are maintained incrementally (no per-step full scans).
    # The library is the authority via `library.anchors` (bounded by |anchors|).
    anchors = getattr(library, "anchors", None)
    if anchors is None:
        raise RuntimeError("Library missing `anchors` set; v5 requires incremental anchor tracking")

    for nid in sorted(int(x) for x in anchors):
        node = nodes.get(int(nid))
        if node is None:
            continue
        if not bool(getattr(node, "is_anchor", False)):
            continue
        cost = _get_node_cost(node)
        anchor_ids.append(int(nid))
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
    retrieved = get_retrieval_candidates(state, cfg, budget_meter=budget_meter)

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

    # Thread-pinned units (planning contexts / forced workloads)
    pinned_units = set(getattr(state, "thread_pinned_units", set()) or set())
    for nid in pinned_units:
        if nid in nodes and nid not in anchor_set:
            candidate_pool.add(int(nid))
    # Optional linger: recently active nodes remain candidates for a short TTL.
    # v5 requirement: linger membership must be maintained incrementally or bounded;
    # never via per-step scans over all nodes.
    linger_steps = int(getattr(cfg, "working_set_linger_steps", 0))
    if linger_steps > 0:
        t_now = int(getattr(state, "t_w", 0))
        # Initialize / advance bounded linger index (expiry-ring).
        ring = getattr(state, "_ws_linger_ring", None)
        exp_map = getattr(state, "_ws_linger_exp", None)
        linger_ids = getattr(state, "_ws_linger_ids", None)
        last_t = getattr(state, "_ws_linger_last_t", None)
        ring_len = int(linger_steps) + 1

        def _reset_linger():
            nonlocal ring, exp_map, linger_ids, last_t
            ring = [set() for _ in range(ring_len)]
            exp_map = {}
            linger_ids = set()
            last_t = int(t_now)

        if ring is None or exp_map is None or linger_ids is None or last_t is None:
            _reset_linger()
        else:
            # If linger_steps changed or time moved backwards/too far, reset.
            if len(ring) != ring_len or int(t_now) < int(last_t) or (int(t_now) - int(last_t)) > ring_len:
                _reset_linger()
            else:
                # Advance step-by-step if we skipped a small number of steps.
                for s in range(int(last_t) + 1, int(t_now) + 1):
                    idx = int(s) % ring_len
                    due = ring[idx]
                    if due:
                        for nid in list(due):
                            if int(exp_map.get(int(nid), -1)) == int(s):
                                exp_map.pop(int(nid), None)
                                linger_ids.discard(int(nid))
                        due.clear()
                last_t = int(t_now)

        # Persist linger index back onto state (bounded structures).
        state._ws_linger_ring = ring
        state._ws_linger_exp = exp_map
        state._ws_linger_ids = linger_ids
        state._ws_linger_last_t = last_t

        # Add current linger set to candidate pool (excluding anchors).
        for nid in list(linger_ids):
            if int(nid) in anchor_set:
                continue
            if int(nid) in nodes:
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

    forced_set = set(int(n) for n in forced_ids)
    remaining_candidates = [int(n) for n in candidate_pool if int(n) not in forced_set]

    selected_non_anchors = forced_ids + _greedy_select(
        candidates=remaining_candidates,
        scores=scores,
        costs=costs,
        budget_load=remaining_load,
        budget_count=remaining_count,
    )

    # De-duplicate while preserving order (forced ids win).
    _seen = set()
    selected_non_anchors = [n for n in selected_non_anchors if not (n in _seen or _seen.add(n))]

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


    # Commit linger membership for the NEXT step (bounded update; no scans).
    if linger_steps > 0:
        t_now = int(getattr(state, "t_w", 0))
        ring_len = int(linger_steps) + 1
        ring = getattr(state, "_ws_linger_ring", None)
        exp_map = getattr(state, "_ws_linger_exp", None)
        linger_ids = getattr(state, "_ws_linger_ids", None)
        if ring is None or exp_map is None or linger_ids is None or len(ring) != ring_len:
            # If the linger index was not initialized above (or config changed), reset now.
            ring = [set() for _ in range(ring_len)]
            exp_map = {}
            linger_ids = set()

        for nid in selected_non_anchors:
            n = int(nid)
            old_exp = exp_map.get(n)
            if old_exp is not None:
                try:
                    ring[int(old_exp) % ring_len].discard(n)
                except Exception:
                    pass
            new_exp = int(t_now) + int(linger_steps) + 1
            exp_map[n] = int(new_exp)
            ring[int(new_exp) % ring_len].add(n)
            linger_ids.add(n)

        state._ws_linger_ring = ring
        state._ws_linger_exp = exp_map
        state._ws_linger_ids = linger_ids
        state._ws_linger_last_t = int(t_now)

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
