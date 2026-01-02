"""nupca3/memory/salience.py

Salience scoring u_j(t), temperature τ^eff(t), and activation weights a_j(t).

Axiom coverage:
- A5.1: Score u_j(t) = α_π·π̄_j + α_deg·(deg⁺_j/deg⁺_max) + α_ctx·relevance(x(t),j)
- A5.2: Temperature τ^eff(t) = (τ_a/(1+β_sharp·s_int^need))·(1+β_open·s^play)
- A5.3: Activation a_j(t) = σ((u_j(t-1) - θ_a)/τ^eff(t))

This module computes salience and activation weights. It does NOT:
- Select the working set (that's in working_set.py per A4.2, A5.4)
- Modify expert parameters (REST-only per A12)

Timing discipline:
- Temperature uses lagged signals: s^ar(t-1), s_int^need(t-1), s_ext^th(t-1)
- Activation uses lagged scores: u_j(t-1)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Set, List, Tuple

import math
import numpy as np

from ..config import AgentConfig
from ..geometry.fovea import block_of_dim
from ..types import AgentState, Node
from ..incumbents import get_incumbent_bucket


# =============================================================================
# Result Types
# =============================================================================


@dataclass
class SalienceResult:
    """Salience computation result for one timestep."""
    
    scores: Dict[int, float]          # u_j(t) per node (A5.1)
    activations: Dict[int, float]     # a_j(t) per node (A5.3)
    temperature: float                 # τ^eff(t) (A5.2)
    s_play: float                      # Play/approach factor
    

@dataclass
class StressSignals:
    """Stress signals from A0.2-A0.3 for salience computation.
    
    These should come from the margin/arousal computation module.
    """
    arousal: float = 0.0              # s^ar(t-1) - hormonal arousal (A0.2)
    s_int_need: float = 0.0           # s_int^need(t-1) - internal need/deficit (A0.3)
    s_ext_th: float = 0.0             # s_ext^th(t-1) - external threat (A0.3)


# =============================================================================
# Helpers
# =============================================================================


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid σ(x) = 1/(1+e^(-x))."""
    if x >= 0:
        return 1.0 / (1.0 + np.exp(-x))
    else:
        exp_x = np.exp(x)
        return exp_x / (1.0 + exp_x)


def _get_node_reliability(node: Node) -> float:
    """Get node reliability π_j ∈ (0,1]."""
    pi = getattr(node, "reliability", None)
    if pi is not None:
        return float(max(0.0, min(1.0, pi)))
    pi = getattr(node, "pi_j", None)
    if pi is not None:
        return float(max(0.0, min(1.0, pi)))
    return 1.0


def _get_node_out_degree(node: Node) -> int:
    """Get out-degree deg⁺_j from DAG structure."""
    children = getattr(node, "children", None)
    if children is not None:
        return len(children)
    out_degree = getattr(node, "out_degree", None)
    if out_degree is not None:
        return int(out_degree)
    return 0


def _get_node_mask(node: Node) -> Optional[np.ndarray]:
    """Get node mask m_j."""
    mask = getattr(node, "mask", None)
    if mask is not None:
        return np.asarray(mask, dtype=np.float64)
    return None


def infer_node_band_level(node: Node, cfg: AgentConfig) -> int:
    """Infer a coarse abstraction level (band) for a node using supported dims."""
    support: Set[int] = set()
    mask = _get_node_mask(node)
    if mask is not None:
        mask_arr = np.asarray(mask, dtype=int).reshape(-1)
        support.update(int(x) for x in mask_arr if x >= 0)
    for idx_name in ("out_idx", "in_idx"):
        idx_arr = getattr(node, idx_name, None)
        if idx_arr is None:
            continue
        idx_np = np.asarray(idx_arr, dtype=int).reshape(-1)
        support.update(int(x) for x in idx_np if x >= 0)
    if not support:
        return 1
    level = math.ceil(math.sqrt(float(len(support))))
    return max(1, int(level))


def context_similarity(vec: np.ndarray, tag: np.ndarray) -> float:
    """Cosine-based similarity between gist and node context tags in [0,1]."""
    a = np.asarray(vec, dtype=float).reshape(-1)
    b = np.asarray(tag, dtype=float).reshape(-1)
    if a.size == 0 or b.size == 0:
        return 0.0
    min_len = min(a.size, b.size)
    if min_len <= 0:
        return 0.0
    if a.size != min_len:
        a = a[:min_len]
    if b.size != min_len:
        b = b[:min_len]
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    sim = float(np.dot(a, b) / (norm_a * norm_b))
    sim = max(-1.0, min(1.0, sim))
    return float(0.5 * (sim + 1.0))


# =============================================================================
# A5.2: Temperature (Need-Sharpened, Play-Opened)
# =============================================================================


def compute_play_factor(stress: StressSignals) -> float:
    """Compute safe play/approach factor s^play(t-1) (A5.2).
    
    s^play(t-1) = s^ar(t-1) · (1 - s_int^need(t-1)) · (1 - s_ext^th(t-1))
    
    Play is enabled when:
    - Aroused (s^ar > 0)
    - Not in deficit (s_int^need low)
    - Not threatened (s_ext^th low)
    """
    arousal = float(stress.arousal)
    s_int_need = float(stress.s_int_need)
    s_ext_th = float(stress.s_ext_th)
    
    s_play = arousal * (1.0 - s_int_need) * (1.0 - s_ext_th)
    return max(0.0, min(1.0, s_play))


def compute_temperature(
    stress: StressSignals,
    cfg: AgentConfig,
) -> tuple[float, float]:
    """Compute effective temperature τ^eff(t) (A5.2).
    
    τ^eff(t) = (τ_a / (1 + β_sharp · s_int^need(t-1))) · (1 + β_open · s^play(t-1))
    
    Temperature is:
    - Sharpened (lowered) by internal need → focus under deficit
    - Opened (raised) by play factor → exploration when safe and aroused
    
    Returns:
        (τ^eff, s^play) tuple
    """
    # Base temperature
    tau_a = float(cfg.tau_a)
    
    # Sharpening gain (focus under need)
    beta_sharp = float(cfg.beta_sharp)
    
    # Opening gain (explore when safe)
    beta_open = float(cfg.beta_open)
    
    # Compute play factor
    s_play = compute_play_factor(stress)
    
    # Apply formula
    s_int_need = float(stress.s_int_need)
    
    # Sharpening term: τ decreases as need increases
    sharpening = 1.0 + beta_sharp * s_int_need
    
    # Opening term: τ increases with safe arousal
    opening = 1.0 + beta_open * s_play
    
    tau_eff = (tau_a / max(sharpening, 1e-6)) * opening
    
    # Clamp to configured bounds
    tau_min = float(cfg.tau_min)
    tau_max = float(cfg.tau_max)
    tau_eff = max(tau_min, min(tau_max, tau_eff))
    
    return tau_eff, s_play


# =============================================================================
# A5.1: Score
# =============================================================================


def compute_relevance(
    node: Node,
    x_current: Optional[np.ndarray],
    observed_dims: Set[int],
    cfg: AgentConfig,
) -> float:
    """Compute context relevance term for u_j(t) (A5.1).
    
    relevance(x(t), j) measures how relevant expert j is to current observation.
    
    Implementation options:
    1. Mask overlap with observed dims (simple)
    2. Prediction error on recent observations (requires history)
    3. Feature similarity (requires embedding)
    
    Here we use mask overlap normalized by mask size.
    """
    mask = _get_node_mask(node)
    if mask is None:
        return 0.0
    
    # Mask support
    mask_dims = set(int(i) for i in np.where(mask > 0.5)[0].tolist())
    if not mask_dims:
        return 0.0
    
    # Overlap with observed dimensions
    if observed_dims:
        overlap = len(mask_dims & observed_dims)
        relevance = float(overlap) / float(len(mask_dims))
    else:
        # If no observation, use uniform relevance
        relevance = 0.5
    
    return relevance


def _gather_salience_block_candidates(
    state: AgentState,
    cfg: AgentConfig,
    nodes: Dict[int, Node],
    active_set: Set[int],
) -> Set[int]:
    """Roadmap the block footprints that should seed retrieval this step."""
    candidate_blocks: Set[int] = set()
    fovea = getattr(state, "fovea", None)
    if fovea is not None:
        candidate_blocks.update(int(b) for b in getattr(fovea, "current_blocks", set()) or set())

    pending = getattr(state, "pending_fovea_selection", {}) or {}
    pending_blocks = pending.get("blocks") or []
    candidate_blocks.update(int(b) for b in pending_blocks if isinstance(b, (int, np.integer)))
    forced_blocks = pending.get("forced_periph_blocks") or []
    candidate_blocks.update(int(b) for b in forced_blocks if isinstance(b, (int, np.integer)))

    observed_dims = set(getattr(state, "observed_dims", set()) or set())
    if observed_dims:
        for dim in observed_dims:
            block_id = block_of_dim(int(dim), cfg)
            candidate_blocks.add(int(block_id))

    if nodes and active_set:
        for nid in active_set:
            node = nodes.get(int(nid))
            if node is None:
                continue
            block_id = getattr(node, "footprint", -1)
            if block_id is None or int(block_id) < 0:
                continue
            candidate_blocks.add(int(block_id))

    B = int(cfg.B)
    if B > 0:
        valid_blocks = {int(b) for b in candidate_blocks if isinstance(b, (int, np.integer)) and 0 <= int(b) < B}
    else:
        valid_blocks = {int(b) for b in candidate_blocks if isinstance(b, (int, np.integer)) and b >= 0}

    if not valid_blocks and B > 0:
        valid_blocks.add(0)

    return valid_blocks


def _gather_dag_neighbors(
    nodes: Dict[int, Node],
    seed_ids: Set[int],
    depth: int,
) -> Set[int]:
    """Collect DAG neighbors within `depth` hops for the salience frontier."""
    neighbors: Set[int] = set()
    frontier = set(int(nid) for nid in seed_ids)
    visited = set(frontier)
    for _ in range(depth):
        next_frontier: Set[int] = set()
        for nid in frontier:
            node = nodes.get(int(nid))
            if node is None:
                continue
            parents = getattr(node, "parents", set()) or set()
            children = getattr(node, "children", set()) or set()
            for adj in parents:
                adj_id = int(adj)
                if adj_id in visited:
                    continue
                visited.add(adj_id)
                next_frontier.add(adj_id)
            for adj in children:
                adj_id = int(adj)
                if adj_id in visited:
                    continue
                visited.add(adj_id)
                next_frontier.add(adj_id)
        neighbors.update(next_frontier)
        frontier = next_frontier
        if not frontier:
            break
    return neighbors


def _prune_salience_history(
    state: AgentState,
    cfg: AgentConfig,
    nodes: Dict[int, Node],
) -> Set[int]:
    """Prune the linger history and surface valid lingering nodes."""
    linger_steps = int(getattr(cfg, "working_set_linger_steps", 0))
    if linger_steps <= 0:
        return set()
    history = getattr(state, "salience_recent_candidates", {}) or {}
    t_now = int(getattr(state, "t_w", 0))
    valid_ids: Set[int] = set()
    stale: List[int] = []
    for nid, last_seen in history.items():
        if t_now - int(last_seen) <= linger_steps:
            if nid in nodes:
                valid_ids.add(int(nid))
        else:
            stale.append(int(nid))
    for nid in stale:
        history.pop(nid, None)
    state.salience_recent_candidates = history
    return valid_ids


def _refresh_salience_history(
    state: AgentState,
    cfg: AgentConfig,
    nodes: Dict[int, Node],
    candidate_ids: Set[int],
    active_set: Set[int],
) -> None:
    """Refresh linger history after defining the current candidate universe."""
    linger_steps = int(getattr(cfg, "working_set_linger_steps", 0))
    if linger_steps <= 0:
        return
    history = getattr(state, "salience_recent_candidates", {}) or {}
    t_now = int(getattr(state, "t_w", 0))
    for nid in candidate_ids.union(active_set):
        if nid in nodes:
            history[int(nid)] = t_now
    state.salience_recent_candidates = history


def _collect_salience_candidate_ids(
    state: AgentState,
    cfg: AgentConfig,
    nodes: Dict[int, Node],
    active_set: Set[int],
) -> Set[int]:
    """Gather the working candidate nodes C_t reachable from the attended frontier."""
    candidate_ids: Set[int] = set()
    block_ids = _gather_salience_block_candidates(state, cfg, nodes, active_set)
    for block_id in block_ids:
        bucket = get_incumbent_bucket(state, int(block_id))
        for nid in bucket or set():
            if nid in nodes:
                candidate_ids.add(int(nid))

    anchors = getattr(state, "library").anchors
    for anchor_id in anchors:
        if anchor_id in nodes:
            candidate_ids.add(int(anchor_id))

    candidate_ids.update(nid for nid in active_set if nid in nodes)

    reach_depth = int(getattr(cfg, "salience_reach_depth", 1))
    if reach_depth > 0 and active_set:
        neighbors = _gather_dag_neighbors(nodes, active_set, reach_depth)
        candidate_ids.update(nid for nid in neighbors if nid in nodes)

    linger_ids = _prune_salience_history(state, cfg, nodes)
    candidate_ids.update(linger_ids)

    explore_budget = int(getattr(cfg, "salience_explore_budget", 0))
    if explore_budget > 0:
        _include_explore_nodes(candidate_ids, nodes, budget=explore_budget)

    _refresh_salience_history(state, cfg, nodes, candidate_ids, active_set)
    return candidate_ids


def _include_explore_nodes(
    candidate_ids: Set[int],
    nodes: Dict[int, Node],
    budget: int,
) -> None:
    """Add a small exploration budget of node IDs to keep retrieval responsive."""
    if budget <= 0:
        return
    added = 0
    for nid in sorted(nodes.keys()):
        if added >= budget:
            break
        if nid in candidate_ids:
            continue
        candidate_ids.add(int(nid))
        added += 1


def compute_scores(
    state: AgentState,
    cfg: AgentConfig,
    observed_dims: Optional[Set[int]] = None,
    *,
    candidate_node_ids: Optional[Iterable[int]] = None,
) -> Dict[int, float]:
    """Compute salience scores u_j(t) for all experts (A5.1) with coverage-context nudges."""
    state.salience_num_nodes_scored = 0
    state.salience_candidate_ids = set()
    library = getattr(state, "library", None)
    if library is None:
        return {}

    nodes = getattr(library, "nodes", {})
    if not nodes:
        return {}

    active_set = {int(nid) for nid in getattr(state, "active_set", set()) or set()}

    # Weights for score terms
    alpha_pi = float(cfg.alpha_pi)
    alpha_deg = float(cfg.alpha_deg)
    alpha_ctx_relevance = float(cfg.alpha_ctx_relevance)
    alpha_ctx_gist = float(cfg.alpha_ctx_gist)
    alpha_cov_exp = float(cfg.alpha_cov_exp)
    alpha_cov_band = float(cfg.alpha_cov_band)

    # Get observed dims
    if observed_dims is None:
        observed_dims = set(getattr(state, "observed_dims", set()))

    # Compute max out-degree for normalization
    node_levels = getattr(state, "node_band_levels", {})
    debug_exhaustive = bool(getattr(cfg, "salience_debug_exhaustive", False))
    if candidate_node_ids is None:
        if debug_exhaustive:
            candidate_ids = {int(nid) for nid in nodes.keys()}
        else:
            candidate_ids = _collect_salience_candidate_ids(state, cfg, nodes, active_set)
    else:
        candidate_ids = {int(nid) for nid in candidate_node_ids}

    candidate_ids = {nid for nid in candidate_ids if nid in nodes}
    raw_candidate_count = len(candidate_ids)

    max_candidates = int(cfg.salience_max_candidates)
    max_candidates = max(1, max_candidates)
    state.salience_candidate_limit = max_candidates
    state.salience_candidate_count_raw = raw_candidate_count

    limit_candidates = candidate_node_ids is None and not debug_exhaustive and raw_candidate_count > max_candidates
    if limit_candidates:
        truncated_list = sorted(candidate_ids)
        candidate_ids = set(truncated_list[:max_candidates])

    state.salience_candidates_truncated = raw_candidate_count > len(candidate_ids)
    state.salience_candidate_ids = set(candidate_ids)

    if candidate_node_ids is not None or debug_exhaustive:
        _refresh_salience_history(state, cfg, nodes, candidate_ids, active_set)
    if not candidate_ids:
        state.node_band_levels = node_levels
        state.salience_num_nodes_scored = 0
        return {}

    candidate_items: list[Tuple[int, Node]] = []
    for nid in candidate_ids:
        node = nodes.get(nid)
        if node is None:
            continue
        candidate_items.append((nid, node))
    if not candidate_items:
        state.node_band_levels = node_levels
        state.salience_num_nodes_scored = 0
        return {}
    state.salience_num_nodes_scored = len(candidate_items)
    max_out_degree = 1
    for _, node in candidate_items:
        deg = _get_node_out_degree(node)
        if deg > max_out_degree:
            max_out_degree = deg

    # Current latent estimate
    x_current = state.buffer.x_last

    # Contextual resources
    context_reg = np.asarray(getattr(state, "context_register", np.zeros(0, dtype=float)), dtype=float).reshape(-1)
    context_tags = getattr(state, "node_context_tags", {})
    expert_debt = getattr(state, "coverage_expert_debt", {})
    band_debt = getattr(state, "coverage_band_debt", {})

    scores: Dict[int, float] = {}

    for nid, node in candidate_items:
        pi_j = _get_node_reliability(node)
        deg_j = _get_node_out_degree(node)
        deg_normalized = float(deg_j) / float(max_out_degree)
        relevance = compute_relevance(node, x_current, observed_dims, cfg)

        level = node_levels.get(nid)
        if level is None:
            level = infer_node_band_level(node, cfg)
            node_levels[nid] = level

        base_score = alpha_pi * pi_j + alpha_deg * deg_normalized

        if alpha_ctx_relevance != 0.0:
            base_score += alpha_ctx_relevance * relevance

        if alpha_ctx_gist != 0.0:
            tag = context_tags.get(nid)
            if tag is None or tag.shape != context_reg.shape:
                tag = np.zeros_like(context_reg)
            base_score += alpha_ctx_gist * context_similarity(context_reg, tag)

        if alpha_cov_exp != 0.0:
            tau_exp = float(expert_debt.get(nid, 0))
            base_score += alpha_cov_exp * math.log1p(max(0.0, tau_exp))

        if alpha_cov_band != 0.0 and level is not None:
            band_tau = float(band_debt.get(level, 0))
            base_score += alpha_cov_band * math.log1p(max(0.0, band_tau))

        scores[nid] = float(base_score)

    state.node_band_levels = node_levels
    return scores


# =============================================================================
# A5.3: Activation Weights
# =============================================================================


def compute_activations(
    scores_prev: Dict[int, float],
    temperature: float,
    cfg: AgentConfig,
) -> Dict[int, float]:
    """Compute activation weights a_j(t) from lagged scores (A5.3).
    
    a_j(t) = σ((u_j(t-1) - θ_a) / τ^eff(t))
    
    This is a per-expert sigmoid, NOT softmax across experts.
    Each expert's activation depends only on its own score vs threshold.
    
    Args:
        scores_prev: Scores u_j(t-1) from previous timestep
        temperature: Effective temperature τ^eff(t)
        cfg: Configuration with threshold θ_a
    
    Returns:
        Dict mapping node_id → a_j(t)
    """
    # Activation threshold
    theta_a = float(cfg.theta_a)
    
    # Ensure temperature is positive
    tau = max(temperature, 1e-6)
    
    activations: Dict[int, float] = {}
    
    for node_id, u_j in scores_prev.items():
        # Per-expert sigmoid activation
        z = (u_j - theta_a) / tau
        a_j = _sigmoid(z)
        activations[int(node_id)] = float(a_j)
    
    return activations


# =============================================================================
# Main Computation Function
# =============================================================================


def compute_salience(
    state: AgentState,
    stress: StressSignals,
    scores_prev: Optional[Dict[int, float]],
    cfg: AgentConfig,
    observed_dims: Optional[Set[int]] = None,
) -> SalienceResult:
    """Compute full salience for current timestep (A5.1-A5.3).
    
    This is the main entry point for salience computation.
    
    Timing discipline:
    - Temperature uses stress signals from t-1
    - Activation uses scores from t-1
    - Current scores u_j(t) are computed for use at t+1
    
    Args:
        state: Current agent state
        stress: Stress signals (arousal, need, threat) from t-1
        scores_prev: Scores u_j(t-1) from previous step (None on first step)
        cfg: Agent configuration
        observed_dims: Current observed dimensions O_t
    
    Returns:
        SalienceResult with current scores, activations, and temperature
    """
    # Compute temperature τ^eff(t) using lagged stress signals
    temperature, s_play = compute_temperature(stress, cfg)
    
    # Compute current scores u_j(t) for use at t+1
    scores = compute_scores(state, cfg, observed_dims)
    
    # Compute activations a_j(t) from lagged scores u_j(t-1)
    if scores_prev is not None and scores_prev:
        activations = compute_activations(scores_prev, temperature, cfg)
    else:
        # First timestep: use current scores as bootstrap
        activations = compute_activations(scores, temperature, cfg)
    
    return SalienceResult(
        scores=scores,
        activations=activations,
        temperature=temperature,
        s_play=s_play,
    )


# =============================================================================
# Convenience Functions
# =============================================================================


def get_stress_signals(state: AgentState) -> StressSignals:
    """Extract stress signals from agent state for salience computation.
    
    These should be the (t-1) lagged values stored in state.
    """
    return StressSignals(
        arousal=float(getattr(state, "arousal_prev")),
        s_int_need=float(getattr(state, "s_int_need_prev")),
        s_ext_th=float(getattr(state, "s_ext_th_prev")),
    )


def bootstrap_scores(state: AgentState, cfg: AgentConfig) -> Dict[int, float]:
    """Compute initial scores for bootstrap on first timestep."""
    return compute_scores(state, cfg, observed_dims=None)
