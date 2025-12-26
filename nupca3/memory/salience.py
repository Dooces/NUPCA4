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
from typing import Dict, Optional, Set, List

import math
import numpy as np

from ..config import AgentConfig
from ..types import AgentState, Node


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
    """Infer a coarse abstraction level (band) for a node based on mask size."""
    mask = _get_node_mask(node)
    if mask is None:
        return 1
    support = int(np.count_nonzero(mask > 0.5))
    if support <= 0:
        return 1
    level = math.ceil(math.sqrt(float(support)))
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
    tau_a = float(getattr(cfg, "tau_a", getattr(cfg, "tau_base", 1.0)))
    
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
    tau_min = float(getattr(cfg, "tau_min", 0.1))
    tau_max = float(getattr(cfg, "tau_max", 10.0))
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


def compute_scores(
    state: AgentState,
    cfg: AgentConfig,
    observed_dims: Optional[Set[int]] = None,
) -> Dict[int, float]:
    """Compute salience scores u_j(t) for all experts (A5.1) with coverage-context nudges."""
    library = getattr(state, "library", None)
    if library is None:
        return {}

    nodes = getattr(library, "nodes", {})
    if not nodes:
        return {}

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
    max_out_degree = 1
    for node in nodes.values():
        deg = _get_node_out_degree(node)
        if deg > max_out_degree:
            max_out_degree = deg

    # Current latent estimate
    x_current = getattr(getattr(state, "buffer", None), "x_last", None)

    # Contextual resources
    context_reg = np.asarray(getattr(state, "context_register", np.zeros(0, dtype=float)), dtype=float).reshape(-1)
    context_tags = getattr(state, "node_context_tags", {})
    expert_debt = getattr(state, "coverage_expert_debt", {})
    band_debt = getattr(state, "coverage_band_debt", {})
    node_levels = getattr(state, "node_band_levels", {})

    scores: Dict[int, float] = {}

    for node_id, node in nodes.items():
        nid = int(node_id)
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
    theta_a = float(getattr(cfg, "theta_a", getattr(cfg, "salience_threshold", 0.5)))
    
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
        arousal=float(getattr(state, "arousal_prev", getattr(state, "arousal", 0.0))),
        s_int_need=float(getattr(state, "s_int_need_prev", getattr(state, "s_int_need", 0.0))),
        s_ext_th=float(getattr(state, "s_ext_th_prev", getattr(state, "s_ext_th", 0.0))),
    )


def bootstrap_scores(state: AgentState, cfg: AgentConfig) -> Dict[int, float]:
    """Compute initial scores for bootstrap on first timestep."""
    return compute_scores(state, cfg, observed_dims=None)


# =============================================================================
# Legacy Compatibility
# =============================================================================


def temperature_legacy(
    norm_margins: np.ndarray,
    arousal: float,
    stress,
    cfg: AgentConfig,
) -> float:
    """Legacy temperature interface.
    
    WARNING: This does not implement A5.2 correctly. Use compute_temperature().
    """
    # Convert legacy stress object to StressSignals
    s_E = float(getattr(stress, "s_E", getattr(stress, "s_E_need", 0.0)))
    s_D = float(getattr(stress, "s_D", getattr(stress, "s_D_need", 0.0)))
    s_int_need = max(s_E, s_D)
    s_ext_th = float(getattr(stress, "s_ext_th", 0.0))
    
    signals = StressSignals(
        arousal=arousal,
        s_int_need=s_int_need,
        s_ext_th=s_ext_th,
    )
    
    tau_eff, _ = compute_temperature(signals, cfg)
    return tau_eff


def score_experts_legacy(
    lib,
    buf,
    O_t: Set[int],
    cfg: AgentConfig,
) -> Dict[int, float]:
    """Legacy score interface.
    
    WARNING: This does not have full state context. Use compute_scores().
    """
    nodes = getattr(lib, "nodes", {})
    if not nodes:
        return {}
    
    # Simplified scoring without full A5.1
    alpha_pi = float(cfg.alpha_pi)
    alpha_ctx = float(cfg.alpha_ctx)
    
    scores: Dict[int, float] = {}
    
    for node_id, node in nodes.items():
        # Reliability
        pi_j = _get_node_reliability(node)
        
        # Context overlap
        mask = _get_node_mask(node)
        if mask is not None:
            mask_dims = set(int(i) for i in np.where(mask > 0.5)[0].tolist())
            overlap = len(mask_dims & O_t) if O_t else 0
            relevance = float(overlap) / max(len(mask_dims), 1)
        else:
            relevance = 0.0
        
        scores[int(node_id)] = alpha_pi * pi_j + alpha_ctx * relevance
    
    return scores


def activation_weights_legacy(
    scores: Dict[int, float],
    tau_eff: float,
    cfg: AgentConfig,
) -> Dict[int, float]:
    """Legacy activation interface.
    
    WARNING: Original used softmax which is not A5.3. This now uses sigmoid.
    """
    return compute_activations(scores, tau_eff, cfg)
