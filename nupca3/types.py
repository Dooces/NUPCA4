#!/usr/bin/env python3
"""nupca3/types.py

VERSION: v1.5b-perf.13 (types-contract reconciliation)

This file is the *shared contract* across NUPCA3 modules.
It exists to keep cross-module imports stable and to prevent "one-missing-symbol"
import cascades. Many modules import these names directly.

WARNING:
- Do not rename or remove symbols from this file without first scanning the repo
  for `from nupca3.types import ...` imports and updating all call sites.
- Do not change field semantics unless you understand the axioms and the
  downstream modules that consume these structures.

Axiom coverage:
- Cross-cutting: A0/A4/A5/A6/A7/A12/A14/A16 (shared data structures referenced by
  those mechanisms). This file intentionally contains no policy logic; only types
  and small, local helpers (e.g., footprint inference).


[AXIOM_CLARIFICATION_ADDENDUM — Representation & Naming]

- Terminology: identifiers like "Expert" in this codebase refer to NUPCA3 **abstraction/resonance nodes** (a "constellation"), not conventional Mixture-of-Experts "experts" or router-based MoE.

- Representation boundary (clarified intent of v1.5b): the completion/fusion operator (A7) is defined over an **encoded, multi-resolution abstraction vector** \(x(t)\). Raw pixels may exist only in a transient observation buffer for the current step; **raw pixel values must never be inserted into long-term storage** (library/cold storage) and must not persist across REST boundaries.

- Decomposition intuition: each node is an operator that *factors out* a predictable/resonant component on its footprint, leaving residual structure for other nodes (or for REST-time proposal) to capture. This is the intended "FFT-like" interpretation of masks/constellations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .geometry.block_spec import BlockSpec, BlockView

# =============================================================================
# Core aliases
# =============================================================================

# Discrete action space (A1): most sims treat action as an integer.
Action = int


# =============================================================================
# Environment observation
# =============================================================================


@dataclass
class EnvObs:
    """External observation at time t (partial, sparse).

    Fields:
      - x_partial: sparse vector {dim -> value} representing observed dims.
      - opp: opportunity proxy (world-supplied).
      - danger: danger proxy (world-supplied).
    """
    x_partial: Dict[int, float]
    opp: float = 0.0
    danger: float = 0.0
    x_full: np.ndarray | None = None
    true_delta: Tuple[int, int] | None = None
    pos_dims: Set[int] = field(default_factory=set)
    selected_blocks: Tuple[int, ...] = field(default_factory=tuple)


# =============================================================================
# State (A0): margins, stress, baselines, macrostate, geometry buffers
# =============================================================================


@dataclass
class Margins:
    """Margin vector v(t) (A0.1)."""
    m_E: float
    m_D: float
    m_L: float
    m_C: float
    m_S: float


@dataclass
class Stress:
    """Stress signals (A0.2-A0.3).

    v1.5b distinguishes internal need (s_int^need) and external threat (s_ext^th).
    We include these as explicit channels with defaults for backward compatibility.
    """
    s_E: float
    s_D: float
    s_L: float
    s_C: float
    s_S: float
    s_int_need: float = 0.0
    s_ext_th: float = 0.0


@dataclass
class Baselines:
    """Running baselines used by stress/novelty computations (A3)."""
    mu: np.ndarray
    var_fast: np.ndarray
    var_slow: np.ndarray
    # Previous normalized margin vector \tilde{v}(t-1) used to compute
    # \Delta\tilde{v}(t) in A0.2.
    tilde_prev: np.ndarray | None = None
    # Step index of the most recent structural edit (A3.3).
    last_struct_edit_t: int = -10**9


@dataclass
class MacrostateVars:
    """Macrostate bookkeeping (A14).

    - rest: whether REST is active (A14.1).
    - Q_struct: structural edit queue, REST-only processing (A14.2).
    - T_since: operating steps counter (A14.2).
    - T_rest: consecutive REST steps counter (A14.2).
    - P_rest: rest pressure accumulator (A14.3).
    """
    rest: bool
    Q_struct: List["EditProposal"] = field(default_factory=list)
    T_since: int = 0
    T_rest: int = 0
    P_rest: float = 0.0
    rest_cooldown: int = 0
    rest_zero_processed_streak: int = 0


@dataclass
class FoveaState:
    """Attention state over blocks (A16.2-A16.4).
    
    - block_residual: smoothed r(b,t) per block (A16.2)
    - block_age: steps since last observed, per block (A16.2)
    - routing_scores: optional routing bias per block (non-axiom, off by default)
    - block_costs: per-block daDoF observation cost (used by budgeted selection)
    - current_blocks: F_t, the currently selected fovea blocks (A16.3)
    """
    block_residual: np.ndarray  # shape (B,)
    block_age: np.ndarray       # shape (B,)
    block_uncertainty: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    block_costs: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    routing_scores: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    routing_last_t: int = -1
    current_blocks: Set[int] = field(default_factory=set)


@dataclass
class ObservationBuffer:
    """Most recent full state estimate (dense) and observation tracking (A16.5)."""
    x_last: np.ndarray
    observed_dims: Set[int] = field(default_factory=set)


@dataclass
class WorkingSet:
    """Working set A_t and associated metadata (A4.2, A5.4–A5.6, A6.2).

    This dataclass is the canonical representation of the working set across
    the repository.

    Fields
    ------
    active:
        Ordered list of active node IDs.
    weights:
        Per-node activation weights a_j(t).
    load:
        Raw load Σ_{j∈A_t} L_j (A4.2).
    effective_load:
        L^eff(t) = Σ a_j(t)·L_j (A5.5).
    anchor_load:
        L^eff_anc(t) (A6.2), i.e., effective load attributable to anchors.
    rollout_load:
        L^eff_roll(t) (A6.2), i.e., effective load attributable to rollout.
        (In the current repo, rollout is a stub; this field is still tracked
        so A6.2/A6.3 remain structurally present.)
    anchor_ids / non_anchor_ids:
        Selection metadata used by A5.6 and by REST-time edit heuristics.
    """

    active: List[int] = field(default_factory=list)
    weights: Dict[int, float] = field(default_factory=dict)

    # Load metrics
    load: float = 0.0
    effective_load: float = 0.0
    anchor_load: float = 0.0
    rollout_load: float = 0.0

    # Selection metadata
    anchor_ids: List[int] = field(default_factory=list)
    non_anchor_ids: List[int] = field(default_factory=list)


# =============================================================================
# Expert library (A4): nodes + footprints + anchors
# =============================================================================


@dataclass
class ExpertNode:
    """Masked linear-Gaussian expert (A4.1).

    Compatibility note:
    - Some modules historically used (pi, L) naming for (reliability, cost).
    - Other modules use (reliability, cost).
    This class supports both, and keeps them synchronized.
    """

    node_id: int
    mask: np.ndarray
    W: np.ndarray
    b: np.ndarray
    Sigma: np.ndarray
    input_mask: Optional[np.ndarray] = None

    # Canonical internal names:
    reliability: float = 1.0
    cost: float = 0.1

    # Back-compat / alternate init names:
    pi: Optional[float] = None
    L: Optional[float] = None

    # Structural properties (A4.4, A5.6)
    is_anchor: bool = False
    footprint: int = -1  # φ(j) = block-id(m_j), -1 if unset

    # DAG bookkeeping
    parents: Set[int] = field(default_factory=set)
    children: Set[int] = field(default_factory=set)

    # Activity tracking (A12.3 PRUNE)
    last_active_step: int = 0
    created_step: int = 0

    def __post_init__(self) -> None:
        # Normalize arrays
        self.mask = np.asarray(self.mask, dtype=float).reshape(-1)
        if self.input_mask is not None:
            self.input_mask = np.asarray(self.input_mask, dtype=float).reshape(-1)
        self.W = np.asarray(self.W, dtype=float)
        self.b = np.asarray(self.b, dtype=float).reshape(-1)
        self.Sigma = np.asarray(self.Sigma, dtype=float)

        # Synchronize (pi, L) <-> (reliability, cost)
        if self.pi is not None:
            self.reliability = float(self.pi)
        else:
            self.pi = float(self.reliability)

        if self.L is not None:
            self.cost = float(self.L)
        else:
            self.L = float(self.cost)


# Many modules still import Node / ExpertNode separately.
Node = ExpertNode


@dataclass
class ExpertLibrary:
    """Expert library container (A4).

    nodes: node_id -> ExpertNode
    anchors: anchor node ids
    footprint_index: footprint_id -> list[node_id]
    next_node_id: counter for ID allocation
    """
    nodes: Dict[int, ExpertNode] = field(default_factory=dict)
    anchors: Set[int] = field(default_factory=set)
    footprint_index: Dict[int, List[int]] = field(default_factory=dict)
    next_node_id: int = 0

    def add_node(self, node: ExpertNode) -> int:
        """Insert node; allocate an id if the provided one collides."""
        node_id = int(getattr(node, "node_id", -1))
        if node_id < 0 or node_id in self.nodes:
            node_id = self.next_node_id
            self.next_node_id += 1
            node.node_id = node_id
        else:
            self.next_node_id = max(self.next_node_id, node_id + 1)
        self.nodes[node_id] = node
        return node_id

    def remove_node(self, node_id: int) -> Optional[ExpertNode]:
        """Remove node and clean indices. Returns removed node or None."""
        node = self.nodes.pop(int(node_id), None)
        if node is None:
            return None
        self.anchors.discard(int(node_id))
        # Best-effort cleanup of footprint index
        for phi, ids in list(self.footprint_index.items()):
            if int(node_id) in ids:
                self.footprint_index[phi] = [i for i in ids if i != int(node_id)]
                if not self.footprint_index[phi]:
                    self.footprint_index.pop(phi, None)
        return node


# Back-compat: some files historically called this "Library".
Library = ExpertLibrary


# =============================================================================
# Budgeting / horizon (A6)
# =============================================================================


@dataclass
class BudgetBreakdown:
    """Budget components (A6.2)."""
    b_enc: float
    b_roll: float
    b_cons: float
    h: int
    x_C: float


# =============================================================================
# Learning cache (step-local)
# =============================================================================


@dataclass
class LearningCache:
    """Ephemeral per-step learning context."""
    x_t: np.ndarray
    yhat_tp1: np.ndarray
    # Diagonal of Σ_global(t+1|t) associated with yhat_tp1.
    # Required for A17 and for precision-weighted residual accounting.
    sigma_tp1_diag: np.ndarray
    A_t: WorkingSet
    permit_param_t: bool
    rest_t: bool


# =============================================================================
# Rollout results (A7/A8)
# =============================================================================


@dataclass
class RolloutResult:
    """Rollout prediction outputs (A7.4)."""
    x_hats: List[np.ndarray] = field(default_factory=list)
    Sigma_hats: List[np.ndarray] = field(default_factory=list)
    H: List[float] = field(default_factory=list)
    c: List[float] = field(default_factory=list)
    n_cov: List[int] = field(default_factory=list)
    rho: List[float] = field(default_factory=list)
    c_qual: List[float] = field(default_factory=list)
    c_cov: List[float] = field(default_factory=list)


# =============================================================================
# Step tracing (diagnostics)
# =============================================================================


@dataclass
class StepTrace:
    """Structured trace for one environment step.

    step_pipeline currently constructs this with a subset of fields; anything that
    might be omitted in the constructor must have a default.
    """
    t: int
    rest: bool
    h: int
    commit: bool

    # Budget / compute slack
    x_C: float
    b_enc: float
    b_roll: float
    b_cons: float

    # Diagnostics
    L_eff: float
    arousal: float
    feel: Dict[str, float] = field(default_factory=dict)

    # Optional gates (some code adds these only to the dict trace)
    permit_param: bool = False
    freeze: bool = False


# =============================================================================
# Structural edits (A12/A14)
# =============================================================================


class EditKind(str, Enum):
    """Structural edit types (A12.3, A12.4)."""
    MERGE = "MERGE"
    PRUNE = "PRUNE"
    SPAWN = "SPAWN"
    SPLIT = "SPLIT"


@dataclass
class MergeEvidence:
    """Evidence for MERGE acceptance (A12.3)."""
    expert_a_id: int
    expert_b_id: int
    footprint: int
    correlation: float
    evaluation_window_start: int
    evaluation_window_end: int
    activation_pairs: List[Tuple[int, float, float]]  # [(τ, a_A(τ), a_B(τ)), ...]
    cost_a: float
    cost_b: float
    estimated_merged_cost: float
    taus: List[int] = field(default_factory=list)


@dataclass
class PruneEvidence:
    """Evidence for PRUNE acceptance (A12.3)."""
    node_id: int
    footprint: int
    reason: str  # 'low_reliability' or 'inactive'
    reliability: float
    time_since_active: int


@dataclass
class SpawnEvidence:
    """Evidence for SPAWN acceptance (A12.4)."""
    footprint: int
    persistent_residual: float
    coverage_visits: int
    block_dims: Set[int]
    recent_transitions: List[int] = field(default_factory=list)


@dataclass
class SplitEvidence:
    """Evidence for SPLIT acceptance (A12.4)."""
    source_node_id: int
    footprint: int
    dims_group_1: Set[int]
    dims_group_2: Set[int]
    cross_correlation: float


@dataclass
class EditProposal:
    """Proposed structural edit for Q_struct (A12, A14.2)."""
    kind: EditKind
    footprint: int
    priority: float
    source_node_ids: List[int]
    proposal_step: int

    merge_evidence: Optional[MergeEvidence] = None
    prune_evidence: Optional[PruneEvidence] = None
    spawn_evidence: Optional[SpawnEvidence] = None
    split_evidence: Optional[SplitEvidence] = None


@dataclass
class AcceptanceResult:
    """Result of acceptance evaluation (A12.1-A12.3)."""
    accepted: bool
    rejection_reason: str = ""

    # Gate signals / diagnostics
    permit_struct: bool = False
    delta_j: float = 0.0
    delta_s: float = 0.0
    delta_c: float = 0.0
    delta_j_ok: bool = False
    delta_s_ok: bool = False
    delta_c_ok: bool = False
    quality_ok: bool = True

    # Anti-aliasing details (A4.4)
    anti_alias_ok: bool = True
    aliased_with: Optional[int] = None
    replace_incumbent: bool = False

    # Merge-specific diagnostics
    merge_domain_a_ok: bool = True
    merge_domain_b_ok: bool = True


# =============================================================================
# Transition logging + residual statistics (edit proposals)
# =============================================================================


@dataclass
class TransitionRecord:
    """Single observed transition for MERGE evaluation (A12.3)."""
    tau: int
    dims: Tuple[int, ...]
    x_tau_block: np.ndarray
    x_tau_plus_1_block: np.ndarray
    observed_dims_tau_plus_1: Set[int]

    def full_vectors(self, state_dim: int) -> tuple[np.ndarray, np.ndarray]:
        """Reconstruct full-state vectors from stored block slices."""
        x_tau = np.zeros(state_dim, dtype=float)
        x_tau_plus_1 = np.zeros(state_dim, dtype=float)
        for idx, dim in enumerate(self.dims):
            if 0 <= dim < state_dim:
                x_tau[dim] = float(self.x_tau_block[idx])
                x_tau_plus_1[dim] = float(self.x_tau_plus_1_block[idx])
        return x_tau, x_tau_plus_1


@dataclass
class PersistentResidualState:
    """Per-footprint R_φ(t) for SPAWN triggering (A12.4).
    
    R_φ(t) = (1-β_R)R_φ(t-1) + β_R·residual_block(φ,t)
    """
    value: float = 0.0
    coverage_visits: int = 0
    last_update_step: int = 0


@dataclass
class FootprintResidualStats:
    """Within-block residual statistics for SPLIT detection (A12.4).
    
    Tracks running correlation structure to detect independent subspaces.
    Uses incremental sum-of-products for memory efficiency.
    """
    dims: List[int] = field(default_factory=list)  # Dimensions in this footprint
    n_updates: int = 0
    
    # Running sums for correlation computation
    # mean_ema[i] = EMA of residual[dims[i]]
    # cov_ema[i,j] = EMA of (residual[dims[i]] - mean[i]) * (residual[dims[j]] - mean[j])
    mean_ema: Optional[np.ndarray] = None   # shape (len(dims),)
    cov_ema: Optional[np.ndarray] = None    # shape (len(dims), len(dims))
    
    def update(self, residual_vec: np.ndarray, beta: float = 0.1) -> None:
        """Update stats with new residual observation.
        
        Args:
            residual_vec: Full residual vector (will be indexed by self.dims)
            beta: EMA decay rate
        """
        if not self.dims:
            return
        
        x = np.asarray(residual_vec, dtype=float).reshape(-1)
        n = len(self.dims)
        
        # Extract values for our dimensions
        vals = np.array([float(x[d]) if d < len(x) else 0.0 for d in self.dims])
        
        # Initialize if needed
        if self.mean_ema is None:
            self.mean_ema = np.zeros(n)
        if self.cov_ema is None:
            self.cov_ema = np.zeros((n, n))
        
        # Update mean EMA
        self.mean_ema = (1 - beta) * self.mean_ema + beta * vals
        
        # Update covariance EMA (centered)
        centered = vals - self.mean_ema
        outer = np.outer(centered, centered)
        self.cov_ema = (1 - beta) * self.cov_ema + beta * outer
        
        self.n_updates += 1


def infer_footprint(mask: np.ndarray, blocks: List[List[int]]) -> int:
    """Infer the unique block footprint for a block-aligned mask.

    Raises ValueError if mask is not exactly one full block (0/1) mask.
    """
    m = np.asarray(mask, dtype=float).reshape(-1)
    on = set(int(i) for i in np.where(m > 0.5)[0].tolist())

    if not on:
        raise ValueError("mask has no active dimensions")

    for phi, dims in enumerate(blocks):
        bd = set(int(d) for d in dims)
        if on == bd:
            return int(phi)
        # Also accept subset (mask within block)
        if on <= bd:
            return int(phi)

    raise ValueError("mask is not contained within any single block footprint")


# =============================================================================
# Agent state container
# =============================================================================


@dataclass
class AgentState:
    """Unified agent state container (A0-A17).

    This is intentionally a superset to satisfy both the step pipeline
    and the structural-edit modules. Modules only rely on the subset of
    fields they touch.

    Primary (step_pipeline) fields:
      - t, margins, stress, arousal, baselines, macro, fovea, buffer, library

    Structural-edit support fields (used by nupca3/edits/*):
      - incumbents, observed_transitions, activation_log, residual_stats,
        persistent_residuals, blocks, active_set

    Initialization per A14.8:
      - μ_k(0), σ_k^fast(0), σ_k^slow(0) = 0 (in baselines)
      - P_rest(0), Q_struct(0), T_since(0), T_rest(0) = 0 (in macro)
      - s^ar(0) = s_inst^ar(0) (arousal)
    """
    # ----- Primary fields -----
    t: int
    # Underlying margin observables (A2.1) evolved by A15 dynamics.
    # These are kept explicitly so A0.3 and A10.2 can use raw headrooms.
    E: float
    D: float
    drift_P: float
    margins: Margins
    stress: Stress
    arousal: float
    baselines: Baselines
    macro: MacrostateVars
    fovea: FoveaState
    buffer: ObservationBuffer
    library: ExpertLibrary
    # Consolidation cost b_cons(t) from REST structural processing (A6.2).
    # During OPERATING this should remain 0.
    b_cons: float = 0.0

    # Lagged A14 predicates evaluated at time t-1 (A14.7).
    rest_permitted_prev: bool = True
    demand_prev: bool = False
    interrupt_prev: bool = False
    learn_cache: Optional[LearningCache] = None

    # ----- Observation geometry (A16) -----
    blocks: List[List[int]] = field(default_factory=list)
    block_specs: List["BlockSpec"] = field(default_factory=list)
    block_view: Optional["BlockView"] = None

    # ----- Memory / edit support (A4, A12) -----
    incumbents: Dict[int, Set[int]] = field(default_factory=dict)
    observed_transitions: Dict[int, List[TransitionRecord]] = field(default_factory=dict)
    activation_log: Dict[int, List[Tuple[int, float]]] = field(default_factory=dict)
    residual_stats: Dict[int, FootprintResidualStats] = field(default_factory=dict)
    persistent_residuals: Dict[int, PersistentResidualState] = field(default_factory=dict)
    active_set: Set[int] = field(default_factory=set)
    # Peripheral gist / coverage bookkeeping
    context_register: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    node_context_tags: Dict[int, np.ndarray] = field(default_factory=dict)
    node_band_levels: Dict[int, int] = field(default_factory=dict)
    coverage_expert_debt: Dict[int, int] = field(default_factory=dict)
    coverage_band_debt: Dict[int, int] = field(default_factory=dict)

    # ----- Lagged values for timing discipline (A5.2, A5.3, A10.2) -----
    arousal_prev: float = 0.0
    scores_prev: Dict[int, float] = field(default_factory=dict)

    # Lagged stress channels for A5.2/A14.7 timing discipline
    s_int_need_prev: float = 0.0
    s_ext_th_prev: float = 0.0

    # Lagged compute slack x_C(t-1) for A10.2 timing discipline
    x_C_prev: float = 0.0

    # Lagged raw headrooms for A10.2 timing discipline.
    rawE_prev: float = 0.0
    rawD_prev: float = 0.0

    # Lagged commitment confidence c_d(t-1) at latency floor d (A8.2 timing discipline).
    c_d_prev: float = 1.0

    coarse_prev: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    coarse_shift: Tuple[int, int] = field(default_factory=lambda: (0, 0))
    grid_prev_mass: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    transport_beliefs: Dict[Tuple[int, int], float] = field(default_factory=dict)
    transport_last_delta: Tuple[int, int] = field(default_factory=lambda: (0, 0))
    transport_confidence: float = 0.0
    transport_margin: float = 0.0
    transport_disagreement_scores: Dict[int, float] = field(default_factory=dict)
    transport_disagreement_margin: float = field(default_factory=lambda: float("inf"))

    # ----- Stability probe/feature variance summaries (A3.3) -----
    # The surrounding pipeline is responsible for feeding probe/feature vectors
    # (e.g., world probes and internal feature summaries). We store rolling
    # windows and scalar variance summaries so stable(t) is well-defined.
    probe_window: List[np.ndarray] = field(default_factory=list)
    feature_window: List[np.ndarray] = field(default_factory=list)
    probe_var: Optional[float] = None
    feature_var: Optional[float] = None

    def __post_init__(self) -> None:
        # Default previous arousal to current arousal on fresh state creation.
        if self.arousal_prev == 0.0 and self.arousal != 0.0:
            self.arousal_prev = float(self.arousal)

    # -------------------------------------------------------------------------
    # Compatibility properties (used by edit modules)
    # -------------------------------------------------------------------------

    @property
    def timestep(self) -> int:
        """Alias for t."""
        return int(self.t)

    @property
    def is_rest(self) -> bool:
        """REST state (A14.1)."""
        return bool(getattr(self.macro, "rest", False))

    @property
    def q_struct(self) -> List[EditProposal]:
        """Structural edit queue (A14.2)."""
        return getattr(self.macro, "Q_struct", [])

    @property
    def state_dim(self) -> int:
        """Observation dimensionality."""
        x = getattr(self.buffer, "x_last", None)
        return int(len(x)) if x is not None else 0

    @property
    def observed_dims(self) -> Set[int]:
        """Currently observed dimensions O_t (A16.5)."""
        return getattr(self.buffer, "observed_dims", set())

    @property
    def current_fovea(self) -> Set[int]:
        """Currently selected fovea blocks F_t (A16.3)."""
        return getattr(self.fovea, "current_blocks", set())


# =============================================================================
# Exports check
# =============================================================================

__all__ = [
    # Core
    "Action",
    "EnvObs",
    # State components
    "Margins",
    "Stress",
    "Baselines",
    "MacrostateVars",
    "FoveaState",
    "ObservationBuffer",
    "WorkingSet",
    # Library
    "ExpertNode",
    "Node",
    "ExpertLibrary",
    "Library",
    # Budget
    "BudgetBreakdown",
    # Learning
    "LearningCache",
    "RolloutResult",
    "StepTrace",
    # Edits
    "EditKind",
    "MergeEvidence",
    "PruneEvidence",
    "SpawnEvidence",
    "SplitEvidence",
    "EditProposal",
    "AcceptanceResult",
    # Tracking
    "TransitionRecord",
    "PersistentResidualState",
    "FootprintResidualStats",
    # Helpers
    "infer_footprint",
    # Main container
    "AgentState",
]
