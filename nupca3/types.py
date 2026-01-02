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

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

import numpy as np


from .geometry.block_spec import BlockSpec, BlockView


# =============================================================================
# Core aliases
# =============================================================================

# Discrete action space (A1): most sims treat action as an integer.
from enum import Enum

class CurriculumCommand(Enum):
    NONE = "none"
    ADD_SHAPE = "add_shape"
    REMOVE_SHAPE = "remove_shape"

@dataclass
class Action:
    command: CurriculumCommand = CurriculumCommand.NONE
    # Preserve scalar action if needed for other domains (stubbed as int)
    value: int = 0


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
    periph_full: np.ndarray | None = None
    true_delta: Tuple[int, int] | None = None
    t_w: int = 0
    wall_ms: int | None = None
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
    block_disagreement: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    block_innovation: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    block_periph_demand: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    block_confidence: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    routing_last_t: int = -1
    current_blocks: Set[int] = field(default_factory=set)
    coverage_cursor: int = 0


@dataclass
class ObservationBuffer:
    """Most recent full state estimate (dense) and observation tracking (A16.5)."""
    x_last: np.ndarray
    x_prior: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    observed_dims: Set[int] = field(default_factory=set)


@dataclass
class WorldHypothesis:
    """Candidate hypothesis over the current state (per-phase multi-world bookkeeping)."""
    delta: Tuple[int, int] = field(default_factory=lambda: (0, 0))
    x_prior: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    x_post: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    sigma_prior_diag: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    weight: float = 0.0
    prior_mae: float = float("nan")
    likelihood: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


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


@dataclass
class PlanningThread:
    """Persistent planning thread metadata (A0.6)."""

    thread_id: int
    focus_blocks: Tuple[int, ...] = field(default_factory=tuple)
    focus_object: Any | None = None
    timeline: Tuple[int, ...] = field(default_factory=tuple)
    status: str = "idle"
    plan_state: Dict[str, Any] = field(default_factory=dict)
    last_progress_t_w: int = 0


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
    out_idx: Optional[np.ndarray] = None
    in_idx: Optional[np.ndarray] = None

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
    # ----- NUPCA5 signature retrieval (A4.3′) -----
    # Immutable 64-bit retrieval address captured at unit creation time from:
    #   committed metadata + ephemeral periphery gist (NOT persisted).
    # This MUST NOT be a per-node salt and MUST NOT depend on node_id.
    unit_sig64: int = 0

    # Structural flags used by retrieval/budgeting; stored explicitly (no setattr shims).
    is_transport: bool = False

    # Bookkeeping for deterministic sig_index removal (set by library on registration).
    sig_index_blocks: Tuple[int, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        # Normalize arrays
        self.mask = np.asarray(self.mask, dtype=float).reshape(-1)
        if self.input_mask is not None:
            self.input_mask = np.asarray(self.input_mask, dtype=float).reshape(-1)
        self.W = np.asarray(self.W, dtype=float)
        self.b = np.asarray(self.b, dtype=float).reshape(-1)
        self.Sigma = np.asarray(self.Sigma, dtype=float)
        if self.out_idx is not None:
            self.out_idx = np.asarray(self.out_idx, dtype=int).reshape(-1)
        if self.in_idx is not None:
            self.in_idx = np.asarray(self.in_idx, dtype=int).reshape(-1)

        # Synchronize (pi, L) <-> (reliability, cost)
        if self.pi is not None:
            self.reliability = float(self.pi)
        else:
            self.pi = float(self.reliability)

        if self.L is not None:
            self.cost = float(self.L)
        else:
            self.L = float(self.cost)

        # NUPCA5: unit_sig64 is a 64-bit stored address (A4.3′). Enforce width.
        try:
            self.unit_sig64 = int(self.unit_sig64) & 0xFFFFFFFFFFFFFFFF
        except Exception:
            self.unit_sig64 = 0

        self.is_transport = bool(self.is_transport)

        if self.sig_index_blocks is None:
            self.sig_index_blocks = tuple()
        else:
            self.sig_index_blocks = tuple(int(b) for b in self.sig_index_blocks)

    @property
    def block_id(self) -> int:
        """Canonical block identifier for this node (A4.4 footprint)."""
        return int(getattr(self, "footprint", -1))

    @block_id.setter
    def block_id(self, value: int) -> None:
        self.footprint = int(value)


# Many modules still import Node / ExpertNode separately.
Node = ExpertNode


@dataclass
class ExpertLibrary:
    """Expert library container (A4).

    nodes: node_id -> ExpertNode
    anchors: anchor node ids
    footprint_index: footprint_id -> list[node_id]
    next_node_id: counter for ID allocation
    revision: monotonic counter incremented on every structural mutation
    """
    nodes: Dict[int, ExpertNode] = field(default_factory=dict)
    anchors: Set[int] = field(default_factory=set)
    footprint_index: Dict[int, List[int]] = field(default_factory=dict)
    next_node_id: int = 0
    revision: int = 0
    # NUPCA5 packed signature index (optional)
    sig_index: Any = None
    def add_node(self, node: ExpertNode) -> int:
        """Insert a node into the library and return its id.

        This must NEVER overwrite an existing node id. In earlier builds,
        `next_node_id` could be stale after `init_library()` populated
        `nodes` directly, causing `add_node()` to reuse id=0 and overwrite
        the anchor (and other incumbents). That is a fatal integrity bug
        and also a major contributor to bloated pickles (large dense W matrices
        being re-created and persisted).
        """
        # Fast path: honor a caller-supplied id if it is unused.
        node_id = int(getattr(node, 'node_id', -1))
        if node_id >= 0 and node_id not in self.nodes:
            self.next_node_id = max(int(self.next_node_id), node_id + 1)
        else:
            # Allocate the next free id, robust to stale `next_node_id`.
            candidate = int(getattr(self, 'next_node_id', 0))
            # Ensure we start above the current max id if next_node_id is stale.
            if self.nodes:
                candidate = max(candidate, max(int(k) for k in self.nodes.keys()) + 1)
            while candidate in self.nodes:
                candidate += 1
            node_id = candidate
            node.node_id = int(node_id)
            self.next_node_id = int(node_id) + 1

        self.nodes[int(node_id)] = node
        self.revision += 1
        return int(node_id)

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
        self.revision += 1
        return node

    # ---------------------------------------------------------------------
    # v5 durability boundary (A0.BUDGET.6)
    # ---------------------------------------------------------------------
    def __getstate__(self) -> dict:
        # Structural compliance: the in-memory dict/set object graph is NOT a durable DB.
        # Persist via PackedExpertLibrary (NPZ / arrays), not pickle.
        raise TypeError(
            "ExpertLibrary is not picklable. Persist via PackedExpertLibrary (arrays/NPZ) "
            "to satisfy v5 A0.BUDGET.6."
        )

    def pack(self) -> "PackedExpertLibrary":
        return pack_expert_library(self)

    @staticmethod
    def unpack(packed: "PackedExpertLibrary") -> "ExpertLibrary":
        return unpack_expert_library(packed)

# =============================================================================
# Packed durability format (v5 A0.BUDGET.6)
# =============================================================================

_PACKED_LIB_VERSION = "v5.packed.1"


@dataclass(frozen=True)
class PackedExpertLibrary:
    """Packed, array-only durable form of the expert library.

    This is the ONLY supported durable representation of the library in v5 mode.
    It contains no Python dict/set/list object graph as stored state; only arrays
    and scalar ints suitable for NPZ persistence.
    """
    version: str

    # Node scalars (aligned by index i)
    node_ids: np.ndarray              # int32 (N,)
    footprint: np.ndarray             # int32 (N,)
    is_anchor: np.ndarray             # uint8 (N,)
    is_transport: np.ndarray          # uint8 (N,)
    reliability: np.ndarray           # float32 (N,)
    cost: np.ndarray                  # float32 (N,)
    created_step: np.ndarray          # int32 (N,)
    last_active_step: np.ndarray      # int32 (N,)
    unit_sig64: np.ndarray            # uint64 (N,)

    next_node_id: int
    revision: int

    # Ragged parameter blobs
    mask_data: np.ndarray             # float32 (sum_i |mask_i|)
    mask_indptr: np.ndarray           # int32 (N+1,)

    input_mask_data: np.ndarray       # float32 (sum_i |input_mask_i|)
    input_mask_indptr: np.ndarray     # int32 (N+1,)
    has_input_mask: np.ndarray        # uint8 (N,)

    W_data: np.ndarray                # float32 (sum_i |W_i|)
    W_indptr: np.ndarray              # int32 (N+1,)
    W_shape0: np.ndarray              # int32 (N,)
    W_shape1: np.ndarray              # int32 (N,)

    b_data: np.ndarray                # float32 (sum_i |b_i|)
    b_indptr: np.ndarray              # int32 (N+1,)

    Sigma_data: np.ndarray            # float32 (sum_i |Sigma_i|)
    Sigma_indptr: np.ndarray          # int32 (N+1,)
    Sigma_shape0: np.ndarray          # int32 (N,)
    Sigma_shape1: np.ndarray          # int32 (N,)

    out_idx_data: np.ndarray          # int32 (sum_i |out_idx_i|)
    out_idx_indptr: np.ndarray        # int32 (N+1,)
    has_out_idx: np.ndarray           # uint8 (N,)

    in_idx_data: np.ndarray           # int32 (sum_i |in_idx_i|)
    in_idx_indptr: np.ndarray         # int32 (N+1,)
    has_in_idx: np.ndarray            # uint8 (N,)

    # Edges (parent -> child) for DAG bookkeeping
    edge_parent: np.ndarray           # int32 (E,)
    edge_child: np.ndarray            # int32 (E,)

    def as_npz_dict(self) -> Dict[str, np.ndarray]:
        """Return an array-only dict suitable for np.savez."""
        return {
            "version": np.asarray([self.version], dtype="U"),
            "node_ids": self.node_ids,
            "footprint": self.footprint,
            "is_anchor": self.is_anchor,
            "is_transport": self.is_transport,
            "reliability": self.reliability,
            "cost": self.cost,
            "created_step": self.created_step,
            "last_active_step": self.last_active_step,
            "unit_sig64": self.unit_sig64,
            "next_node_id": np.asarray([int(self.next_node_id)], dtype=np.int64),
            "revision": np.asarray([int(self.revision)], dtype=np.int64),
            "mask_data": self.mask_data,
            "mask_indptr": self.mask_indptr,
            "input_mask_data": self.input_mask_data,
            "input_mask_indptr": self.input_mask_indptr,
            "has_input_mask": self.has_input_mask,
            "W_data": self.W_data,
            "W_indptr": self.W_indptr,
            "W_shape0": self.W_shape0,
            "W_shape1": self.W_shape1,
            "b_data": self.b_data,
            "b_indptr": self.b_indptr,
            "Sigma_data": self.Sigma_data,
            "Sigma_indptr": self.Sigma_indptr,
            "Sigma_shape0": self.Sigma_shape0,
            "Sigma_shape1": self.Sigma_shape1,
            "out_idx_data": self.out_idx_data,
            "out_idx_indptr": self.out_idx_indptr,
            "has_out_idx": self.has_out_idx,
            "in_idx_data": self.in_idx_data,
            "in_idx_indptr": self.in_idx_indptr,
            "has_in_idx": self.has_in_idx,
            "edge_parent": self.edge_parent,
            "edge_child": self.edge_child,
        }

    @staticmethod
    def from_npz_dict(d: Dict[str, np.ndarray]) -> "PackedExpertLibrary":
        v = str(np.asarray(d["version"]).reshape(-1)[0])
        next_node_id = int(np.asarray(d["next_node_id"]).reshape(-1)[0])
        revision = int(np.asarray(d["revision"]).reshape(-1)[0])
        return PackedExpertLibrary(
            version=v,
            node_ids=np.asarray(d["node_ids"], dtype=np.int32),
            footprint=np.asarray(d["footprint"], dtype=np.int32),
            is_anchor=np.asarray(d["is_anchor"], dtype=np.uint8),
            is_transport=np.asarray(d["is_transport"], dtype=np.uint8),
            reliability=np.asarray(d["reliability"], dtype=np.float32),
            cost=np.asarray(d["cost"], dtype=np.float32),
            created_step=np.asarray(d["created_step"], dtype=np.int32),
            last_active_step=np.asarray(d["last_active_step"], dtype=np.int32),
            unit_sig64=np.asarray(d["unit_sig64"], dtype=np.uint64),
            next_node_id=next_node_id,
            revision=revision,
            mask_data=np.asarray(d["mask_data"], dtype=np.float32),
            mask_indptr=np.asarray(d["mask_indptr"], dtype=np.int32),
            input_mask_data=np.asarray(d["input_mask_data"], dtype=np.float32),
            input_mask_indptr=np.asarray(d["input_mask_indptr"], dtype=np.int32),
            has_input_mask=np.asarray(d["has_input_mask"], dtype=np.uint8),
            W_data=np.asarray(d["W_data"], dtype=np.float32),
            W_indptr=np.asarray(d["W_indptr"], dtype=np.int32),
            W_shape0=np.asarray(d["W_shape0"], dtype=np.int32),
            W_shape1=np.asarray(d["W_shape1"], dtype=np.int32),
            b_data=np.asarray(d["b_data"], dtype=np.float32),
            b_indptr=np.asarray(d["b_indptr"], dtype=np.int32),
            Sigma_data=np.asarray(d["Sigma_data"], dtype=np.float32),
            Sigma_indptr=np.asarray(d["Sigma_indptr"], dtype=np.int32),
            Sigma_shape0=np.asarray(d["Sigma_shape0"], dtype=np.int32),
            Sigma_shape1=np.asarray(d["Sigma_shape1"], dtype=np.int32),
            out_idx_data=np.asarray(d["out_idx_data"], dtype=np.int32),
            out_idx_indptr=np.asarray(d["out_idx_indptr"], dtype=np.int32),
            has_out_idx=np.asarray(d["has_out_idx"], dtype=np.uint8),
            in_idx_data=np.asarray(d["in_idx_data"], dtype=np.int32),
            in_idx_indptr=np.asarray(d["in_idx_indptr"], dtype=np.int32),
            has_in_idx=np.asarray(d["has_in_idx"], dtype=np.uint8),
            edge_parent=np.asarray(d["edge_parent"], dtype=np.int32),
            edge_child=np.asarray(d["edge_child"], dtype=np.int32),
        )


def _pack_ragged_f32(arrs: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    lens = np.asarray([int(np.asarray(a).size) for a in arrs], dtype=np.int32)
    indptr = np.empty((lens.size + 1,), dtype=np.int32)
    indptr[0] = 0
    if lens.size:
        indptr[1:] = np.cumsum(lens, dtype=np.int64).astype(np.int32)
    else:
        indptr[1:] = 0
    total = int(indptr[-1])
    if total == 0:
        return np.zeros((0,), dtype=np.float32), indptr
    data = np.empty((total,), dtype=np.float32)
    k = 0
    for a in arrs:
        a1 = np.asarray(a, dtype=np.float32).reshape(-1)
        n = int(a1.size)
        if n:
            data[k:k+n] = a1
            k += n
    return data, indptr


def _pack_ragged_i32(arrs: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    lens = np.asarray([int(np.asarray(a).size) for a in arrs], dtype=np.int32)
    indptr = np.empty((lens.size + 1,), dtype=np.int32)
    indptr[0] = 0
    if lens.size:
        indptr[1:] = np.cumsum(lens, dtype=np.int64).astype(np.int32)
    else:
        indptr[1:] = 0
    total = int(indptr[-1])
    if total == 0:
        return np.zeros((0,), dtype=np.int32), indptr
    data = np.empty((total,), dtype=np.int32)
    k = 0
    for a in arrs:
        a1 = np.asarray(a, dtype=np.int32).reshape(-1)
        n = int(a1.size)
        if n:
            data[k:k+n] = a1
            k += n
    return data, indptr


def _unpack_ragged(data: np.ndarray, indptr: np.ndarray, i: int) -> np.ndarray:
    s = int(indptr[i])
    e = int(indptr[i + 1])
    if e <= s:
        return np.zeros((0,), dtype=data.dtype)
    return np.asarray(data[s:e])


def pack_expert_library(lib: ExpertLibrary) -> PackedExpertLibrary:
    node_ids = np.asarray(sorted(int(k) for k in lib.nodes.keys()), dtype=np.int32)
    N = int(node_ids.size)

    footprint = np.empty((N,), dtype=np.int32)
    is_anchor = np.zeros((N,), dtype=np.uint8)
    is_transport = np.zeros((N,), dtype=np.uint8)
    reliability = np.empty((N,), dtype=np.float32)
    cost = np.empty((N,), dtype=np.float32)
    created_step = np.empty((N,), dtype=np.int32)
    last_active_step = np.empty((N,), dtype=np.int32)
    unit_sig64 = np.empty((N,), dtype=np.uint64)

    masks: List[np.ndarray] = []
    input_masks: List[np.ndarray] = []
    has_input_mask = np.zeros((N,), dtype=np.uint8)

    Ws: List[np.ndarray] = []
    W_shape0 = np.zeros((N,), dtype=np.int32)
    W_shape1 = np.zeros((N,), dtype=np.int32)

    bs: List[np.ndarray] = []
    Sigmas: List[np.ndarray] = []
    Sigma_shape0 = np.zeros((N,), dtype=np.int32)
    Sigma_shape1 = np.zeros((N,), dtype=np.int32)

    out_idxs: List[np.ndarray] = []
    in_idxs: List[np.ndarray] = []
    has_out_idx = np.zeros((N,), dtype=np.uint8)
    has_in_idx = np.zeros((N,), dtype=np.uint8)

    # Edge list (parent -> child)
    e_parent: List[int] = []
    e_child: List[int] = []

    for i, nid in enumerate(node_ids.tolist()):
        n = lib.nodes[int(nid)]
        footprint[i] = int(getattr(n, "footprint", -1))
        is_anchor[i] = 1 if bool(getattr(n, "is_anchor", False)) else 0
        is_transport[i] = 1 if bool(getattr(n, "is_transport", False)) else 0
        reliability[i] = float(getattr(n, "reliability", 1.0))
        cost[i] = float(getattr(n, "cost", getattr(n, "L", 0.0)))
        created_step[i] = int(getattr(n, "created_step", 0))
        last_active_step[i] = int(getattr(n, "last_active_step", 0))
        unit_sig64[i] = np.uint64(int(getattr(n, "unit_sig64", 0)) & 0xFFFFFFFFFFFFFFFF)

        masks.append(np.asarray(getattr(n, "mask", np.zeros((0,), dtype=np.float32)), dtype=np.float32).reshape(-1))

        im = getattr(n, "input_mask", None)
        if im is None:
            input_masks.append(np.zeros((0,), dtype=np.float32))
            has_input_mask[i] = 0
        else:
            input_masks.append(np.asarray(im, dtype=np.float32).reshape(-1))
            has_input_mask[i] = 1

        W = np.asarray(getattr(n, "W", np.zeros((0, 0), dtype=np.float32)), dtype=np.float32)
        if W.ndim != 2:
            W = W.reshape((W.shape[0], -1))
        W_shape0[i] = int(W.shape[0])
        W_shape1[i] = int(W.shape[1])
        Ws.append(W.reshape(-1))

        b = np.asarray(getattr(n, "b", np.zeros((0,), dtype=np.float32)), dtype=np.float32).reshape(-1)
        bs.append(b)

        S = np.asarray(getattr(n, "Sigma", np.zeros((0, 0), dtype=np.float32)), dtype=np.float32)
        if S.ndim != 2:
            S = S.reshape((S.shape[0], -1))
        Sigma_shape0[i] = int(S.shape[0])
        Sigma_shape1[i] = int(S.shape[1])
        Sigmas.append(S.reshape(-1))

        oi = getattr(n, "out_idx", None)
        if oi is None:
            out_idxs.append(np.zeros((0,), dtype=np.int32))
            has_out_idx[i] = 0
        else:
            out_idxs.append(np.asarray(oi, dtype=np.int32).reshape(-1))
            has_out_idx[i] = 1

        ii = getattr(n, "in_idx", None)
        if ii is None:
            in_idxs.append(np.zeros((0,), dtype=np.int32))
            has_in_idx[i] = 0
        else:
            in_idxs.append(np.asarray(ii, dtype=np.int32).reshape(-1))
            has_in_idx[i] = 1

        # DAG edges
        for c in getattr(n, "children", set()):
            e_parent.append(int(nid))
            e_child.append(int(c))

    mask_data, mask_indptr = _pack_ragged_f32(masks)
    input_mask_data, input_mask_indptr = _pack_ragged_f32(input_masks)
    W_data, W_indptr = _pack_ragged_f32(Ws)
    b_data, b_indptr = _pack_ragged_f32(bs)
    Sigma_data, Sigma_indptr = _pack_ragged_f32(Sigmas)
    out_idx_data, out_idx_indptr = _pack_ragged_i32(out_idxs)
    in_idx_data, in_idx_indptr = _pack_ragged_i32(in_idxs)

    edge_parent = np.asarray(e_parent, dtype=np.int32)
    edge_child = np.asarray(e_child, dtype=np.int32)

    return PackedExpertLibrary(
        version=_PACKED_LIB_VERSION,
        node_ids=node_ids,
        footprint=footprint,
        is_anchor=is_anchor,
        is_transport=is_transport,
        reliability=reliability,
        cost=cost,
        created_step=created_step,
        last_active_step=last_active_step,
        unit_sig64=unit_sig64,
        next_node_id=int(getattr(lib, "next_node_id", int(node_ids.max() + 1 if N else 0))),
        revision=int(getattr(lib, "revision", 0)),
        mask_data=mask_data,
        mask_indptr=mask_indptr,
        input_mask_data=input_mask_data,
        input_mask_indptr=input_mask_indptr,
        has_input_mask=has_input_mask,
        W_data=W_data,
        W_indptr=W_indptr,
        W_shape0=W_shape0,
        W_shape1=W_shape1,
        b_data=b_data,
        b_indptr=b_indptr,
        Sigma_data=Sigma_data,
        Sigma_indptr=Sigma_indptr,
        Sigma_shape0=Sigma_shape0,
        Sigma_shape1=Sigma_shape1,
        out_idx_data=out_idx_data,
        out_idx_indptr=out_idx_indptr,
        has_out_idx=has_out_idx,
        in_idx_data=in_idx_data,
        in_idx_indptr=in_idx_indptr,
        has_in_idx=has_in_idx,
        edge_parent=edge_parent,
        edge_child=edge_child,
    )


def unpack_expert_library(packed: PackedExpertLibrary) -> ExpertLibrary:
    if packed.version != _PACKED_LIB_VERSION:
        raise ValueError(f"Unsupported PackedExpertLibrary version: {packed.version}")

    lib = ExpertLibrary()
    lib.nodes = {}
    lib.anchors = set()
    lib.footprint_index = {}
    lib.next_node_id = int(packed.next_node_id)
    lib.revision = int(packed.revision)

    N = int(packed.node_ids.size)
    # First pass: create nodes
    for i in range(N):
        nid = int(packed.node_ids[i])
        mask = _unpack_ragged(packed.mask_data, packed.mask_indptr, i).astype(np.float32, copy=False)

        im = None
        if int(packed.has_input_mask[i]) == 1:
            im = _unpack_ragged(packed.input_mask_data, packed.input_mask_indptr, i).astype(np.float32, copy=False)

        w_flat = _unpack_ragged(packed.W_data, packed.W_indptr, i).astype(np.float32, copy=False)
        w0 = int(packed.W_shape0[i]); w1 = int(packed.W_shape1[i])
        W = w_flat.reshape((w0, w1)) if w0 * w1 == int(w_flat.size) else w_flat.reshape((w0, -1))

        b = _unpack_ragged(packed.b_data, packed.b_indptr, i).astype(np.float32, copy=False)

        s_flat = _unpack_ragged(packed.Sigma_data, packed.Sigma_indptr, i).astype(np.float32, copy=False)
        s0 = int(packed.Sigma_shape0[i]); s1 = int(packed.Sigma_shape1[i])
        Sigma = s_flat.reshape((s0, s1)) if s0 * s1 == int(s_flat.size) else s_flat.reshape((s0, -1))

        oi = None
        if int(packed.has_out_idx[i]) == 1:
            oi = _unpack_ragged(packed.out_idx_data, packed.out_idx_indptr, i).astype(np.int32, copy=False)
        ii = None
        if int(packed.has_in_idx[i]) == 1:
            ii = _unpack_ragged(packed.in_idx_data, packed.in_idx_indptr, i).astype(np.int32, copy=False)

        node = ExpertNode(
            node_id=nid,
            mask=mask,
            W=W,
            b=b,
            Sigma=Sigma,
            input_mask=im,
            out_idx=oi,
            in_idx=ii,
            reliability=float(packed.reliability[i]),
            cost=float(packed.cost[i]),
            is_anchor=bool(int(packed.is_anchor[i])),
            footprint=int(packed.footprint[i]),
            created_step=int(packed.created_step[i]),
            last_active_step=int(packed.last_active_step[i]),
            unit_sig64=int(packed.unit_sig64[i]),
            is_transport=bool(int(packed.is_transport[i])),
            sig_index_blocks=tuple(),
        )
        lib.nodes[nid] = node

        if node.is_anchor:
            lib.anchors.add(nid)
        phi = int(node.footprint)
        if phi not in lib.footprint_index:
            lib.footprint_index[phi] = []
        lib.footprint_index[phi].append(nid)

    # Second pass: edges
    E = int(packed.edge_parent.size)
    for k in range(E):
        p_id = int(packed.edge_parent[k])
        c_id = int(packed.edge_child[k])
        if p_id in lib.nodes and c_id in lib.nodes:
            lib.nodes[p_id].children.add(c_id)
            lib.nodes[c_id].parents.add(p_id)

    return lib


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
    """Proposed structural edit for Q_struct (A12, A14.2).

    v5 note: proposals that create new units (SPAWN/SPLIT/MERGE) must carry the
    propose-time signature snapshot used to deterministically set the created
    unit's stored address (A4.3′).
    """
    kind: EditKind
    footprint: int
    priority: float
    source_node_ids: List[int]
    proposal_step: int

    # Stored 64-bit address snapshot captured at propose time (from committed metadata + ephemeral gist).
    proposal_sig64: int

    merge_evidence: Optional[MergeEvidence] = None
    prune_evidence: Optional[PruneEvidence] = None
    spawn_evidence: Optional[SpawnEvidence] = None
    split_evidence: Optional[SplitEvidence] = None

    def __post_init__(self) -> None:
        try:
            self.proposal_sig64 = int(self.proposal_sig64) & 0xFFFFFFFFFFFFFFFF
        except Exception as e:
            raise ValueError("proposal_sig64 must be a 64-bit int") from e

        if self.kind in (EditKind.SPAWN, EditKind.SPLIT, EditKind.MERGE) and self.proposal_sig64 == 0:
            raise ValueError("proposal_sig64 must be nonzero for SPAWN/SPLIT/MERGE (v5 A4.3′)")


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
# NUPCA5 deferred validation queue item
# =============================================================================


@dataclass
class PendingValidationRecord:
    """Deferred validation record for a candidate recall.

    This stays SMALL and strictly bounded:
      - stores only node_id, horizon bin, and lightweight scalars
      - does NOT store dense vectors or raw observations
    """

    node_id: int
    h_bin: int
    t_emit: int
    dist: int = 0
    err: float = 0.0

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
      - incumbents_by_block, incumbents_revision, observed_transitions,
        activation_log, residual_stats, persistent_residuals, blocks, active_set

    Initialization per A14.8:
      - μ_k(0), σ_k^fast(0), σ_k^slow(0) = 0 (in baselines)
      - P_rest(0), Q_struct(0), T_since(0), T_rest(0) = 0 (in macro)
      - s^ar(0) = s_inst^ar(0) (arousal)
    """
    # ----- Primary fields -----
    t_w: int
    k_op: int
    wall_ms: int
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
    world_hypotheses: List[WorldHypothesis] = field(default_factory=list)
    observed_history: Deque[Set[int]] = field(default_factory=lambda: deque())
    scan_counter: int = 0

    # Lagged A14 predicates evaluated at time t-1 (A14.7).
    rest_permitted_prev: bool = True
    demand_prev: bool = False
    interrupt_prev: bool = False
    learn_cache: Optional[LearningCache] = None


    # ----- NUPCA5 signature retrieval state -----
    # scan-proof signature computed at decision time from committed metadata +
    # ephemeral periphery gist (do not persist dense vectors).
    last_sig64: Optional[int] = None
    # previous committed metadata (t-1) for motion-sensitive delta term
    sig_prev_counts: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int16))
    sig_prev_hist: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.uint16))
    # bounded deferred validation queue
    pending_validation: Deque[PendingValidationRecord] = field(default_factory=lambda: deque())
    pred_store: Any = field(default=None, repr=False)
    trace_cache: Any = field(default=None, repr=False)
    value_of_compute: float = 0.0
    hazard_pressure: float = 0.0
    novelty_pressure: float = 0.0
    P_nov_state: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    U_prev_state: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    g_contemplate: bool = False
    focus_mode: str = "operate"
    planning_budget: float = 0.0
    planning_target_blocks: Tuple[int, ...] | None = None

    # ----- Observation geometry (A16) -----
    blocks: List[List[int]] = field(default_factory=list)
    block_specs: List["BlockSpec"] = field(default_factory=list)
    block_view: Optional["BlockView"] = None

    # ----- Memory / edit support (A4, A12) -----
    incumbents_by_block: List[Set[int]] = field(default_factory=list)
    incumbents_revision: int = 0
    observed_transitions: Dict[int, List[TransitionRecord]] = field(default_factory=dict)
    activation_log: Dict[int, List[Tuple[int, float]]] = field(default_factory=dict)
    residual_stats: Dict[int, FootprintResidualStats] = field(default_factory=dict)
    persistent_residuals: Dict[int, PersistentResidualState] = field(default_factory=dict)
    active_set: Set[int] = field(default_factory=set)
    thread_pinned_units: Set[int] = field(default_factory=set)
    planning_threads: Dict[int, "PlanningThread"] = field(default_factory=dict)
    planning_thread_stage: Dict[int, "PlanningThread"] = field(default_factory=dict)
    # Peripheral gist / coverage bookkeeping
    context_register: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    node_context_tags: Dict[int, np.ndarray] = field(default_factory=dict)
    node_band_levels: Dict[int, int] = field(default_factory=dict)
    coverage_expert_debt: Dict[int, int] = field(default_factory=dict)
    coverage_band_debt: Dict[int, int] = field(default_factory=dict)
    salience_recent_candidates: Dict[int, int] = field(default_factory=dict)
    salience_num_nodes_scored: int = 0
    salience_candidate_ids: Set[int] = field(default_factory=set)
    salience_candidate_limit: int = 0
    salience_candidate_count_raw: int = 0
    salience_candidates_truncated: bool = False
    learning_candidates_prev: Dict[str, Any] = field(default_factory=dict)
    proposals_prev: int = 0

    # ----- Lagged values for timing discipline (A5.2, A5.3, A10.2) -----
    arousal_prev: float = 0.0
    scores_prev: Dict[int, float] = field(default_factory=dict)

    # Lagged stress channels for A5.2/A14.7 timing discipline
    s_int_need_prev: float = 0.0
    s_ext_th_prev: float = 0.0

    # Lagged compute slack x_C(t-1) for A10.2 timing discipline
    x_C_prev: float = 0.0

    # Budget governor telemetry
    budget_degradation_level: int = 0
    budget_degradation_history: Tuple[str, ...] = field(default_factory=tuple)
    budget_hat_max: float = 0.0

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
    transport_biases: Dict[Tuple[int, int, int], float] = field(default_factory=dict)
    transport_offsets: List[Tuple[int, int]] = field(default_factory=list)
    peripheral_prior: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    peripheral_obs: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    peripheral_confidence: float = 0.0
    peripheral_residual: float = 0.0

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
    "WorldHypothesis",
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
    # NUPCA5
    "PendingValidationRecord",
    "PlanningThread",
    # Main container
    "AgentState",
]
