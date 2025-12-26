"""nupca3/edits/edit_types.py

NUPCA3 — Structural Edit Type Definitions (A12/A14) — AXIOM-FAITHFUL

VERSION: v1.5b-perf.7 (2025-12-20)

PURPOSE
  This module holds ONLY the structural-edit datatypes used by the edit pipeline:
    - EditKind, EditProposal
    - Evidence bundles for MERGE/SPAWN/SPLIT/PRUNE
    - AcceptanceResult

  It intentionally does NOT live in `nupca3/types.py`.

WHY THIS FILE EXISTS (CRITICAL)
  Your repo's `nupca3/types.py` is a core public API imported by NUPCA3Agent/AgentConfig.
  Overwriting it with "edit-only" types breaks imports like:
    `from nupca3.types import Stress, Baselines, ...`

  Therefore: keep core types in `nupca3/types.py`, and keep edit-only types here.

AXIOM COVERAGE
  - A12: Structural edits (MERGE/PRUNE/SPAWN/SPLIT)
  - A14: Q_struct proposal queue content and typing (the queue holds EditProposal objects)

HARD WARNINGS (DO NOT VIOLATE)
  - Do NOT add unrelated core agent state types here (Stress, Baselines, etc.) — those belong to `nupca3/types.py`.
  - Do NOT mutate AgentState here. These are pure datatypes.
  - Any change to acceptance semantics requires explicit permission and a read-through of the axiom list.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional, Set, Tuple


class EditKind(Enum):
    """Structural edit types (A12.3, A12.4)."""
    SPLIT = "split"      # Partition expert into two within same footprint/block
    MERGE = "merge"      # Combine two same-footprint experts
    PRUNE = "prune"      # Remove low-utility expert
    SPAWN = "spawn"      # Create new expert for persistent-residual footprint


# -----------------------------------------------------------------------------
# Evidence bundles (A12.3 / A12.4)
# -----------------------------------------------------------------------------

@dataclass
class MergeEvidence:
    """Evidence bundle for MERGE acceptance test (A12.3).

    NOTE: `taus` pins the evaluation set to avoid silent drift between propose-time and REST-time.
    """
    expert_a_id: int
    expert_b_id: int
    footprint: int
    correlation: float

    taus: List[int] = field(default_factory=list)
    evaluation_window_start: int = 0
    evaluation_window_end: int = 0
    activation_pairs: List[Tuple[int, float, float]] = field(default_factory=list)

    cost_a: float = 0.0
    cost_b: float = 0.0
    estimated_merged_cost: float = 0.0


@dataclass
class SpawnEvidence:
    """Evidence bundle for SPAWN acceptance (A12.4)."""
    footprint: int
    persistent_residual: float
    coverage_visits: int
    block_dims: Set[int] = field(default_factory=set)

    recent_transitions: List[Any] = field(default_factory=list)


@dataclass
class SplitEvidence:
    """Evidence bundle for SPLIT acceptance (A12.4 special case)."""
    source_node_id: int
    footprint: int
    dims_group_1: Set[int] = field(default_factory=set)
    dims_group_2: Set[int] = field(default_factory=set)
    cross_correlation: float = 1.0


@dataclass
class PruneEvidence:
    """Evidence bundle for PRUNE acceptance (A12.3)."""
    node_id: int
    footprint: int
    reason: str
    reliability: float
    time_since_active: int


# -----------------------------------------------------------------------------
# Proposal and acceptance result types (A14 / A12)
# -----------------------------------------------------------------------------

@dataclass
class EditProposal:
    """Proposed structural edit for Q_struct (A14.2)."""
    kind: EditKind
    footprint: int
    priority: float = 0.0
    proposal_step: int = 0
    source_node_ids: List[int] = field(default_factory=list)

    merge_evidence: Optional[MergeEvidence] = None
    spawn_evidence: Optional[SpawnEvidence] = None
    split_evidence: Optional[SplitEvidence] = None
    prune_evidence: Optional[PruneEvidence] = None


@dataclass
class AcceptanceResult:
    """Decision record for REST-time evaluation (A12)."""
    accepted: bool
    reason: str = ""

    delta_f: float = 0.0
    delta_mdl: float = 0.0
    delta_j: float = 0.0
    delta_s: float = 0.0
    delta_c: float = 0.0

    permit_struct: bool = False
    anti_alias_ok: bool = True
