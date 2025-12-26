"""nupca3/agent.py

Public agent wrapper for NUPCA3.

This file MUST remain a thin orchestrator:
- It owns `AgentState` and exposes a stable `step()` API.
- It performs A14.8 initialization for all axiom-required state.
- It delegates the single authoritative step order to
  `nupca3.step_pipeline.step_pipeline`.

Axiom intent notes
------------------
- Long-term memory is the expert library (A4). `reset(clear_memory=False)` must
  preserve it; otherwise Color×Shape holdout evaluation is invalid.
- This wrapper must not introduce alternate authorities (e.g., writing into
  Q_struct directly, bypassing REST gating, or storing pixel data in long-term
  fields). Observations are passed into the pipeline and not retained here.

Representation boundary (clarified project intent)
-------------------------------------------------
Raw pixels may exist only in transient observation buffers for the current step.
This wrapper does not store raw observation payloads beyond the pipeline call.

"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .config import AgentConfig
from .types import (
    Action,
    AgentState,
    EnvObs,
    ExpertLibrary,
    PersistentResidualState,
)

from .step_pipeline import step_pipeline

from .geometry.buffer import init_observation_buffer
from .geometry.block_spec import BlockSpec, BlockView, build_block_specs
from .geometry.fovea import init_fovea_state
from .geometry.streams import coarse_bin_count

from .memory.library import init_library

from .state.baselines import init_baselines
from .state.macrostate import init_macro, rest_permitted
from .state.margins import init_margins, init_stress


# =============================================================================
# Helper utilities
# =============================================================================


def _field_names(cls: type) -> Set[str]:
    """Return dataclass field names for `cls` (empty set if not a dataclass)."""
    if not is_dataclass(cls):
        return set()
    return {f.name for f in fields(cls)}


def _build_blocks(state_dim: int, n_blocks: int) -> List[List[int]]:
    """Build a DoF-aligned block partition with exactly `n_blocks` blocks (A16.1).

    Partitions {0..D-1} into disjoint contiguous blocks. If D% B != 0 we distribute
    the remainder so block sizes differ by at most 1.

    NOTE
    ----
    If B > D, we fall back to D singleton blocks.
    #ITOOKASHORTCUT: A16.1 presumes meaningful blocks; for B>D we cannot allocate
    non-empty disjoint blocks without overlap.
    """
    D = int(state_dim)
    B = max(1, int(n_blocks))

    if D <= 0:
        return [[]]

    base = D // B
    rem = D % B

    if B > D:
        B = D
        base = 1
        rem = 0

    blocks: List[List[int]] = []
    start = 0
    for b in range(B):
        size = base + (1 if b < rem else 0)
        end = start + size
        blocks.append(list(range(start, end)))
        start = end

    return blocks


def _build_incumbents_from_library(library: ExpertLibrary, blocks: List[List[int]]) -> Dict[int, Set[int]]:
    """Build incumbents I_phi per footprint from library nodes (A4.4).

    For each node, infer its footprint by checking whether its mask support lies
    entirely within a block.

    This is a best-effort initializer; footprints for existing nodes should
    ideally be set where nodes are constructed.
    """
    incumbents: Dict[int, Set[int]] = {i: set() for i in range(len(blocks))}

    nodes = getattr(library, "nodes", {}) or {}
    for node_id, node in nodes.items():
        mask = getattr(node, "mask", None)
        if mask is None:
            continue

        mask_arr = np.asarray(mask)
        if mask_arr.ndim != 1:
            mask_arr = mask_arr.reshape(-1)

        support = set(int(i) for i in np.where(mask_arr > 0.5)[0].tolist())
        if not support:
            continue

        for block_id, block_dims in enumerate(blocks):
            block_set = set(int(d) for d in block_dims)
            if support <= block_set:
                incumbents[block_id].add(int(node_id))
                # If node objects are mutable, annotate footprint for later modules.
                try:
                    if getattr(node, "footprint", -1) < 0:
                        node.footprint = int(block_id)
                except Exception:
                    pass
                break

    return incumbents


def _init_persistent_residuals(n_blocks: int) -> Dict[int, PersistentResidualState]:
    """Initialize persistent residual accumulator R_phi(t) = 0 per footprint (A12.4, A14.8)."""
    return {
        int(block_id): PersistentResidualState(value=0.0, coverage_visits=0)
        for block_id in range(int(n_blocks))
    }


# =============================================================================
# Agent
# =============================================================================


class NUPCA3Agent:
    """Stateful agent wrapper. The pipeline is the authority."""

    def __init__(self, cfg: AgentConfig, init_state: Optional[AgentState] = None):
        self.cfg = cfg
        self.state = init_state if init_state is not None else self._fresh_state(clear_memory=True)

    def _fresh_state(
        self,
        *,
        clear_memory: bool,
        preserve_library: Optional[ExpertLibrary] = None,
    ) -> AgentState:
        """Construct a new AgentState instance following A14.8."""
        cfg = self.cfg

        # Core dimensions
        D = int(getattr(cfg, "D", getattr(cfg, "state_dim", getattr(cfg, "obs_dim", 64))))
        B = int(getattr(cfg, "B", getattr(cfg, "n_blocks", 2)))
        blocks = _build_blocks(D, B)
        block_specs = build_block_specs(blocks)
        block_view = BlockView(block_specs)
        block_costs = np.asarray([float(spec.cost) for spec in block_specs], dtype=float)
        periph_blocks = max(0, min(int(getattr(cfg, "periph_blocks", 0)), int(B)))
        periph_bins = max(1, int(getattr(cfg, "periph_bins", 0)))
        if periph_blocks > 0:
            periph_cost = float(max(1, periph_bins * periph_bins))
            start_idx = max(0, int(B) - periph_blocks)
            for idx in range(start_idx, int(B)):
                block_costs[idx] = periph_cost

        # Underlying observables (A15/A2): initialize to safe midpoints.
        E_min = float(getattr(cfg, "E_min", 0.0))
        E_max = float(getattr(cfg, "E_max", 1.0))
        D_min = float(getattr(cfg, "D_min", 0.0))
        D_max = float(getattr(cfg, "D_max", 1.0))

        E0 = 0.5 * (E_min + E_max)
        D0 = 0.5 * (D_min + D_max)
        drift0 = 0.0

        # Subsystems (A14.8 init)
        margins0 = init_margins(E=E0, D=D0, drift_P=drift0, cfg=cfg)
        stress0 = init_stress(cfg)
        rest_perm0, _ = rest_permitted(stress0, 0.0, cfg, arousal=0.0)
        baselines0 = init_baselines(cfg)
        macro0 = init_macro(cfg)
        fovea0 = init_fovea_state(cfg, block_costs=block_costs)
        buffer0 = init_observation_buffer(cfg)

        # Library (A4)
        if clear_memory or preserve_library is None:
            library0 = init_library(cfg)
        else:
            library0 = preserve_library

        # A4.4 incumbents and A12.4 persistent residuals
        incumbents0 = _build_incumbents_from_library(library0, blocks)
        residuals0 = _init_persistent_residuals(len(blocks))

        # Build AgentState using schema-tolerant construction (keeps branch compatibility).
        state_fields = _field_names(AgentState)
        kw: Dict[str, Any] = {}

        # Required primary fields
        kw["t"] = 0
        kw["E"] = float(E0)
        kw["D"] = float(D0)
        kw["drift_P"] = float(drift0)
        kw["margins"] = margins0
        kw["stress"] = stress0
        kw["arousal"] = 0.0
        kw["baselines"] = baselines0
        kw["macro"] = macro0
        kw["fovea"] = fovea0
        kw["buffer"] = buffer0
        kw["library"] = library0

        # Optional fields present in this branch
        if "blocks" in state_fields:
            kw["blocks"] = blocks
        if "block_specs" in state_fields:
            kw["block_specs"] = block_specs
        if "block_view" in state_fields:
            kw["block_view"] = block_view
        if "incumbents" in state_fields:
            kw["incumbents"] = incumbents0
        if "persistent_residuals" in state_fields:
            kw["persistent_residuals"] = residuals0
        if "active_set" in state_fields:
            kw["active_set"] = set()
        if "b_cons" in state_fields:
            kw["b_cons"] = 0.0

        # Lagged values for timing discipline (A5.2/A10.2) and A14.6 gate
        if "arousal_prev" in state_fields:
            kw["arousal_prev"] = 0.0
        if "scores_prev" in state_fields:
            kw["scores_prev"] = {}
        if "rest_permitted_prev" in state_fields:
            kw["rest_permitted_prev"] = bool(rest_perm0)
        if "demand_prev" in state_fields:
            kw["demand_prev"] = False
        if "interrupt_prev" in state_fields:
            kw["interrupt_prev"] = False
        if "s_int_need_prev" in state_fields:
            kw["s_int_need_prev"] = 0.0
        if "s_ext_th_prev" in state_fields:
            kw["s_ext_th_prev"] = 0.0
        if "x_C_prev" in state_fields:
            kw["x_C_prev"] = float(getattr(cfg, "B_rt", 0.0))
        if "rawE_prev" in state_fields:
            kw["rawE_prev"] = float(E0 - E_min)
        if "rawD_prev" in state_fields:
            kw["rawD_prev"] = float(D_max - D0)
        if "c_d_prev" in state_fields:
            kw["c_d_prev"] = 1.0
        coarse_len = coarse_bin_count(cfg)
        if "coarse_prev" in state_fields:
            kw["coarse_prev"] = np.zeros(coarse_len, dtype=float)
        if "coarse_shift" in state_fields:
            kw["coarse_shift"] = (0, 0)
        if "context_register" in state_fields:
            kw["context_register"] = np.zeros(coarse_len, dtype=float)
        if "node_context_tags" in state_fields:
            kw["node_context_tags"] = {}
        if "node_band_levels" in state_fields:
            kw["node_band_levels"] = {}
        if "coverage_expert_debt" in state_fields:
            kw["coverage_expert_debt"] = {}
        if "coverage_band_debt" in state_fields:
            kw["coverage_band_debt"] = {}
        grid_side = int(getattr(cfg, "grid_side", 0))
        grid_cells = max(0, grid_side * grid_side)
        if "grid_prev_mass" in state_fields:
            kw["grid_prev_mass"] = np.zeros(grid_cells, dtype=float)

        return AgentState(**kw)

    def reset(self, seed: Optional[int] = None, *, clear_memory: bool = False) -> None:
        """Reset episodic state.

        - clear_memory=False preserves long-term library (A4).
        - clear_memory=True performs a cold start.

        Note: this resets episodic traces/hidden state, not learned parameters,
        unless explicitly requested.
        """
        preserve_library = None if clear_memory else getattr(self.state, "library", None)
        self.state = self._fresh_state(clear_memory=clear_memory, preserve_library=preserve_library)

        if seed is not None:
            #ITOOKASHORTCUT: global RNG seed. A stricter implementation would plumb
            # a per-agent Generator through all stochastic subsystems.
            np.random.seed(int(seed))

    def step(self, env_obs: EnvObs) -> Tuple[Action, Dict[str, Any]]:
        """Advance one environment step by delegating to the step pipeline."""
        action, next_state, trace = step_pipeline(self.state, env_obs, self.cfg)
        self.state = next_state
        return action, trace


__all__ = ["NUPCA3Agent"]
