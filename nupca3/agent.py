#!/usr/bin/env python3
"""nupca3/agent.py

NUPCA3Agent implementation used by the v5 harness (test5.py).

This module is intentionally thin:
  - State initialization lives here (because it must be consistent and
    pickleable for harness save/restore).
  - Per-step logic is delegated to ``nupca3.step_pipeline.core.step_pipeline``.

Critical ordering note
----------------------
The v5 harness expects the sig_* fields to be present in AgentConfig.
This repo is v5-only; no import-time compatibility patching is supported.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from .config import AgentConfig
from .types import (
    Action,
    AgentState,
    Baselines,
    EnvObs,
    FoveaState,
    MacrostateVars,
    Margins,
    ObservationBuffer,
    Stress,
)

from .geometry.block_spec import build_block_specs, BlockView
from .geometry.fovea import build_blocks_from_cfg, init_fovea_state
from .incumbents import rebuild_incumbents_by_block
from .memory.library import init_library
from .step_pipeline.core import step_pipeline
from .step_pipeline.fovea import apply_signals_and_select


def _init_primary_state(cfg: AgentConfig) -> AgentState:
    """Create a fresh AgentState aligned to cfg.

    This is the canonical initializer for harnesses that pickle state.
    """

    D = int(getattr(cfg, "D", 0))
    B = int(getattr(cfg, "B", 0))
    if D < 0:
        raise ValueError(f"cfg.D must be >= 0, got {D}")
    if B <= 0:
        raise ValueError(f"cfg.B must be > 0, got {B}")
    if D > 0 and B > D:
        # Multiple subsystems assume B<=D (block partitions, per-block arrays,
        # footprint indexing). Fail fast instead of silently mis-sizing.
        raise ValueError(f"cfg.B ({B}) must be <= cfg.D ({D})")

    # Observation geometry.
    blocks = build_blocks_from_cfg(cfg)
    if blocks and len(blocks) != B:
        # build_blocks_from_cfg should match cfg.B; if it doesn't, downstream
        # array shapes will diverge.
        raise ValueError(f"Block partition mismatch: len(blocks)={len(blocks)} != cfg.B={B}")
    block_specs = build_block_specs(blocks, cost_fn=lambda dims: float(max(1, len(dims))))
    block_view = BlockView(block_specs)

    # Per-block costs for budgeted fovea selection.
    block_costs = np.array([max(1.0, float(len(spec.dims))) for spec in block_specs], dtype=float)
    if block_costs.size != B:
        block_costs = np.resize(block_costs, (B,))

    fovea: FoveaState = init_fovea_state(cfg, block_costs=block_costs)

    # Dense belief buffer (allowed to persist per v1.5b; v5 only bans persisting
    # ephemeral gist vectors, not the belief state itself).
    x0 = np.zeros(D, dtype=float) if D > 0 else np.zeros(0, dtype=float)
    buf = ObservationBuffer(x_last=x0.copy(), x_prior=x0.copy(), observed_dims=set())

    margins = Margins(0.0, 0.0, 0.0, 0.0, 0.0)
    stress = Stress(0.0, 0.0, 0.0, 0.0, 0.0)
    baselines = Baselines(mu=np.zeros(5, dtype=float), var_fast=np.zeros(5, dtype=float), var_slow=np.zeros(5, dtype=float))
    macro = MacrostateVars(rest=False)

    library = init_library(cfg)

    state = AgentState(
        t=0,
        E=0.0,
        D=0.0,
        drift_P=0.0,
        margins=margins,
        stress=stress,
        arousal=0.0,
        baselines=baselines,
        macro=macro,
        fovea=fovea,
        buffer=buf,
        library=library,
    )

    # Attach geometry + incumbents indices.
    state.blocks = [dims.copy() for dims in blocks]
    state.block_specs = block_specs
    state.block_view = block_view
    state.incumbents_by_block = rebuild_incumbents_by_block(library, state.blocks)
    state.incumbents_revision = int(getattr(library, "revision", 0))

    # NUPCA5: ensure sig prev vectors exist (motion-sensitive delta term).
    # Sizes are derived from the commit metadata path (see step_pipeline/core).
    # Leave empty; core will resize on first use.
    state.sig_prev_counts = np.zeros(0, dtype=np.int16)
    state.sig_prev_hist = np.zeros(0, dtype=np.uint16)

    return state


class NUPCA3Agent:
    """Unified agent wrapper expected by test5.py."""

    def __init__(self, cfg: AgentConfig, *, state: Optional[AgentState] = None):
        self.cfg: AgentConfig = cfg
        self.state: AgentState = state if state is not None else _init_primary_state(cfg)

    # ------------------------------------------------------------------
    # Harness-facing API
    # ------------------------------------------------------------------

    def step(self, env_obs: EnvObs) -> Tuple[Action, Dict[str, Any]]:
        action, new_state, trace = step_pipeline(self.state, env_obs, self.cfg)
        self.state = new_state
        return action, trace

    def prepare_fovea_selection(self, *, periph_full: np.ndarray | None = None) -> Dict[str, Any]:
        """Precompute the next fovea selection.

        test5.py uses this to decide which dims to reveal in env_obs.x_partial.
        """
        prev_dims = set(getattr(self.state.buffer, "observed_dims", set()) or set())
        pending = apply_signals_and_select(self.state, self.cfg, periph_full=periph_full, prev_observed_dims=prev_dims)
        return pending

    def reset(self, *, clear_memory: bool = False) -> None:
        """Reset episode-level state.

        If clear_memory is True, reinitialize library + indices.
        Otherwise, keep learned memory and only clear transient episode fields.
        """

        if clear_memory:
            self.state = _init_primary_state(self.cfg)
            return

        # Soft reset: keep library + learned params.
        D = int(getattr(self.cfg, "D", 0))
        x0 = np.zeros(D, dtype=float) if D > 0 else np.zeros(0, dtype=float)
        self.state.t = 0
        self.state.E = 0.0
        self.state.D = 0.0
        self.state.drift_P = 0.0
        self.state.margins = Margins(0.0, 0.0, 0.0, 0.0, 0.0)
        self.state.stress = Stress(0.0, 0.0, 0.0, 0.0, 0.0)
        self.state.arousal = 0.0
        self.state.arousal_prev = 0.0
        self.state.baselines = Baselines(mu=np.zeros(5, dtype=float), var_fast=np.zeros(5, dtype=float), var_slow=np.zeros(5, dtype=float))
        self.state.macro = MacrostateVars(rest=False)
        self.state.fovea = init_fovea_state(self.cfg, block_costs=getattr(self.state.fovea, "block_costs", None))
        self.state.buffer = ObservationBuffer(x_last=x0.copy(), x_prior=x0.copy(), observed_dims=set())
        self.state.active_set = set()
        self.state.pending_validation.clear()
        self.state.last_sig64 = None
        self.state.pending_fovea_selection = None
        self.state.pending_fovea_signals = None
        self.state.context_register = np.zeros(0, dtype=float)
        self.state.node_context_tags = {}
        self.state.observed_history.clear()

    # ------------------------------------------------------------------
    # Optional persistence helpers (used by some older harnesses).
    # ------------------------------------------------------------------

    def save_state(self, path: str | Path) -> None:
        p = Path(path)
        payload = {"cfg": self.cfg, "state": self.state}
        p.write_bytes(pickle.dumps(payload))

    @classmethod
    def load_state(cls, path: str | Path) -> "NUPCA3Agent":
        p = Path(path)
        payload = pickle.loads(p.read_bytes())
        cfg = payload.get("cfg")
        state = payload.get("state")
        if cfg is None or state is None:
            raise ValueError("Invalid agent state payload")
        return cls(cfg=cfg, state=state)


# Back-compat alias used by some older scripts.
NUPCAAgent = NUPCA3Agent
