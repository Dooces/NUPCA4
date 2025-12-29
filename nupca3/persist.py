"""Persistence helpers for NUPCA3.

This module centralizes "REST-safe" checkpointing.

Key invariant:
- Persisted snapshots must not contain raw pixel/observation buffers beyond REST
  (A16.5). Therefore, we temporarily zero `state.buffer` content and force
  `state.macro.rest=True` while pickling.

The in-memory agent is restored after the checkpoint is written.
"""

from __future__ import annotations

from pathlib import Path
import pickle
from typing import Optional, Tuple, Set

import numpy as np

from nupca3.agent import NUPCA3Agent


def persist_state(
    agent: NUPCA3Agent,
    path: Path,
    *,
    drain_rest: bool = True,
    compact: bool = True,
    max_rest_cycles: int = 256,
) -> None:
    """Persist agent.state to `path` in a REST-safe manner.

    Enhancements:
    - Optionally *drain* the structural queue under REST semantics before saving,
      so the checkpoint is consistent with A14's "REST does consolidation" intent.
    - Optionally *compact* accidental dense (D x D) matrices on non-anchor nodes
      into footprint-local matrices for serialization. This materially reduces
      checkpoint size without changing operational semantics for mask-local nodes.

    Both behaviors are best-effort and will never throw on failure.
    """
    state = agent.state

    # ---------------------------------------------------------------------
    # Best-effort consolidation before persist ("quit" should REST then save)
    # ---------------------------------------------------------------------
    if drain_rest:
        try:
            from nupca3.edits.rest_processor import process_struct_queue
            from nupca3.state.macrostate import update_queue

            cfg = getattr(agent, 'cfg', None)
            if cfg is not None and getattr(state, 'macro', None) is not None:
                max_edits = int(getattr(cfg, 'max_edits_per_rest_step', 1))
                cycles = 0
                # Force REST flag while draining.
                saved_rest = bool(getattr(state.macro, 'rest', False))
                state.macro.rest = True
                try:
                    while cycles < int(max_rest_cycles):
                        Q_prev = list(getattr(state.macro, 'Q_struct', []) or [])
                        if not Q_prev:
                            break
                        res = process_struct_queue(state, cfg, queue=Q_prev, max_edits=max_edits)
                        processed = int(getattr(res, 'proposals_processed', 0))
                        # Apply the authoritative queue pop via macrostate rule.
                        state.macro.Q_struct = update_queue(
                            Q_prev=Q_prev,
                            rest_t=True,
                            proposals_t=[],
                            edits_processed_t=processed,
                        )
                        cycles += 1
                        if processed <= 0:
                            break
                finally:
                    state.macro.rest = saved_rest
        except Exception:
            # Never fail persistence due to best-effort drain.
            pass

    # ---------------------------------------------------------------------
    # Serialization compaction (dense W -> compact W for mask-local nodes)
    # ---------------------------------------------------------------------
    if compact:
        try:
            lib = getattr(state, 'library', None)
            buf = getattr(state, 'buffer', None)
            D = int(getattr(buf, 'x_last', np.zeros(0)).size) if buf is not None else int(getattr(state, 'state_dim', 0))
            if lib is not None and D > 0:
                for node in getattr(lib, 'nodes', {}).values():
                    if node is None or bool(getattr(node, 'is_anchor', False)):
                        continue
                    W = getattr(node, 'W', None)
                    if not isinstance(W, np.ndarray) or W.ndim != 2:
                        continue
                    if W.shape != (D, D):
                        continue

                    # Only compact "mask-local" nodes (no explicit indices provided).
                    if getattr(node, 'out_idx', None) is not None or getattr(node, 'in_idx', None) is not None:
                        continue

                    mask = np.asarray(getattr(node, 'mask', np.zeros(D)), dtype=float).reshape(-1)
                    if mask.size != D:
                        continue
                    out_idx = np.where(mask > 0.5)[0].astype(int)
                    if out_idx.size == 0 or out_idx.size >= D:
                        continue
                    # Inputs follow input_mask if provided, else mask.
                    in_mask = getattr(node, 'input_mask', None)
                    if in_mask is None:
                        in_mask = mask
                    in_mask = np.asarray(in_mask, dtype=float).reshape(-1)
                    in_idx = np.where(in_mask > 0.5)[0].astype(int)
                    if in_idx.size == 0 or in_idx.size >= D:
                        continue

                    node.W = W[np.ix_(out_idx, in_idx)]
                    node.out_idx = out_idx
                    node.in_idx = in_idx
                    if getattr(node, 'input_mask', None) is None:
                        node.input_mask = mask.copy()
        except Exception:
            pass

    # ---------------------------------------------------------------------
    # A16.5: ensure no raw buffers are persisted
    # ---------------------------------------------------------------------
    buffer = getattr(state, "buffer", None)
    saved_buffer: Optional[Tuple[np.ndarray, np.ndarray, Set[int]]] = None
    if buffer is not None:
        saved_buffer = (
            buffer.x_last.copy(),
            buffer.x_prior.copy(),
            set(buffer.observed_dims),
        )
        buffer.x_last = np.zeros_like(buffer.x_last)
        buffer.x_prior = np.zeros_like(buffer.x_prior)
        buffer.observed_dims = set()

    macro = getattr(state, "macro", None)
    saved_macro: Optional[Tuple[bool, int, int]] = None
    if macro is not None:
        saved_macro = (
            bool(macro.rest),
            int(getattr(macro, "T_rest", 0)),
            int(getattr(macro, "T_since", 0)),
        )
        macro.rest = True
        macro.T_rest = max(1, saved_macro[1])
        macro.T_since = 0

    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fid:
            pickle.dump(state, fid)
        print(f"Persisted restful state to {path}")
    finally:
        if buffer is not None and saved_buffer is not None:
            buffer.x_last = saved_buffer[0]
            buffer.x_prior = saved_buffer[1]
            buffer.observed_dims = saved_buffer[2]
        if macro is not None and saved_macro is not None:
            macro.rest, macro.T_rest, macro.T_since = saved_macro



def load_state(agent: NUPCA3Agent, path: Path) -> bool:
    """Load a persisted agent.state from `path`.

    Returns True if loaded, False if the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        return False
    with path.open("rb") as fid:
        state = pickle.load(fid)
    agent.state = state
    print(f"Loaded persisted state from {path}")
    return True
