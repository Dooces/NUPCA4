"""nupca3/memory/library.py

Expert library DAG + footprint indices + anchors.

Axiom coverage: A4.

Key constraints
--------------
- Initialization must avoid pathological parameter allocations.
  In particular, a global anchor must not allocate a dense (D×D) weight matrix.
- Structural IDs must be monotonic and never overwritten.

Representation boundary
-----------------------
These nodes are "abstraction/resonance" operators over the encoded state vector
x(t). They are not router-based MoE experts.

"""

from __future__ import annotations

import logging
from dataclasses import replace as dc_replace
from typing import Dict, List, Optional

import numpy as np

from ..config import AgentConfig
from ..types import ExpertLibrary, ExpertNode

LIBRARY_LOGGER = logging.getLogger("nupca3.library")


def _build_blocks(D: int, B: int) -> List[np.ndarray]:
    """Partition D dims into B contiguous blocks, returning list of index arrays."""
    if B <= 0:
        return []
    B = min(B, D)
    base = D // B
    rem = D % B
    blocks: List[np.ndarray] = []
    start = 0
    for b in range(B):
        width = base + (1 if b < rem else 0)
        idx = np.arange(start, start + width, dtype=int)
        blocks.append(idx)
        start += width
    return blocks


def _total_params(lib: ExpertLibrary) -> int:
    """Count parameters for diagnostics (not used by core logic)."""
    total = 0
    for node in lib.nodes.values():
        total += int(np.asarray(getattr(node, "W", np.zeros(0))).size)
        total += int(np.asarray(getattr(node, "b", np.zeros(0))).size)
        total += int(np.asarray(getattr(node, "Sigma", np.zeros(0))).size)
    return total


def _make_global_anchor(*, D: int, anchor_inputs: int, base_cost: float, dim_cost: float) -> ExpertNode:
    """Create a lightweight global anchor without allocating a dense (D×D) matrix.

    We keep anchor prediction effectively zero-initialized, but we ensure
    predict()/sgd_update() stay on the *compact* code path.

    Design:
      - mask covers all dims (anchor has global applicability).
      - in_idx is a tiny fixed subset (size = anchor_inputs, clipped to [1, D]).
      - W has shape (D, anchor_inputs), zero-initialized.
      - b is length D (kept for compatibility with existing predict path).
      - Sigma is diagonal (1D vector) length D.

    This avoids O(D^2) memory while preserving an always-defined anchor node.
    """
    if D <= 0:
        raise ValueError("D must be positive.")
    k = int(max(1, min(int(anchor_inputs), D)))
    in_idx = np.arange(k, dtype=int)
    input_mask = np.zeros(D, dtype=float)
    input_mask[in_idx] = 1.0
    mask = np.ones(D, dtype=float)
    out_idx = np.arange(D, dtype=int)
    W = np.zeros((D, k), dtype=float)
    b = np.zeros(D, dtype=float)
    Sigma = np.ones(D, dtype=float)
    return ExpertNode(
        node_id=0,
        mask=mask,
        W=W,
        b=b,
        Sigma=Sigma,
        input_mask=input_mask,
        out_idx=out_idx,
        in_idx=in_idx,
        reliability=1.0,
        cost=float(base_cost) + float(dim_cost) * float(D),
        is_anchor=True,
        footprint=-1,
    )


def init_library(cfg: AgentConfig) -> ExpertLibrary:
    """Initialize the expert library.

    By default, this seeds:
      - a lightweight global anchor (always), and
      - one per-block expert for each of the B footprint blocks (optional).

    Seeding controls (config)
    -------------------------
    - library_seed_block_experts (bool, default True):
        if False, only the anchor is created at init.
    - library_anchor_inputs (int, default 1):
        number of input dims used by the anchor to stay on compact W path.
    """
    D = int(getattr(cfg, "D", 0))
    B = int(getattr(cfg, "B", 0))
    if D <= 0 or B <= 0:
        raise ValueError(f"Invalid geometry for init_library: D={D} B={B}")

    seed_blocks = bool(getattr(cfg, "library_seed_block_experts", True))
    anchor_inputs = int(getattr(cfg, "library_anchor_inputs", 1))

    base_cost = float(getattr(cfg, "b_enc_base", 0.0))
    dim_cost = float(getattr(cfg, "b_enc_dim", 0.0))

    lib = ExpertLibrary()

    # ------------------------------------------------------------------
    # Anchor (always)
    # ------------------------------------------------------------------
    anchor = _make_global_anchor(D=D, anchor_inputs=anchor_inputs, base_cost=base_cost, dim_cost=dim_cost)
    lib.nodes[anchor.node_id] = anchor
    lib.anchors.add(anchor.node_id)

    # ------------------------------------------------------------------
    # Block experts (optional seed)
    # ------------------------------------------------------------------
    next_id = 1
    if seed_blocks:
        blocks = _build_blocks(D, B)
        for block_id, idx in enumerate(blocks):
            mask = np.zeros(D, dtype=float)
            mask[idx] = 1.0
            out_idx = idx.copy()
            in_idx = idx.copy()
            W = np.zeros((out_idx.size, in_idx.size), dtype=float)
            b = np.zeros(D, dtype=float)
            Sigma = np.ones(D, dtype=float)
            node = ExpertNode(
                node_id=next_id,
                mask=mask,
                W=W,
                b=b,
                Sigma=Sigma,
                out_idx=out_idx,
                in_idx=in_idx,
                reliability=1.0,
                cost=float(base_cost) + float(dim_cost) * float(idx.size),
                is_anchor=False,
                footprint=int(block_id),
            )
            lib.nodes[node.node_id] = node
            lib.footprint_index.setdefault(int(block_id), []).append(node.node_id)
            next_id += 1

    # Ensure next_node_id is consistent for add_node()
    lib.next_node_id = int(max(lib.nodes.keys(), default=-1) + 1)

    # Basic DAG neighbor links (optional; keep it extremely conservative)
    # We only link nodes that share the same footprint (if any). More
    # sophisticated linking happens via structural edits in macrostate.
    for fp, ids in lib.footprint_index.items():
        if len(ids) < 2:
            continue
        # Simple chain
        for a, b in zip(ids[:-1], ids[1:]):
            lib.nodes[a].children.add(b)
            lib.nodes[b].parents.add(a)

    LIBRARY_LOGGER.info(
        "init_library: D=%d B=%d seed_blocks=%s anchor_W_shape=%s nodes=%d params=%d next_node_id=%d",
        D,
        B,
        seed_blocks,
        tuple(anchor.W.shape),
        len(lib.nodes),
        _total_params(lib),
        lib.next_node_id,
    )
    return lib


def nodes_in_library(lib: ExpertLibrary) -> List[int]:
    """Return node IDs in the library (stable ordering)."""
    return sorted(int(k) for k in lib.nodes.keys())


def link_node_neighbors(*, lib: ExpertLibrary, node_id: int, footprint: int) -> None:
    """Best-effort DAG neighbor linking by shared footprint."""
    if node_id not in lib.nodes:
        return
    if footprint < 0:
        return
    candidates = list(lib.footprint_index.get(int(footprint), []))
    candidates = [int(c) for c in candidates if int(c) != int(node_id)]
    if not candidates:
        return
    # Link to the most recent node in the same footprint.
    parent_id = max(candidates)
    lib.nodes[parent_id].children.add(int(node_id))
    lib.nodes[int(node_id)].parents.add(int(parent_id))
    if candidates and parent_id not in candidates:
        LIBRARY_LOGGER.warning(
            "DAG link: unexpected parent selection node=%s footprint=%s candidates=%s",
            node_id,
            footprint,
            candidates,
        )
