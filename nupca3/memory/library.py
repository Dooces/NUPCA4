"""nupca3/memory/library.py

Expert library DAG + footprint indices + anchors.

Axiom coverage: A4.


[AXIOM_CLARIFICATION_ADDENDUM — Representation & Naming]

- Terminology: identifiers like "Expert" in this codebase refer to NUPCA3 **abstraction/resonance nodes** (a "constellation"), not conventional Mixture-of-Experts "experts" or router-based MoE.

- Representation boundary (clarified intent of v1.5b): the completion/fusion operator (A7) is defined over an **encoded, multi-resolution abstraction vector** \(x(t)\). Raw pixels may exist only in a transient observation buffer for the current step; **raw pixel values must never be inserted into long-term storage** (library/cold storage) and must not persist across REST boundaries.

- Decomposition intuition: each node is an operator that *factors out* a predictable/resonant component on its footprint, leaving residual structure for other nodes (or for REST-time proposal) to capture. This is the intended "FFT-like" interpretation of masks/constellations.
"""

from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Set

import numpy as np

from ..config import AgentConfig
from ..types import ExpertLibrary, ExpertNode


def _build_blocks(state_dim: int, n_blocks: int) -> List[List[int]]:
    """Build a DoF-aligned block partition with remainder distribution."""
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


def init_library(cfg: AgentConfig) -> ExpertLibrary:
    """Initialize a minimal library with a global anchor and per-block experts.

    This is a convenience initialization, not an axiom requirement.
    """
    lib = ExpertLibrary(nodes={}, anchors=set(), footprint_index={})
    node_id = 0
    D = int(cfg.D)
    B = int(cfg.B)
    blocks = _build_blocks(D, B)
    span = int(getattr(cfg, "transport_span_blocks", 0))
    base_cost = float(getattr(cfg, "expert_base_cost", 1.0))
    dim_cost = float(getattr(cfg, "expert_dim_cost", 0.05))
    mask = np.ones(D, dtype=float)
    anchor_cost = base_cost + dim_cost * float(np.count_nonzero(mask))
    W = np.zeros((D, D), dtype=float)
    Sigma = np.ones(D, dtype=float)
    node = ExpertNode(
        node_id=node_id,
        mask=mask,
        input_mask=None,
        W=W,
        b=np.zeros(D),
        Sigma=Sigma,
        reliability=1.0,
        cost=float(anchor_cost),
        is_anchor=True,
        footprint=-1,
    )
    lib.nodes[node_id] = node
    lib.anchors.add(node_id)
    node_id += 1

    for b, block_dims in enumerate(blocks):
        mask = np.zeros(D, dtype=float)
        if block_dims:
            mask[np.array(block_dims, dtype=int)] = 1.0
        block_cost = base_cost + dim_cost * float(np.count_nonzero(mask))

        out_idx = np.array(block_dims, dtype=int) if block_dims else np.zeros(0, dtype=int)
        in_mask = mask.copy()
        in_idx = out_idx.copy()
        W = np.zeros((len(out_idx), len(in_idx)), dtype=float)
        Sigma = np.ones(D, dtype=float)
        block_anchor = bool(getattr(cfg, "force_block_anchors", False))
        local_node = ExpertNode(
            node_id=node_id,
            mask=mask,
            input_mask=in_mask,
            W=W,
            b=np.zeros(D),
            Sigma=Sigma,
            reliability=1.0,
            cost=float(block_cost),
            is_anchor=block_anchor,
            footprint=int(b),
            out_idx=out_idx,
            in_idx=in_idx,
        )
        lib.nodes[node_id] = local_node
        lib.footprint_index.setdefault(b, []).append(node_id)
        if block_anchor:
            lib.anchors.add(node_id)
        node_id += 1

        if span > 0:
            input_mask = np.zeros(D, dtype=float)
            lo = max(0, b - span)
            hi = min(len(blocks) - 1, b + span)
            for bb in range(lo, hi + 1):
                dims = blocks[bb]
                if dims:
                    input_mask[np.array(dims, dtype=int)] = 1.0
            in_idx = np.where(input_mask > 0.5)[0].astype(int)
            transport_node = ExpertNode(
                node_id=node_id,
                mask=mask,
                input_mask=input_mask,
                W=np.zeros((len(out_idx), len(in_idx)), dtype=float),
                b=np.zeros(D),
                Sigma=np.ones(D, dtype=float),
                reliability=1.0,
                cost=float(block_cost),
                is_anchor=False,
                footprint=int(b),
                out_idx=out_idx,
                in_idx=in_idx,
            )
            setattr(transport_node, "transport", True)
            lib.nodes[node_id] = transport_node
            lib.footprint_index.setdefault(b, []).append(node_id)
            node_id += 1
    # IMPORTANT: init_library populates lib.nodes directly (not via lib.add_node),
    # so we must advance next_node_id to avoid id collisions that overwrite incumbents
    # (especially the anchor at node_id=0).
    if getattr(lib, 'nodes', None):
        lib.next_node_id = max(int(k) for k in lib.nodes.keys()) + 1
    else:
        lib.next_node_id = 0

    _initialize_dag_neighbors(lib)
    return lib


def nodes_in_library(lib: ExpertLibrary) -> List[ExpertNode]:
    return list(lib.nodes.values())


LIBRARY_LOGGER = logging.getLogger("nupca3.library")


def _initialize_dag_neighbors(lib: ExpertLibrary) -> None:
    """Connect existing nodes within each footprint to seed the DAG."""
    for footprint, node_ids in lib.footprint_index.items():
        for nid in node_ids:
            link_node_neighbors(lib, nid, int(footprint))


def link_node_neighbors(
    lib: ExpertLibrary,
    node_id: int,
    footprint: int,
    *,
    exclude_ids: Iterable[int] | None = None,
) -> None:
    """Link `node_id` to other nodes in the same footprint."""
    node = lib.nodes.get(int(node_id))
    if node is None:
        return
    if footprint < 0:
        return
    bucket = lib.footprint_index.get(int(footprint), [])
    if not bucket:
        return
    exclude_set: Set[int] = set(int(x) for x in (exclude_ids or []) if x is not None)
    candidates: list[int] = [
        int(nid)
        for nid in bucket
        if int(nid) not in exclude_set and int(nid) != int(node_id)
    ]
    if not candidates:
        return
    linked = 0
    for neighbor_id in candidates:
        neighbor = lib.nodes.get(neighbor_id)
        if neighbor is None:
            continue
        node.parents.add(neighbor_id)
        neighbor.children.add(int(node_id))
        linked += 1
    if candidates and linked == 0:
        LIBRARY_LOGGER.warning(
            "DAG link failed for node=%s footprint=%s candidates=%s",
            node_id,
            footprint,
            candidates,
        )
