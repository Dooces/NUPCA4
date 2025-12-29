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
    """Initialize a minimal library of per-footprint experts.

    Fixes:
      - Removes the dense global anchor (D×D) initialization.
        A7.2 already provides the persistence fallback for uncovered dims.
        A dense, all-dims anchor is both budget-violating (O(D^2)) and
        destructive under the current fusion rule (it "covers" everything and
        overwrites stale persistence, even when untrained).
    """
    lib = ExpertLibrary(nodes={}, anchors=set(), footprint_index={})
    node_id = 0
    D = int(cfg.D)
    B = int(cfg.B)
    blocks = _build_blocks(D, B)
    span = int(getattr(cfg, "transport_span_blocks", 0))
    base_cost = float(getattr(cfg, "expert_base_cost", 1.0))
    dim_cost = float(getattr(cfg, "expert_dim_cost", 0.05))
    # dtype: keep params lightweight; step_pipeline casts to float as needed.
    f_dtype = np.float32
    sigma_init_untrained = float(getattr(cfg, "sigma_init_untrained", float("inf")))

    for b, block_dims in enumerate(blocks):
        out_idx = np.array(block_dims, dtype=int) if block_dims else np.zeros(0, dtype=int)
        mask = np.zeros(D, dtype=f_dtype)
        if out_idx.size:
            mask[out_idx] = 1.0

        # Default: local node reads and writes only its footprint.
        in_mask = mask.copy()
        in_idx = out_idx.copy()

        # Cost proportional to output dims (legacy behavior); optionally include
        # input dims if the config enables it.
        cost_include_inputs = bool(getattr(cfg, "expert_cost_include_inputs", False))
        if cost_include_inputs:
            eff_dims = float(out_idx.size + in_idx.size)
        else:
            eff_dims = float(out_idx.size)
        block_cost = base_cost + dim_cost * eff_dims

        W = np.zeros((int(out_idx.size), int(in_idx.size)), dtype=f_dtype)
        # Bias and uncertainty are full-length for compatibility with existing
        # expert.py routines. (A future refactor may store these compactly.)
        b_full = np.zeros(D, dtype=f_dtype)
        Sigma_full = np.full(D, sigma_init_untrained, dtype=f_dtype)
        block_anchor = bool(getattr(cfg, "force_block_anchors", False))
        local_node = ExpertNode(
            node_id=node_id,
            mask=mask,
            input_mask=in_mask,
            W=W,
            b=b_full,
            Sigma=Sigma_full,
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
            input_mask = np.zeros(D, dtype=f_dtype)
            lo = max(0, b - span)
            hi = min(len(blocks) - 1, b + span)
            for bb in range(lo, hi + 1):
                dims = blocks[bb]
                if dims:
                    input_mask[np.array(dims, dtype=int)] = 1.0
            in_idx = np.where(input_mask > 0.5)[0].astype(int)
            if cost_include_inputs:
                eff_dims = float(out_idx.size + in_idx.size)
            else:
                eff_dims = float(out_idx.size)
            t_cost = base_cost + dim_cost * eff_dims
            transport_node = ExpertNode(
                node_id=node_id,
                mask=mask,
                input_mask=input_mask,
                W=np.zeros((int(out_idx.size), int(in_idx.size)), dtype=f_dtype),
                b=np.zeros(D, dtype=f_dtype),
                Sigma=np.full(D, sigma_init_untrained, dtype=f_dtype),
                reliability=1.0,
                cost=float(t_cost),
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
    # so we must advance next_node_id to avoid id collisions that overwrite incumbents.
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
