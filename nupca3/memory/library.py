# /mnt/data/NUPCA4/NUPCA4/nupca3/memory/library.py

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
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from ..config import AgentConfig
from ..types import ExpertLibrary, ExpertNode

from .sig64 import splitmix64
from .sig_index import PackedSigIndex


_MASK64 = (1 << 64) - 1


class V5ExpertLibrary(ExpertLibrary):
    """ExpertLibrary with v5 unit-lifecycle hooks.

    Key property:
      - sig_index bucket updates occur ONLY on unit creation/eviction, never in
        the online step loop.

    Requirements:
      - New units MUST arrive with `unit_sig64` already set to their stored
        64-bit retrieval address (computed from committed metadata + ephemeral
        gist at creation time).
    """

    def add_node(self, node: ExpertNode) -> int:  # type: ignore[override]
        # Enforce v5 address semantics (no silent salts).
        if not hasattr(node, "unit_sig64"):
            raise ValueError("NUPCA5: node must define unit_sig64 (stored 64-bit address)")
        addr = int(getattr(node, "unit_sig64")) & _MASK64
        if addr == 0:
            raise ValueError("NUPCA5: node.unit_sig64 must be a non-zero stored address")

        node_id = super().add_node(node)
        # Register in the packed signature index (if enabled).
        dim2block = getattr(self, "_sig_dim2block", None)
        register_unit_in_sig_index(self, node, dim2block=dim2block)
        return int(node_id)

    def remove_node(self, node_id: int) -> Optional[ExpertNode]:  # type: ignore[override]
        node = self.nodes.get(int(node_id))
        if node is not None:
            unregister_unit_from_sig_index(self, node)
        return super().remove_node(int(node_id))


def _template_unit_addr(
    *,
    seed: int,
    block_id: int,
    is_anchor: bool,
    is_transport: bool,
) -> int:
    """Deterministic address for *synthetic* init-time units.

    v5 intent: learned units must store the 64-bit address derived from
    (committed metadata + ephemeral gist) *at creation time*. The seed library
    units created in `init_library()` do not have an observation context, so we
    give them a deterministic address derived only from their committed
    template metadata (block_id + flags) and the configured sig_seed.

    This is NOT the old pseudo-random per-node salt (it does not depend on
    node_id). Real spawned/merged/split units must set their unit address from
    the observation-time sig64.
    """
    s = int(seed) & _MASK64
    b = int(block_id) & _MASK64
    flags = (1 if bool(is_anchor) else 0) | ((1 if bool(is_transport) else 0) << 1)

    # Pack a tiny committed-metadata payload and mix.
    # Layout: [block_id (u32), flags (u32), constant (u64)]
    payload = (
        int(b & 0xFFFFFFFF).to_bytes(4, "little", signed=False)
        + int(flags & 0xFFFFFFFF).to_bytes(4, "little", signed=False)
        + int(0xA5A5A5A5A5A5A5A5).to_bytes(8, "little", signed=False)
    )
    x = splitmix64(s ^ int.from_bytes(payload[:8], "little", signed=False))
    x = splitmix64(x ^ int.from_bytes(payload[8:16], "little", signed=False))
    return int(x & _MASK64)


def _sig_index_blocks_for_node(
    *,
    node: ExpertNode,
    dim2block: Optional[np.ndarray],
) -> Tuple[int, ...]:
    """Return the block ids under which the node should be registered.

    We register under the union of block_ids touched by (in_idx ∪ out_idx),
    falling back to node.footprint if no indices exist.

    This is computed at unit lifecycle time (creation) and stored on the node
    as a small tuple for deterministic removal on eviction.
    """
    bids: Set[int] = set()
    if dim2block is not None:
        for idx_name in ("in_idx", "out_idx"):
            arr = getattr(node, idx_name, None)
            if arr is None:
                continue
            a = np.asarray(arr, dtype=int).reshape(-1)
            if a.size:
                a = a[(a >= 0) & (a < dim2block.size)]
                if a.size:
                    bids.update(int(x) for x in np.unique(dim2block[a]).tolist())

    if not bids:
        bids.add(int(getattr(node, "footprint", -1)))
    bids = {int(b) for b in bids if int(b) >= 0}
    if not bids:
        return tuple()
    return tuple(sorted(bids))


def register_unit_in_sig_index(
    lib: ExpertLibrary,
    node: ExpertNode,
    *,
    dim2block: Optional[np.ndarray] = None,
) -> None:
    """Register a unit in the packed signature index.

    This is a unit-lifecycle operation (creation), not an online step-loop
    operation.
    """
    sig_index = getattr(lib, "sig_index", None)
    if sig_index is None:
        return
    addr = int(getattr(node, "unit_sig64", 0)) & _MASK64
    bids = _sig_index_blocks_for_node(node=node, dim2block=dim2block)
    setattr(node, "sig_index_blocks", bids)
    nid = int(getattr(node, "node_id", -1))
    for b in bids:
        sig_index.insert(int(addr), int(b), int(nid))


def unregister_unit_from_sig_index(
    lib: ExpertLibrary,
    node: ExpertNode,
) -> None:
    """Unregister a unit from the packed signature index (eviction)."""
    sig_index = getattr(lib, "sig_index", None)
    if sig_index is None:
        return
    addr = int(getattr(node, "unit_sig64", 0)) & _MASK64
    bids = tuple(int(b) for b in (getattr(node, "sig_index_blocks", ()) or ()))
    nid = int(getattr(node, "node_id", -1))
    for b in bids:
        sig_index.remove(int(addr), int(b), int(nid))


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
    lib: ExpertLibrary = V5ExpertLibrary(nodes={}, anchors=set(), footprint_index={})
    # NUPCA5: create packed signature index for scan-proof retrieval.
    lib.sig_index = PackedSigIndex.from_cfg_obj(cfg)
    node_id = 0
    D = int(cfg.D)
    B = int(cfg.B)
    blocks = _build_blocks(D, B)

    # Precompute dim->block map for bounded, lifecycle-time sig_index registration.
    dim2block: Optional[np.ndarray]
    if D > 0 and blocks and blocks[0] != []:
        dim2block = np.empty(D, dtype=np.int32)
        for bi, dims in enumerate(blocks):
            for d in dims:
                if 0 <= int(d) < D:
                    dim2block[int(d)] = int(bi)
    else:
        dim2block = None
    # Stash for bounded sig_index lifecycle registration (creation/eviction).
    setattr(lib, "_sig_dim2block", dim2block)
    span = int(cfg.transport_span_blocks)
    base_cost = float(cfg.expert_base_cost)
    dim_cost = float(cfg.expert_dim_cost)
    # dtype: keep params lightweight; step_pipeline casts to float as needed.
    f_dtype = np.float32
    sigma_init_untrained = float(cfg.sigma_init_untrained)

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
        cost_include_inputs = bool(cfg.expert_cost_include_inputs)
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
        block_anchor = bool(cfg.force_block_anchors)
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
        # v5: `unit_sig64` is the unit's stored 64-bit retrieval address.
        # Seed units created at init have no observation context; assign a
        # deterministic template-derived address (no node_id dependence).
        local_node.unit_sig64 = _template_unit_addr(
            seed=int(cfg.sig_seed),
            block_id=int(b),
            is_anchor=bool(block_anchor),
            is_transport=False,
        )
        # Register in scan-proof retrieval index at unit lifecycle time.
        register_unit_in_sig_index(lib, local_node, dim2block=dim2block)
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
            transport_node.unit_sig64 = _template_unit_addr(
                seed=int(cfg.sig_seed),
                block_id=int(b),
                is_anchor=False,
                is_transport=True,
            )
            register_unit_in_sig_index(lib, transport_node, dim2block=dim2block)
            lib.nodes[node_id] = transport_node
            lib.footprint_index.setdefault(b, []).append(node_id)
            node_id += 1

    # IMPORTANT: init_library populates lib.nodes directly (not via lib.add_node),
    # so we must advance next_node_id to avoid id collisions that overwrite incumbents.
    if getattr(lib, "nodes", None):
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
