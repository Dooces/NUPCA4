"""Incumbents index helpers for NUPCA3."""

from __future__ import annotations

from typing import Iterator, List, Optional, Sequence, Set, Tuple

from .types import AgentState, ExpertLibrary, Node, infer_footprint


def _ensure_bucket_capacity(buckets: List[Set[int]], block_id: int) -> Set[int]:
    """Ensure the bucket list can contain `block_id` (modifies in place)."""
    while len(buckets) <= block_id:
        buckets.append(set())
    return buckets[block_id]


def _resolve_block_id(node: Node, blocks: Sequence[Sequence[int]]) -> Optional[int]:
    """Return a block id for `node`, inferring from mask if needed."""
    raw_block_id = getattr(node, "block_id", getattr(node, "footprint", -1))
    if raw_block_id is not None and int(raw_block_id) >= 0:
        return int(raw_block_id)

    mask = getattr(node, "mask", None)
    if mask is None or not blocks:
        return None

    try:
        block_id = infer_footprint(mask, list(blocks))
    except Exception:
        return None

    node.block_id = block_id
    return block_id


def rebuild_incumbents_by_block(library: ExpertLibrary, blocks: Sequence[Sequence[int]]) -> List[Set[int]]:
    """Recreate block-keyed incumbent buckets from the library."""
    buckets: List[Set[int]] = [set() for _ in range(len(blocks))]
    nodes = getattr(library, "nodes", {}) or {}

    for node_id, node in nodes.items():
        block_id = _resolve_block_id(node, blocks)
        if block_id is None or block_id < 0:
            continue
        bucket = _ensure_bucket_capacity(buckets, block_id)
        bucket.add(int(node_id))

    return buckets


def ensure_incumbents_index(state: AgentState) -> None:
    """Rebuild the cached incumbents_by_block index when the library revision changes."""
    library = getattr(state, "library", None)
    if library is None:
        return

    revision = int(getattr(library, "revision", 0))
    if getattr(state, "incumbents_revision", -1) == revision:
        return

    blocks = getattr(state, "blocks", []) or []
    state.incumbents_by_block = rebuild_incumbents_by_block(library, blocks)
    state.incumbents_revision = revision


def get_incumbent_bucket(state: AgentState, block_id: int, *, create: bool = False) -> Optional[Set[int]]:
    """Return the incumbent set for `block_id` (creating it if requested)."""
    if block_id < 0:
        return None

    buckets = getattr(state, "incumbents_by_block", [])
    if block_id < len(buckets):
        return buckets[block_id]

    if not create:
        return None

    bucket = _ensure_bucket_capacity(buckets, block_id)
    return bucket


def iter_incumbent_buckets(state: AgentState) -> Iterator[Tuple[int, Set[int]]]:
    """Yield (block_id, bucket) pairs for the current index."""
    buckets = getattr(state, "incumbents_by_block", [])
    for block_id, bucket in enumerate(buckets):
        yield block_id, bucket
