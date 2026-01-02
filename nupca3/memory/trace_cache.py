"""Foveated TraceCache supporting bounded, OPERATING-only cues."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, Tuple

import numpy as np

from ..config import AgentConfig
from ..types import AgentState  # type: ignore[assignment]


@dataclass(frozen=True)
class TraceEntry:
    """Foveated cue snapshot stored in the trace cache."""

    t_w: int
    selected_blocks: Tuple[int, ...]
    dims_idx: np.ndarray
    dims_vals: np.ndarray
    sig64: int | None
    meta: Dict[str, int]


class TraceCache:
    """Bounded ring buffer for committed trace entries (A16.8)."""

    def __init__(self, max_entries: int, max_cues_per_entry: int, block_cap: int) -> None:
        self.max_entries = max(1, int(max_entries))
        self.max_cues_per_entry = max(1, int(max_cues_per_entry))
        self.block_cap = max(1, int(block_cap))
        self.entries: Deque[TraceEntry] = deque()
        self.block_refcounts: Dict[int, int] = {}
        self.total_cues = 0

    def clear(self) -> None:
        self.entries.clear()
        self.block_refcounts.clear()
        self.total_cues = 0

    @property
    def size(self) -> int:
        return len(self.entries)

    @property
    def block_count(self) -> int:
        return sum(1 for count in self.block_refcounts.values() if count > 0)

    @property
    def cue_mass(self) -> int:
        return int(self.total_cues)

    def append(self, entry: TraceEntry) -> bool:
        if entry.dims_idx.size == 0:
            return False
        self._evict_to_capacity()
        active_blocks = {b for b, cnt in self.block_refcounts.items() if cnt > 0}
        new_blocks = {b for b in entry.selected_blocks if b not in active_blocks}
        if self.block_count + len(new_blocks) > self.block_cap:
            return False
        self.entries.append(entry)
        self.total_cues += int(entry.dims_idx.size)
        for block_id in entry.selected_blocks:
            block = int(block_id)
            self.block_refcounts[block] = int(self.block_refcounts.get(block, 0)) + 1
        return True

    def _evict_to_capacity(self) -> None:
        while len(self.entries) >= self.max_entries:
            oldest = self.entries.popleft()
            self.total_cues -= int(oldest.dims_idx.size)
            for block_id in oldest.selected_blocks:
                block = int(block_id)
                current = int(self.block_refcounts.get(block, 0)) - 1
                if current <= 0:
                    self.block_refcounts.pop(block, None)
                else:
                    self.block_refcounts[block] = current


def _prepare_entry(
    tick: int,
    blocks: Iterable[int],
    obs_idx: np.ndarray,
    obs_vals: np.ndarray,
    sig64: int | None,
    max_cues: int,
) -> TraceEntry | None:
    dims = np.asarray(obs_idx, dtype=int).reshape(-1)
    if dims.size == 0:
        return None
    order = np.argsort(dims)
    dims = dims[order]
    vals = np.asarray(obs_vals, dtype=float).reshape(-1)[order]
    if dims.size > max_cues:
        dims = dims[:max_cues]
        vals = vals[:max_cues]
    blocks_arr = tuple(sorted({int(b) for b in blocks if b is not None}))
    if not blocks_arr:
        return None
    meta = {"cue_mass": int(dims.size), "block_count": len(blocks_arr)}
    return TraceEntry(
        t_w=int(tick),
        selected_blocks=blocks_arr,
        dims_idx=dims.copy(),
        dims_vals=vals.copy(),
        sig64=int(sig64) if sig64 is not None else None,
        meta=meta,
    )


def init_trace_cache(cfg: AgentConfig) -> TraceCache:
    return TraceCache(
        max_entries=int(getattr(cfg, "trace_cache_max_entries", 16)),
        max_cues_per_entry=int(getattr(cfg, "trace_cache_max_cues_per_entry", 16)),
        block_cap=int(getattr(cfg, "trace_cache_block_cap", 32)),
    )


def cache_observation(
    state: AgentState,
    cfg: AgentConfig,
    *,
    tick: int,
    blocks: Iterable[int],
    obs_idx: np.ndarray,
    obs_vals: np.ndarray,
    sig64: int | None,
) -> bool:
    cache = getattr(state, "trace_cache", None)
    if cache is None:
        return False
    entry = _prepare_entry(
        tick,
        blocks,
        obs_idx,
        obs_vals,
        sig64,
        max_cues=int(getattr(cfg, "trace_cache_max_cues_per_entry", 16)),
    )
    if entry is None:
        return False
    return cache.append(entry)
