# /mnt/data/NUPCA4/NUPCA4/nupca3/memory/sig_index.py

"""nupca3/memory/sig_index.py

NUPCA5: scan-proof, bounded signature retrieval index.

v5 requirements implemented here
------------------------------
1) No size-dependent loops:
   - Query cost is strictly bounded by |block_ids| * tables * bucket_cap.
   - Insert/remove cost is strictly bounded by tables * bucket_cap.

2) Overflow/eviction is NOT usage-based and NOT arbitrary.
   When a bucket is full, replacement is driven by outcome-vetted priority:
     - Each node_id has an error estimate updated only by deferred validation
       (via `update_error(...)`).
     - Bucket overflow keeps the best bucket_cap entries by that priority.

3) The optional error cache belongs here only to support (2).
   It is never updated by query/activation; only by explicit deferred
   validation calls.

This structure is a bounded candidate generator (LSH-like); it is not an exact
address map and does not guarantee recall for every node.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from .sig64 import splitmix64

_I32 = np.int32
_F32 = np.float32


@dataclass
class SigIndexCfg:
    # Hash tables (independent salts)
    tables: int = 4
    # Buckets per table: 2^bucket_bits
    bucket_bits: int = 10
    # Slots per bucket
    bucket_cap: int = 8
    # Deterministic salt seed
    seed: int = 0

    # Outcome-vetted priority cache.
    # NOTE: v5 overflow semantics require this to be enabled.
    enable_err_cache: bool = True
    err_bins: int = 3  # NEG/ZERO/POS bins
    # Unknown/unvetted default error; should be pessimistic.
    init_err: float = 1e6
    # Which validation bin to use for eviction priority (default: POS).
    eviction_bin: int = 2


class PackedSigIndex:
    """Packed signature index with bounded query and priority-driven overflow."""

    def __init__(self, cfg: SigIndexCfg):
        self.cfg = cfg
        self.tables = int(cfg.tables)
        self.bucket_bits = int(cfg.bucket_bits)
        self.bucket_cap = int(cfg.bucket_cap)
        if self.bucket_bits < 1:
            raise ValueError("bucket_bits must be >= 1")
        if self.bucket_cap < 1:
            raise ValueError("bucket_cap must be >= 1")
        if self.tables < 1:
            raise ValueError("tables must be >= 1")

        self.n_buckets = 1 << self.bucket_bits
        self.bucket_mask = self.n_buckets - 1

        # Buckets: [tables, n_buckets, bucket_cap] of node_id (int32), -1 empty.
        self.buckets = np.full(
            (self.tables, self.n_buckets, self.bucket_cap),
            fill_value=-1,
            dtype=_I32,
        )

        # Per-table salts (deterministic).
        base = int(cfg.seed) & ((1 << 64) - 1)
        self.salts = np.array(
            [splitmix64(base ^ splitmix64(t + 1)) for t in range(self.tables)],
            dtype=np.uint64,
        )

        # Outcome-vetted error cache used to drive overflow replacement.
        self.enable_err_cache = bool(cfg.enable_err_cache)
        if not self.enable_err_cache:
            raise ValueError("NUPCA5: sig_index overflow requires enable_err_cache=True")
        self.err_bins = int(cfg.err_bins)
        if self.err_bins < 1:
            raise ValueError("err_bins must be >= 1")
        self.eviction_bin = int(cfg.eviction_bin)
        if self.eviction_bin < 0 or self.eviction_bin >= self.err_bins:
            raise ValueError("eviction_bin out of range")
        self._err_cache: Optional[np.ndarray] = None
        self._err_init = float(cfg.init_err)

    @classmethod
    def from_cfg_obj(cls, cfg_obj) -> "PackedSigIndex":
        """Construct from an AgentConfig-like object with sig_* fields."""
        cfg = SigIndexCfg(
            tables=int(getattr(cfg_obj, "sig_tables", 4)),
            bucket_bits=int(getattr(cfg_obj, "sig_bucket_bits", 10)),
            bucket_cap=int(getattr(cfg_obj, "sig_bucket_cap", 8)),
            seed=int(getattr(cfg_obj, "sig_seed", 0)),
            enable_err_cache=bool(getattr(cfg_obj, "sig_enable_err_cache", True)),
            err_bins=int(getattr(cfg_obj, "sig_err_bins", 3)),
            init_err=float(getattr(cfg_obj, "sig_err_init", 1e6)),
            eviction_bin=int(getattr(cfg_obj, "sig_eviction_bin", 2)),
        )
        return cls(cfg)

    def _bucket_index(self, table: int, sig64: int, block_id: int) -> int:
        """Map (sig64, block_id, table) -> bucket index in [0, 2^bucket_bits)."""
        s = int(sig64) & ((1 << 64) - 1)
        b = int(block_id) & ((1 << 64) - 1)
        salt = int(self.salts[table])
        # Two-stage mix to reduce structured collisions.
        x = splitmix64(s ^ salt)
        y = splitmix64(b ^ salt ^ x)
        return int(y) & self.bucket_mask

    def _ensure_err_cache(self, node_id: int) -> None:
        n = int(node_id) + 1
        if self._err_cache is None:
            self._err_cache = np.full((max(n, 1024), self.err_bins), self._err_init, dtype=_F32)
            return
        if n <= self._err_cache.shape[0]:
            return
        new_n = max(n, int(self._err_cache.shape[0] * 2))
        new = np.full((new_n, self.err_bins), self._err_init, dtype=_F32)
        new[: self._err_cache.shape[0], :] = self._err_cache
        self._err_cache = new

    def _priority(self, node_id: int) -> float:
        """Higher is better. Derived only from deferred-validation error."""
        self._ensure_err_cache(int(node_id))
        assert self._err_cache is not None
        nid = int(node_id)
        if nid < 0 or nid >= self._err_cache.shape[0]:
            return float("-inf")
        err = float(self._err_cache[nid, self.eviction_bin])
        if not np.isfinite(err):
            return float("-inf")
        # Lower error => higher priority.
        return -err

    def insert(self, sig64: int, block_id: int, node_id: int) -> None:
        """Insert node_id into each table bucket for (sig64, block_id).

        Overflow behavior (v5): keep the best `bucket_cap` entries by
        outcome-vetted priority; do NOT shift-left/overwrite.
        """
        nid = int(node_id)
        if nid < 0:
            return
        self._ensure_err_cache(nid)
        p_new = self._priority(nid)

        for t in range(self.tables):
            bi = self._bucket_index(t, sig64, block_id)
            row = self.buckets[t, bi]  # shape (bucket_cap,)

            # No duplicates.
            if np.any(row == nid):
                continue

            # First empty slot.
            empties = np.where(row < 0)[0]
            if empties.size:
                row[int(empties[0])] = nid
                continue

            # Bucket full: evict the lowest-priority incumbent IF the new entry
            # is better than that worst incumbent.
            worst_j = 0
            worst_p = float("inf")
            for j in range(self.bucket_cap):
                inc = int(row[j])
                p = self._priority(inc)
                if p < worst_p:
                    worst_p = p
                    worst_j = j

            if p_new > worst_p:
                row[int(worst_j)] = nid

    def remove(self, sig64: int, block_id: int, node_id: int) -> None:
        """Best-effort removal from buckets for (sig64, block_id)."""
        nid = int(node_id)
        if nid < 0:
            return
        for t in range(self.tables):
            bi = self._bucket_index(t, sig64, block_id)
            row = self.buckets[t, bi]
            hits = np.where(row == nid)[0]
            if hits.size:
                row[int(hits[0])] = -1

    def query(
        self,
        sig64: int,
        block_ids: Sequence[int],
        *,
        cand_cap: int = 64,
    ) -> List[int]:
        """Return bounded candidate node_ids for (sig64, block_ids).

        Complexity: O(len(block_ids) * tables * bucket_cap) worst-case, bounded.

        Dedup is done with a bounded linear check (cand_cap is small by design).
        """
        cap = int(cand_cap)
        if cap <= 0:
            return []

        out: List[int] = []

        def _push(nid: int) -> None:
            if nid < 0:
                return
            if nid in out:
                return
            out.append(int(nid))

        for b in block_ids:
            bid = int(b)
            for t in range(self.tables):
                bi = self._bucket_index(t, sig64, bid)
                row = self.buckets[t, bi]
                for j in range(self.bucket_cap):
                    _push(int(row[j]))
                    if len(out) >= cap:
                        return out
        return out

    # ---- Deferred-validation hooks (priority cache) ----

    def update_error(self, node_id: int, h_bin: int, err_ema: float) -> None:
        """Update the outcome-vetted error cache.

        This MUST be called only from deferred validation (i.e., once an
        outcome has been observed), not merely because a unit was activated.
        """
        nid = int(node_id)
        if nid < 0:
            return
        hb = int(h_bin)
        if hb < 0 or hb >= self.err_bins:
            return
        self._ensure_err_cache(nid)
        assert self._err_cache is not None
        self._err_cache[nid, hb] = float(err_ema)

    def get_error(self, node_id: int, h_bin: int) -> float:
        if self._err_cache is None:
            return float(self._err_init)
        hb = int(h_bin)
        if hb < 0 or hb >= self.err_bins:
            return float(self._err_init)
        nid = int(node_id)
        if nid < 0 or nid >= self._err_cache.shape[0]:
            return float(self._err_init)
        return float(self._err_cache[nid, hb])
