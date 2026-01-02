# /mnt/data/NUPCA4/NUPCA4/nupca3/memory/sig_index.py

"""nupca3/memory/sig_index.py

NUPCA5: scan-proof, bounded signature retrieval index using banded LSH
compatible with Hamming distance on sig64.

Key constraints:
  - Bucket keys are derived directly from sig64 band bits (no avalanche hash).
  - Block scope is explicit: buckets are keyed by (table, block_id, bucket_id).
  - Query/insert/remove touch a constant number of buckets:
      O(|block_ids| * tables * bucket_cap).
  - Overflow uses outcome-vetted error cache only (no usage-based scoring).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set

import numpy as np

_I32 = np.int32
_F32 = np.float32


@dataclass
class SigIndexCfg:
    # Hash tables (bands)
    tables: int = 4
    # Buckets per table: 2^bucket_bits
    bucket_bits: int = 10
    # Slots per bucket
    bucket_cap: int = 8
    # Deterministic seed (used only for err cache sizing symmetry; no hashing)
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
    """Packed signature index with banded LSH and outcome-vetted overflow."""

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
        # Use fixed-width bands to preserve Hamming locality (no avalanche hash).
        self.band_bits = max(1, 64 // self.tables)
        self.band_mask = (1 << self.band_bits) - 1

        # buckets[table][block_id] -> np.ndarray shape (n_buckets, bucket_cap)
        self.buckets: List[Dict[int, np.ndarray]] = [dict() for _ in range(self.tables)]

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

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _band_bucket(self, table: int, sig64: int) -> int:
        """Return band-derived bucket index (banded LSH, no avalanche)."""
        shift = min(63, int(table) * self.band_bits)
        band_val = (int(sig64) >> shift) & self.band_mask
        return int(band_val) & self.bucket_mask

    def _ensure_block_row(self, table: int, block_id: int, bucket_idx: int) -> np.ndarray:
        tbl = self.buckets[table]
        arr = tbl.get(int(block_id))
        if arr is None:
            arr = np.full((self.n_buckets, self.bucket_cap), fill_value=-1, dtype=_I32)
            tbl[int(block_id)] = arr
        return arr[int(bucket_idx)]

    def _get_block_row(self, table: int, block_id: int, bucket_idx: int) -> Optional[np.ndarray]:
        tbl = self.buckets[table]
        arr = tbl.get(int(block_id))
        if arr is None:
            return None
        return arr[int(bucket_idx)]

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

    # ------------------------------------------------------------------ #
    # Public interface
    # ------------------------------------------------------------------ #

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
            bi = self._band_bucket(t, sig64)
            row = self._ensure_block_row(t, block_id, bi)

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
            bi = self._band_bucket(t, sig64)
            row = self._get_block_row(t, block_id, bi)
            if row is None:
                continue
            row[row == nid] = -1

    def query(
        self,
        sig64: int,
        block_ids: Sequence[int],
        *,
        cand_cap: int = 64,
    ) -> List[int]:
        """Return bounded candidate node_ids for (sig64, block_ids)."""
        cap = int(cand_cap)
        if cap <= 0:
            return []

        out: List[int] = []
        seen: Set[int] = set()

        for block_id in block_ids:
            b = int(block_id)
            if b < 0:
                continue
            for t in range(self.tables):
                bi = self._band_bucket(t, sig64)
                row = self._get_block_row(t, b, bi)
                if row is None:
                    continue
                for nid in row:
                    if nid < 0:
                        continue
                    if int(nid) in seen:
                        continue
                    seen.add(int(nid))
                    out.append(int(nid))
                    if len(out) >= cap:
                        return out

        return out

    # ---- Deferred-validation hooks (priority cache) ----

    def update_error(self, node_id: int, h_bin: int, err_ema: float) -> None:
        """Update the outcome-vetted error cache."""
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
