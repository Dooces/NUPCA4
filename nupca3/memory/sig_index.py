# /mnt/data/NUPCA4/NUPCA4/nupca3/memory/sig_index.py

"""nupca3/memory/sig_index.py

NUPCA5: token-overlap evidence index (multi-evidence collisions), replacing
Hamming(sig64) with overlap counting:

  - Each unit is keyed by a small set of discrete tokens (e.g., block ids).
  - An inverted index maps token -> node ids.
  - DF stoplist excludes background-like tokens (df/N >= df_stop_frac).
  - Candidate admission requires >= min_evidence distinct eligible tokens.
  - Stage-2 can still use deferred-validation error cache to rank.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set

import numpy as np

_I32 = np.int32
_F32 = np.float32


@dataclass
class SigIndexCfg:
    # Evidence gate: minimum distinct eligible tokens required for admission.
    min_evidence: int = 2
    # Document-frequency stoplist threshold (df/N >= tau_df => drop token).
    df_stop_frac: float = 0.2
    # Number of sketch components (if used downstream for compression).
    sketch_K: int = 0
    # Legacy params retained for compatibility/persistence (unused here).
    tables: int = 4
    bucket_bits: int = 10
    bucket_cap: int = 8
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
        self.min_evidence = max(1, int(cfg.min_evidence))
        self.df_stop_frac = max(0.0, min(1.0, float(cfg.df_stop_frac)))
        self.sketch_K = max(0, int(cfg.sketch_K))
        # Legacy/Hamming fields retained for persistence compatibility.
        self.tables = int(getattr(cfg, "tables", 0))
        self.bucket_bits = int(getattr(cfg, "bucket_bits", 0))
        self.bucket_cap = int(getattr(cfg, "bucket_cap", 0))
        self.n_buckets = 1 << max(1, self.bucket_bits or 1)
        self.bucket_mask = self.n_buckets - 1
        self.band_bits = max(1, 64 // max(1, self.tables or 1))
        self.band_mask = (1 << self.band_bits) - 1

        # Token inverted index: token -> set(node_id)
        self.token_index: Dict[int, Set[int]] = {}
        # Document frequencies for df-stoplist.
        self.df_counts: Dict[int, int] = {}
        self.total_units: int = 0
        # Keep placeholders for legacy persistence callers.
        self.buckets: List[Dict[int, np.ndarray]] = [dict() for _ in range(max(1, self.tables or 1))]
        self.salts = np.zeros(0, dtype=np.uint64)

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
            min_evidence=int(getattr(cfg_obj, "sig_min_evidence", 2)),
            df_stop_frac=float(getattr(cfg_obj, "sig_df_stop_frac", 0.2)),
            sketch_K=int(getattr(cfg_obj, "sig_sketch_K", 0)),
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

    def _eligible_tokens(self, tokens: Sequence[int]) -> List[int]:
        """Apply df-stoplist to provided tokens."""
        if not tokens or self.total_units <= 0:
            return []
        out: List[int] = []
        for t in tokens:
            tt = int(t)
            if tt < 0:
                continue
            df = int(self.df_counts.get(tt, 0))
            frac = float(df) / float(max(1, self.total_units))
            if frac >= self.df_stop_frac:
                continue
            out.append(tt)
        return out

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

    def insert_tokens(self, node_id: int, tokens: Sequence[int]) -> None:
        """Insert node_id keyed by the provided tokens (token-overlap evidence)."""
        nid = int(node_id)
        if nid < 0:
            return
        self._ensure_err_cache(nid)
        uniq_tokens = sorted({int(t) for t in tokens if int(t) >= 0})
        self.total_units += 1
        for t in uniq_tokens:
            self.df_counts[t] = int(self.df_counts.get(t, 0)) + 1
            bucket = self.token_index.get(t)
            if bucket is None:
                bucket = set()
                self.token_index[t] = bucket
            bucket.add(nid)

    def remove_tokens(self, node_id: int, tokens: Sequence[int]) -> None:
        """Remove node_id from the provided tokens."""
        nid = int(node_id)
        if nid < 0:
            return
        uniq_tokens = sorted({int(t) for t in tokens if int(t) >= 0})
        if self.total_units > 0:
            self.total_units -= 1
        for t in uniq_tokens:
            if t in self.df_counts:
                self.df_counts[t] = max(0, int(self.df_counts[t]) - 1)
                if self.df_counts[t] == 0:
                    self.df_counts.pop(t, None)
            bucket = self.token_index.get(t)
            if bucket is None:
                continue
            bucket.discard(nid)
            if not bucket:
                self.token_index.pop(t, None)

    def query(
        self,
        query_tokens: Sequence[int],
        *,
        cand_cap: int = 64,
    ) -> List[tuple[int, int]]:
        """Return candidate node_ids with evidence counts (token overlaps)."""
        cap = int(cand_cap)
        if cap <= 0:
            return []

        eligible = self._eligible_tokens(sorted(set(int(t) for t in query_tokens)))
        if not eligible:
            return []

        evidence: Dict[int, int] = {}
        for t in eligible:
            bucket = self.token_index.get(int(t))
            if bucket is None:
                continue
            for nid in bucket:
                evidence[nid] = evidence.get(nid, 0) + 1

        # Apply evidence gate.
        min_e = self.min_evidence
        filtered = [(nid, cnt) for nid, cnt in evidence.items() if cnt >= min_e]
        if not filtered:
            return []

        # Highest evidence first; tie-breaker by node id.
        filtered.sort(key=lambda x: (x[1], -int(x[0])), reverse=True)
        if len(filtered) > cap:
            filtered = filtered[:cap]
        return filtered

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
