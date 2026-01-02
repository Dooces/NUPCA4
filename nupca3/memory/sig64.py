"""nupca3/memory/sig64.py

NUPCA5 v5.02(3) signature (sig64) with the closed metadata schema and
Hamming-local SimHash sketch:

  - Metadata = block_mask bitvector || total_dims (u16) || anchor_count (u8)
    || block_counts[≤F_MAX] (u8, sorted by block_id, zero-padded).
  - Sketch = SimHash over the fixed-size metadata bytes concatenated with the
    ephemeral periphery gist p(t).to_bytes().

Constraints:
  - No avalanche hashes; Hamming distance must reflect locality.
  - Input size is fixed by (B, F_MAX, |gist|); cost is constant in library size.
  - Callers must not persist gist; it is ephemeral and non-serialized.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np


_MASK64 = (1 << 64) - 1


_SIMHASH_CACHE: dict[Tuple[int, int], np.ndarray] = {}


def splitmix64(x: int) -> int:
    """Deterministic 64-bit mixer (used for synthetic/template addresses)."""
    x = (x + 0x9E3779B97F4A7C15) & _MASK64
    z = x
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9 & _MASK64
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB & _MASK64
    z = (z ^ (z >> 31)) & _MASK64
    return z


@dataclass(frozen=True)
class Sig64Meta:
    """Closed metadata schema for sig64.

    Fields are already quantized/padded to fixed sizes:
      - block_mask_bytes: bitvector of length ceil(B/8)
      - total_dims: number of committed dims in O_t
      - anchor_count: number of anchor blocks in O_t
      - block_counts: uint8[F_max] sorted by block_id (padded with zeros)
      - B, F_max retained for auditing (not used in the sketch directly)

    For backward compatibility with older state snapshots, counts/hist fields
    are retained but set to empty arrays.
    """

    block_mask_bytes: np.ndarray
    total_dims: int
    anchor_count: int
    block_counts: np.ndarray
    B: int
    F_max: int
    counts: np.ndarray = np.zeros(0, dtype=np.int16)
    hist: np.ndarray = np.zeros(0, dtype=np.uint16)

    def to_bytes(self) -> bytes:
        parts = [
            np.asarray(self.block_mask_bytes, dtype=np.uint8).reshape(-1),
            np.array([int(self.total_dims)], dtype=np.uint16).view(np.uint8),
            np.array([int(self.anchor_count)], dtype=np.uint8),
            np.asarray(self.block_counts, dtype=np.uint8).reshape(-1),
        ]
        return b"".join(p.tobytes(order="C") for p in parts)


def compute_sig64(
    meta_t: Sig64Meta,
    gist_u8_t: np.ndarray,
    *,
    seed: int = 0,
) -> Tuple[int, Sig64Meta]:
    """Compute sig64(t) using SimHash over metadata || gist bytes."""
    meta_bytes = meta_t.to_bytes()
    gist = np.asarray(gist_u8_t, dtype=np.uint8).reshape(-1)
    payload = np.concatenate(
        [np.frombuffer(meta_bytes, dtype=np.uint8), gist], axis=0
    ).astype(np.int16)
    centered = payload - 128  # center to symmetric integers

    key = (int(seed), int(centered.size))
    R = _SIMHASH_CACHE.get(key)
    if R is None:
        rng = np.random.default_rng(int(seed))
        R = rng.choice(np.array([-1, 1], dtype=np.int8), size=(64, centered.size))
        _SIMHASH_CACHE[key] = R

    proj = R.astype(np.int32) @ centered.astype(np.int32)
    bits = (proj >= 0).astype(np.uint8)
    sig = 0
    for i, bit in enumerate(bits.tolist()):
        if bit:
            sig |= 1 << i
    return int(sig & _MASK64), meta_t


def compute_sig64_from_blocks(
    block_counts: Mapping[int, int],
    *,
    B: int,
    F_max: int,
    anchor_blocks: Iterable[int],
    gist_u8_t: np.ndarray,
    seed: int = 0,
) -> Tuple[int, Sig64Meta]:
    """Build the closed metadata schema from block counts and compute sig64."""
    B = max(0, int(B))
    F_max = max(0, int(F_max))
    block_ids = sorted(int(b) for b in block_counts.keys() if 0 <= int(b) < B)

    # Block mask bitvector (ceil(B/8) bytes).
    mask_bytes = np.zeros((max(1, (B + 7) // 8),), dtype=np.uint8)
    for b in block_ids:
        byte_idx = b // 8
        bit_idx = b % 8
        mask_bytes[byte_idx] |= np.uint8(1 << bit_idx)

    # Block counts array (sorted by block_id, padded to F_max).
    counts_arr = np.zeros((F_max,), dtype=np.uint8)
    for i, b in enumerate(block_ids[:F_max]):
        counts_arr[i] = np.uint8(max(0, min(255, int(block_counts.get(b, 0)))))

    total_dims = int(sum(int(block_counts.get(b, 0)) for b in block_ids))
    anchor_set = {int(a) for a in anchor_blocks}
    anchor_count = sum(1 for b in block_ids if b in anchor_set)

    meta = Sig64Meta(
        block_mask_bytes=mask_bytes,
        total_dims=total_dims,
        anchor_count=anchor_count,
        block_counts=counts_arr,
        B=B,
        F_max=F_max,
        counts=np.zeros(0, dtype=np.int16),
        hist=np.zeros(0, dtype=np.uint16),
    )
    return compute_sig64(meta, np.asarray(gist_u8_t, dtype=np.uint8), seed=seed)
