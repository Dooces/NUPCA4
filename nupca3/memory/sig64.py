"""nupca3/memory/sig64.py

NUPCA5: scan-proof signature input (sig64).

The signature is a deterministic 64-bit value computed ONLY from:

  (1) Small committed metadata derived from the current observed cue values.
  (2) A small *ephemeral* periphery gist (e.g., blockwise max-pool occupancy).
  (3) A motion-sensitive delta term between committed metadata at (t) and (t-1).

Axiom intent constraints:
  - No dense per-dimension raw state is stored.
  - No dependency on library size (sig computation is fixed cost in input size).
  - The periphery gist is ephemeral: callers must NOT persist it.

This module deliberately does not know anything about blocks/geometry; the
caller provides the per-step ephemeral gist as a small uint8 vector.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


_MASK64 = (1 << 64) - 1


def splitmix64(x: int) -> int:
    """SplitMix64 mixer (deterministic 64-bit)."""
    x = (x + 0x9E3779B97F4A7C15) & _MASK64
    z = x
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9 & _MASK64
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB & _MASK64
    z = (z ^ (z >> 31)) & _MASK64
    return z


def _sat_int16(x: np.ndarray) -> np.ndarray:
    x32 = np.asarray(x, dtype=np.int32)
    return np.clip(x32, -32768, 32767).astype(np.int16, copy=False)


@dataclass(frozen=True)
class Sig64Meta:
    """Committed metadata used as sig64 input.

    counts: small fixed-length int16 vector
    hist:   small fixed-length uint16 histogram
    """

    counts: np.ndarray
    hist: np.ndarray


def build_committed_meta(
    obs_vals: np.ndarray,
    *,
    value_bins: int = 8,
    vmax: float = 4.0,
) -> Sig64Meta:
    """Derive small committed metadata from current observed values.

    This is intentionally low-dimensional and stable:
      counts = [n_obs, n_occ, sum_q, max_q]
      hist   = histogram over quantized values (including 0 bin)

    Args:
        obs_vals: observed cue values (float or int)
        value_bins: number of quantization bins
        vmax: maximum value used for quantization scale

    Returns:
        Sig64Meta with counts (int16) and hist (uint16).
    """
    vals = np.asarray(obs_vals, dtype=np.float32).reshape(-1)
    n_obs = int(vals.size)
    if n_obs == 0:
        counts = np.zeros(4, dtype=np.int16)
        hist = np.zeros(int(value_bins), dtype=np.uint16)
        return Sig64Meta(counts=counts, hist=hist)

    n_occ = int(np.count_nonzero(vals))
    if vmax <= 0:
        q = np.zeros_like(vals, dtype=np.int32)
    else:
        q = np.floor((np.clip(vals, 0.0, float(vmax)) / float(vmax)) * float(value_bins)).astype(np.int32)
        q = np.clip(q, 0, int(value_bins) - 1)

    sum_q = int(q.sum())
    max_q = int(q.max()) if q.size else 0

    counts32 = np.array([n_obs, n_occ, sum_q, max_q], dtype=np.int32)
    counts = _sat_int16(counts32)

    hist32 = np.bincount(q, minlength=int(value_bins)).astype(np.int32)
    hist = np.clip(hist32, 0, 65535).astype(np.uint16)

    return Sig64Meta(counts=counts, hist=hist)


def compute_sig64(
    meta_t: Sig64Meta,
    gist_u8_t: np.ndarray,
    *,
    prev_meta: Optional[Sig64Meta] = None,
    seed: int = 0,
) -> Tuple[int, Sig64Meta]:
    """Compute sig64(t).

    Args:
        meta_t: committed metadata at time t
        gist_u8_t: ephemeral gist at time t (uint8 vector). Caller must NOT persist.
        prev_meta: committed metadata at time t-1 (for delta term)
        seed: deterministic seed (per-agent config)

    Returns:
        (sig64, meta_t) where meta_t is returned for convenience.
    """
    counts_t = np.asarray(meta_t.counts, dtype=np.int16).reshape(-1)
    hist_t = np.asarray(meta_t.hist, dtype=np.uint16).reshape(-1)
    gist = np.asarray(gist_u8_t, dtype=np.uint8).reshape(-1)

    if prev_meta is None:
        d_counts = np.zeros_like(counts_t, dtype=np.int16)
        d_hist = np.zeros_like(hist_t, dtype=np.int16)
    else:
        prev_counts = np.asarray(prev_meta.counts, dtype=np.int16).reshape(-1)
        prev_hist = np.asarray(prev_meta.hist, dtype=np.uint16).reshape(-1)
        d_counts = _sat_int16(counts_t.astype(np.int32) - prev_counts.astype(np.int32))
        d_hist = _sat_int16(hist_t.astype(np.int32) - prev_hist.astype(np.int32))

    # Pack into a byte payload deterministically.
    # int16 arrays are viewed as uint8; uint16 hist is viewed as uint8; delta hist uses int16 view.
    parts = [
        counts_t.view(np.uint8),
        d_counts.view(np.uint8),
        hist_t.view(np.uint8),
        d_hist.view(np.uint8),
        gist,
    ]
    payload = np.concatenate(parts, axis=0).tobytes(order="C")

    # Mix 8-byte chunks.
    x = splitmix64((int(seed) & _MASK64) ^ (len(payload) & _MASK64))
    step_const = 0x9E3779B97F4A7C15
    for i in range(0, len(payload), 8):
        chunk = payload[i : i + 8]
        if len(chunk) < 8:
            chunk = chunk + b"\x00" * (8 - len(chunk))
        v = int.from_bytes(chunk, byteorder="little", signed=False)
        x = splitmix64((x ^ ((v + ((i // 8 + 1) * step_const)) & _MASK64)) & _MASK64)

    # Final avalanching.
    sig = splitmix64(x)
    return int(sig & _MASK64), meta_t


def compute_sig64_from_obs(
    obs_vals: np.ndarray,
    gist_u8_t: np.ndarray,
    *,
    prev_meta: Optional[Sig64Meta] = None,
    value_bins: int = 8,
    vmax: float = 4.0,
    seed: int = 0,
) -> Tuple[int, Sig64Meta]:
    """Convenience wrapper: build committed metadata from obs_vals and compute sig64."""
    meta_t = build_committed_meta(obs_vals, value_bins=value_bins, vmax=vmax)
    return compute_sig64(meta_t, gist_u8_t, prev_meta=prev_meta, seed=seed)
