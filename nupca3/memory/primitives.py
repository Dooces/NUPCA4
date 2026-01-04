"""Primitive extractor for NUPCA5 retrieval keys (A16.X).

Implements a fixed, discrete vocabulary of primitive IDs derived only from:
  - observed block IDs (O_t),
  - bounded per-block summaries (q_b(t-1), p_b(t), U_prev_state),
  - fixed schema metadata.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Set

import numpy as np

from ..config import AgentConfig
from ..geometry.fovea import dims_for_block
from ..types import AgentState


# Primitive family IDs (stable, small integers).
_F_BLOCK = 1
_F_PMEAN = 2
_F_U = 3
_F_DMEAN = 4
_F_META_DIMS = 5
_F_META_ANCHOR = 6


def _quantize_linear(value: float, *, bins: int, vmin: float, vmax: float) -> int:
    if bins <= 1:
        return 0
    if not np.isfinite(value):
        return 0
    if vmax <= vmin:
        return 0
    v = max(vmin, min(vmax, float(value)))
    pos = (v - vmin) / (vmax - vmin)
    return int(min(bins - 1, max(0, int(round(pos * (bins - 1))))))


def _block_count(state: AgentState, cfg: AgentConfig) -> int:
    blocks = getattr(state, "blocks", None)
    if blocks:
        return int(len(blocks))
    return max(0, int(getattr(cfg, "B", 0)))


def _build_dim2block(blocks: Sequence[Sequence[int]], D: int) -> np.ndarray:
    m = np.full(int(D), -1, dtype=np.int32)
    for bi, dims in enumerate(blocks):
        for d in dims:
            dd = int(d)
            if 0 <= dd < m.size:
                m[dd] = int(bi)
    return m


def _get_dim2block(state: AgentState, cfg: AgentConfig) -> np.ndarray | None:
    lib = getattr(state, "library", None)
    dim2block = getattr(lib, "_sig_dim2block", None) if lib is not None else None
    if isinstance(dim2block, np.ndarray) and dim2block.size:
        return dim2block

    blocks = getattr(state, "blocks", None)
    if not blocks:
        return None
    D = int(getattr(cfg, "D", 0))
    if D <= 0:
        return None
    dim2block = _build_dim2block(blocks, D)
    return dim2block


def _observed_blocks(state: AgentState, cfg: AgentConfig) -> List[int]:
    dim2block = _get_dim2block(state, cfg)
    obs_dims = getattr(getattr(state, "buffer", None), "observed_dims", set()) or set()
    if dim2block is None or not obs_dims:
        return sorted(int(b) for b in (getattr(state, "current_fovea", set()) or set()))
    blocks: Set[int] = set()
    for d in obs_dims:
        dd = int(d)
        if 0 <= dd < dim2block.size:
            b = int(dim2block[dd])
            if b >= 0:
                blocks.add(b)
    return sorted(blocks)


def _block_dims(state: AgentState, b: int, cfg: AgentConfig) -> Sequence[int]:
    blocks = getattr(state, "blocks", None)
    if blocks and 0 <= int(b) < len(blocks):
        return blocks[int(b)]
    return dims_for_block(int(b), cfg)


def _block_mean_from_dense(x_vec: np.ndarray, block_dims: Sequence[int]) -> float:
    if x_vec.size == 0 or not block_dims:
        return 0.0
    idx = np.asarray([int(d) for d in block_dims if 0 <= int(d) < x_vec.size], dtype=int)
    if idx.size == 0:
        return 0.0
    return float(np.mean(x_vec[idx]))


def _token_stride(cfg: AgentConfig, *, B: int, bins: int) -> int:
    # Bounded stride keeps IDs small and stable across runs.
    block_stride = max(1, int(bins))
    return max(1, int(B)) * block_stride


def _encode_token(
    family: int,
    block_id: int,
    bin_id: int,
    *,
    stride: int,
    bins: int,
    offset: int,
) -> int:
    if int(family) == _F_BLOCK:
        return int(block_id)
    block_stride = max(1, int(bins))
    fam = max(1, int(family)) - 1
    return int(offset) + int(fam) * int(stride) + int(block_id) * block_stride + int(bin_id)


def compute_primitive_tokens(
    state: AgentState,
    cfg: AgentConfig,
    *,
    block_ids: Iterable[int] | None = None,
) -> List[int]:
    """Return the primitive ID set K(·) from the committed audited bundle."""
    B = _block_count(state, cfg)
    bins = max(1, int(getattr(cfg, "sig_value_bins", 8)))
    vmax = float(getattr(cfg, "sig_vmax", 4.0))
    stride = _token_stride(cfg, B=B, bins=bins)
    offset = max(1, int(B))

    if block_ids is None:
        blocks = _observed_blocks(state, cfg)
    else:
        blocks = sorted({int(b) for b in block_ids})
    if B > 0:
        blocks = [b for b in blocks if 0 <= int(b) < B]

    q_mean = np.asarray(getattr(state, "q_block_mean", np.zeros(0, dtype=float)), dtype=float).reshape(-1)
    u_prev = np.asarray(getattr(state, "U_prev_state", np.zeros(0, dtype=float)), dtype=float).reshape(-1)
    x_last = np.asarray(getattr(getattr(state, "buffer", None), "x_last", np.zeros(0, dtype=float)), dtype=float).reshape(-1)

    tokens: Set[int] = set()

    for b in blocks:
        p_mean = _block_mean_from_dense(x_last, _block_dims(state, b, cfg))
        q_val = float(q_mean[b]) if b < q_mean.size else 0.0
        u_val = float(u_prev[b]) if b < u_prev.size else 0.0

        tokens.add(_encode_token(_F_BLOCK, b, 0, stride=stride, bins=bins, offset=offset))
        tokens.add(
            _encode_token(
                _F_PMEAN,
                b,
                _quantize_linear(p_mean, bins=bins, vmin=-vmax, vmax=vmax),
                stride=stride,
                bins=bins,
                offset=offset,
            )
        )
        tokens.add(
            _encode_token(
                _F_U,
                b,
                _quantize_linear(u_val, bins=bins, vmin=0.0, vmax=1.0),
                stride=stride,
                bins=bins,
                offset=offset,
            )
        )
        tokens.add(
            _encode_token(
                _F_DMEAN,
                b,
                _quantize_linear(p_mean - q_val, bins=bins, vmin=-vmax, vmax=vmax),
                stride=stride,
                bins=bins,
                offset=offset,
            )
        )

    # Fixed schema metadata (bounded, closed-form).
    total_dims = int(len(getattr(getattr(state, "buffer", None), "observed_dims", set()) or set()))
    total_dims_bin = _quantize_linear(total_dims, bins=bins, vmin=0.0, vmax=max(1.0, float(getattr(cfg, "D", 1))))
    tokens.add(
        _encode_token(
            _F_META_DIMS,
            0,
            total_dims_bin,
            stride=stride,
            bins=bins,
            offset=offset,
        )
    )

    anchor_blocks = {int(b) for b in (getattr(cfg, "contemplate_anchor_blocks", ()) or ())}
    anchor_count = int(sum(1 for b in blocks if int(b) in anchor_blocks))
    anchor_bin = _quantize_linear(anchor_count, bins=bins, vmin=0.0, vmax=max(1.0, float(B)))
    tokens.add(
        _encode_token(
            _F_META_ANCHOR,
            0,
            anchor_bin,
            stride=stride,
            bins=bins,
            offset=offset,
        )
    )

    return sorted(tokens)


def update_q_block_mean(
    state: AgentState,
    obs_idx: np.ndarray,
    obs_vals: np.ndarray,
    cfg: AgentConfig,
) -> None:
    """Update per-block committed mean summary q_b(t)."""
    B = _block_count(state, cfg)
    if B <= 0:
        state.q_block_mean = np.zeros(0, dtype=float)
        return
    q_mean = np.asarray(getattr(state, "q_block_mean", np.zeros(B, dtype=float)), dtype=float).reshape(-1)
    if q_mean.size != B:
        q_mean = np.resize(q_mean, (B,))

    if obs_idx.size == 0 or obs_vals.size == 0:
        state.q_block_mean = q_mean
        return

    dim2block = _get_dim2block(state, cfg)
    if dim2block is None or dim2block.size == 0:
        state.q_block_mean = q_mean
        return

    sums = np.zeros(B, dtype=float)
    counts = np.zeros(B, dtype=np.int32)
    for dim, val in zip(obs_idx.tolist(), obs_vals.tolist()):
        d = int(dim)
        if 0 <= d < dim2block.size:
            b = int(dim2block[d])
            if 0 <= b < B:
                sums[b] += float(val)
                counts[b] += 1

    for b in np.where(counts > 0)[0].tolist():
        q_mean[int(b)] = float(sums[int(b)] / float(counts[int(b)]))

    state.q_block_mean = q_mean
