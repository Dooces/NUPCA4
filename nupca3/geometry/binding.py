"""Binding transforms for location-variant constellations.

Implements shift/rotation binding by permuting mask coordinates at evaluation time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np


@dataclass
class BindingMap:
    forward: np.ndarray  # canonical -> world (len D), -1 for out-of-bounds
    inverse: np.ndarray  # world -> canonical (len D), -1 for out-of-bounds
    base_dim: int


def _rot_coord(x: int, y: int, side: int, rot: int) -> Tuple[int, int]:
    if rot == 90:
        return side - 1 - y, x
    if rot == 180:
        return side - 1 - x, side - 1 - y
    if rot == 270:
        return y, side - 1 - x
    return x, y


def build_binding_maps(
    *,
    D: int,
    side: int,
    channels: int,
    base_dim: int,
    shift_radius: int,
    rotations: Sequence[int],
) -> List[BindingMap]:
    maps: List[BindingMap] = []
    side = int(side)
    channels = int(channels)
    base_dim = int(base_dim)
    if side <= 0 or channels <= 0 or base_dim <= 0:
        return maps
    if base_dim < side * side * channels:
        return maps

    def _shift_seq(r: int) -> List[int]:
        seq = [0]
        for i in range(1, r + 1):
            seq.extend([i, -i])
        return seq

    shift_seq = _shift_seq(shift_radius)
    for rot in rotations:
        for dy in shift_seq:
            for dx in shift_seq:
                fwd = np.full(D, -1, dtype=int)
                inv = np.full(D, -1, dtype=int)
                # Base grid dims
                for y in range(side):
                    for x in range(side):
                        xr, yr = _rot_coord(x, y, side, int(rot))
                        xt = xr + dx
                        yt = yr + dy
                        if xt < 0 or yt < 0 or xt >= side or yt >= side:
                            continue
                        cell_c = y * side + x
                        cell_w = yt * side + xt
                        for c in range(channels):
                            i = (cell_c * channels) + c
                            j = (cell_w * channels) + c
                            if i >= D or j >= D:
                                continue
                            fwd[i] = j
                            inv[j] = i
                # Non-grid dims pass through unchanged.
                for i in range(base_dim, D):
                    fwd[i] = i
                    inv[i] = i
                maps.append(BindingMap(forward=fwd, inverse=inv, base_dim=base_dim))
    return maps


def select_best_binding(
    *,
    mask: np.ndarray,
    observed_dims: Set[int],
    maps: Sequence[BindingMap],
) -> Optional[BindingMap]:
    if not maps or mask is None:
        return None
    on = np.where(np.asarray(mask, dtype=float) > 0.5)[0]
    if on.size == 0:
        return None
    best: Optional[BindingMap] = None
    best_score = -1
    for m in maps:
        fwd = m.forward
        count = 0
        for i in on:
            if i < 0 or i >= fwd.shape[0]:
                continue
            j = int(fwd[i])
            if j >= 0 and j in observed_dims:
                count += 1
        if count > best_score:
            best_score = count
            best = m
    return best


def select_best_binding_by_fit(
    *,
    mask: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
    input_mask: Optional[np.ndarray],
    x_prev: np.ndarray,
    cue_t: Dict[int, float],
    maps: Sequence[BindingMap],
) -> Optional[BindingMap]:
    if not maps or mask is None:
        return None
    if not cue_t:
        observed_dims = set(int(k) for k in cue_t.keys())
        return select_best_binding(mask=mask, observed_dims=observed_dims, maps=maps)
    mask_arr = np.asarray(mask, dtype=float)
    mask_idx = np.where(mask_arr > 0.5)[0]
    if mask_idx.size == 0:
        return None
    in_mask = input_mask if input_mask is not None else mask_arr
    in_mask = np.asarray(in_mask, dtype=float)
    obs_items = list(cue_t.items())
    best: Optional[BindingMap] = None
    best_err = float("inf")
    best_count = -1
    for m in maps:
        fwd = np.asarray(m.forward, dtype=int)
        inv = np.asarray(m.inverse, dtype=int)
        x_canon = np.zeros_like(x_prev)
        valid = fwd >= 0
        x_canon[valid] = x_prev[fwd[valid]]
        x_masked = x_canon * in_mask
        preds: Dict[int, float] = {}
        for i in mask_idx:
            preds[int(i)] = float(np.dot(W[i, :], x_masked) + b[i])
        err_sum = 0.0
        count = 0
        for j, obs_val in obs_items:
            jj = int(j)
            if jj < 0 or jj >= inv.shape[0]:
                continue
            ii = int(inv[jj])
            if ii in preds:
                err_sum += abs(float(obs_val) - preds[ii])
                count += 1
        if count == 0:
            continue
        mean_err = err_sum / float(count)
        if mean_err < best_err:
            best_err = mean_err
            best = m
            best_count = count
        elif mean_err == best_err and count > best_count:
            best = m
            best_count = count
    return best
