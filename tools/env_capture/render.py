"""ASCII rendering helpers for environment screen capture."""

from __future__ import annotations

from typing import List

import numpy as np


def real_grid(vec: np.ndarray, *, side: int, channels: int) -> List[List[str]]:
    grid_cells = int(side) * int(side)
    if grid_cells <= 0 or channels <= 0:
        return [[".." for _ in range(int(side))] for _ in range(int(side))]
    length = grid_cells * channels
    arr = np.asarray(vec[:length], dtype=float).reshape(-1)
    if arr.size < length:
        pad = np.zeros(length - arr.size, dtype=float)
        arr = np.concatenate([arr, pad])
    grid: List[List[str]] = []
    for y in range(int(side)):
        row: List[str] = []
        for x in range(int(side)):
            idx = (y * int(side) + x) * channels
            slice_vals = arr[idx : idx + channels]
            occupied = bool(np.any(np.abs(slice_vals) > 1e-6))
            row.append("##" if occupied else "..")
        grid.append(row)
    return grid


def render_env(
    *,
    step_idx: int,
    env_vec: np.ndarray,
    side: int,
    channels: int,
) -> List[str]:
    lines = [f"[ENV_CAPTURE step={step_idx}]"]
    grid = real_grid(env_vec, side=side, channels=channels)
    for row_idx, row in enumerate(grid):
        prefix = "ENV" if row_idx == 0 else "   "
        lines.append(f"{prefix} " + " ".join(row))
    vec_str = " ".join(f"{val:.6f}" for val in np.asarray(env_vec, dtype=float).reshape(-1).tolist())
    lines.append(f"VEC {vec_str}")
    return lines
