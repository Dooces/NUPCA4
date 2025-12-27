"""Rendering and visualization helpers for the harness."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import numpy as np


def _grid_transform(x: int, y: int, side: int, transform: str | None) -> tuple[int, int]:
    if side <= 0 or not transform:
        return x, y
    max_idx = side - 1
    if transform == "rotate_cw":
        return max_idx - y, x
    if transform == "rotate_ccw":
        return y, max_idx - x
    if transform == "mirror_x":
        return max_idx - x, y
    if transform == "mirror_y":
        return x, max_idx - y
    return x, y


ROTATE_COMMANDS = {"rotate_cw", "rotate_ccw", "mirror_x", "mirror_y"}
PUSH_OFFSETS = {
    "push_up": (0, -1),
    "push_down": (0, 1),
    "push_left": (-1, 0),
    "push_right": (1, 0),
}
PULL_OFFSETS = {
    "pull_up": (0, 1),
    "pull_down": (0, -1),
    "pull_left": (1, 0),
    "pull_right": (-1, 0),
}
COMMAND_OFFSETS = {**PUSH_OFFSETS, **PULL_OFFSETS}


def _mass_per_cell(
    vec: np.ndarray,
    *,
    side: int,
    n_colors: int,
    n_shapes: int,
    base_dim: int,
) -> List[float]:
    grid_cells = int(side) * int(side)
    masses: List[float] = [0.0] * grid_cells
    if grid_cells <= 0 or base_dim <= 0:
        return masses
    arr = np.asarray(vec[:base_dim], dtype=float).reshape(-1)
    cell_size = max(1, int(math.ceil(float(base_dim) / max(1, grid_cells))))
    idx = 0
    for cell in range(grid_cells):
        end = min(idx + cell_size, arr.size)
        if end > idx:
            masses[cell] = float(np.sum(np.abs(arr[idx:end])))
        idx = end
    return masses


def _mass_grid(
    vec: np.ndarray,
    *,
    side: int,
    base_dim: int,
    n_colors: int,
    n_shapes: int,
) -> List[List[int]]:
    """Quantize per-cell activation mass into the [0,9] range for visualization."""
    grid_cells = int(side) * int(side)
    if grid_cells <= 0:
        return [[0 for _ in range(int(side))] for _ in range(int(side))]
    masses = _mass_per_cell(vec, side=side, n_colors=n_colors, n_shapes=n_shapes, base_dim=base_dim)
    max_mass = max(masses) if masses else 0.0
    grid: List[List[int]] = []
    for y in range(int(side)):
        row: List[int] = []
        for x in range(int(side)):
            idx = y * int(side) + x
            value = masses[idx] if idx < len(masses) else 0.0
            if value <= 0.0 or max_mass <= 0.0:
                row.append(0)
            else:
                normalized = float(value) / float(max_mass)
                quant = max(1, min(9, int(round(normalized * 9))))
                row.append(quant)
        grid.append(row)
    return grid


def _mass_center(
    vec: np.ndarray,
    *,
    side: int,
    base_dim: int,
    n_colors: int,
    n_shapes: int,
) -> tuple[int, int]:
    masses = _mass_per_cell(vec, side=side, n_colors=n_colors, n_shapes=n_shapes, base_dim=base_dim)
    if not masses:
        return (0, 0)
    max_idx = int(np.argmax(masses))
    return (max_idx % int(side), max_idx // int(side))


def _real_grid(vec: np.ndarray, *, side: int, channels: int) -> List[List[str]]:
    """Return occupancy grid showing which cells have nonzero activity."""
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


def _env_full_grid(vec: np.ndarray, *, side: int) -> List[List[str]]:
    """Return integer grid of the environment state ignoring channels."""
    grid_cells = int(side) * int(side)
    if grid_cells <= 0:
        return [[".." for _ in range(int(side))] for _ in range(int(side))]
    arr = np.asarray(vec, dtype=float).reshape(-1)
    if arr.size < grid_cells:
        pad = np.zeros(grid_cells - arr.size, dtype=float)
        arr = np.concatenate([arr, pad])
    else:
        arr = arr[:grid_cells]
    grid: List[List[str]] = []
    for y in range(int(side)):
        row: List[str] = []
        for x in range(int(side)):
            val = float(arr[y * int(side) + x])
            row.append(str(int(round(val))) if abs(val) >= 0.5 else "..")
        grid.append(row)
    return grid


def _occupancy_grid(
    vec: np.ndarray,
    *,
    side: int,
    base_dim: int,
    n_colors: int = 0,
    n_shapes: int = 0,
) -> List[List[str]]:
    """Return a binary occupancy grid derived from the true environment state."""
    grid_cells = int(side) * int(side)
    if grid_cells <= 0:
        return [[".." for _ in range(int(side))] for _ in range(int(side))]
    occ = _occupancy_array(
        vec,
        side=side,
        base_dim=base_dim,
        n_colors=n_colors,
        n_shapes=n_shapes,
    )
    grid: List[List[str]] = []
    for y in range(int(side)):
        row: List[str] = []
        for x in range(int(side)):
            idx = y * int(side) + x
            row.append("##" if occ[idx] else "..")
        grid.append(row)
    return grid


def _occupancy_array(
    vec: np.ndarray,
    *,
    side: int,
    base_dim: int,
    n_colors: int = 0,
    n_shapes: int = 0,
) -> np.ndarray:
    """Return a flat boolean occupancy mask for `side×side` cells."""
    grid_cells = int(side) * int(side)
    arr = np.asarray(vec[:base_dim], dtype=float).reshape(-1)
    if arr.size < base_dim:
        pad = np.zeros(int(base_dim) - arr.size, dtype=float)
        arr = np.concatenate([arr, pad])
    tol = 1e-6
    if grid_cells <= 0:
        return np.zeros(0, dtype=bool)
    mask = np.zeros(grid_cells, dtype=bool)
    if n_colors > 0 or n_shapes > 0:
        color_chunk = grid_cells * n_colors
        color_section = arr[:color_chunk]
        if color_section.size < color_chunk:
            color_section = np.pad(
                color_section, (0, color_chunk - color_section.size), mode="constant"
            )
        shape_chunk = grid_cells * n_shapes
        shape_section = arr[color_chunk : color_chunk + shape_chunk]
        if shape_section.size < shape_chunk:
            shape_section = np.pad(
                shape_section, (0, shape_chunk - shape_section.size), mode="constant"
            )
        masses = np.zeros(grid_cells, dtype=float)
        for cell in range(grid_cells):
            cell_mass = 0.0
            if n_colors > 0:
                start = cell * n_colors
                end = start + n_colors
                cell_mass += float(np.sum(np.abs(color_section[start:end])))
            if n_shapes > 0:
                start = cell * n_shapes
                end = start + n_shapes
                cell_mass += float(np.sum(np.abs(shape_section[start:end])))
            masses[cell] = cell_mass
        max_mass = float(np.max(masses)) if masses.size else 0.0
        threshold = max(tol, max_mass * 0.25)
        mask = masses >= threshold
        if not mask.any() and masses.size:
            mask[int(np.argmax(masses))] = True
        return mask
    cell_size = max(1, int(math.ceil(float(base_dim) / max(1, grid_cells))))
    idx = 0
    masses = np.zeros(grid_cells, dtype=float)
    for cell in range(grid_cells):
        end = min(idx + cell_size, arr.size)
        masses[cell] = float(np.sum(np.abs(arr[idx:end])))
        idx = end
    max_mass = float(np.max(masses)) if masses.size else 0.0
    if max_mass <= tol:
        return mask
    threshold = max(tol, max_mass * 0.25)
    mask = masses >= threshold
    return mask


def _obs_mask_grid(
    obs_dims: set[int],
    *,
    side: int,
    n_colors: int,
    n_shapes: int,
) -> List[List[str]]:
    grid_cells = int(side) * int(side)
    channels = int(n_colors) + int(n_shapes)
    if grid_cells <= 0 or channels <= 0:
        return [[".." for _ in range(int(side))] for _ in range(int(side))]
    grid: List[List[str]] = []
    for y in range(int(side)):
        row: List[str] = []
        for x in range(int(side)):
            cell = y * int(side) + x
            dims = []
            if n_colors > 0:
                dims.extend(range(cell * n_colors, cell * n_colors + n_colors))
            if n_shapes > 0:
                base_shape = grid_cells * n_colors
                dims.extend(range(base_shape + cell * n_shapes, base_shape + cell * n_shapes + n_shapes))
            observed = any(dim in obs_dims for dim in dims)
            row.append("##" if observed else "..")
        grid.append(row)
    return grid


def _check_block_partition(blocks: List[List[int]], D_agent: int) -> Tuple[bool, str]:
    dims_seen: set[int] = set()
    for block_id, dims in enumerate(blocks):
        for dim in dims:
            if dim in dims_seen:
                return False, f"dim {dim} appears in multiple blocks (first seen before block {block_id})"
            dims_seen.add(int(dim))
    missing = set(range(int(D_agent))) - dims_seen
    extra = dims_seen - set(range(int(D_agent)))
    if missing:
        missing_head = sorted(missing)[:8]
        return False, f"missing dims (head)={missing_head} count={len(missing)}"
    if extra:
        extra_head = sorted(extra)[:8]
        return False, f"extra dims (head)={extra_head} count={len(extra)}"
    return True, "ok"


def _print_visualization(
    step_idx: int,
    true_delta: tuple[int, int],
    transport_delta: tuple[int, int],
    env_vec: np.ndarray,
    obs_dims: set[int],
    prev_vec: np.ndarray,
    pred_vec: np.ndarray,
    side: int,
    n_colors: int,
    n_shapes: int,
    base_dim: int,
    trace: Dict[str, Any],
    true_env_vec: np.ndarray | None = None,
    true_env_dim: int | None = None,
) -> None:
    env_source_vec = env_vec
    env_source_dim = base_dim
    if true_env_vec is not None and int(true_env_dim or 0) > 0:
        env_source_vec = true_env_vec
        env_source_dim = int(true_env_dim)

    env_pos = _mass_center(
        env_source_vec[:env_source_dim],
        side=side,
        base_dim=env_source_dim,
        n_colors=n_colors,
        n_shapes=n_shapes,
    )
    prev_pos = _mass_center(prev_vec[:base_dim], side=side, base_dim=base_dim, n_colors=n_colors, n_shapes=n_shapes)
    pred_pos = _mass_center(pred_vec[:base_dim], side=side, base_dim=base_dim, n_colors=n_colors, n_shapes=n_shapes)
    grid_kwargs = dict(
        side=side,
        base_dim=base_dim,
        n_colors=n_colors,
        n_shapes=n_shapes,
    )
    env_grid = _occupancy_grid(
        env_source_vec[:env_source_dim],
        side=side,
        base_dim=env_source_dim,
        n_colors=n_colors,
        n_shapes=n_shapes,
    )
    prev_grid = _occupancy_grid(prev_vec[:base_dim], **grid_kwargs)
    pred_grid = _occupancy_grid(pred_vec[:base_dim], **grid_kwargs)
    obs_grid = _obs_mask_grid(obs_dims, side=side, n_colors=n_colors, n_shapes=n_shapes)
    if len(obs_grid) != int(side) or any(len(row) != int(side) for row in obs_grid):
        raise AssertionError(
            f"Observation mask grid malformed: expected {side}x{side}, got "
            f"{len(obs_grid)}x{max((len(r) for r in obs_grid), default=0)}"
        )
    env_occ = _occupancy_array(
        env_source_vec[:env_source_dim],
        side=side,
        base_dim=env_source_dim,
        n_colors=n_colors,
        n_shapes=n_shapes,
    )
    pred_occ = _occupancy_array(
        pred_vec[:base_dim],
        side=side,
        base_dim=base_dim,
        n_colors=n_colors,
        n_shapes=n_shapes,
    )
    diff_grid: List[List[str]] = []
    for y in range(int(side)):
        diff_row: List[str] = []
        for x in range(int(side)):
            idx = y * int(side) + x
            diff_row.append(".." if env_occ[idx] == pred_occ[idx] else "!!")
        diff_grid.append(diff_row)

    mae_pre = trace.get("transport_mae_pre", float("nan"))
    mae_post = trace.get("transport_mae_post", float("nan"))
    drop = mae_pre - mae_post if not math.isnan(mae_pre) and not math.isnan(mae_post) else float("nan")
    cand = trace.get("permit_param_info", {}).get("candidate_count", 0)
    print(
        f"[VISUAL step={step_idx}] true_delta={true_delta} transport_delta={transport_delta} "
        f"env_pos={env_pos} prev_pos={prev_pos} pred_pos={pred_pos} "
        f"mae_drop={drop if not math.isnan(drop) else 'nan'} candidates={cand}"
    )
    for label, grid in (
        ("ENV ", env_grid),
        ("OBS ", obs_grid),
        ("PREV", prev_grid),
        ("PRED", pred_grid),
        ("DIFF", diff_grid),
    ):
        for row_idx, row in enumerate(grid):
            prefix = label if row_idx == 0 else "    "
            print(f"{prefix} " + " ".join(row))

    mass_grids = [
        (
            "EMASS",
            _mass_grid(
                env_source_vec[:env_source_dim],
                side=side,
                base_dim=env_source_dim,
                n_colors=n_colors,
                n_shapes=n_shapes,
            ),
        ),
        (
            "PMASS",
            _mass_grid(
                prev_vec[:base_dim],
                side=side,
                base_dim=base_dim,
                n_colors=n_colors,
                n_shapes=n_shapes,
            ),
        ),
        (
            "DMASS",
            _mass_grid(
                pred_vec[:base_dim],
                side=side,
                base_dim=base_dim,
                n_colors=n_colors,
                n_shapes=n_shapes,
            ),
        ),
    ]
    for label, grid in mass_grids:
        for row_idx, row in enumerate(grid):
            prefix = label if row_idx == 0 else "     "
            formatted = " ".join(".." if val == 0 else f"{val}{val}" for val in row)
            print(f"{prefix} " + formatted)
    real_grid = _real_grid(
        env_source_vec[:env_source_dim],
        side=side,
        channels=max(1, n_colors + n_shapes),
    )
    for row_idx, row in enumerate(real_grid):
        prefix = "REAL" if row_idx == 0 else "    "
        print(f"{prefix} " + " ".join(row))
    full_grid = _env_full_grid(env_source_vec[:env_source_dim], side=side)
    for row_idx, row in enumerate(full_grid):
        prefix = "FULL" if row_idx == 0 else "    "
        print(f"{prefix} " + " ".join(row))
    true_dim = (
        int(true_env_dim)
        if true_env_dim is not None
        else (int(true_env_vec.size) if true_env_vec is not None else int(base_dim))
    )
    if true_env_vec is not None and true_dim > 0:
        occ_grid = _occupancy_grid(
            true_env_vec,
            side=side,
            base_dim=true_dim,
            n_colors=n_colors,
            n_shapes=n_shapes,
        )
        for row_idx, row in enumerate(occ_grid):
            prefix = "OCCU" if row_idx == 0 else "    "
            print(f"{prefix} " + " ".join(row))
        indices = np.where(np.abs(true_env_vec[:true_dim]) > 1e-6)[0]
        if indices.size:
            coords = [(int(idx % side), int(idx // side)) for idx in indices]
            xs = [x for x, _ in coords]
            ys = [y for _, y in coords]
            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)
            preview = coords if len(coords) <= 8 else coords[:8]
            more = "" if len(coords) <= 8 else " ..."
            print(
                f"SQR  size={len(coords)} bbox=x[{xmin},{xmax}] y[{ymin},{ymax}] "
                f"cells={preview}{more}"
            )


def _block_grid_bounds(
    block_dims: List[int],
    *,
    side: int,
    channels: int,
    base_dim: int,
) -> tuple[int, int, int, int] | None:
    if not block_dims or side <= 0 or channels <= 0 or base_dim <= 0:
        return None
    grid_cells = int(side) * int(side)
    max_cell = min(grid_cells - 1, (int(base_dim) - 1) // int(channels)) if grid_cells > 0 else -1
    cells = []
    for dim in block_dims:
        if 0 <= int(dim) < int(base_dim):
            cell = int(dim) // int(channels)
            if 0 <= cell <= max_cell:
                cells.append(cell)
    if not cells:
        return None
    rows = [cell // int(side) for cell in cells]
    cols = [cell % int(side) for cell in cells]
    return min(rows), max(rows), min(cols), max(cols)
