"""Coarse/fine stream helpers (Down/Up/Transport) for Stage 2."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ..config import AgentConfig


def _grid_dims(cfg: AgentConfig) -> Tuple[int, int, int]:
    grid_w = int(getattr(cfg, "grid_width", 0))
    grid_h = int(getattr(cfg, "grid_height", 0))
    channels = int(getattr(cfg, "grid_channels", 0))
    return grid_w, grid_h, channels


def periph_block_size(cfg: AgentConfig) -> int:
    D = int(getattr(cfg, "D", 0))
    B = max(1, int(getattr(cfg, "B", 0)))
    block_size = D // B if B > 0 else 0
    periph_blocks = max(0, int(getattr(cfg, "periph_blocks", 0)))
    return periph_blocks * block_size


def coarse_bin_count(cfg: AgentConfig) -> int:
    bins = max(0, int(getattr(cfg, "periph_bins", 0)))
    blocks = max(0, int(getattr(cfg, "periph_blocks", 0)))
    channels = max(1, int(getattr(cfg, "periph_channels", 1)))
    if bins <= 0 or blocks <= 0:
        return 0
    return int(blocks * bins * bins * channels)


def _coarse_slice(cfg: AgentConfig) -> Tuple[int, int]:
    periph_size = periph_block_size(cfg)
    if periph_size <= 0:
        return (int(getattr(cfg, "D", 0)), 0)
    start = max(0, int(getattr(cfg, "D", 0)) - periph_size)
    count = min(periph_size, coarse_bin_count(cfg))
    return start, count


def extract_coarse(state_vec: np.ndarray, cfg: AgentConfig) -> np.ndarray:
    start, count = _coarse_slice(cfg)
    if count <= 0 or start >= state_vec.size:
        return np.zeros(0, dtype=float)
    slice_vec = state_vec[start : min(state_vec.size, start + count)]
    if slice_vec.shape[0] < count:
        pad = np.zeros(count - slice_vec.shape[0], dtype=float)
        slice_vec = np.concatenate([slice_vec, pad])
    return np.asarray(slice_vec, dtype=float)


def up_project(state_vec: np.ndarray, cfg: AgentConfig) -> np.ndarray:
    base_dim = int(getattr(cfg, "D", 0)) - periph_block_size(cfg)
    base_dim = max(0, min(base_dim, state_vec.size))
    bins = max(0, int(getattr(cfg, "periph_bins", 0)))
    if bins <= 0 or base_dim <= 0:
        return np.zeros(0, dtype=float)

    grid_w, grid_h, channels = _grid_dims(cfg)
    if grid_w <= 0 or grid_h <= 0 or channels <= 0:
        return np.zeros(0, dtype=float)
    cell_count = grid_w * grid_h
    max_cells = min(cell_count, base_dim // channels) if channels > 0 else 0
    if max_cells <= 0:
        return np.zeros(0, dtype=float)

    data = np.asarray(state_vec[: max_cells * channels], dtype=float)
    cells = data.reshape(max_cells, channels)
    mass = np.sum(cells, axis=1)

    bin_counts = np.zeros(bins * bins, dtype=int)
    bin_mass = np.zeros(bins * bins, dtype=float)
    tile_w = max(1, grid_w // bins)
    tile_h = max(1, grid_h // bins)
    for cell_idx in range(max_cells):
        y = cell_idx // grid_w
        x = cell_idx % grid_w
        bin_x = min(bins - 1, x // tile_w)
        bin_y = min(bins - 1, y // tile_h)
        idx = bin_y * bins + bin_x
        bin_mass[idx] += float(mass[cell_idx])
        bin_counts[idx] += 1

    coarse = np.zeros(bins * bins, dtype=float)
    for idx in range(bins * bins):
        denom = max(1, bin_counts[idx])
        coarse[idx] = bin_mass[idx] / float(denom)
    return coarse


def down_project(coarse: np.ndarray, cfg: AgentConfig) -> np.ndarray:
    bins = max(0, int(getattr(cfg, "periph_bins", 0)))
    if bins <= 0 or coarse.size <= 0:
        return np.zeros(0, dtype=float)

    grid_w, grid_h, channels = _grid_dims(cfg)
    if grid_w <= 0 or grid_h <= 0 or channels <= 0:
        return np.zeros(0, dtype=float)
    periph_len = bins * bins
    values = coarse.reshape(min(periph_len, coarse.size))
    base_dim = int(getattr(cfg, "D", 0)) - periph_block_size(cfg)
    if base_dim <= 0:
        return np.zeros(0, dtype=float)

    cell_count = grid_w * grid_h
    data = np.zeros(cell_count * channels, dtype=float)
    tile_w = max(1, grid_w // bins)
    tile_h = max(1, grid_h // bins)
    value_map = np.zeros(periph_len, dtype=float)
    value_map[: values.size] = values

    for cell_idx in range(cell_count):
        y = cell_idx // grid_w
        x = cell_idx % grid_w
        bin_x = min(bins - 1, x // tile_w)
        bin_y = min(bins - 1, y // tile_h)
        idx = bin_y * bins + bin_x
        val = value_map[idx]
        start = cell_idx * channels
        end = start + channels
        data[start:end] = float(val)

    result = np.zeros(base_dim, dtype=float)
    length = min(base_dim, data.size)
    result[:length] = data[:length]
    return result


def _coarse_grid(vec: np.ndarray, bins: int) -> np.ndarray:
    if bins <= 0:
        return np.zeros((0, 0), dtype=float)
    size = bins * bins
    arr = np.asarray(vec, dtype=float).reshape(-1)
    if arr.size < size:
        pad = np.zeros(size - arr.size, dtype=float)
        arr = np.concatenate([arr, pad])
    else:
        arr = arr[:size]
    return arr.reshape(bins, bins)


def _normalize_grid(grid: np.ndarray) -> np.ndarray:
    total = float(np.sum(grid))
    if total <= 0.0:
        return grid
    return grid / total


def _best_coarse_shift(prev_grid: np.ndarray, curr_grid: np.ndarray) -> Tuple[int, int]:
    bins = prev_grid.shape[0]
    if bins == 0 or curr_grid.shape != prev_grid.shape:
        return (0, 0)
    norm_prev = _normalize_grid(prev_grid)
    norm_curr = _normalize_grid(curr_grid)
    best_err = float("inf")
    best_shift = (0, 0)
    best_mag = float("inf")
    max_offset = bins - 1
    for dy in range(-max_offset, max_offset + 1):
        rolled_y = np.roll(norm_prev, shift=dy, axis=0)
        for dx in range(-max_offset, max_offset + 1):
            rolled = np.roll(rolled_y, shift=dx, axis=1)
            err = float(np.linalg.norm(norm_curr - rolled))
            magnitude = float(abs(dx) + abs(dy))
            if err < best_err or (err == best_err and magnitude < best_mag):
                best_err = err
                best_mag = magnitude
                best_shift = (dx, dy)
    return best_shift


def compute_transport_shift(prev: np.ndarray, curr: np.ndarray, cfg: AgentConfig) -> Tuple[int, int]:
    if prev.size == 0 or curr.size == 0:
        return (0, 0)
    bins = max(0, int(getattr(cfg, "periph_bins", 0)))
    if bins <= 0:
        return (0, 0)
    prev_grid = _coarse_grid(prev, bins)
    curr_grid = _coarse_grid(curr, bins)
    return _best_coarse_shift(prev_grid, curr_grid)


def _rotate_grid_segment(fine_vec: np.ndarray, rotation: int, cfg: AgentConfig) -> np.ndarray:
    rotation = int(rotation) % 4
    if rotation == 0:
        return fine_vec.copy()

    grid_w, grid_h, channels = _grid_dims(cfg)
    base_dim = int(getattr(cfg, "D", 0)) - periph_block_size(cfg)
    if grid_w <= 0 or grid_h <= 0 or channels <= 0 or base_dim <= 0:
        return fine_vec.copy()

    cell_count = grid_w * grid_h
    data = np.zeros(cell_count * channels, dtype=float)
    length = min(base_dim, data.size)
    data[:length] = fine_vec[:length]
    grid = data.reshape(grid_h, grid_w, channels)
    rotated = np.rot90(grid, k=rotation, axes=(0, 1))
    flat = rotated.reshape(-1)

    result = fine_vec.copy()
    fill_len = min(base_dim, flat.size)
    result[:fill_len] = flat[:fill_len]
    return result


def apply_transport(
    fine_vec: np.ndarray,
    shift: Tuple[int, int],
    cfg: AgentConfig,
    *,
    rotation: int = 0,
) -> np.ndarray:
    """Apply translation (and optional rotation) to the fine-grid part of the state."""
    dx, dy = shift
    rot = int(rotation) % 4
    rotated_vec = _rotate_grid_segment(fine_vec, rot, cfg) if rot else fine_vec.copy()
    if dx == 0 and dy == 0 and rot == 0:
        return rotated_vec.copy()
    if dx == 0 and dy == 0:
        # Only rotation was applied, no translation required after rotation.
        return rotated_vec
    grid_w, grid_h, channels = _grid_dims(cfg)
    base_dim = int(getattr(cfg, "D", 0)) - periph_block_size(cfg)
    if grid_w <= 0 or grid_h <= 0 or channels <= 0 or base_dim <= 0:
        return fine_vec.copy()
    cell_count = grid_w * grid_h
    data_size = min(cell_count * channels, base_dim)
    data = np.zeros(cell_count * channels, dtype=float)
    data[:data_size] = rotated_vec[:data_size]
    data = data.reshape(cell_count, channels)
    data = data.reshape(grid_h, grid_w, channels)
    dx_cells = int(dx)
    dy_cells = int(dy)
    if dy_cells != 0:
        data = np.roll(data, shift=dy_cells, axis=0)
    if dx_cells != 0:
        data = np.roll(data, shift=dx_cells, axis=1)
    out = fine_vec.copy()
    flat = data.reshape(-1)
    length = min(flat.size, base_dim)
    out[:length] = flat[:length]
    return out


def grid_cell_mass(state_vec: np.ndarray, cfg: AgentConfig) -> np.ndarray:
    """Compute per-cell aggregate mass from the fine part of the state vector."""
    grid_w, grid_h, channels = _grid_dims(cfg)
    if grid_w <= 0 or grid_h <= 0:
        return np.zeros(0, dtype=float)
    base_dim = int(getattr(cfg, "D", 0)) - periph_block_size(cfg)
    base_dim = max(0, base_dim)
    if base_dim <= 0:
        return np.zeros(grid_w * grid_h, dtype=float)

    cell_count = grid_w * grid_h
    vec = np.asarray(state_vec, dtype=float).reshape(-1)

    def _slice_chunk(data: np.ndarray, start: int, length: int) -> np.ndarray:
        if length <= 0:
            return np.zeros(0, dtype=float)
        chunk = np.zeros(length, dtype=float)
        if start < data.size:
            copy_len = min(length, data.size - start)
            if copy_len > 0:
                chunk[:copy_len] = data[start : start + copy_len]
        return chunk

    mass = np.zeros(cell_count, dtype=float)
    offset = 0
    color_channels = max(0, int(getattr(cfg, "grid_color_channels", 0)))
    shape_channels = max(0, int(getattr(cfg, "grid_shape_channels", 0)))
    if color_channels > 0:
        length = cell_count * color_channels
        chunk = _slice_chunk(vec, offset, length)
        if chunk.size == length:
            chunk = chunk.reshape(cell_count, color_channels)
            mass += np.sum(chunk, axis=1)
        offset += length
    if shape_channels > 0:
        length = cell_count * shape_channels
        chunk = _slice_chunk(vec, offset, length)
        if chunk.size == length:
            chunk = chunk.reshape(cell_count, shape_channels)
            mass += np.sum(chunk, axis=1)
        offset += length
    if color_channels + shape_channels > 0:
        return mass

    channels = max(1, channels)
    max_cells = min(cell_count, base_dim // channels if channels > 0 else 0)
    if max_cells <= 0:
        return np.zeros(cell_count, dtype=float)
    data_len = max_cells * channels
    data = _slice_chunk(vec, 0, data_len)
    if data.size < data_len:
        padded = np.zeros(data_len, dtype=float)
        padded[: data.size] = data
        data = padded
    data = data.reshape(max_cells, channels)
    mass_slice = np.sum(data, axis=1)
    if max_cells == cell_count:
        return mass_slice
    result = np.zeros(cell_count, dtype=float)
    result[:max_cells] = mass_slice
    return result


def _search_grid_shift(prev_grid: np.ndarray, curr_grid: np.ndarray, radius: int) -> Tuple[int, int]:
    if prev_grid.size == 0 or curr_grid.size == 0 or prev_grid.shape != curr_grid.shape:
        return (0, 0)
    best_err = float("inf")
    best_mag = float("inf")
    best_shift: Tuple[int, int] = (0, 0)
    max_offset = max(0, min(radius, prev_grid.shape[0] - 1))
    for dy in range(-max_offset, max_offset + 1):
        rolled_y = np.roll(prev_grid, shift=dy, axis=0)
        for dx in range(-max_offset, max_offset + 1):
            rolled = np.roll(rolled_y, shift=dx, axis=1)
            err = float(np.linalg.norm(curr_grid - rolled))
            magnitude = float(abs(dx) + abs(dy))
            if err < best_err or (err == best_err and magnitude < best_mag):
                best_err = err
                best_mag = magnitude
                best_shift = (dx, dy)
    return best_shift


def _mask_shift(prev_grid: np.ndarray, curr_grid: np.ndarray, radius: int, threshold: float = 0.5) -> Tuple[Tuple[int, int], int]:
    if prev_grid.size == 0 or curr_grid.size == 0 or prev_grid.shape != curr_grid.shape:
        return (0, 0), 0
    prev_max = float(np.max(prev_grid)) if prev_grid.size else 0.0
    curr_max = float(np.max(curr_grid)) if curr_grid.size else 0.0
    if prev_max <= 0.0 or curr_max <= 0.0:
        return (0, 0), 0
    prev_mask = prev_grid >= (threshold * prev_max)
    curr_mask = curr_grid >= (threshold * curr_max)
    best_overlap = -1
    best_mag = float("inf")
    best_shift: Tuple[int, int] = (0, 0)
    max_offset = max(0, min(radius, prev_grid.shape[0] - 1))
    for dy in range(-max_offset, max_offset + 1):
        rolled_y = np.roll(prev_mask, shift=dy, axis=0)
        for dx in range(-max_offset, max_offset + 1):
            rolled = np.roll(rolled_y, shift=dx, axis=1)
            overlap = int(np.count_nonzero(curr_mask & rolled))
            magnitude = float(abs(dx) + abs(dy))
            if overlap > best_overlap or (overlap == best_overlap and magnitude < best_mag):
                best_overlap = overlap
                best_mag = magnitude
                best_shift = (dx, dy)
    return best_shift, best_overlap


def compute_grid_shift(
    prev: np.ndarray,
    curr: np.ndarray,
    cfg: AgentConfig,
    *,
    radius: int | None = None,
) -> Tuple[int, int]:
    """Align two grid mass vectors using bounded translation search."""
    if prev.size == 0 or curr.size == 0:
        return (0, 0)
    grid_w, grid_h, _channels = _grid_dims(cfg)
    if grid_w <= 0 or grid_h <= 0:
        return (0, 0)
    expected_size = grid_w * grid_h
    if prev.size != expected_size or curr.size != expected_size:
        prev = np.resize(prev, (expected_size,))
        curr = np.resize(curr, (expected_size,))
    grid_prev = prev.reshape(grid_h, grid_w)
    grid_curr = curr.reshape(grid_h, grid_w)
    max_radius = radius if radius is not None else int(getattr(cfg, "transport_search_radius", 1))
    max_radius = max(0, max_radius)
    shift, overlap = _mask_shift(grid_prev, grid_curr, max_radius)
    if overlap <= 0:
        shift = _search_grid_shift(grid_prev, grid_curr, max_radius)
    shift_x = max(-max_radius, min(max_radius, int(shift[0])))
    shift_y = max(-max_radius, min(max_radius, int(shift[1])))
    return (shift_x, shift_y)
