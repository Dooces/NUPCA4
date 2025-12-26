#!/usr/bin/env python3
"""
NUPCA3 axioms harness (uses in-repo implementation).

This harness exercises the real nupca3 pipeline and enforces A16 foveation
discipline by filtering observations to the greedy_cov-selected blocks *before*
each agent step.

It intentionally avoids re-implementing learning logic. Any improvements or
failures in output reflect the current codebase, not a parallel harness.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict, replace
from typing import Dict, List, Tuple, Any
import math

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


def _mass_per_cell(vec: np.ndarray, *, side: int, n_colors: int, n_shapes: int, base_dim: int) -> List[float]:
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


def _mass_grid(vec: np.ndarray, *, side: int, base_dim: int, n_colors: int, n_shapes: int) -> List[List[int]]:
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


def _mass_center(vec: np.ndarray, *, side: int, base_dim: int, n_colors: int, n_shapes: int) -> tuple[int, int]:
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
            mask[np.argmax(masses)] = True
        return mask
    if arr.size < grid_cells:
        padded = np.pad(arr, (0, grid_cells - arr.size), mode="constant")
    else:
        padded = arr[:grid_cells]
    mask = np.abs(padded) > tol
    return np.asarray(mask, dtype=bool)


def _obs_mask_grid(obs_dims: set[int], *, side: int, n_colors: int, n_shapes: int) -> List[List[str]]:
    grid_cells = int(side) * int(side)
    grid: List[List[str]] = []
    for y in range(int(side)):
        row: List[str] = []
        for x in range(int(side)):
            cell = y * int(side) + x
            dims = []
            dims.extend(range(cell * n_colors, cell * n_colors + n_colors))
            base_shape = grid_cells * n_colors
            dims.extend(range(base_shape + cell * n_shapes, base_shape + cell * n_shapes + n_shapes))
            observed = any(dim in obs_dims for dim in dims)
            row.append("##" if observed else "..")
        grid.append(row)
    return grid


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


REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from nupca3.agent import NUPCA3Agent
from nupca3.config import AgentConfig
from nupca3.geometry.fovea import make_observation_set, select_fovea, update_fovea_routing_scores
from nupca3.step_pipeline import (
    _enforce_motion_probe_blocks,
    _enforce_peripheral_blocks,
    _peripheral_block_ids,
    _select_motion_probe_blocks,
    step_pipeline,
)
from nupca3.types import EnvObs, Action


class LinearARWorld:
    """Simple AR(1) latent state with Gaussian noise."""

    def __init__(self, D: int, seed: int, rho: float = 0.9, noise_std: float = 0.05):
        self.D = int(D)
        self.rho = float(rho)
        self.noise_std = float(noise_std)
        self.rng = np.random.default_rng(int(seed))
        self.x = self.rng.normal(size=self.D).astype(float)

    def reset(self) -> np.ndarray:
        self.x = self.rng.normal(size=self.D).astype(float)
        return self.x.copy()

    def step(self) -> np.ndarray:
        noise = self.rng.normal(scale=self.noise_std, size=self.D)
        self.x = self.rho * self.x + noise
        return self.x.copy()


class MovingColorShapeWorld:
    """Grid world with moving colored shapes encoded as per-cell color/shape channels."""

    def __init__(
        self,
        side: int,
        n_colors: int,
        n_shapes: int,
        seed: int,
        p_color_shift: float = 0.05,
        p_shape_shift: float = 0.05,
        periph_bins: int = 2,
        speed_range: tuple[int, int] = (1, 1),
        occlusion_prob: float = 0.0,
        occlusion_size_range: tuple[int, int] = (0, 0),
        object_size: int = 3,
    ):
        self.side = int(side)
        self.n_colors = int(n_colors)
        self.n_shapes = int(n_shapes)
        self.rng = np.random.default_rng(int(seed))
        self.p_color_shift = float(p_color_shift)
        self.p_shape_shift = float(p_shape_shift)
        self.speed_range = tuple(int(v) for v in speed_range)
        self.speed_min = max(1, min(self.speed_range))
        self.speed_max = max(self.speed_min, max(self.speed_range))
        self.occlusion_prob = float(occlusion_prob)
        self.occlusion_size_range = tuple(int(v) for v in occlusion_size_range)
        self.occlusion_min = max(0, min(self.occlusion_size_range))
        self.occlusion_max = max(self.occlusion_min, max(self.occlusion_size_range))
        self.periph_bins = int(periph_bins)

        self.x = 0
        self.y = 0
        self.color = 0
        self.shape = 0
        self._last_state: np.ndarray | None = None
        self.last_dx = 0
        self.last_dy = 0
        self.object_size = max(1, int(object_size))

    @property
    def D(self) -> int:
        return self.side * self.side * (self.n_colors + self.n_shapes)

    def reset(self) -> np.ndarray:
        self.x = int(self.rng.integers(self.side))
        self.y = int(self.rng.integers(self.side))
        self.color = int(self.rng.integers(self.n_colors))
        self.shape = int(self.rng.integers(self.n_shapes))
        self.last_dx = 0
        self.last_dy = 0
        return self._encode()

    def step(self) -> np.ndarray:
        dx, dy = int(self.rng.integers(-1, 2)), int(self.rng.integers(-1, 2))
        speed = int(self.rng.integers(self.speed_min, self.speed_max + 1))
        dx *= speed
        dy *= speed
        self.last_dx = dx
        self.last_dy = dy
        self.x = (self.x + dx) % self.side
        self.y = (self.y + dy) % self.side

        if self.rng.random() < self.p_color_shift:
            self.color = int(self.rng.integers(self.n_colors))
        if self.rng.random() < self.p_shape_shift:
            self.shape = int(self.rng.integers(self.n_shapes))

        return self._encode()

    def _encode(self) -> np.ndarray:
        vec = np.zeros(self.D, dtype=float)
        color_offset = 0
        shape_offset = self.side * self.side * self.n_colors
        half = self.object_size // 2
        x_start = self.x - half
        y_start = self.y - half
        for oy in range(self.object_size):
            for ox in range(self.object_size):
                xx = (x_start + ox) % self.side
                yy = (y_start + oy) % self.side
                cell = yy * self.side + xx
                if self.n_colors > 0:
                    idx = color_offset + cell * self.n_colors + self.color
                    vec[idx] = 1.0
                if self.n_shapes > 0:
                    idx = shape_offset + cell * self.n_shapes + self.shape
                    vec[idx] = 1.0
        self._last_state = vec.copy()
        return vec

    def default_control_target(self) -> str:
        return "shape"

    def get_object_position(self, target: str) -> tuple[int, int]:
        if target == "shape":
            return (self.x, self.y)
        return (self.x, self.y)

    def apply_control_command(self, command: str, target: str) -> bool:
        if target != "shape":
            return False
        if command in ROTATE_COMMANDS:
            self.x, self.y = _grid_transform(self.x, self.y, self.side, command)
            self._last_state = None
            return True
        if command in COMMAND_OFFSETS:
            dx, dy = COMMAND_OFFSETS[command]
            self.x = (self.x + dx) % self.side
            self.y = (self.y + dy) % self.side
            self._last_state = None
            return True
        return False

    def apply_transform(self, transform: str | None) -> None:
        if not transform:
            return
        self.x, self.y = _grid_transform(self.x, self.y, self.side, transform)

    def encode_peripheral(self) -> np.ndarray:
        """Coarse peripheral summary (encoded abstraction) over a low-res grid."""
        vec = getattr(self, "_last_state", None)
        if vec is None or vec.size == 0:
            vec = self._encode()
        return self._pooled_peripheral(vec)

    def _pooled_peripheral(self, state_vec: np.ndarray) -> np.ndarray:
        bins = max(1, int(self.periph_bins))
        periph = np.zeros(bins * bins, dtype=float)
        counts = np.zeros(bins * bins, dtype=int)
        total_cells = self.side * self.side
        vec = np.asarray(state_vec, dtype=float).reshape(-1)
        channels = self.n_colors + self.n_shapes
        if channels <= 0:
            channels = 1
        mass = np.zeros(total_cells, dtype=float)
        offset = 0
        if self.n_colors > 0:
            color_chunk = total_cells * self.n_colors
            color_vals = vec[offset : offset + color_chunk]
            color_vals = np.resize(color_vals, (color_chunk,))
            mass += np.sum(color_vals.reshape(total_cells, self.n_colors), axis=1)
            offset += color_chunk
        if self.n_shapes > 0:
            shape_chunk = total_cells * self.n_shapes
            shape_vals = vec[offset : offset + shape_chunk]
            shape_vals = np.resize(shape_vals, (shape_chunk,))
            mass += np.sum(shape_vals.reshape(total_cells, self.n_shapes), axis=1)
            offset += shape_chunk

        tile_w = max(1, self.side // bins)
        tile_h = max(1, self.side // bins)
        for cell in range(total_cells):
            y = cell // self.side
            x = cell % self.side
            bin_x = min(bins - 1, x // tile_w)
            bin_y = min(bins - 1, y // tile_h)
            idx = bin_y * bins + bin_x
            periph[idx] += float(mass[cell])
            counts[idx] += 1
        denom_channels = float(max(1, channels))
        for idx in range(bins * bins):
            denom = max(1, counts[idx]) * denom_channels
            periph[idx] /= denom
        return periph

    def last_move(self) -> Tuple[int, int]:
        return (int(self.last_dx), int(self.last_dy))

    def last_move(self) -> Tuple[int, int]:
        return (int(self.last_dx), int(self.last_dy))


class LinearSquareWorld:
    """Simple world with a moving square that alternates size in a fixed pattern."""

    def __init__(
        self,
        side: int,
        seed: int,
        square_small: int = 1,
        square_big: int = 2,
        pattern_period: int = 20,
        dx: int = 1,
        dy: int = 0,
        periph_bins: int = 2,
    ):
        self.side = int(side)
        self.rng = np.random.default_rng(int(seed))
        min_square = max(2, int(square_small))
        self.square_small = min_square
        self.square_big = max(min_square, max(2, int(square_big)))
        self.pattern_period = max(1, int(pattern_period))
        self.dx = int(dx)
        self.dy = int(dy)
        self.periph_bins = int(periph_bins)
        self.x = 0
        self.y = 0
        self.t = 0
        self._last_state: np.ndarray | None = None
        self.last_dx = 0
        self.last_dy = 0

    @property
    def D(self) -> int:
        return self.side * self.side

    def reset(self) -> np.ndarray:
        self.x = int(self.rng.integers(self.side))
        self.y = int(self.rng.integers(self.side))
        self.t = 0
        self.last_dx = 0
        self.last_dy = 0
        return self._encode()

    def step(self) -> np.ndarray:
        self.t += 1
        self.last_dx = self.dx
        self.last_dy = self.dy
        self.x = (self.x + self.dx) % self.side
        self.y = (self.y + self.dy) % self.side
        return self._encode()

    def _square_size(self) -> int:
        if (self.t // self.pattern_period) % 2 == 0:
            return self.square_small
        return self.square_big

    def _encode(self) -> np.ndarray:
        vec = np.zeros(self.D, dtype=float)
        size = self._square_size()
        half = size // 2
        for oy in range(-half, half + 1):
            for ox in range(-half, half + 1):
                xx = (self.x + ox) % self.side
                yy = (self.y + oy) % self.side
                vec[yy * self.side + xx] = 1.0
        return vec

    def encode_peripheral(self) -> np.ndarray:
        vec = getattr(self, "_last_state", None)
        if vec is None or vec.size == 0:
            vec = self._encode()
        return self._pooled_peripheral(vec)

    def _pooled_peripheral(self, state_vec: np.ndarray) -> np.ndarray:
        bins = max(1, int(self.periph_bins))
        periph = np.zeros(bins * bins, dtype=float)
        counts = np.zeros(bins * bins, dtype=int)
        total_cells = self.side * self.side
        vec = np.asarray(state_vec, dtype=float).reshape(-1)
        if vec.size < total_cells:
            vec = np.resize(vec, (total_cells,))
        tile_w = max(1, self.side // bins)
        tile_h = max(1, self.side // bins)
        for cell in range(total_cells):
            y = cell // self.side
            x = cell % self.side
            bin_x = min(bins - 1, x // tile_w)
            bin_y = min(bins - 1, y // tile_h)
            idx = bin_y * bins + bin_x
            periph[idx] += float(vec[cell])
            counts[idx] += 1
        for idx in range(bins * bins):
            denom = max(1, counts[idx])
            periph[idx] /= float(denom)
        return periph

    def last_move(self) -> Tuple[int, int]:
        return (int(self.last_dx), int(self.last_dy))


def _tile_to_dim(vec: np.ndarray, target_dim: int) -> np.ndarray:
    arr = np.asarray(vec, dtype=float).reshape(-1)
    if target_dim <= 0:
        return np.zeros(0, dtype=float)
    if arr.size == 0:
        return np.zeros(target_dim, dtype=float)
    repeats = int(math.ceil(float(target_dim) / float(arr.size)))
    tiled = np.tile(arr, repeats)[: target_dim]
    return tiled


def build_partial_obs(full_x: np.ndarray, obs_dims: List[int]) -> Dict[int, float]:
    return {int(k): float(full_x[int(k)]) for k in obs_dims}



def run_task(
    *,
    D: int,
    B: int,
    steps: int,
    seed: int,
    world: str,
    side: int,
    n_colors: int,
    n_shapes: int,
    square_small: int,
    square_big: int,
    pattern_period: int,
    dx: int,
    dy: int,
    p_color_shift: float,
    p_shape_shift: float,
    obs_budget: float,
    obs_cost: float,
    obs_budget_mode: str,
    obs_budget_min: float,
    obs_budget_max: float,
    coverage_debt_target: float,
    pred_only_start: int,
    pred_only_len: int,
    dense_world: bool,
    dense_sigma: float,
    fovea_residual_only: bool,
    binding_enabled: bool,
    binding_shift_radius: int,
    binding_rotations: bool,
    periph_blocks: int,
    periph_bins: int,
    object_size: int,
    alpha_cov: float,
    coverage_cap_G: int,
    fovea_residual_ema: float,
    fovea_use_age: bool,
    fovea_age_min_inc: float,
    fovea_age_resid_scale: float,
    fovea_age_resid_thresh: float,
    fovea_routing_weight: float,
    fovea_routing_ema: float,
    occlude_start: int,
    occlude_len: int,
    occlude_period: int,
    working_set_linger_steps: int,
    transport_span_blocks: int,
    min_fovea_blocks: int,
    train_active_only: bool,
    train_active_threshold: float,
    train_weight_by_value: bool,
    train_value_power: float,
    lr_expert: float,
    sigma_ema: float,
    theta_learn: float,
    theta_ar_rest: float,
    nu_max: float,
    xi_max: float,
    stability_window: int,
    theta_ar: float,
    kappa_ar: float,
    scan_steps: int,
    warm_steps: int,
    warm_fovea_blocks: int,
    log_every: int,
    n_max: int,
    l_work_max: float,
    force_block_anchors: bool,
    diagnose_coverage: bool,
    coverage_log_every: int,
    rest_test_period: int,
    rest_test_length: int,
    periph_test: bool,
    transport_test: bool,
    transport_force_true_delta: bool,
    visualize_steps: int = 0,
) -> Dict[str, float]:
    D_int = int(D)
    B_int = max(1, int(B))
    if D_int <= 0:
        raise ValueError("D must be > 0")
    obs_cost_val = max(float(obs_cost), 1e-9)
    avg_block_size = float(D_int) / float(B_int) if B_int > 0 else float(D_int)
    avg_block_size = max(1.0, avg_block_size)

    def _budget_to_blocks(budget: float) -> int:
        if budget <= 0.0:
            return 1
        dims = float(budget) / obs_cost_val
        ratio = float(dims) / avg_block_size
        ratio = max(0.0, ratio)
        blocks = int(math.ceil(max(1.0, ratio)))
        return max(1, min(B_int, blocks))

    k_fixed = _budget_to_blocks(float(obs_budget))
    k_min = _budget_to_blocks(float(obs_budget_min)) if obs_budget_min > 0 else k_fixed
    k_max = _budget_to_blocks(float(obs_budget_max)) if obs_budget_max > 0 else k_fixed
    if k_min > k_max:
        k_max = k_min
    if int(min_fovea_blocks) > 0:
        k_fixed = max(k_fixed, min(int(B), int(min_fovea_blocks)))
    step_idx = 0
    obs_dims: List[int] = []
    transport_span_effective = int(transport_span_blocks)
    if transport_span_effective <= 0 and int(k_fixed) >= B_int:
        transport_span_effective = B_int
    train_weight_by_value_effective = bool(train_weight_by_value)
    if not train_weight_by_value_effective and world == "square" and int(k_fixed) >= B_int:
        train_weight_by_value_effective = True
    theta_learn_effective = float(theta_learn)
    if world == "square" and int(k_fixed) >= B_int and theta_learn_effective < 0.5:
        theta_learn_effective = 0.5
    moving = None
    square = None
    linear_world = None
    periph_dim = 0
    base_dim = D_int
    world_dim = D_int
    if world == "moving":
        moving = MovingColorShapeWorld(
            side=side,
            n_colors=n_colors,
            n_shapes=n_shapes,
            seed=seed,
            p_color_shift=p_color_shift,
            p_shape_shift=p_shape_shift,
            periph_bins=periph_bins,
            object_size=object_size,
        )
        world_dim = moving.D
        block_size = int(D_int) // int(B_int)
        if block_size * int(B_int) != int(D_int):
            raise ValueError("D must be divisible by B")
        periph_dim = int(periph_blocks) * block_size
        base_dim = int(D_int) - periph_dim
        cfg = AgentConfig(
            D=int(D),
            B=int(B),
            fovea_blocks_per_step=int(k_fixed),
            fovea_residual_only=bool(fovea_residual_only),
            alpha_cov=float(alpha_cov),
            coverage_cap_G=int(coverage_cap_G),
            fovea_residual_ema=float(fovea_residual_ema),
            fovea_use_age=bool(fovea_use_age),
            fovea_age_min_inc=float(fovea_age_min_inc),
            fovea_age_resid_scale=float(fovea_age_resid_scale),
            fovea_age_resid_thresh=float(fovea_age_resid_thresh),
            working_set_linger_steps=int(working_set_linger_steps),
            transport_span_blocks=int(transport_span_effective),
            train_active_only=bool(train_active_only),
            train_active_threshold=float(train_active_threshold),
            train_weight_by_value=train_weight_by_value_effective,
            train_value_power=float(train_value_power),
            lr_expert=float(lr_expert),
            sigma_ema=float(sigma_ema),
            theta_learn=float(theta_learn_effective),
            theta_ar_rest=float(theta_ar_rest),
            nu_max=float(nu_max),
            xi_max=float(xi_max),
            W=int(stability_window),
            theta_ar=float(theta_ar),
            kappa_ar=float(kappa_ar),
            N_max=int(n_max),
            L_work_max=float(l_work_max),
            force_block_anchors=bool(force_block_anchors),
            binding_enabled=bool(binding_enabled),
            binding_shift_radius=int(binding_shift_radius),
            binding_rotations=bool(binding_rotations),
            grid_side=int(side),
            grid_channels=int(n_colors + n_shapes),
            grid_base_dim=int(base_dim),
            periph_bins=int(periph_bins),
            periph_blocks=int(periph_blocks),
            periph_channels=1,
            transport_use_true_full=bool(transport_test),
            transport_force_true_delta=bool(transport_force_true_delta),
            fovea_routing_weight=float(fovea_routing_weight),
            fovea_routing_ema=float(fovea_routing_ema),
        )
        agent = NUPCA3Agent(cfg)
        moving.reset()
        vis_n_colors = int(n_colors)
        vis_n_shapes = int(n_shapes)
    elif world == "square":
        square = LinearSquareWorld(
            side=side,
            seed=seed,
            square_small=square_small,
            square_big=square_big,
            pattern_period=pattern_period,
            dx=dx,
            dy=dy,
            periph_bins=periph_bins,
        )
        world_dim = square.D
        block_size = int(D_int) // int(B_int)
        if block_size * int(B_int) != int(D_int):
            raise ValueError("D must be divisible by B")
        periph_dim = int(periph_blocks) * block_size
        base_dim = int(D_int) - periph_dim
        cfg = AgentConfig(
            D=int(D),
            B=int(B),
            fovea_blocks_per_step=int(k_fixed),
            fovea_residual_only=bool(fovea_residual_only),
            alpha_cov=float(alpha_cov),
            coverage_cap_G=int(coverage_cap_G),
            fovea_residual_ema=float(fovea_residual_ema),
            fovea_use_age=bool(fovea_use_age),
            fovea_age_min_inc=float(fovea_age_min_inc),
            fovea_age_resid_scale=float(fovea_age_resid_scale),
            fovea_age_resid_thresh=float(fovea_age_resid_thresh),
            working_set_linger_steps=int(working_set_linger_steps),
            transport_span_blocks=int(transport_span_effective),
            train_active_only=bool(train_active_only),
            train_active_threshold=float(train_active_threshold),
            train_weight_by_value=train_weight_by_value_effective,
            train_value_power=float(train_value_power),
            lr_expert=float(lr_expert),
            sigma_ema=float(sigma_ema),
            theta_learn=float(theta_learn_effective),
            theta_ar_rest=float(theta_ar_rest),
            nu_max=float(nu_max),
            xi_max=float(xi_max),
            W=int(stability_window),
            theta_ar=float(theta_ar),
            kappa_ar=float(kappa_ar),
            N_max=int(n_max),
            L_work_max=float(l_work_max),
            force_block_anchors=bool(force_block_anchors),
            grid_side=int(side),
            grid_channels=1,
            grid_base_dim=int(base_dim),
            periph_bins=int(periph_bins),
            periph_blocks=int(periph_blocks),
            periph_channels=1,
            transport_use_true_full=bool(transport_test),
            transport_force_true_delta=bool(transport_force_true_delta),
            fovea_routing_weight=float(fovea_routing_weight),
            fovea_routing_ema=float(fovea_routing_ema),
        )
        agent = NUPCA3Agent(cfg)
        square.reset()
        vis_n_colors = 1
        vis_n_shapes = 0
    else:
        cfg = AgentConfig(
            D=int(D),
            B=int(B),
            fovea_blocks_per_step=int(k_fixed),
            fovea_residual_only=bool(fovea_residual_only),
            alpha_cov=float(alpha_cov),
            coverage_cap_G=int(coverage_cap_G),
            fovea_residual_ema=float(fovea_residual_ema),
            fovea_use_age=bool(fovea_use_age),
            fovea_age_min_inc=float(fovea_age_min_inc),
            fovea_age_resid_scale=float(fovea_age_resid_scale),
            fovea_age_resid_thresh=float(fovea_age_resid_thresh),
            working_set_linger_steps=int(working_set_linger_steps),
            transport_span_blocks=int(transport_span_effective),
            train_active_only=bool(train_active_only),
            train_active_threshold=float(train_active_threshold),
            train_weight_by_value=train_weight_by_value_effective,
            train_value_power=float(train_value_power),
            lr_expert=float(lr_expert),
            sigma_ema=float(sigma_ema),
            theta_learn=float(theta_learn_effective),
            theta_ar_rest=float(theta_ar_rest),
            nu_max=float(nu_max),
            xi_max=float(xi_max),
            W=int(stability_window),
            theta_ar=float(theta_ar),
            kappa_ar=float(kappa_ar),
            N_max=int(n_max),
            L_work_max=float(l_work_max),
            force_block_anchors=bool(force_block_anchors),
            binding_enabled=bool(binding_enabled),
            binding_shift_radius=int(binding_shift_radius),
            binding_rotations=bool(binding_rotations),
            periph_bins=int(periph_bins),
            periph_blocks=int(periph_blocks),
            periph_channels=1,
            transport_use_true_full=bool(transport_test),
            transport_force_true_delta=bool(transport_force_true_delta),
            fovea_routing_weight=float(fovea_routing_weight),
            fovea_routing_ema=float(fovea_routing_ema),
        )
        agent = NUPCA3Agent(cfg)
        linear_world = LinearARWorld(D=int(D), seed=int(seed))
        linear_world.reset()
        vis_n_colors = 0
        vis_n_shapes = 0
    periph_ids = _peripheral_block_ids(cfg) if periph_blocks > 0 else []
    coverage_diag_enabled = bool(diagnose_coverage) and world == "square"
    coverage_steps_total = 0
    coverage_hits = 0
    coverage_square_blocks_seen = set()
    coverage_covered_blocks_seen = set()
    coverage_dim_to_block: Dict[int, int] = {}
    if coverage_diag_enabled:
        blocks_partition = getattr(agent.state, "blocks", []) or []
        for block_id, dims in enumerate(blocks_partition):
            for dim in dims:
                coverage_dim_to_block[int(dim)] = block_id
    coverage_log_every = int(coverage_log_every)
    coverage_log_every = coverage_log_every if coverage_log_every > 0 else 0
    rest_test_period = max(0, int(rest_test_period))
    rest_test_length = max(0, int(rest_test_length))
    periph_test_active = bool(periph_test and periph_blocks > 0)
    transport_test_active = bool(transport_test)
    rest_test_forced_steps = 0
    rest_test_edits_processed = 0
    periph_missing_steps = 0
    periph_present_steps = 0
    transport_test_total = 0
    transport_test_matches = 0
    mae_obs = []
    mae_unobs = []
    mae_unobs_predonly = []
    mae_pos = []
    mae_pos_obs = []
    mae_pos_unobs = []
    mae_pos_predonly = []
    corr_obs = []
    corr_unobs = []
    corr_unobs_predonly = []
    pos_frac = []
    mae_zero = []
    same_block_err = []
    diff_block_err = []
    block_changes = 0
    block_steps = 0
    prev_active_block = None
    prev_active_block_step = None
    obs_active_hits = 0
    obs_active_steps = 0
    first_seen_err = []
    reappear_err = []
    was_occluded = False
    step_log_limit = 25
    permit_param_true = 0
    permit_param_total = 0

    for step_idx in range(int(steps)):
        rest_permitted_prev = bool(getattr(agent.state, "rest_permitted_prev", False))
        demand_prev = bool(getattr(agent.state, "demand_prev", False))
        interrupt_prev = bool(getattr(agent.state, "interrupt_prev", False))
        s_int_need_prev = float(getattr(agent.state, "s_int_need_prev", 0.0))
        s_ext_th_prev = float(getattr(agent.state, "s_ext_th_prev", 0.0))
        x_C_prev = float(getattr(agent.state, "x_C_prev", 0.0))
        rawE_prev = float(getattr(agent.state, "rawE_prev", 0.0))
        rawD_prev = float(getattr(agent.state, "rawD_prev", 0.0))
        coverage_debt_prev_val = float(getattr(agent.state, "coverage_debt_prev", 0.0))
        b_cons_prev = float(getattr(agent.state, "b_cons", 0.0))
        active_idx = np.array([], dtype=int)
        active_idx = np.array([], dtype=int)
        obs_active_idx = np.array([], dtype=int)
        active_blocks = set()
        obs_active_count = 0
        force_rest = False
        if rest_test_period > 0 and rest_test_length > 0:
            cycle_phase = step_idx % rest_test_period
            force_rest = cycle_phase < rest_test_length
        if force_rest:
            agent.state.rest_permitted_prev = True
            agent.state.demand_prev = True
            agent.state.interrupt_prev = False
        # A16.3: select fovea blocks from current agent state (t-1 stats).
        k_eff = int(k_fixed)
        if str(obs_budget_mode).lower() == "coverage":
            debt = float(getattr(agent.state, "coverage_debt_prev", 0.0))
            denom = float(coverage_debt_target) if coverage_debt_target > 0 else float(max(1, B))
            ratio = 0.0 if denom <= 0 else min(1.0, max(0.0, debt / denom))
            k_eff = int(round(k_min + (k_max - k_min) * ratio))
            k_eff = max(1, min(int(B), k_eff))
        cfg_step = agent.cfg
        if step_idx < int(scan_steps):
            k_eff = int(B)
        elif warm_steps > 0 and step_idx < (int(scan_steps) + int(warm_steps)):
            k_eff = int(warm_fovea_blocks) if int(warm_fovea_blocks) > 0 else int(k_fixed)
            k_eff = max(1, min(int(B), k_eff))
        if int(min_fovea_blocks) > 0:
            k_eff = max(k_eff, min(int(B), int(min_fovea_blocks)))
        cfg_step = replace(agent.cfg, fovea_blocks_per_step=int(k_eff))
        update_fovea_routing_scores(agent.state.fovea, agent.state.buffer.x_last, cfg_step, t=step_idx)
        fovea_state = agent.state.fovea
        ages_snapshot = np.asarray(getattr(fovea_state, "block_age", []), dtype=float)
        resid_snapshot = np.asarray(getattr(fovea_state, "block_residual", []), dtype=float)
        top_k = int(min(int(k_eff), int(B))) if int(B) > 0 else 0
        if top_k > 0 and ages_snapshot.size:
            top_age_blocks = [int(i) for i in np.argsort(-ages_snapshot)[:top_k]]
        else:
            top_age_blocks = []
        if top_k > 0 and resid_snapshot.size:
            top_resid_blocks = [int(i) for i in np.argsort(-resid_snapshot)[:top_k]]
        else:
            top_resid_blocks = []
        blocks = select_fovea(agent.state.fovea, cfg_step)
        periph_candidates = list(range(max(0, int(B) - int(periph_blocks)), int(B)))
        blocks, forced_periph_blocks = _enforce_peripheral_blocks(
            blocks,
            cfg_step,
            periph_candidates,
        )
        prev_observed_dims = set(int(k) for k in getattr(agent.state.buffer, "observed_dims", set()) or set())
        motion_probe_budget = max(0, int(getattr(cfg_step, "motion_probe_blocks", 0)))
        motion_probe_blocks = _select_motion_probe_blocks(prev_observed_dims, cfg_step, motion_probe_budget)
        blocks, motion_probe_blocks_used = _enforce_motion_probe_blocks(blocks, cfg_step, motion_probe_blocks)
        selected_blocks_tuple = tuple(int(b) for b in blocks)
        if periph_test_active:
            periph_present = all(int(b) in blocks for b in periph_ids)
            if periph_present:
                periph_present_steps += 1
            else:
                periph_missing_steps += 1
            print(
                f"[periph check] step={step_idx} n_blocks={len(blocks)} "
                f"periph_present={periph_present} missing={periph_missing_steps}"
            )
        obs_dims = sorted(make_observation_set(blocks, cfg))
        pred_only = pred_only_len > 0 and pred_only_start <= step_idx < (pred_only_start + pred_only_len)
        occluding = False
        if occlude_len > 0 and occlude_period > 0 and step_idx >= occlude_start:
            if ((step_idx - occlude_start) % occlude_period) < occlude_len:
                occluding = True
        if pred_only:
            obs_dims = []
        if occluding:
            obs_dims = []

        # Environment evolves, then we reveal only the selected dims.
        true_delta: Tuple[int, int] = (0, 0)
        true_env_vec = np.zeros(0, dtype=float)
        true_env_dim = int(world_dim)
        if world == "moving":
            env_state_full = moving.step()
            true_env_vec = env_state_full.copy()
            true_env_dim = int(moving.D)
            base_x = env_state_full
            true_delta = moving.last_move()
            if int(base_dim) != int(world_dim):
                if int(base_dim) > int(world_dim):
                    base_x = np.pad(base_x, (0, int(base_dim) - int(world_dim)), mode="constant")
                else:
                    base_x = base_x[: int(base_dim)]
            periph = moving.encode_peripheral() if periph_blocks > 0 else np.zeros(0, dtype=float)
            if periph_blocks > 0:
                pad = periph_dim - periph.shape[0]
                if pad < 0:
                    periph = periph[:periph_dim]
                elif pad > 0:
                    periph = np.pad(periph, (0, pad), mode="constant")
            full_x = np.concatenate([base_x, periph]) if periph_blocks > 0 else base_x
        elif world == "square":
            env_state_full = square.step()
            true_env_vec = env_state_full.copy()
            true_env_dim = int(square.D)
            base_x = env_state_full
            true_delta = square.last_move()
            if int(base_dim) != int(world_dim):
                if int(base_dim) > int(world_dim):
                    base_x = np.pad(base_x, (0, int(base_dim) - int(world_dim)), mode="constant")
                else:
                    base_x = base_x[: int(base_dim)]
            periph = square.encode_peripheral() if periph_blocks > 0 else np.zeros(0, dtype=float)
            if periph_blocks > 0:
                pad = periph_dim - periph.shape[0]
                if pad < 0:
                    periph = periph[:periph_dim]
                elif pad > 0:
                    periph = np.pad(periph, (0, pad), mode="constant")
            full_x = np.concatenate([base_x, periph]) if periph_blocks > 0 else base_x
        else:
            env_state_full = linear_world.step()
            true_env_vec = env_state_full.copy()
            true_env_dim = int(linear_world.D)
            base_x = env_state_full
            full_x = base_x
        pos_dims = set(int(idx) for idx in np.where(full_x > 0.0)[0])
        obs = EnvObs(
            x_partial=build_partial_obs(full_x, obs_dims),
            opp=0.0,
            danger=0.0,
            x_full=full_x.copy(),
            true_delta=true_delta,
            pos_dims=pos_dims,
            selected_blocks=selected_blocks_tuple,
        )

        buffer_prev = agent.state.buffer.x_last.copy()

        # Pre-step prior for evaluation (A16.2 residual definition).
        prior = getattr(agent.state.learn_cache, "yhat_tp1", None)
        prior_arr: np.ndarray | None = None
        if prior is not None:
            prior_arr = np.asarray(prior, dtype=float).reshape(-1)
        action, next_state, trace = step_pipeline(agent.state, obs, cfg_step)
        agent.state = next_state
        if force_rest:
            rest_test_forced_steps += 1
            rest_test_edits_processed += int(trace.get("edits_processed", 0))
            agent.state.rest_permitted_prev = True
            agent.state.demand_prev = True
            agent.state.interrupt_prev = False
        if coverage_diag_enabled:
            square_blocks = {
                coverage_dim_to_block.get(int(dim))
                for dim in np.where(full_x > 0.0)[0]
                if coverage_dim_to_block.get(int(dim)) is not None
            }
            square_blocks.discard(None)
            library_nodes = getattr(agent.state.library, "nodes", {}) or {}
            active_set_ids = set(getattr(agent.state, "active_set", set()))
            covered_blocks = set()
            for nid in active_set_ids:
                node = library_nodes.get(int(nid))
                if node is None:
                    continue
                footprint = int(getattr(node, "footprint", -1))
                if footprint >= 0:
                    covered_blocks.add(footprint)
            coverage_steps_total += 1
            if square_blocks:
                coverage_square_blocks_seen.update(square_blocks)
            coverage_covered_blocks_seen.update(covered_blocks)
            hit = bool(square_blocks & covered_blocks)
            coverage_hits += int(hit)
            if coverage_log_every > 0 and (step_idx % coverage_log_every == 0 or coverage_log_every == 1):
                print(
                    f"[coverage] step={trace['t']} square_blocks={sorted(square_blocks)} "
                    f"covered_blocks={sorted(covered_blocks)} hit={int(hit)}"
                )
        permit_param_total += 1
        if bool(trace.get("permit_param", False)):
            permit_param_true += 1

        if prior_arr is not None:
            active_mask = full_x > 0.0
            active_idx = np.where(active_mask)[0]
            active_blocks = set()
            if active_idx.size:
                block_size = int(D) // int(B)
                active_blocks = set(int(i) // block_size for i in active_idx)
            obs_active_idx = np.array([i for i in obs_dims if active_mask[i]], dtype=int)
            obs_active_count = int(obs_active_idx.size)
            if obs_active_count:
                obs_active_hits += 1
            if active_idx.size:
                obs_active_steps += 1
            if np.any(active_mask):
                pos_frac.append(float(np.mean(active_mask)))
                mae_zero.append(float(np.mean(np.abs(full_x))))
                err_pos = np.abs(prior_arr[active_mask] - full_x[active_mask])
                mae_pos.append(float(np.mean(err_pos)))
                if pred_only:
                    mae_pos_predonly.append(float(np.mean(err_pos)))
                if not occluding and not first_seen_err:
                    first_seen_err.append(float(np.mean(err_pos)))
                if was_occluded and not occluding:
                    reappear_err.append(float(np.mean(err_pos)))
                if active_idx.size:
                    active_blocks = np.array([int(i) // int(D // B) for i in active_idx], dtype=int)
                    active_block_now = int(np.bincount(active_blocks).argmax())
                    if prev_active_block is not None:
                        block_steps += 1
                        if active_block_now != prev_active_block:
                            block_changes += 1
                            diff_block_err.append(float(np.mean(err_pos)))
                        else:
                            same_block_err.append(float(np.mean(err_pos)))
                    prev_active_block = active_block_now
                    prev_active_block_step = step_idx
            if obs_dims:
                err = np.abs(prior_arr[obs_dims] - full_x[obs_dims])
                mae_obs.append(float(np.mean(err)))
                if len(obs_dims) > 1:
                    obs_pred = prior_arr[obs_dims]
                    obs_true = full_x[obs_dims]
                    if np.std(obs_pred) > 0 and np.std(obs_true) > 0:
                        corr_obs.append(float(np.corrcoef(obs_pred, obs_true)[0, 1]))
            if prior_arr.shape[0] == full_x.shape[0]:
                mask = np.ones(prior_arr.shape[0], dtype=bool)
                if obs_dims:
                    mask[np.asarray(obs_dims, dtype=int)] = False
                if np.any(mask):
                    err_unobs = np.abs(prior_arr[mask] - full_x[mask])
                    mae_unobs.append(float(np.mean(err_unobs)))
                    if pred_only:
                        mae_unobs_predonly.append(float(np.mean(err_unobs)))
                    if np.sum(mask) > 1:
                        unobs_pred = prior_arr[mask]
                        unobs_true = full_x[mask]
                        if np.std(unobs_pred) > 0 and np.std(unobs_true) > 0:
                            corr_unobs.append(float(np.corrcoef(unobs_pred, unobs_true)[0, 1]))
                            if pred_only:
                                corr_unobs_predonly.append(float(np.corrcoef(unobs_pred, unobs_true)[0, 1]))
                if np.any(active_mask):
                    if obs_dims:
                        obs_mask = np.zeros_like(active_mask, dtype=bool)
                        obs_mask[np.asarray(obs_dims, dtype=int)] = True
                    else:
                        obs_mask = np.zeros_like(active_mask, dtype=bool)
                    pos_obs_mask = active_mask & obs_mask
                    pos_unobs_mask = active_mask & ~obs_mask
                    if np.any(pos_obs_mask):
                        err_pos_obs = np.abs(prior_arr[pos_obs_mask] - full_x[pos_obs_mask])
                        mae_pos_obs.append(float(np.mean(err_pos_obs)))
                    if np.any(pos_unobs_mask):
                        err_pos_unobs = np.abs(prior_arr[pos_unobs_mask] - full_x[pos_unobs_mask])
                        mae_pos_unobs.append(float(np.mean(err_pos_unobs)))
        perc = build_partial_obs(full_x, obs_dims)
        pred_vec = prior_arr if prior_arr is not None else agent.state.buffer.x_last
        pred_vec = np.asarray(pred_vec, dtype=float).reshape(-1)
        occ_env = _occupancy_array(
            full_x[:int(base_dim)],
            side=side,
            base_dim=int(base_dim),
            n_colors=vis_n_colors,
            n_shapes=vis_n_shapes,
        )
        occ_pred = _occupancy_array(
            pred_vec[:int(base_dim)],
            side=side,
            base_dim=int(base_dim),
            n_colors=vis_n_colors,
            n_shapes=vis_n_shapes,
        )
        diff_mask = np.asarray(occ_env, dtype=bool) != np.asarray(occ_pred, dtype=bool)
        diff_count = int(np.sum(diff_mask)) if diff_mask.size else 0
        trace["diff_count"] = diff_count
        print(f"[diff check] step={step_idx} diff_count={diff_count}")
        preds = prior_arr.tolist() if prior_arr is not None else None
        print(
            f"[TRACE step={step_idx}] EXACT_ENV={full_x.tolist()} "
            f"EXACT_AGENT_PERCEIVES={perc} "
            f"EXACT_AGENT_PREDICTS={preds}"
        )
        if visualize_steps and step_idx < visualize_steps:
            obs_set = {int(dim) for dim in obs_dims if 0 <= int(dim) < int(D)}
            transport_delta = tuple(trace.get("transport_delta", (0, 0)))
            _print_visualization(
                step_idx=step_idx,
                true_delta=true_delta,
                transport_delta=transport_delta,
                env_vec=full_x,
                obs_dims=obs_set,
                prev_vec=buffer_prev,
                pred_vec=pred_vec,
                side=side,
                n_colors=vis_n_colors,
                n_shapes=vis_n_shapes,
                base_dim=int(base_dim),
                trace=trace,
                true_env_vec=true_env_vec,
                true_env_dim=true_env_dim,
            )
        was_occluded = bool(occluding)

        block_change_rate = float(block_changes) / float(block_steps) if block_steps > 0 else 0.0
        if world == "square":
            env_pos = (int(square.x), int(square.y))
            print(
                f"[env square] step={step_idx} pos={env_pos} true_delta={true_delta} "
                f"block_change_rate={block_change_rate:.6f} coverage_debt={trace['coverage_debt']:.6f} "
                f"forced_rest={force_rest}"
            )
        elif world == "moving":
            env_pos = (int(moving.x), int(moving.y))
            print(
                f"[env moving] step={step_idx} pos={env_pos} true_delta={true_delta} "
                f"color={int(moving.color)} shape={int(moving.shape)} "
                f"block_change_rate={block_change_rate:.6f} coverage_debt={trace['coverage_debt']:.6f} "
                f"forced_rest={force_rest}"
            )

        if transport_test_active:
            tdelta = tuple(trace.get("transport_delta", (0, 0)))
            match = tdelta == tuple(true_delta)
            transport_test_total += 1
            if match:
                transport_test_matches += 1
            print(
                f"[transport check] step={step_idx} true_delta={true_delta} "
                f"transport_delta={tdelta} match={match}"
            )
            if not match:
                coarse_prev_norm = trace.get("coarse_prev_norm", 0.0)
                coarse_curr_norm = trace.get("coarse_curr_norm", 0.0)
                periph_block_ids = trace.get("periph_block_ids", ())
                periph_missing_head = trace.get("periph_dims_missing_head", ())
                periph_missing_count = trace.get("periph_dims_missing_count", 0)
                periph_dims_in_req = trace.get("periph_dims_in_req", 0)
                coarse_prev_head = trace.get("coarse_prev_head", ())
                coarse_curr_head = trace.get("coarse_curr_head", ())
                print(
                    f"[transport diag] step={step_idx} coarse_prev_norm={coarse_prev_norm:.3f} "
                    f"coarse_curr_norm={coarse_curr_norm:.3f} "
                    f"periph_block_ids={periph_block_ids} periph_dims_in_req={periph_dims_in_req} "
                    f"periph_dims_missing_count={periph_missing_count} "
                    f"periph_dims_missing_head={periph_missing_head}"
                )
                print(
                    f"[transport diag] step={step_idx} coarse_prev_head={coarse_prev_head} "
                    f"coarse_curr_head={coarse_curr_head}"
                )

        emit = step_idx < step_log_limit or (int(log_every) > 0 and step_idx % int(log_every) == 0)
        if emit:
            print(
                f"[D{D} seed{seed}] step={trace['t']} rest={trace['rest']} forced_rest={force_rest} "
                f"rest_permitted_prev={rest_permitted_prev} rest_permitted_t={trace['rest_permitted_t']} "
                f"demand_prev={demand_prev} demand_t={trace['demand_t']} "
                f"interrupt_prev={interrupt_prev} interrupt_t={trace['interrupt_t']} "
                f"coverage_debt={trace['coverage_debt']:.6f} coverage_debt_prev={coverage_debt_prev_val:.6f} "
                f"coverage_debt_delta={trace['coverage_debt_delta']:.6f} "
                f"s_int_need_prev={s_int_need_prev:.3f} s_int_need={trace['s_int_need']:.3f} "
                f"s_ext_th_prev={s_ext_th_prev:.3f} s_ext_th={trace['s_ext_th']:.3f} "
                f"mE={trace['mE']:.3f} mD={trace['mD']:.3f} mL={trace['mL']:.3f} mC={trace['mC']:.6f} "
                f"mS={trace['mS']:.3f} P_rest={trace['P_rest']:.3f} P_rest_eff={trace['P_rest_eff']:.3f} "
                f"P_wake={trace['P_wake']:.3f} maint_debt={trace['maint_debt']:.3f} "
                f"b_cons_prev={b_cons_prev:.3f} Q_struct_len={trace['Q_struct_len']} "
                f"last_struct_edit_t={trace.get('last_struct_edit_t', -999999)} "
                f"W={trace.get('W_window', 50)} "
                f"observed_dims={trace['observed_dims']} edits_processed={trace['edits_processed']} "
                f"permit_param={trace['permit_param']} x_C_prev={x_C_prev:.3f} "
                f"rawE_prev={rawE_prev:.3f} rawD_prev={rawD_prev:.3f} "
                f"arousal={trace.get('arousal', 0.0):.3f} "
                f"arousal_prev={trace.get('arousal_prev', 0.0):.3f} "
                f"permit_struct={trace.get('permit_struct', False)} "
                f"permit_reason={trace.get('permit_struct_reason','')} "
                f"probe_var={trace.get('probe_var', float('nan')):.6f} "
                f"feature_var={trace.get('feature_var', float('nan')):.6f}"
            )
            learning_info = trace.get("learning_candidates", {}) or {}
            if learning_info:
                samples = learning_info.get("samples", [])
                sample_str = ", ".join(
                    f"n{int(s['node'])}@b{int(s['footprint'])} err={s['err']:.3f} clamped={s['clamped']}"
                    for s in samples
                )
                print(
                    f"[D{D} seed{seed}] learning_info candidates={learning_info.get('candidates',0)} "
                    f"clamped={learning_info.get('clamped',0)} err_max={learning_info.get('err_max',float('nan')):.6f} "
                    f"samples=[{sample_str}]"
                )
            permit_meta = trace.get("permit_param_info", {}) or {}
            print(
                f"[D{D} seed{seed}] permit_param_summary theta_learn={permit_meta.get('theta_learn',0.0):.3f} "
                f"permit={permit_meta.get('permit',False)} "
                f"cand={permit_meta.get('candidate_count',0)} clamped={permit_meta.get('clamped',0)} "
                f"updated={permit_meta.get('updated',0)}"
            )
            if pred_only:
                print(f"[D{D} seed{seed}] pred_only_step={trace['t']} obs_dims=0")
            if occluding:
                print(f"[D{D} seed{seed}] occluding_step={trace['t']} obs_dims=0")
        if emit and step_idx < step_log_limit:
            obs_preview = obs_dims[: min(32, len(obs_dims))]
            active_preview = active_idx[: min(32, active_idx.size)].tolist() if active_idx.size else []
            obs_active_preview = obs_active_idx[: min(32, obs_active_idx.size)].tolist() if obs_active_idx.size else []
            print(f"[D{D} seed{seed}] obs_dims_count={len(obs_dims)} obs_dims_head={obs_preview}")
            print(f"[D{D} seed{seed}] active_dims_count={int(active_idx.size)} active_dims_head={active_preview}")
            print(f"[D{D} seed{seed}] obs_active_count={obs_active_count} obs_active_head={obs_active_preview}")
            blocks_preview = blocks[: min(16, len(blocks))]
            print(f"[D{D} seed{seed}] fovea_blocks_count={len(blocks)} fovea_blocks_head={blocks_preview} k_eff={k_eff}")
            top_age_preview = top_age_blocks[: min(16, len(top_age_blocks))]
            top_resid_preview = top_resid_blocks[: min(16, len(top_resid_blocks))]
            print(f"[D{D} seed{seed}] fovea_top_age_head={top_age_preview} fovea_top_resid_head={top_resid_preview}")
        active_count = int(active_idx.size) if active_idx.size else 0
        obs_active_rate = float(obs_active_hits) / float(obs_active_steps) if obs_active_steps else 0.0
        obs_active_blocks = sorted({int(i) // (int(D) // int(B)) for i in obs_active_idx})
        active_blocks_list = sorted(list(active_blocks))
        if len(active_blocks_list) > 8:
            active_blocks_list = active_blocks_list[:8]
        if len(obs_active_blocks) > 8:
            obs_active_blocks = obs_active_blocks[:8]
        if emit:
            print(
                f"[D{D} seed{seed}] active_obs_metrics active_count={active_count} "
                f"obs_active_count={obs_active_count} obs_active_rate={obs_active_rate:.3f} "
                f"active_blocks={active_blocks_list} obs_active_blocks={obs_active_blocks}"
            )
            top_age_hits = len(set(blocks) & set(top_age_blocks)) if top_age_blocks else 0
            top_resid_hits = len(set(blocks) & set(top_resid_blocks)) if top_resid_blocks else 0
            print(
                f"[D{D} seed{seed}] fovea_overlap top_age_hits={top_age_hits} "
                f"top_resid_hits={top_resid_hits} top_k={top_k}"
            )
            full_obs = int(len(obs_dims) >= int(D) and k_eff >= int(B))
            cov_debt = float(trace.get("coverage_debt", 0.0))
            cov_violation = int(full_obs and cov_debt > 1e-6)
            print(
                f"[D{D} seed{seed}] coverage_check full_obs={full_obs} "
                f"coverage_debt={cov_debt:.6f} coverage_debt_full_obs_violation={cov_violation}"
            )
        mae_pos_avg = float(np.mean(mae_pos)) if mae_pos else 0.0
        mae_pos_unobs_avg = float(np.mean(mae_pos_unobs)) if mae_pos_unobs else 0.0
        pos_frac_avg = float(np.mean(pos_frac)) if pos_frac else 0.0
        mae_zero_avg = float(np.mean(mae_zero)) if mae_zero else 0.0
        change_rate = float(block_changes) / float(block_steps) if block_steps > 0 else 0.0
        same_block_avg = float(np.mean(same_block_err)) if same_block_err else 0.0
        diff_block_avg = float(np.mean(diff_block_err)) if diff_block_err else 0.0
        if emit:
            print(
                f"[D{D} seed{seed}] sparse_metrics mae_pos={mae_pos_avg:.6f} "
                f"mae_pos_unobs={mae_pos_unobs_avg:.6f} "
                f"pos_frac={pos_frac_avg:.6f} mae_zero={mae_zero_avg:.6f}"
            )
            print(
                f"[D{D} seed{seed}] block_metrics change_rate={change_rate:.6f} "
                f"mae_same_block={same_block_avg:.6f} mae_diff_block={diff_block_avg:.6f}"
            )
            permit_rate = float(permit_param_true) / float(permit_param_total) if permit_param_total else 0.0
            print(f"[D{D} seed{seed}] permit_param_rate={permit_rate:.6f} permit_param_true={permit_param_true}")
        first_avg = float(np.mean(first_seen_err)) if first_seen_err else 0.0
        reap_avg = float(np.mean(reappear_err)) if reappear_err else 0.0
        ratio = (reap_avg / first_avg) if first_avg > 0 else 0.0
        if emit:
            print(
                f"[D{D} seed{seed}] occlusion_metrics first_err={first_avg:.6f} "
                f"reappear_err={reap_avg:.6f} ratio={ratio:.6f}"
            )

    coverage_hit_rate = float(coverage_hits) / float(coverage_steps_total) if coverage_steps_total else 0.0
    if coverage_diag_enabled:
        print(
            f"[coverage summary] steps={coverage_steps_total} hit_rate={coverage_hit_rate:.6f} "
            f"square_blocks_seen={sorted(coverage_square_blocks_seen)} "
            f"covered_blocks_seen={sorted(coverage_covered_blocks_seen)}"
        )

    return {
        "mae_obs": float(np.mean(mae_obs)) if mae_obs else 0.0,
        "mae_unobs": float(np.mean(mae_unobs)) if mae_unobs else 0.0,
        "corr_obs": float(np.mean(corr_obs)) if corr_obs else 0.0,
        "corr_unobs": float(np.mean(corr_unobs)) if corr_unobs else 0.0,
        "mae_unobs_predonly": float(np.mean(mae_unobs_predonly)) if mae_unobs_predonly else 0.0,
        "corr_unobs_predonly": float(np.mean(corr_unobs_predonly)) if corr_unobs_predonly else 0.0,
        "mae_pos": float(np.mean(mae_pos)) if mae_pos else 0.0,
        "mae_pos_obs": float(np.mean(mae_pos_obs)) if mae_pos_obs else 0.0,
        "mae_pos_unobs": float(np.mean(mae_pos_unobs)) if mae_pos_unobs else 0.0,
        "mae_pos_predonly": float(np.mean(mae_pos_predonly)) if mae_pos_predonly else 0.0,
        "pos_frac": float(np.mean(pos_frac)) if pos_frac else 0.0,
        "mae_zero": float(np.mean(mae_zero)) if mae_zero else 0.0,
        "block_change_rate": float(block_changes) / float(block_steps) if block_steps > 0 else 0.0,
        "mae_same_block": float(np.mean(same_block_err)) if same_block_err else 0.0,
        "mae_diff_block": float(np.mean(diff_block_err)) if diff_block_err else 0.0,
        "binding_enabled": bool(binding_enabled),
        "binding_shift_radius": int(binding_shift_radius),
        "binding_rotations": bool(binding_rotations),
        "periph_blocks": int(periph_blocks),
        "periph_bins": int(periph_bins),
        "first_seen_err": float(np.mean(first_seen_err)) if first_seen_err else 0.0,
        "reappear_err": float(np.mean(reappear_err)) if reappear_err else 0.0,
        "steps": int(steps),
        "coverage_hit_rate": coverage_hit_rate,
        "coverage_steps": int(coverage_steps_total),
        "rest_test_forced_steps": int(rest_test_forced_steps),
        "rest_test_edits_processed": int(rest_test_edits_processed),
        "periph_test_active": bool(periph_test_active),
        "periph_missing_steps": int(periph_missing_steps),
        "periph_present_steps": int(periph_present_steps),
        "transport_test_active": bool(transport_test_active),
        "transport_test_total": int(transport_test_total),
        "transport_test_matches": int(transport_test_matches),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world", choices=["linear", "moving", "square"], default="linear")
    parser.add_argument("--D", type=int, default=16)
    parser.add_argument("--B", type=int, default=4)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--obs-budget", type=float, default=2.0)
    parser.add_argument("--obs-cost", type=float, default=1.0)
    parser.add_argument("--obs-budget-mode", choices=["fixed", "coverage"], default="fixed")
    parser.add_argument("--obs-budget-min", type=float, default=0.0)
    parser.add_argument("--obs-budget-max", type=float, default=0.0)
    parser.add_argument("--coverage-debt-target", type=float, default=0.0)
    parser.add_argument("--pred-only-start", type=int, default=-1)
    parser.add_argument("--pred-only-len", type=int, default=0)
    parser.add_argument("--dense-world", action="store_true")
    parser.add_argument("--dense-sigma", type=float, default=1.5)
    parser.add_argument("--fovea-residual-only", action="store_true")
    parser.add_argument("--binding-enabled", action="store_true")
    parser.add_argument("--binding-shift-radius", type=int, default=1)
    parser.add_argument("--binding-rotations", action="store_true")
    parser.add_argument("--periph-blocks", type=int, default=0)
    parser.add_argument("--periph-bins", type=int, default=2)
    parser.add_argument("--object-size", type=int, default=3)
    parser.add_argument("--rest-test-period", type=int, default=0)
    parser.add_argument("--rest-test-length", type=int, default=0)
    parser.add_argument("--periph-test", action="store_true")
    parser.add_argument("--transport-test", action="store_true")
    parser.add_argument("--transport-force-true-delta", action="store_true")
    parser.add_argument("--alpha-cov", type=float, default=0.10)
    parser.add_argument("--coverage-cap-G", type=int, default=50)
    parser.add_argument("--fovea-residual-ema", type=float, default=0.10)
    parser.add_argument("--fovea-use-age", action="store_true")
    parser.add_argument("--fovea-age-min-inc", type=float, default=0.05)
    parser.add_argument("--fovea-age-resid-scale", type=float, default=0.05)
    parser.add_argument("--fovea-age-resid-thresh", type=float, default=0.01)
    parser.add_argument("--fovea-routing-weight", type=float, default=1.0)
    parser.add_argument("--fovea-routing-ema", type=float, default=0.0)
    parser.add_argument("--occlude-start", type=int, default=-1)
    parser.add_argument("--occlude-len", type=int, default=0)
    parser.add_argument("--occlude-period", type=int, default=0)
    parser.add_argument("--working-set-linger-steps", type=int, default=0)
    parser.add_argument("--transport-span-blocks", type=int, default=0)
    parser.add_argument("--min-fovea-blocks", type=int, default=0)
    parser.add_argument("--train-active-only", action="store_true")
    parser.add_argument("--train-active-threshold", type=float, default=0.0)
    parser.add_argument("--train-weight-by-value", action="store_true")
    parser.add_argument("--train-value-power", type=float, default=1.0)
    parser.add_argument("--theta-learn", type=float, default=0.02)
    parser.add_argument("--lr-expert", type=float, default=0.01)
    parser.add_argument("--sigma-ema", type=float, default=0.01)
    parser.add_argument("--theta-ar-rest", type=float, default=1.0)
    parser.add_argument("--theta-ar", type=float, default=0.5)
    parser.add_argument("--kappa-ar", type=float, default=0.2)
    parser.add_argument("--nu-max", type=float, default=1.0)
    parser.add_argument("--xi-max", type=float, default=12.0)
    parser.add_argument("--stability-window", type=int, default=50)
    parser.add_argument("--scan-steps", type=int, default=0)
    parser.add_argument("--warm-steps", type=int, default=0)
    parser.add_argument("--warm-fovea-blocks", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--visualize-steps", type=int, default=0, help="Print ASCII environment/agent/prediction grids for the first N steps.")
    parser.add_argument("--n-max", type=int, default=256)
    parser.add_argument("--l-work-max", type=float, default=48.0)
    parser.add_argument("--force-block-anchors", action="store_true")
    parser.add_argument("--diagnose-coverage", action="store_true")
    parser.add_argument("--coverage-log-every", type=int, default=0)
    parser.add_argument("--side", type=int, default=8)
    parser.add_argument("--n-colors", type=int, default=4)
    parser.add_argument("--n-shapes", type=int, default=4)
    parser.add_argument("--p-color-shift", type=float, default=0.05)
    parser.add_argument("--p-shape-shift", type=float, default=0.05)
    parser.add_argument("--square-small", type=int, default=2)
    parser.add_argument("--square-big", type=int, default=3)
    parser.add_argument("--pattern-period", type=int, default=20)
    parser.add_argument("--dx", type=int, default=1)
    parser.add_argument("--dy", type=int, default=0)
    args = parser.parse_args()

    summary = run_task(
        D=args.D,
        B=args.B,
        steps=args.steps,
        seed=args.seed,
        world=args.world,
        side=args.side,
        n_colors=args.n_colors,
        n_shapes=args.n_shapes,
        square_small=args.square_small,
        square_big=args.square_big,
        pattern_period=args.pattern_period,
        dx=args.dx,
        dy=args.dy,
        p_color_shift=args.p_color_shift,
        p_shape_shift=args.p_shape_shift,
        obs_budget=args.obs_budget,
        obs_cost=args.obs_cost,
        obs_budget_mode=args.obs_budget_mode,
        obs_budget_min=args.obs_budget_min,
        obs_budget_max=args.obs_budget_max,
        coverage_debt_target=args.coverage_debt_target,
        pred_only_start=args.pred_only_start,
        pred_only_len=args.pred_only_len,
        dense_world=args.dense_world,
        dense_sigma=args.dense_sigma,
        fovea_residual_only=args.fovea_residual_only,
        binding_enabled=args.binding_enabled,
        binding_shift_radius=args.binding_shift_radius,
        binding_rotations=args.binding_rotations,
        periph_blocks=args.periph_blocks,
        periph_bins=args.periph_bins,
        object_size=args.object_size,
        alpha_cov=args.alpha_cov,
        coverage_cap_G=args.coverage_cap_G,
        fovea_residual_ema=args.fovea_residual_ema,
        fovea_use_age=args.fovea_use_age,
        fovea_age_min_inc=args.fovea_age_min_inc,
        fovea_age_resid_scale=args.fovea_age_resid_scale,
        fovea_age_resid_thresh=args.fovea_age_resid_thresh,
        fovea_routing_weight=args.fovea_routing_weight,
        fovea_routing_ema=args.fovea_routing_ema,
        occlude_start=args.occlude_start,
        occlude_len=args.occlude_len,
        occlude_period=args.occlude_period,
        working_set_linger_steps=args.working_set_linger_steps,
        transport_span_blocks=args.transport_span_blocks,
        min_fovea_blocks=args.min_fovea_blocks,
        train_active_only=args.train_active_only,
        train_active_threshold=args.train_active_threshold,
        train_weight_by_value=args.train_weight_by_value,
        train_value_power=args.train_value_power,
        lr_expert=args.lr_expert,
        sigma_ema=args.sigma_ema,
        theta_learn=args.theta_learn,
        theta_ar_rest=args.theta_ar_rest,
        nu_max=args.nu_max,
        xi_max=args.xi_max,
        stability_window=args.stability_window,
        theta_ar=args.theta_ar,
        kappa_ar=args.kappa_ar,
        scan_steps=args.scan_steps,
        warm_steps=args.warm_steps,
        warm_fovea_blocks=args.warm_fovea_blocks,
        log_every=args.log_every,
        n_max=args.n_max,
        l_work_max=args.l_work_max,
        force_block_anchors=args.force_block_anchors,
        diagnose_coverage=args.diagnose_coverage,
        coverage_log_every=args.coverage_log_every,
        rest_test_period=args.rest_test_period,
        rest_test_length=args.rest_test_length,
        periph_test=args.periph_test,
        transport_test=args.transport_test,
        transport_force_true_delta=args.transport_force_true_delta,
        visualize_steps=args.visualize_steps,
    )
    print("[SUMMARY]", vars(args), summary)


if __name__ == "__main__":
    main()


  
