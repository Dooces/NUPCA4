"""Toy worlds for the harness."""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from .render import COMMAND_OFFSETS, ROTATE_COMMANDS, _grid_transform


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
        self.vx = 0
        self.vy = 0

    def _sample_velocity(self) -> tuple[int, int]:
        while True:
            dx = int(self.rng.integers(-1, 2))
            dy = int(self.rng.integers(-1, 2))
            if dx != 0 or dy != 0:
                break
        speed = int(self.rng.integers(self.speed_min, self.speed_max + 1))
        return dx * speed, dy * speed

    @property
    def D(self) -> int:
        return self.side * self.side * (self.n_colors + self.n_shapes)

    def reset(self) -> np.ndarray:
        self.x = int(self.rng.integers(self.side))
        self.y = int(self.rng.integers(self.side))
        self.color = int(self.rng.integers(self.n_colors))
        self.shape = int(self.rng.integers(self.n_shapes))
        self.vx, self.vy = self._sample_velocity()
        self.last_dx = 0
        self.last_dy = 0
        return self._encode()

    def step(self) -> np.ndarray:
        self.last_dx = self.vx
        self.last_dy = self.vy
        self.x = (self.x + self.vx) % self.side
        self.y = (self.y + self.vy) % self.side

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
