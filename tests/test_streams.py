import numpy as np

from nupca3.config import AgentConfig
from nupca3.geometry.streams import (
    apply_transport,
    compute_grid_shift,
    down_project,
    grid_cell_mass,
    compute_transport_shift,
    up_project,
)


def make_cfg() -> AgentConfig:
    return AgentConfig(
        D=72,
        B=9,
        periph_bins=2,
        periph_blocks=1,
        periph_channels=1,
        grid_side=8,
        grid_channels=1,
        grid_base_dim=64,
    )


def test_up_down_project_match_bins() -> None:
    cfg = make_cfg()
    base_dim = int(cfg.D) - (cfg.periph_blocks * (cfg.D // cfg.B))
    state_vec = np.zeros(cfg.D, dtype=float)
    bins = cfg.periph_bins
    tile_w = max(1, cfg.grid_side // bins)
    tile_h = max(1, cfg.grid_side // bins)
    for cell in range(cfg.grid_side * cfg.grid_side):
        y = cell // cfg.grid_side
        x = cell % cfg.grid_side
        bin_x = min(bins - 1, x // tile_w)
        bin_y = min(bins - 1, y // tile_h)
        state_vec[cell] = float(bin_y * bins + bin_x + 1)

    coarse = up_project(state_vec, cfg)
    assert coarse.shape[0] == bins * bins
    for idx in range(bins * bins):
        assert np.isclose(coarse[idx], float(idx + 1))

    fine_replica = down_project(coarse, cfg)
    assert fine_replica.size == base_dim
    for cell in range(cfg.grid_side * cfg.grid_side):
        y = cell // cfg.grid_side
        x = cell % cfg.grid_side
        bin_x = min(bins - 1, x // tile_w)
        bin_y = min(bins - 1, y // tile_h)
        expected = float(bin_y * bins + bin_x + 1)
        channel_idx = cell * cfg.grid_channels
        assert np.isclose(fine_replica[channel_idx], expected)


def test_transport_shift_and_apply() -> None:
    cfg = make_cfg()
    vec = np.zeros(cfg.D, dtype=float)
    vec[0] = 1.0
    coarse_prev = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    coarse_curr = np.array([0.0, 1.0, 0.0, 0.0], dtype=float)
    shift = compute_transport_shift(coarse_prev, coarse_curr, cfg)
    assert shift[0] != 0 or shift[1] != 0

    transported = apply_transport(vec, shift, cfg)
    assert not np.allclose(transported[: cfg.grid_side * cfg.grid_channels], vec[: cfg.grid_side * cfg.grid_channels])


def test_grid_mass_shift() -> None:
    cfg = make_cfg()
    prev_vec = np.zeros(cfg.D, dtype=float)
    curr_vec = np.zeros(cfg.D, dtype=float)
    prev_vec[0] = 1.0
    curr_vec[cfg.grid_channels] = 1.0

    prev_mass = grid_cell_mass(prev_vec, cfg)
    curr_mass = grid_cell_mass(curr_vec, cfg)
    assert prev_mass.shape == curr_mass.shape
    assert np.count_nonzero(prev_mass) == 1
    assert np.count_nonzero(curr_mass) == 1

    shift = compute_grid_shift(prev_mass, curr_mass, cfg)
    assert shift[0] != 0 or shift[1] != 0
