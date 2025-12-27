#!/usr/bin/env python3
"""Standalone environment screen capture utility.

Runs a toy environment and captures an ASCII rendering for each tick.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np

from test import LinearARWorld, LinearSquareWorld, MovingColorShapeWorld


def _real_grid(vec: np.ndarray, *, side: int, channels: int) -> List[List[str]]:
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


def _render_env(
    *,
    step_idx: int,
    env_vec: np.ndarray,
    side: int,
    channels: int,
) -> List[str]:
    lines = [f"[ENV_CAPTURE step={step_idx}]"]
    grid = _real_grid(env_vec, side=side, channels=channels)
    for row_idx, row in enumerate(grid):
        prefix = "ENV" if row_idx == 0 else "   "
        lines.append(f"{prefix} " + " ".join(row))
    vec_str = " ".join(f"{val:.6f}" for val in np.asarray(env_vec, dtype=float).reshape(-1).tolist())
    lines.append(f"VEC {vec_str}")
    return lines


def _write_frame(lines: List[str], *, out_file: Path, echo: bool) -> None:
    payload = "\n".join(lines) + "\n"
    with out_file.open("a", encoding="utf-8") as handle:
        handle.write(payload)
    if echo:
        print(payload, end="")


def _init_env(args: argparse.Namespace):
    if args.world == "moving":
        env = MovingColorShapeWorld(
            side=args.side,
            n_colors=args.n_colors,
            n_shapes=args.n_shapes,
            seed=args.seed,
            p_color_shift=args.p_color_shift,
            p_shape_shift=args.p_shape_shift,
            periph_bins=args.periph_bins,
            object_size=args.object_size,
        )
        channels = max(1, int(args.n_colors + args.n_shapes))
        return env, channels
    if args.world == "square":
        env = LinearSquareWorld(
            side=args.side,
            seed=args.seed,
            square_small=args.square_small,
            square_big=args.square_big,
            pattern_period=args.pattern_period,
            dx=args.dx,
            dy=args.dy,
            periph_bins=args.periph_bins,
        )
        return env, 1
    env = LinearARWorld(D=int(args.D), seed=args.seed)
    return env, 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world", choices=["linear", "moving", "square"], default="moving")
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--side", type=int, default=8)
    parser.add_argument("--D", type=int, default=16)
    parser.add_argument("--n-colors", type=int, default=4)
    parser.add_argument("--n-shapes", type=int, default=4)
    parser.add_argument("--p-color-shift", type=float, default=0.05)
    parser.add_argument("--p-shape-shift", type=float, default=0.05)
    parser.add_argument("--periph-bins", type=int, default=2)
    parser.add_argument("--object-size", type=int, default=3)
    parser.add_argument("--square-small", type=int, default=2)
    parser.add_argument("--square-big", type=int, default=3)
    parser.add_argument("--pattern-period", type=int, default=20)
    parser.add_argument("--dx", type=int, default=1)
    parser.add_argument("--dy", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("env_capture.txt"))
    parser.add_argument("--no-echo", action="store_true", help="Do not echo capture to stdout.")
    args = parser.parse_args()

    env, channels = _init_env(args)
    if hasattr(env, "reset"):
        env.reset()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        args.out.unlink()

    for step_idx in range(int(args.steps)):
        if args.world == "linear":
            env_state = env.step()
        else:
            env_state = env.step()
        env_vec = np.asarray(env_state, dtype=float).reshape(-1)
        lines = _render_env(step_idx=step_idx, env_vec=env_vec, side=args.side, channels=channels)
        _write_frame(lines, out_file=args.out, echo=not args.no_echo)


if __name__ == "__main__":
    main()
