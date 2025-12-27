"""Environment initialization for screen capture."""

from __future__ import annotations

import argparse
from typing import Tuple

from nupca3.harness.worlds import LinearARWorld, LinearSquareWorld, MovingColorShapeWorld


def init_env(args: argparse.Namespace) -> Tuple[object, int]:
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
