#!/usr/bin/env python3
"""Standalone environment screen capture utility."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tools.env_capture.io import write_frame
from tools.env_capture.render import render_env
from tools.env_capture.worlds import init_env


def build_parser() -> argparse.ArgumentParser:
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
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    env, channels = init_env(args)
    if hasattr(env, "reset"):
        env.reset()
    if args.out.exists():
        args.out.unlink()

    for step_idx in range(int(args.steps)):
        env_state = env.step()
        env_vec = np.asarray(env_state, dtype=float).reshape(-1)
        lines = render_env(
            step_idx=step_idx,
            env_vec=env_vec,
            side=args.side,
            channels=channels,
        )
        write_frame(lines, out_file=args.out, echo=not args.no_echo)


if __name__ == "__main__":
    main()
