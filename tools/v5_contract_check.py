#!/usr/bin/env python3
"""tools/v5_contract_check.py

Contract enforcement runner for Step 3 of the v5 rewrite.

It performs the following checks:
  1. Forbidden symbols (`x_full`, `allow_full_state`, `NUPCAAgent`, `Library = ExpertLibrary`,
     `nupca5_enabled`) are absent from Python modules.
  2. A short 30-step run using the v5 kernel emits the expected kernel and ordering markers.
  3. The scan counter stays zero during OPERATING ticks.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
import argparse
from typing import Iterable, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from nupca3.agent import NUPCA3Agent
from nupca3.config import AgentConfig
from nupca3.sim.worlds import ToyWorld
from nupca3.types import EnvObs

EXPECTED_ORDERING_MARKERS = ("A13.transport", "A13.complete", "A13.validation")
FORBIDDEN_PATTERNS = [
    "x_full",
    "allow_full_state",
    "NUPCAAgent",
    "Library = ExpertLibrary",
    "nupca5_enabled",
]


def repo_root() -> Path:
    return REPO_ROOT


def search_forbidden(
    root: Path,
    patterns: Sequence[str],
    *,
    exclude: Sequence[Path] | None = None,
) -> List[Tuple[Path, int, str, str]]:
    hits: List[Tuple[Path, int, str, str]] = []
    excluded = {p.resolve() for p in (exclude or [])}
    for path in root.rglob("*.py"):
        if path.resolve() in excluded:
            continue
        if "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        for idx, line in enumerate(text.splitlines(), start=1):
            for pattern in patterns:
                if pattern in line:
                    hits.append((path, idx, pattern, line.strip()))
    return hits


def build_obs(vec: np.ndarray, *, t_w: int = 0, wall_ms: int | None = None) -> EnvObs:
    arr = np.asarray(vec, dtype=float).reshape(-1)
    D = arr.size
    mask = range(min(4, D))
    x_partial = {int(i): float(arr[int(i)]) for i in mask}
    pos_dims = {int(idx) for idx, val in enumerate(arr) if abs(val) > 0.0}
    return EnvObs(
        x_partial=x_partial,
        opp=0.0,
        danger=0.0,
        periph_full=arr.copy(),
        t_w=t_w,
        wall_ms=wall_ms,
        pos_dims=pos_dims,
    )


def run_kernel_check(cfg: AgentConfig, steps: int = 30) -> None:
    agent = NUPCA3Agent(cfg)
    world = ToyWorld(D=int(cfg.D), x=np.zeros(int(cfg.D)))
    world.reset(seed=0)
    obs = build_obs(world.x, t_w=1, wall_ms=int(time.perf_counter() * 1000))

    for step_idx in range(steps):
        action, trace = agent.step(obs)
        if not isinstance(trace, dict):
            raise RuntimeError(f"Trace is not a dict at step {step_idx}")
        kernel_tag = trace.get("kernel")
        if kernel_tag != "v5":
            raise RuntimeError(f"Unexpected kernel tag at step {step_idx}: {kernel_tag!r}")
        ordering = trace.get("ordering_markers")
        if tuple(ordering or ()) != EXPECTED_ORDERING_MARKERS:
            raise RuntimeError(f"Ordering markers mismatch at step {step_idx}: {ordering!r}")
        budget_use = trace.get("budget_B_use")
        budget_limit = trace.get("budget_limit")
        if budget_use is None or budget_limit is None:
            raise RuntimeError(f"Budget info missing at step {step_idx}: {trace.keys()}")
        if float(budget_use) > float(budget_limit) + 1e-9:
            raise RuntimeError(
                f"Budget use {budget_use} exceeds limit {budget_limit} at step {step_idx}"
            )
        state = agent.state
        if not state.is_rest and int(getattr(state, "scan_counter", 0)) != 0:
            raise RuntimeError(
                f"Scan counter non-zero during OPERATING at step {step_idx}: {state.scan_counter}"
            )
        if state.k_op != 0:
            raise RuntimeError(f"k_op not reset at step {step_idx}: {state.k_op}")
        cache = getattr(state, "trace_cache", None)
        if cache is not None:
            max_entries = int(getattr(cfg, "trace_cache_max_entries", 16))
            max_cues = int(getattr(cfg, "trace_cache_max_cues_per_entry", 12))
            block_cap = int(getattr(cfg, "trace_cache_block_cap", 32))
            if cache.size > max_entries:
                raise RuntimeError(f"TraceCache size {cache.size} exceeds limit {max_entries}")
            if cache.cue_mass > max_entries * max_cues:
                raise RuntimeError(
                    f"TraceCache cue mass {cache.cue_mass} exceeds max_entries*max_cues ({max_entries * max_cues})"
                )
            if cache.block_count > block_cap:
                raise RuntimeError(
                    f"TraceCache block count {cache.block_count} exceeds cap {block_cap}"
                )
        world.step(0)
        obs = build_obs(
            world.x,
            t_w=step_idx + 2,
            wall_ms=int(time.perf_counter() * 1000),
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run v5 contract checks")
    parser.add_argument(
        "--force-contemplate",
        action="store_true",
        help="Set cfg.contemplate_force=True to exercise the planning-only write fence",
    )
    args = parser.parse_args(argv)

    root = repo_root()

    forbidden_hits = search_forbidden(
        root, FORBIDDEN_PATTERNS, exclude=(Path(__file__).resolve(),)
    )
    if forbidden_hits:
        print("Forbidden pattern(s) detected:")
        for path, line_no, pattern, line in forbidden_hits:
            print(f"  {path.relative_to(root)}:{line_no}: contains {pattern!r} -> {line}")
        return 1

    cfg = AgentConfig(D=32, B=8, fovea_blocks_per_step=2, fovea_visit_window=32, sig_gist_bins=4)
    if args.force_contemplate:
        cfg = cfg.replace(contemplate_force=True)
    run_kernel_check(cfg, steps=30)
    print("tools/v5_contract_check.py: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
