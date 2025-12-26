"""
Evaluate the ARC-AGI-2 solver versus the old input-copy baseline to show how the
transport-aware priors affect actual ARC tasks. The harness can optionally tune
a confidence threshold so only predictions with strong structural support leave
the template pool.

Usage:

    PYTHONPATH=. python tools/arc_solver_eval.py \
        --data-dir ../ARC-AGI-2/data/training \
        --output outputs/arc_predictions/eval_summary.json \
        [--limit N] [--tune-threshold]

When `--tune-threshold` is provided, the script probes several candidate
confidence thresholds on a small sample of tasks (default 80) before running
on the full set. The baseline prediction simply copies the input into the
output shape via zero-padded cropping.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

from tools.arc_agent_runner import ArcAgentRunner, _align_grid

AGENT_D = 256
AGENT_B = 16
PERIPH_BLOCKS = 8
THRESHOLD_CANDIDATES = [0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 1.0, 1.05, 1.1, 1.15, 1.2]
TUNING_SAMPLE_LIMIT = 80


def _task_files(path: Path) -> Iterable[Path]:
    return sorted(p for p in path.iterdir() if p.suffix.lower() == ".json")


def _mae(a: List[List[int]], b: List[List[int]]) -> float:
    if not a or not b or not a[0] or not b[0]:
        return 0.0
    h = min(len(a), len(b))
    w = min(len(a[0]), len(b[0]))
    total = 0.0
    for y in range(h):
        for x in range(w):
            total += abs(a[y][x] - b[y][x])
    if h == 0 or w == 0:
        return 0.0
    return total / (h * w)


def _align_pred(pred: List[List[int]], target: List[List[int]]) -> List[List[int]]:
    return _align_grid(pred, len(target), len(target[0]) if target and target[0] else 0)


def _init_stats() -> dict:
    return {
        "total_tasks": 0,
        "total_tests": 0,
        "solver_perfect_tasks": 0,
        "baseline_perfect_tasks": 0,
        "solver_perfect_tests": 0,
        "baseline_perfect_tests": 0,
        "solver_cells": 0,
        "baseline_cells": 0,
        "solver_abs_error": 0.0,
        "baseline_abs_error": 0.0,
        "solver_better_tests": 0,
        "baseline_better_tests": 0,
        "equal_tests": 0,
    }


def _run_tasks(
    runner: ArcAgentRunner,
    task_paths: List[Path],
    limit: int | None,
    dump_wins: Path | None = None,
) -> tuple[dict, list]:
    stats = _init_stats()
    win_records: list = []
    for idx, task_path in enumerate(task_paths):
        if limit is not None and idx >= limit:
            break
        with task_path.open("r", encoding="utf-8") as fid:
            task = json.load(fid)
        stats["total_tasks"] += 1
        training_pairs = task.get("train", [])
        tests = task.get("test", [])
        runner.ingest_examples(training_pairs, AGENT_D)

        solver_task_perfect = True
        baseline_task_perfect = True

        for test_idx, pair in enumerate(tests):
            stats["total_tests"] += 1
            target = pair["output"]
            solver_pred = runner.predict(pair["input"], AGENT_D)
            solver_aligned = _align_pred(solver_pred, target)
            baseline_pred = _align_pred(pair["input"], target)

            cells = len(target) * len(target[0]) if target and target[0] else 0
            if cells > 0:
                stats["solver_cells"] += cells
                stats["baseline_cells"] += cells
                solver_err = _mae(solver_aligned, target)
                baseline_err = _mae(baseline_pred, target)
                stats["solver_abs_error"] += solver_err * cells
                stats["baseline_abs_error"] += baseline_err * cells
                if solver_err == 0.0:
                    stats["solver_perfect_tests"] += 1
                else:
                    solver_task_perfect = False
                if baseline_err == 0.0:
                    stats["baseline_perfect_tests"] += 1
                else:
                    baseline_task_perfect = False
                if solver_err < baseline_err:
                    if dump_wins is not None:
                        win_records.append(
                            {
                                "task": task_path.name,
                                "test_index": test_idx,
                                "solver_err": solver_err,
                                "baseline_err": baseline_err,
                                "delta": baseline_err - solver_err,
                            }
                        )
                    stats["solver_better_tests"] += 1
                elif baseline_err < solver_err:
                    stats["baseline_better_tests"] += 1
                else:
                    stats["equal_tests"] += 1
            else:
                solver_task_perfect = False
                baseline_task_perfect = False

        if solver_task_perfect and tests:
            stats["solver_perfect_tasks"] += 1
        if baseline_task_perfect and tests:
            stats["baseline_perfect_tasks"] += 1

    return stats, win_records


def _make_summary(stats: dict) -> dict:
    solver_mae = stats["solver_abs_error"] / stats["solver_cells"] if stats["solver_cells"] else 0.0
    baseline_mae = stats["baseline_abs_error"] / stats["baseline_cells"] if stats["baseline_cells"] else 0.0
    return {
        "tasks_processed": stats["total_tasks"],
        "tests_processed": stats["total_tests"],
        "solver_mae": solver_mae,
        "baseline_mae": baseline_mae,
        "solver_perfect_tests": stats["solver_perfect_tests"],
        "baseline_perfect_tests": stats["baseline_perfect_tests"],
        "solver_perfect_tasks": stats["solver_perfect_tasks"],
        "baseline_perfect_tasks": stats["baseline_perfect_tasks"],
        "solver_cells": stats["solver_cells"],
        "baseline_cells": stats["baseline_cells"],
        "solver_better_tests": stats["solver_better_tests"],
        "baseline_better_tests": stats["baseline_better_tests"],
        "equal_tests": stats["equal_tests"],
    }


def _tune_threshold(task_paths: List[Path], sample_limit: int) -> tuple[float, float]:
    best_threshold = THRESHOLD_CANDIDATES[0]
    best_delta = -float("inf")
    limit = min(sample_limit, len(task_paths))
    if limit == 0:
        return best_threshold, 0.0
    for threshold in THRESHOLD_CANDIDATES:
        runner = ArcAgentRunner(
            D=AGENT_D,
            B=AGENT_B,
            periph_blocks=PERIPH_BLOCKS,
            confidence_threshold=threshold,
        )
        stats, _ = _run_tasks(runner, task_paths, limit=limit)
        summary = _make_summary(stats)
        delta = summary["baseline_mae"] - summary["solver_mae"]
        if delta > best_delta:
            best_delta = delta
            best_threshold = threshold
    return best_threshold, best_delta


def evaluate(
    data_dir: Path,
    limit: int | None,
    output_path: Path,
    tune_threshold: bool = False,
    dump_wins: Path | None = None,
    force_threshold: float | None = None,
    warm_state: Path | None = None,
) -> None:
    data_dir = data_dir.expanduser().resolve()
    task_paths = list(_task_files(data_dir))
    best_threshold = 0.7
    tuning_delta: float | None = None
    if force_threshold is not None:
        best_threshold = force_threshold
        tuning_delta = None
        print(f"Forcing confidence threshold → {best_threshold}")
    elif tune_threshold and task_paths:
        best_threshold, tuning_delta = _tune_threshold(task_paths, sample_limit=TUNING_SAMPLE_LIMIT)
        print(f"Tuned confidence threshold → {best_threshold} (baseline MAE - solver MAE = {tuning_delta:.4f})")
    runner = ArcAgentRunner(
        D=AGENT_D,
        B=AGENT_B,
        periph_blocks=PERIPH_BLOCKS,
        confidence_threshold=best_threshold,
    )
    if warm_state is not None:
        runner.load_state(warm_state)
    stats, win_records = _run_tasks(runner, task_paths, limit=limit, dump_wins=dump_wins)
    summary = _make_summary(stats)
    summary["dataset"] = str(data_dir)
    summary["confidence_threshold"] = best_threshold
    summary["threshold_tuning_delta"] = tuning_delta
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fid:
        json.dump(summary, fid, indent=2)
    if dump_wins and win_records:
        dump_wins.parent.mkdir(parents=True, exist_ok=True)
        with dump_wins.open("w", encoding="utf-8") as fid:
            json.dump(win_records, fid, indent=2)
    print("Evaluation complete")
    for key, value in summary.items():
        print(f"{key}: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the ARC solver vs. a baseline.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tune-threshold", action="store_true")
    parser.add_argument("--force-threshold", type=float, default=None, help="Use a fixed confidence threshold instead of tuning.")
    parser.add_argument("--warm-state", type=Path, default=None, help="Optional persisted agent state to warm-start the solver.")
    parser.add_argument("--dump-wins", type=Path, default=None, help="Optional path to write solver win records.")
    args = parser.parse_args()
    evaluate(
        args.data_dir,
        args.limit,
        args.output,
        tune_threshold=args.tune_threshold,
        dump_wins=args.dump_wins,
        force_threshold=args.force_threshold,
        warm_state=args.warm_state,
    )


if __name__ == "__main__":
    main()
