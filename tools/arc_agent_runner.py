"""
arc_agent_runner.py

Simple harness that feeds ARC-AGI-2 tasks through the NUPCA3 agent so it can
track the grid-based priors, then falls back to nearest-template matching to
produce candidate answers. This demonstrates how to “plug this agent into ARC”
and generates outputs that you can inspect or submit.

Usage:

    python tools/arc_agent_runner.py \
        --data-dir ../ARC-AGI-2/data/training \
        --output-dir outputs/arc_predictions \
        --mode training

The script runs over all JSON tasks in the specified directory, builds a
training corpus mapping observed input grids to outputs, and uses the agent to
“remember” each observation before producing a prediction for every test input.
Predictions are saved task-by-task as JSON under `output_dir`.

This is not a full ARC solution (the agent has no ARC bespoke utilities), but it
shows how to bootstrap the pipeline and keep the transport-aware prior engaged
while the dataset iterates. You can replace the `solve_test` heuristic with a
more sophisticated planner later.
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import math
import time

import numpy as np

from nupca3.agent import NUPCA3Agent
from nupca3.config import AgentConfig
from nupca3.types import EnvObs


def _flatten_grid(grid: Sequence[Sequence[int]], D: int) -> np.ndarray:
    arr = np.asarray(grid, dtype=float).reshape(-1)
    if arr.size < D:
        padded = np.zeros(D, dtype=float)
        padded[: arr.size] = arr
        return padded
    return arr[:D]


def _grid_to_obs(x: Sequence[Sequence[int]], D: int) -> EnvObs:
    vec = _flatten_grid(x, D)
    cue = {i: float(val) for i, val in enumerate(vec) if val != 0.0}
    return EnvObs(
        x_partial=cue,
        opp=0.0,
        danger=0.0,
        periph_full=vec.copy(),
    )


def _grid_mask_coords(grid: Sequence[Sequence[int]]) -> Set[Tuple[int, int]]:
    return {
        (y, x)
        for y, row in enumerate(grid)
        for x, val in enumerate(row)
        if val != 0
    }


def _bounding_box(coords: Set[Tuple[int, int]]) -> tuple[int, int, int, int]:
    if not coords:
        return 0, 0, -1, -1
    ys = [y for y, _ in coords]
    xs = [x for _, x in coords]
    return min(ys), min(xs), max(ys), max(xs)


def _bbox_valid(bbox: tuple[int, int, int, int]) -> bool:
    min_y, min_x, max_y, max_x = bbox
    return max_y >= min_y and max_x >= min_x


def _bbox_center(bbox: tuple[int, int, int, int]) -> tuple[float, float] | None:
    if not _bbox_valid(bbox):
        return None
    min_y, min_x, max_y, max_x = bbox
    return (min_y + max_y) / 2.0, (min_x + max_x) / 2.0


def _align_grid(grid: Sequence[Sequence[int]], target_h: int, target_w: int) -> List[List[int]]:
    if target_h <= 0 or target_w <= 0:
        return [[0] * max(0, target_w) for _ in range(max(0, target_h))]
    if not grid or not grid[0]:
        return [[0] * target_w for _ in range(target_h)]
    height = len(grid)
    width = len(grid[0])

    crop_top = max(0, (height - target_h) // 2)
    crop_left = max(0, (width - target_w) // 2)
    crop_bottom = min(height, crop_top + min(target_h, height))
    crop_right = min(width, crop_left + min(target_w, width))
    cropped: List[List[int]] = [
        row[crop_left:crop_right] for row in grid[crop_top:crop_bottom]
    ]

    padded: List[List[int]] = [[0] * target_w for _ in range(target_h)]
    pad_top = max(0, (target_h - len(cropped)) // 2)
    pad_left = 0
    if cropped and cropped[0]:
        pad_left = max(0, (target_w - len(cropped[0])) // 2)
    for y, row in enumerate(cropped):
        for x, val in enumerate(row):
            dst_y = pad_top + y
            dst_x = pad_left + x
            if dst_y < target_h and dst_x < target_w:
                padded[dst_y][dst_x] = val
    return padded


def _rotate_grid(grid: List[List[int]], times: int) -> List[List[int]]:
    rotated = grid
    for _ in range(times % 4):
        if not rotated or not rotated[0]:
            return rotated
        rotated = [list(row) for row in zip(*rotated[::-1])]
    return rotated


def _apply_transform(grid: List[List[int]], transform: "CoordTransform") -> List[List[int]]:
    rotated = _rotate_grid(grid, transform.rotation // 90)
    if transform.flip_y:
        rotated = rotated[::-1]
    if transform.flip_x:
        rotated = [list(reversed(row)) for row in rotated]
    return rotated


@dataclass
class CoordTransform:
    rotation: int
    flip_y: bool
    flip_x: bool
    offset_y: int
    offset_x: int


def _build_transform_candidates() -> List[CoordTransform]:
    candidates: List[CoordTransform] = []
    offsets = (-1, 0, 1)
    for rotation in (0, 90, 180, 270):
        for flip_y in (False, True):
            for flip_x in (False, True):
                for offset_y in offsets:
                    for offset_x in offsets:
                        candidates.append(
                            CoordTransform(
                                rotation=rotation,
                                flip_y=flip_y,
                                flip_x=flip_x,
                                offset_y=offset_y,
                                offset_x=offset_x,
                            )
                        )
    return candidates


TRANSFORM_CANDIDATES = _build_transform_candidates()


@dataclass
class TrainingCase:
    input_shape: tuple[int, int]
    input_grid: List[List[int]]
    output_shape: tuple[int, int]
    value_map: Dict[int, int]
    default_value: int
    output_mask: List[List[bool]]
    input_values: Set[int]
    input_mask_coords: Set[Tuple[int, int]]
    input_bbox: tuple[int, int, int, int]
    value_freq: Counter[int]
    coord_transform: CoordTransform
    value_map_coverage: float
    value_map_consistency: float
    is_tiling: bool
    case_id: int
    transform_score: float

    def score(self, grid: Sequence[Sequence[int]]) -> float:
        if not grid or not grid[0]:
            shape_score = self.input_shape[0] + self.input_shape[1]
        else:
            shape_score = abs(len(grid) - self.input_shape[0]) + abs(len(grid[0]) - self.input_shape[1])
        grid_values = {val for row in grid for val in row if val != 0}
        value_score = len(grid_values.symmetric_difference(self.input_values))
        grid_mask = _grid_mask_coords(grid)
        union = self.input_mask_coords | grid_mask
        overlap = self.input_mask_coords & grid_mask
        if union:
            mask_penalty = (len(union) - len(overlap)) / len(union)
        else:
            mask_penalty = 0.0 if not grid_mask and not self.input_mask_coords else 1.0
        grid_bbox = _bounding_box(grid_mask)
        grid_center = _bbox_center(grid_bbox)
        case_center = _bbox_center(self.input_bbox)
        if grid_center and case_center:
            dy = grid_center[0] - case_center[0]
            dx = grid_center[1] - case_center[1]
            case_height = self.input_bbox[2] - self.input_bbox[0] + 1
            case_width = self.input_bbox[3] - self.input_bbox[1] + 1
            norm = math.hypot(max(case_height, 1), max(case_width, 1))
            bbox_penalty = min(1.0, math.hypot(dy, dx) / max(norm, 1e-6))
        else:
            bbox_penalty = 0.0
        grid_freq = Counter(val for row in grid for val in row if val != 0)
        freq_keys = set(grid_freq) | set(self.value_freq)
        total_grid = sum(grid_freq.values())
        total_case = sum(self.value_freq.values())
        if freq_keys and (total_grid + total_case) > 0:
            diff_sum = sum(abs(grid_freq[k] - self.value_freq.get(k, 0)) for k in freq_keys)
            freq_penalty = diff_sum / max(1.0, total_grid + total_case)
        else:
            freq_penalty = 0.0
        mask_overlap_ratio = len(overlap) / (len(union) + 1e-6)
        mask_bonus = -mask_overlap_ratio * 3.0
        value_map_bonus = -self.value_map_coverage * 2.0
        consistency_bonus = -self.value_map_consistency * 2.0
        aligned_case_input = _align_grid(grid, self.input_shape[0], self.input_shape[1])
        input_match_total = 0
        input_match_hits = 0
        for y in range(len(aligned_case_input)):
            for x in range(len(aligned_case_input[y])):
                val = aligned_case_input[y][x]
                if val == 0:
                    continue
                input_match_total += 1
                if val == self.input_grid[y][x]:
                    input_match_hits += 1
        if input_match_total:
            input_match_ratio = input_match_hits / input_match_total
        else:
            input_match_ratio = 0.0
        input_match_bonus = -input_match_ratio * 2.0
        case_area = len(self.input_mask_coords)
        grid_area = len(grid_mask)
        if case_area or grid_area:
            area_penalty = abs(case_area - grid_area) / max(case_area, grid_area, 1)
        else:
            area_penalty = 0.0
        case_bbox_area = 0
        grid_bbox_area = 0
        if _bbox_valid(self.input_bbox):
            case_bbox_area = (self.input_bbox[2] - self.input_bbox[0] + 1) * (
                self.input_bbox[3] - self.input_bbox[1] + 1
            )
        if _bbox_valid(grid_bbox):
            grid_bbox_area = (grid_bbox[2] - grid_bbox[0] + 1) * (
                grid_bbox[3] - grid_bbox[1] + 1
            )
        if case_bbox_area or grid_bbox_area:
            size_penalty = abs(case_bbox_area - grid_bbox_area) / max(
                case_bbox_area, grid_bbox_area, 1
            )
        else:
            size_penalty = 0.0
        transform_bonus = -self.transform_score * 2.5
        return (
            shape_score
            + value_score * 1.5
            + mask_penalty * 3.0
            + bbox_penalty * 2.0
            + freq_penalty * 1.5
            + transform_bonus
            + area_penalty * 2.0
            + size_penalty * 1.5
            + mask_bonus
            + value_map_bonus
            + input_match_bonus
            + consistency_bonus
        )

    def apply(self, grid: Sequence[Sequence[int]]) -> List[List[int]]:
        target_h, target_w = self.output_shape
        input_h, input_w = self.input_shape
        aligned_input = _align_grid(grid, input_h, input_w)
        transformed_input = _apply_transform(aligned_input, self.coord_transform)
        trans_h = len(transformed_input)
        trans_w = len(transformed_input[0]) if trans_h else 0
        result: List[List[int]] = []
        for y in range(target_h):
            row: List[int] = []
            for x in range(target_w):
                if not self.output_mask[y][x]:
                    row.append(0)
                    continue
                if trans_h == 0 or trans_w == 0:
                    row.append(self.default_value)
                    continue
                src_y = y + self.coord_transform.offset_y
                src_x = x + self.coord_transform.offset_x
                if 0 <= src_y < trans_h and 0 <= src_x < trans_w:
                    val = transformed_input[src_y][src_x]
                else:
                    val = 0
                mapped = self.value_map.get(val, self.default_value)
                row.append(mapped)
            result.append(row)
        return result


class ArcAgentRunner:
    def __init__(self, *, D: int, B: int, periph_blocks: int, seed: int = 0, confidence_threshold: float = 0.7):
        cfg = AgentConfig(
            D=D,
            B=B,
            fovea_blocks_per_step=min(B, 16),
            periph_blocks=periph_blocks,
            periph_bins=2,
        )
        self.agent = NUPCA3Agent(cfg)
        self.agent.reset(seed=seed)
        self._cases: List[TrainingCase] = []
        self.confidence_threshold = confidence_threshold

    def _prepare_obs(self, obs: EnvObs) -> EnvObs:
        obs.t_w = self.agent.state.t_w + 1
        obs.wall_ms = int(time.perf_counter() * 1000)
        return obs

    def load_state(self, path: Path) -> bool:
        if not path.exists():
            return False
        with path.open("rb") as fid:
            state = pickle.load(fid)
        self.agent.state = state
        print(f"Loaded warm state from {path}")
        return True

    def ingest_examples(self, pairs: Iterable[Dict[str, List[List[int]]]], D: int) -> None:
        for pair in pairs:
            obs_in = self._prepare_obs(_grid_to_obs(pair["input"], D))
            self.agent.step(obs_in)
            self._record_training_case(pair["input"], pair["output"])

    def predict(self, grid: Sequence[Sequence[int]], D: int) -> List[List[int]]:
        obs = self._prepare_obs(_grid_to_obs(grid, D))
        self.agent.step(obs)
        return self._predict_from_cases(grid)

    def set_confidence_threshold(self, value: float) -> None:
        self.confidence_threshold = value

    def _predict_from_cases(self, grid: Sequence[Sequence[int]]) -> List[List[int]]:
        if not self._cases:
            return [list(row) for row in grid]
        best = min(self._cases, key=lambda case: case.score(grid))
        if best.transform_score < self.confidence_threshold or not best.is_tiling:
            return [list(row) for row in grid]
        predicted = best.apply(grid)
        target_h, target_w = best.output_shape
        baseline_aligned = _align_grid(grid, target_h, target_w)
        merged: List[List[int]] = []
        for y in range(target_h):
            row: List[int] = []
            for x in range(target_w):
                if best.output_mask[y][x]:
                    row.append(predicted[y][x])
                else:
                    row.append(baseline_aligned[y][x])
            merged.append(row)
        return merged

    def _record_training_case(self, inp: List[List[int]], out: List[List[int]]) -> None:
        input_grid = [list(row) for row in inp]
        output_grid = [list(row) for row in out]
        input_h = len(input_grid)
        input_w = len(input_grid[0]) if input_grid and input_grid[0] else 0
        output_h = len(output_grid)
        output_w = len(output_grid[0]) if output_grid and output_grid[0] else 0
        value_cells: Dict[int, List[int]] = {}
        zero_targets: List[int] = []
        min_h = min(input_h, output_h)
        min_w = min(input_w, output_w)
        for y in range(min_h):
            for x in range(min_w):
                in_val = input_grid[y][x]
                out_val = output_grid[y][x]
                if out_val == 0:
                    continue
                value_cells.setdefault(in_val, []).append(out_val)
                if in_val == 0:
                    zero_targets.append(out_val)
        value_map: Dict[int, int] = {}
        for k, values in value_cells.items():
            most_common = Counter(values).most_common(1)
            if most_common:
                value_map[k] = most_common[0][0]
        nonzeros = [val for row in output_grid for val in row if val != 0]
        default_value = Counter(zero_targets).most_common(1)[0][0] if zero_targets else (nonzeros[0] if nonzeros else 0)
        mask = [[val != 0 for val in row] for row in output_grid]
        input_values = {val for row in input_grid for val in row if val != 0}
        mask_coords = _grid_mask_coords(input_grid)
        bbox = _bounding_box(mask_coords)
        value_freq = Counter(val for row in input_grid for val in row if val != 0)
        value_map_coverage = len(value_map) / max(1, len(input_values))
        consistent_mappings = sum(1 for values in value_cells.values() if len(set(values)) == 1)
        value_map_consistency = consistent_mappings / max(1, len(value_cells))
        is_tiling = False
        if input_h > 0 and input_w > 0 and output_h % input_h == 0 and output_w % input_w == 0:
            is_tiling = value_map_consistency >= 0.95
        best_transform = TRANSFORM_CANDIDATES[0]
        best_score = -1.0
        base_grid = _align_grid(input_grid, input_h, input_w)
        for transform in TRANSFORM_CANDIDATES:
            transformed_input = _apply_transform(base_grid, transform)
            trans_h = len(transformed_input)
            trans_w = len(transformed_input[0]) if trans_h else 0
            if trans_h == 0 or trans_w == 0:
                continue
            hits = 0
            total = 0
            for y in range(output_h):
                for x in range(output_w):
                    if not mask[y][x]:
                        continue
                    total += 1
                    src_y = (y + transform.offset_y) % trans_h
                    src_x = (x + transform.offset_x) % trans_w
                    if transformed_input[src_y][src_x] == output_grid[y][x]:
                        hits += 1
            if total == 0:
                continue
            score = (hits / total) + min(total, 10) * 0.01
            if score > best_score:
                best_score = score
                best_transform = transform
        if best_score < 0:
            best_score = 0.0
        case = TrainingCase(
            input_shape=(input_h, input_w),
            input_grid=input_grid,
            output_shape=(output_h, output_w),
            value_map=value_map,
            default_value=default_value,
            output_mask=mask,
            input_values=input_values,
            input_mask_coords=mask_coords,
            input_bbox=bbox,
            value_freq=value_freq,
            coord_transform=best_transform,
            value_map_coverage=value_map_coverage,
            value_map_consistency=value_map_consistency,
            is_tiling=is_tiling,
            transform_score=best_score,
            case_id=len(self._cases),
        )
        self._cases.append(case)


def _task_files(path: Path) -> Iterable[Path]:
    return sorted(p for p in path.iterdir() if p.suffix.lower() == ".json")


def run_tasks(data_dir: Path, output_dir: Path, mode: str, limit: int | None = None, warm_state: Path | None = None) -> None:
    data_dir = data_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    task_paths = list(_task_files(data_dir))
    if limit is not None:
        task_paths = task_paths[:limit]

    D = 256
    B = 16
    runner = ArcAgentRunner(D=D, B=B, periph_blocks=8)
    if warm_state:
        runner.load_state(warm_state)

    for task_path in task_paths:
        with task_path.open("r", encoding="utf-8") as fid:
            task = json.load(fid)

        training_pairs = task.get("train", [])
        runner.ingest_examples(training_pairs, D)

        results = []
        for pair in task.get("test", []):
            output_grid = runner.predict(pair["input"], D)
            results.append({"input": pair["input"], "predicted": output_grid})

        summary = {
            "task": task_path.name,
            "mode": mode,
            "train_pairs": len(training_pairs),
            "test_pairs": len(task.get("test", [])),
            "predictions": results,
        }

        out_file = output_dir / task_path.name
        with out_file.open("w", encoding="utf-8") as fid:
            json.dump(summary, fid, indent=2)

        print(f"Wrote predictions for {task_path.name} ({len(results)} outputs)")


def main() -> None:
    parser = argparse.ArgumentParser(description="ARC-AGI-2 runner using the NUPCA3 agent.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing ARC JSON tasks.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Where to dump prediction JSONs.")
    parser.add_argument("--mode", choices=["training", "evaluation"], default="training")
    parser.add_argument("--limit", type=int, default=None, help="How many tasks to process.")
    parser.add_argument("--warm-state", type=Path, default=None, help="Optional path to persisted agent state.")
    args = parser.parse_args()

    run_tasks(
        args.data_dir,
        args.output_dir,
        mode=args.mode,
        limit=args.limit,
        warm_state=args.warm_state,
    )


if __name__ == "__main__":
    main()
