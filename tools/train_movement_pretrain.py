"""
train_movement_pretrain.py

Pre-train the NUPCA3 agent on simulated moving environments before tackling ARC-
AGI-2. This script runs through a series of “movement phase” curricula, saves the
agent state for persistence, and exposes checkpoints that downstream runners can
re-use.

1. Train on a narrow “building-block” world (simple grid, few colors/shapes)
   so the agent learns the core peripheral transport and block-level residuals.
2. Increase difficulty (more colors/shapes, quicker changes) so the agent
   exposes itself to richer translations and candidate gating before ARC evidence.

Each phase logs candidate counts + transport delta statistics to `stdout`.
Running script with the same `--persist` path resumes from the saved state.
"""

from __future__ import annotations

import argparse
import pickle
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

from nupca3.agent import NUPCA3Agent
from nupca3.config import AgentConfig
from nupca3.memory.completion import complete, cue_from_env_obs
from nupca3.types import Action, EnvObs
from test import MovingColorShapeWorld


def build_obs(vec: Sequence[float]) -> EnvObs:
    flat = np.asarray(vec, dtype=float).reshape(-1)
    nonzero = np.nonzero(flat)[0]
    x_partial: Dict[int, float] = {}
    for idx in nonzero:
        x_partial[int(idx)] = float(flat[int(idx)])
    return EnvObs(
        x_partial=x_partial,
        opp=0.0,
        danger=0.0,
        x_full=flat.copy(),
        pos_dims=set(int(idx) for idx in nonzero),
    )


def _transform_coords(x: int, y: int, side: int, transform: str | None) -> tuple[int, int]:
    """Map (x,y) through the chosen transform around the grid center."""
    if side <= 0 or transform is None:
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


ROTATE_COMMANDS = ("rotate_cw", "rotate_ccw", "mirror_x", "mirror_y")
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
CONTROL_COMMANDS = (
    "idle",
    *ROTATE_COMMANDS,
    *PUSH_OFFSETS.keys(),
    *PULL_OFFSETS.keys(),
)
COMMAND_OFFSETS = {**PUSH_OFFSETS, **PULL_OFFSETS}


class CompositeMovingColorShapeWorld:
    """Two simultaneous shapes (background + foreground) moving over the same grid."""

    def __init__(
        self,
        side: int,
        n_colors: int,
        n_shapes: int,
        bg_color_span: int,
        fg_color_span: int,
        bg_shape_span: int,
        fg_shape_span: int,
        seed: int,
        p_color_shift_bg: float = 0.02,
        p_color_shift_fg: float = 0.1,
        p_shape_shift_bg: float = 0.02,
        p_shape_shift_fg: float = 0.1,
        dense: bool = False,
        dense_sigma: float = 1.5,
        bg_speed_range: tuple[int, int] = (1, 1),
        fg_speed_range: tuple[int, int] = (1, 1),
        occlusion_prob: float = 0.0,
        occlusion_size_range: tuple[int, int] = (0, 0),
    ):
        assert bg_color_span + fg_color_span == n_colors, "color spans must add to n_colors"
        assert bg_shape_span + fg_shape_span == n_shapes, "shape spans must add to n_shapes"
        self.side = int(side)
        self.n_colors = int(n_colors)
        self.n_shapes = int(n_shapes)
        self.bg_color_span = int(bg_color_span)
        self.fg_color_span = int(fg_color_span)
        self.bg_shape_span = int(bg_shape_span)
        self.fg_shape_span = int(fg_shape_span)
        self.bg_color_offset = 0
        self.fg_color_offset = self.bg_color_span
        self.bg_shape_offset = 0
        self.fg_shape_offset = self.bg_shape_span
        self.rng = np.random.default_rng(int(seed))
        self.p_color_shift_bg = float(p_color_shift_bg)
        self.p_color_shift_fg = float(p_color_shift_fg)
        self.p_shape_shift_bg = float(p_shape_shift_bg)
        self.p_shape_shift_fg = float(p_shape_shift_fg)
        self.dense = bool(dense)
        self.dense_sigma = float(dense_sigma)

        self.bg_x = 0
        self.bg_y = 0
        self.bg_color = 0
        self.bg_shape = 0
        self.fg_x = 0
        self.fg_y = 0
        self.fg_color = 0
        self.fg_shape = 0
        self.bg_last_dx = 0
        self.bg_last_dy = 0
        self.fg_last_dx = 0
        self.fg_last_dy = 0
        self.bg_vx = 0
        self.bg_vy = 0
        self.fg_vx = 0
        self.fg_vy = 0
        self.bg_speed_range = tuple(int(v) for v in bg_speed_range)
        self.bg_speed_min = max(1, min(self.bg_speed_range))
        self.bg_speed_max = max(self.bg_speed_min, max(self.bg_speed_range))
        self.fg_speed_range = tuple(int(v) for v in fg_speed_range)
        self.fg_speed_min = max(1, min(self.fg_speed_range))
        self.fg_speed_max = max(self.fg_speed_min, max(self.fg_speed_range))
        self.occlusion_prob = float(occlusion_prob)
        occ_min = max(0, min(occlusion_size_range))
        occ_max = max(occ_min, max(occlusion_size_range))
        self.occlusion_min = occ_min
        self.occlusion_max = occ_max
        self.last_occlusion: tuple[int, int, int] | None = None

    def _sample_velocity(self, speed_min: int, speed_max: int) -> tuple[int, int]:
        while True:
            dx = int(self.rng.integers(-1, 2))
            dy = int(self.rng.integers(-1, 2))
            if dx != 0 or dy != 0:
                break
        speed = int(self.rng.integers(speed_min, speed_max + 1))
        return dx * speed, dy * speed

    @property
    def D(self) -> int:
        return self.side * self.side * (self.n_colors + self.n_shapes)

    def reset(self) -> np.ndarray:
        self.bg_x = int(self.rng.integers(self.side))
        self.bg_y = int(self.rng.integers(self.side))
        self.bg_color = int(self.rng.integers(self.bg_color_span))
        self.bg_shape = int(self.rng.integers(self.bg_shape_span))
        self.fg_x = int(self.rng.integers(self.side))
        self.fg_y = int(self.rng.integers(self.side))
        self.fg_color = int(self.rng.integers(self.fg_color_span))
        self.fg_shape = int(self.rng.integers(self.fg_shape_span))
        self.bg_vx, self.bg_vy = self._sample_velocity(self.bg_speed_min, self.bg_speed_max)
        self.fg_vx, self.fg_vy = self._sample_velocity(self.fg_speed_min, self.fg_speed_max)
        self.bg_last_dx = 0
        self.bg_last_dy = 0
        self.fg_last_dx = 0
        self.fg_last_dy = 0
        return self._encode()

    def step(self) -> np.ndarray:
        self.bg_last_dx = self.bg_vx
        self.bg_last_dy = self.bg_vy
        self.bg_x = (self.bg_x + self.bg_vx) % self.side
        self.bg_y = (self.bg_y + self.bg_vy) % self.side
        self.fg_last_dx = self.fg_vx
        self.fg_last_dy = self.fg_vy
        self.fg_x = (self.fg_x + self.fg_vx) % self.side
        self.fg_y = (self.fg_y + self.fg_vy) % self.side

        if self.rng.random() < self.p_color_shift_bg:
            self.bg_color = int(self.rng.integers(self.bg_color_span))
        if self.rng.random() < self.p_shape_shift_bg:
            self.bg_shape = int(self.rng.integers(self.bg_shape_span))
        if self.rng.random() < self.p_color_shift_fg:
            self.fg_color = int(self.rng.integers(self.fg_color_span))
        if self.rng.random() < self.p_shape_shift_fg:
            self.fg_shape = int(self.rng.integers(self.fg_shape_span))

        return self._encode()

    def _encode(self) -> np.ndarray:
        vec = np.zeros(self.D, dtype=float)
        color_offset = 0
        shape_offset = self.side * self.side * self.n_colors
        # Background contribution
        cell_bg = self.bg_y * self.side + self.bg_x
        if self.bg_color_span > 0:
            vec[color_offset + cell_bg * self.n_colors + self.bg_color_offset + self.bg_color] += 1.0
        if self.bg_shape_span > 0:
            vec[shape_offset + cell_bg * self.n_shapes + self.bg_shape_offset + self.bg_shape] += 1.0
        # Foreground contribution
        cell_fg = self.fg_y * self.side + self.fg_x
        if self.fg_color_span > 0:
            vec[color_offset + cell_fg * self.n_colors + self.fg_color_offset + self.fg_color] += 1.0
        if self.fg_shape_span > 0:
            vec[shape_offset + cell_fg * self.n_shapes + self.fg_shape_offset + self.fg_shape] += 1.0
        self._maybe_occlude(vec)
        return vec

    def _maybe_occlude(self, vec: np.ndarray) -> None:
        self.last_occlusion = None
        if self.occlusion_prob <= 0.0 or self.occlusion_max <= 0:
            return
        if self.rng.random() >= self.occlusion_prob:
            return
        size = int(self.rng.integers(self.occlusion_min, self.occlusion_max + 1))
        if size <= 0 or size > self.side:
            return
        oy = int(self.rng.integers(0, self.side - size + 1))
        ox = int(self.rng.integers(0, self.side - size + 1))
        color_offset = 0
        shape_offset = self.side * self.side * self.n_colors
        for dy in range(size):
            for dx in range(size):
                cell = (oy + dy) * self.side + (ox + dx)
                if self.n_colors > 0:
                    start = color_offset + cell * self.n_colors
                    vec[start : start + self.n_colors] = 0.0
                if self.n_shapes > 0:
                    start = shape_offset + cell * self.n_shapes
                    vec[start : start + self.n_shapes] = 0.0
        self.last_occlusion = (ox, oy, size)

    def apply_transform(self, transform: str | None) -> None:
        """Apply a discrete transform (rotate/mirror) to both shapes."""
        if not transform:
            return
        self.bg_x, self.bg_y = _transform_coords(self.bg_x, self.bg_y, self.side, transform)
        self.fg_x, self.fg_y = _transform_coords(self.fg_x, self.fg_y, self.side, transform)
        self.last_occlusion = None

    def last_background_move(self) -> Tuple[int, int]:
        return (self.bg_last_dx, self.bg_last_dy)

    def last_foreground_move(self) -> Tuple[int, int]:
        return (self.fg_last_dx, self.fg_last_dy)

    def default_control_target(self) -> str:
        return "fg"

    def get_object_position(self, target: str) -> tuple[int, int]:
        if target == "fg":
            return (self.fg_x, self.fg_y)
        if target == "bg":
            return (self.bg_x, self.bg_y)
        return (self.fg_x, self.fg_y)

    def apply_control_command(self, command: str, target: str) -> bool:
        if target not in {"fg", "bg"}:
            return False
        if command in ROTATE_COMMANDS:
            self._rotate_object(target, command)
            self.last_occlusion = None
            return True
        if command in COMMAND_OFFSETS:
            dx, dy = COMMAND_OFFSETS[command]
            self._move_object(target, dx, dy)
            self.last_occlusion = None
            return True
        return False

    def _move_object(self, target: str, dx: int, dy: int) -> None:
        if target == "fg":
            self.fg_x = (self.fg_x + dx) % self.side
            self.fg_y = (self.fg_y + dy) % self.side
            return
        if target == "bg":
            self.bg_x = (self.bg_x + dx) % self.side
            self.bg_y = (self.bg_y + dy) % self.side

    def _rotate_object(self, target: str, transform: str) -> None:
        if target == "fg":
            self.fg_x, self.fg_y = _transform_coords(self.fg_x, self.fg_y, self.side, transform)
            return
        if target == "bg":
            self.bg_x, self.bg_y = _transform_coords(self.bg_x, self.bg_y, self.side, transform)

class MovementTrainer:
    CONTROL_COMMANDS = CONTROL_COMMANDS

    def __init__(self, *, agent: NUPCA3Agent, log_interval: int = 50):
        self.agent = agent
        self.log_interval = log_interval
        self.control_x = 0
        self.control_y = 0
        self.control_pos: Optional[tuple[int, int]] = None
        self.control_command = "idle"
        self.control_target: str | None = None

    def run_phase(self, *, world: object, steps: int, phase_name: str) -> None:
        print(f"\n--- Starting phase {phase_name}: {steps} steps ---")
        vec = world.reset()
        side = getattr(world, "side", 0)
        self.control_command = "idle"
        self.control_target = self._select_control_target(world)
        self._update_control_pos(world)
        vec, control_pos = self._apply_control(vec, side, -1, control_pos=self.control_pos)
        self.control_pos = control_pos
        obs = build_obs(vec)
        action, trace = self.agent.step(obs)
        self.control_command = self._interpret_action(action)
        transport_matches = 0
        candidate_counts: list[int] = []
        pr_diffs: list[float] = []
        rp_diffs: list[float] = []
        start_info = self._describe_world(world, self.control_pos, self.control_command)
        print(f"[{phase_name}] start constellations: {start_info}")

        transform_effects: list[tuple[str, tuple[tuple[str, int], ...], tuple[tuple[str, int], ...]]] = []
        control_hist = Counter()
        for step in range(steps):
            current_command = self.control_command
            control_hist[current_command] += 1
            pre_state = self._snapshot_positions(world)
            command_applied = self._apply_control_command(world, current_command)
            post_state = self._snapshot_positions(world)
            if command_applied and pre_state != post_state:
                transform_effects.append((current_command, pre_state, post_state))
            vec = world.step()
            self._update_control_pos(world)
            vec, control_pos = self._apply_control(vec, side, step, control_pos=self.control_pos)
            self.control_pos = control_pos
            obs = build_obs(vec)
            action, trace = self.agent.step(obs)
            self.control_command = self._interpret_action(action)
            tdelta = tuple(trace.get("transport_delta", (0, 0)))
            if tdelta != (0, 0):
                transport_matches += 1
            candidate_counts.append(int(trace.get("permit_param_info", {}).get("candidate_count", 0)))
            pr_diff, rp_diff = self._probe_completion(obs)
            pr_diffs.append(pr_diff)
            rp_diffs.append(rp_diff)

            if (step + 1) % self.log_interval == 0:
                last_candidates = candidate_counts[-self.log_interval:]
                avg_candidate = sum(last_candidates) / len(last_candidates)
                avg_pr = sum(pr_diffs[-self.log_interval:]) / len(pr_diffs[-self.log_interval:])
                avg_rp = sum(rp_diffs[-self.log_interval:]) / len(rp_diffs[-self.log_interval:])
                print(
                    f"[{phase_name}] step={step+1} transport_delta={tdelta} "
                    f"avg_cand_last={avg_candidate:.1f} pr_diff={avg_pr:.4f} rp_diff={avg_rp:.4f}"
                )

        total_actions = sum(control_hist.values())
        nonidle = total_actions - control_hist.get("idle", 0)
        effective = sum(1 for entry in transform_effects if entry[1] != entry[2])
        print(
            f"[{phase_name}] control histogram={dict(control_hist)} "
            f"nonidle_ratio={(nonidle/total_actions if total_actions else 0.0):.3f} "
            f"control_effective={effective}/{nonidle if nonidle else 1}"
        )
        end_info = self._describe_world(world, self.control_pos, self.control_command)
        avg_pr_total = sum(pr_diffs) / len(pr_diffs) if pr_diffs else 0.0
        avg_rp_total = sum(rp_diffs) / len(rp_diffs) if rp_diffs else 0.0
        print(
            f"[{phase_name}] phase complete transport_delta_nonzero_steps={transport_matches} "
            f"avg_candidates={sum(candidate_counts)/len(candidate_counts):.2f} "
            f"pr_diff_avg={avg_pr_total:.4f} rp_diff_avg={avg_rp_total:.4f}"
        )
        print(f"[{phase_name}] constellations: {start_info} → {end_info}")

    def _probe_completion(self, obs: EnvObs) -> tuple[float, float]:
        state = self.agent.state
        cfg = self.agent.cfg
        cue = cue_from_env_obs(obs)
        x_perc, _, _ = complete(cue, mode="perception", state=state, cfg=cfg)
        x_rec, _, _ = complete(cue, mode="recall", state=state, cfg=cfg)
        x_pred, _, _ = complete(None, mode="prediction", state=state, cfg=cfg)
        pr_diff = float(np.linalg.norm(x_perc - x_rec))
        rp_diff = float(np.linalg.norm(x_rec - x_pred))
        return pr_diff, rp_diff

    def _interpret_action(self, action: Action) -> str:
        idx = int(action) % len(self.CONTROL_COMMANDS) if self.CONTROL_COMMANDS else 0
        return self.CONTROL_COMMANDS[idx]

    def _apply_control_command(self, world: object, command: str) -> bool:
        if command == "idle" or self.control_target is None:
            self._update_control_pos(world)
            return False
        if hasattr(world, "apply_control_command"):
            applied = bool(world.apply_control_command(command, self.control_target))
            self._update_control_pos(world)
            return applied
        self._update_control_pos(world)
        return False

    def _select_control_target(self, world: object) -> str | None:
        if hasattr(world, "default_control_target"):
            return world.default_control_target()
        return None

    def _update_control_pos(self, world: object) -> None:
        if self.control_target and hasattr(world, "get_object_position"):
            pos = world.get_object_position(self.control_target)
            self.control_pos = (int(pos[0]), int(pos[1]))
        elif self.control_pos is None:
            self.control_pos = (0, 0)


    def _snapshot_positions(self, world: object) -> tuple[tuple[str, int], ...]:
        fields = []
        for attr in ("bg_x", "bg_y", "fg_x", "fg_y", "x", "y"):
            if hasattr(world, attr):
                fields.append((attr, int(getattr(world, attr))))
        return tuple(fields)


    def _apply_control(
        self,
        vec: np.ndarray,
        side: int,
        step: int,
        control_pos: tuple[int, int] | None = None,
    ) -> tuple[np.ndarray, tuple[int, int]]:
        if control_pos is None:
            dx, dy = self._control_move(step)
            if side > 0:
                self.control_x = (self.control_x + dx) % side
                self.control_y = (self.control_y + dy) % side
        else:
            x, y = control_pos
            if side > 0:
                self.control_x = x % side
                self.control_y = y % side
            else:
                self.control_x = int(x)
                self.control_y = int(y)
        control_layer = np.zeros(max(side * side, 0), dtype=float)
        if side > 0:
            idx = self.control_y * side + self.control_x
            control_layer[idx] = 1.0
        padded = np.concatenate([np.asarray(vec, dtype=float).reshape(-1), control_layer])
        return padded, (self.control_x, self.control_y)

    def _control_move(self, step: int) -> tuple[int, int]:
        if step < 0:
            return 0, 0
        pattern = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        return pattern[step % len(pattern)]

    def _describe_world(
        self,
        world: object,
        control_pos: Optional[tuple[int, int]] = None,
        control_command: str | None = None,
    ) -> str:
        descriptors: list[str] = []
        if hasattr(world, "x") and hasattr(world, "y"):
            descriptors.append(f"pos=({getattr(world,'x')},{getattr(world,'y')})")
        if hasattr(world, "color"):
            descriptors.append(f"color={getattr(world,'color')}")
        if hasattr(world, "shape"):
            descriptors.append(f"shape={getattr(world,'shape')}")
        if hasattr(world, "bg_x"):
            descriptors.append(
                f"bg=({getattr(world,'bg_x')},{getattr(world,'bg_y')})"
                f" c={getattr(world,'bg_color')} s={getattr(world,'bg_shape')}"
            )
        if hasattr(world, "fg_x"):
            descriptors.append(
                f"fg=({getattr(world,'fg_x')},{getattr(world,'fg_y')})"
                f" c={getattr(world,'fg_color')} s={getattr(world,'fg_shape')}"
            )
        occ = getattr(world, "last_occlusion", None)
        if occ:
            descriptors.append(f"occ=({occ[0]},{occ[1]}) size={occ[2]}")
        if self.control_target is not None:
            descriptors.append(f"control_target={self.control_target}")
        if control_pos is not None:
            descriptors.append(f"control=({control_pos[0]},{control_pos[1]})")
        if control_command is not None:
            descriptors.append(f"control_cmd={control_command}")
        return " | ".join(descriptors) if descriptors else "world-state=unknown"


def persist_state(agent: NUPCA3Agent, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fid:
        pickle.dump(agent.state, fid)
    print(f"Persisted state to {path}")


def load_state(agent: NUPCA3Agent, path: Path) -> bool:
    if not path.exists():
        return False
    with path.open("rb") as fid:
        state = pickle.load(fid)
    agent.state = state
    print(f"Loaded persisted state from {path}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-train NUPCA3 on moving environments.")
    parser.add_argument("--persist", type=Path, default=Path("persistent_agent.pkl"), help="Where to save/load agent state.")
    parser.add_argument("--phase1-steps", type=int, default=400, help="Steps for the building-block phase.")
    parser.add_argument("--phase2-steps", type=int, default=800, help="Steps for the advanced phase.")
    parser.add_argument("--log-interval", type=int, default=50, help="Log cadence of candidate stats.")
    args = parser.parse_args()

    cfg = AgentConfig(D=256, B=16, fovea_blocks_per_step=8, periph_blocks=4, periph_bins=2)
    agent = NUPCA3Agent(cfg)
    if not load_state(agent, args.persist):
        agent.reset(seed=0)

    trainer = MovementTrainer(agent=agent, log_interval=args.log_interval)

    phase1_world = MovingColorShapeWorld(
        side=4,
        n_colors=2,
        n_shapes=2,
        seed=0,
        p_color_shift=0.05,
        p_shape_shift=0.05,
        dense=True,
        dense_sigma=1.2,
        periph_bins=2,
        speed_range=(1, 2),
        occlusion_prob=0.1,
        occlusion_size_range=(1, 2),
    )
    trainer.run_phase(world=phase1_world, steps=args.phase1_steps, phase_name="block_discovery")

    phase2_world = CompositeMovingColorShapeWorld(
        side=6,
        n_colors=4,
        n_shapes=4,
        bg_color_span=2,
        fg_color_span=2,
        bg_shape_span=2,
        fg_shape_span=2,
        seed=1,
        p_color_shift_bg=0.05,
        p_shape_shift_bg=0.02,
        p_color_shift_fg=0.25,
        p_shape_shift_fg=0.25,
        dense=True,
        dense_sigma=1.5,
        bg_speed_range=(1, 2),
        fg_speed_range=(1, 3),
        occlusion_prob=0.15,
        occlusion_size_range=(1, 3),
    )
    trainer.run_phase(world=phase2_world, steps=args.phase2_steps, phase_name="multi_shape_permanence")

    persist_state(agent, args.persist)


if __name__ == "__main__":
    main()
