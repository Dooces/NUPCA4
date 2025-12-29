import json
import logging
import numpy as np
from pathlib import Path
import sys
import select


try:
    import termios
    import tty
except Exception:  # pragma: no cover
    termios = None
    tty = None

try:
    import msvcrt  # type: ignore
except Exception:  # pragma: no cover
    msvcrt = None

from dataclasses import dataclass

from rich.live import Live
from rich.panel import Panel
from rich.columns import Columns
from rich.text import Text
from rich.console import Group

from nupca3.agent import NUPCA3Agent
from nupca3.persist import persist_state
from nupca3.config import AgentConfig
from nupca3.types import EnvObs, Action, CurriculumCommand
from nupca3.geometry.fovea import make_observation_set

# ============================================================
# OPTIONS (edit these)
# ============================================================
HEIGHT = 24
WIDTH = 24      
FPS = 120

MAX_SHAPES = 8
MIN_SHAPES = 1

STREAK_STEPS = 20
ADD_DELTA_THRESHOLD = 0.09

# High-streak threshold (requested): if mean_delta stays ABOVE this for STREAK_STEPS -> remove a shape
HIGH_STREAK_THRESHOLD = 0.20

# Which shape-types are "elastic" (participate in object-object collisions).
# Green squares + magenta circles are elastic; cyan rectangles are non-elastic occluders (foreground).
ELASTIC_KINDS = {"square", "circle"}

# Fovea budget (scaled for 32x64)
BUDGET = 256
PROBE_K = 64
MAX_LOST = 240

# ============================================================
# Codes / styling
# ============================================================
# 0 empty
# 1 square shell, 2 square core
# 3 rect shell,   4 rect core
# 5 circle shell, 6 circle core
STYLE = {
    0: (".", "dim"),
    1: ("s", "green"),
    2: ("S", "bright_green"),
    3: ("r", "cyan"),
    4: ("R", "bright_cyan"),
    5: ("c", "magenta"),
    6: ("C", "bright_magenta"),
}

def _glyph_style(v: int):
    return STYLE.get(int(v), ("?", "yellow"))

def grid_text(
    vec: np.ndarray,
    H: int,
    W: int,
    *,
    unknown_mask=None,
    highlight_idx=None,
    unknown_as_blank=False,
    underlay_mask=None,
):
    g = vec.reshape(H, W)
    txt = Text()
    hi = set(highlight_idx) if highlight_idx is not None else set()

    for y in range(H):
        for x in range(W):
            i = y * W + x

            if unknown_mask is not None and unknown_mask[y, x]:
                if unknown_as_blank:
                    ch, st = " ", "black"
                else:
                    ch, st = "·", "bright_black"
                txt.append(ch, style=(st + " on grey15") if i in hi else st)
                continue

            v = int(g[y, x])
            ch, st = _glyph_style(v)

            if underlay_mask is not None and underlay_mask[y, x]:
                st = st + " on navy_blue"

            txt.append(ch, style=(st + " on grey15") if i in hi else st)

        if y != H - 1:
            txt.append("\n")
    return txt


# ============================================================
# Environment: deterministic motion + semantic occlusion + curriculum add/remove
# ============================================================

@dataclass
class Obj:
    kind: str
    z: float
    x: float
    y: float
    vx: float
    vy: float
    w: int = 0
    h: int = 0
    r: int = 0
    shell_code: int = 0
    core_code: int = 0

    @property
    def radius(self) -> float:
        if self.kind == "circle":
            return float(self.r)
        return 0.5 * float(max(self.w, self.h))

    @property
    def elastic(self) -> bool:
        return self.kind in ELASTIC_KINDS


class SemanticOcclusionEnv:
    """
    Latent objects persist regardless of visibility.
    Visible frame uses z-order overwrite (occlusion).
    Motion is deterministic: constant velocities + wall bounces.
    Elastic collisions only apply to selected kinds (ELASTIC_KINDS).
    Curriculum is enacted by agent-issued Action commands (ADD_SHAPE / REMOVE_SHAPE).
    """
    def __init__(self, H=HEIGHT, W=WIDTH, seed=1):
        self.H = int(H)
        self.W = int(W)
        self.rng = np.random.default_rng(seed)
        self.t = 0
        self.objs: list[Obj] = []

    def apply_action(self, action: Action):
        if action.command == CurriculumCommand.ADD_SHAPE and len(self.objs) < MAX_SHAPES:
            self._spawn_obj()
        elif action.command == CurriculumCommand.REMOVE_SHAPE and len(self.objs) > MIN_SHAPES:
            self._remove_one()

    def _new_velocity(self):
        vx = float(self.rng.choice([-0.75, -0.5, 0.5, 0.75]))
        vy = float(self.rng.choice([-0.75, -0.5, 0.5, 0.75]))
        if abs(vx) + abs(vy) < 0.9:
            vx = 0.75 if vx >= 0 else -0.75
        return vx, vy

    def _spawn_obj(self):
        # Favor occluders (rectangles) as complexity increases
        if len(self.objs) >= 2:
            kinds = ["square", "circle", "rect", "rect"]
        else:
            kinds = ["square", "circle", "rect"]
        kind = str(self.rng.choice(kinds))

        if kind == "square":
            s = int(self.rng.integers(3, 7))  # 3..6
            w, h, r = s, s, 0
            shell_code, core_code = 1, 2
        elif kind == "rect":
            w = int(self.rng.integers(6, 14))  # 6..13
            h = int(self.rng.integers(3, 8))   # 3..7
            if w == h:
                w += 1
            r = 0
            shell_code, core_code = 3, 4
        else:
            r = int(self.rng.integers(3, 6))  # 3..5
            w, h = 0, 0
            shell_code, core_code = 5, 6

        H, W = self.H, self.W
        if kind == "circle":
            x = float(self.rng.uniform(r, W - 1 - r))
            y = float(self.rng.uniform(r, H - 1 - r))
        else:
            x = float(self.rng.uniform(0, W - w))
            y = float(self.rng.uniform(0, H - h))

        vx, vy = self._new_velocity()

        # Rectangles as FOREGROUND occluders (high z); others lower z
        if kind == "rect":
            z = 100.0 + float(self.rng.uniform(0.0, 1.0))
        else:
            z = float(self.rng.uniform(0.0, 10.0))

        self.objs.append(
            Obj(kind=kind, z=z, x=x, y=y, vx=vx, vy=vy, w=w, h=h, r=r,
                shell_code=shell_code, core_code=core_code)
        )
        self.objs.sort(key=lambda o: o.z)

    def _remove_one(self):
        if len(self.objs) <= MIN_SHAPES:
            return False
        # Prefer removing occluders first
        for i in range(len(self.objs) - 1, -1, -1):
            if self.objs[i].kind == "rect":
                del self.objs[i]
                return True
        self.objs.sort(key=lambda o: o.z)
        self.objs.pop()
        return True

    def _bounce_walls(self, o: Obj):
        H, W = self.H, self.W
        if o.kind == "circle":
            r = o.r
            if o.x < r:
                o.x = r; o.vx = abs(o.vx)
            if o.x > W - 1 - r:
                o.x = W - 1 - r; o.vx = -abs(o.vx)
            if o.y < r:
                o.y = r; o.vy = abs(o.vy)
            if o.y > H - 1 - r:
                o.y = H - 1 - r; o.vy = -abs(o.vy)
        else:
            if o.x < 0:
                o.x = 0; o.vx = abs(o.vx)
            if o.x > W - o.w:
                o.x = W - o.w; o.vx = -abs(o.vx)
            if o.y < 0:
                o.y = 0; o.vy = abs(o.vy)
            if o.y > H - o.h:
                o.y = H - o.h; o.vy = -abs(o.vy)

    def _elastic_collisions(self):
        els = [o for o in self.objs if o.elastic]
        n = len(els)
        if n < 2:
            return
        for i in range(n):
            for j in range(i + 1, n):
                a, b = els[i], els[j]
                dx = b.x - a.x
                dy = b.y - a.y
                dist2 = dx * dx + dy * dy
                rad = a.radius + b.radius
                if dist2 <= (rad * rad) and dist2 > 1e-9:
                    dist = float(np.sqrt(dist2))
                    nx, ny = dx / dist, dy / dist
                    rvx = b.vx - a.vx
                    rvy = b.vy - a.vy
                    rel = rvx * nx + rvy * ny
                    if rel >= 0:
                        continue
                    imp = -2.0 * rel / 2.0
                    a.vx -= imp * nx; a.vy -= imp * ny
                    b.vx += imp * nx; b.vy += imp * ny
                    overlap = rad - dist
                    a.x -= 0.5 * overlap * nx; a.y -= 0.5 * overlap * ny
                    b.x += 0.5 * overlap * nx; b.y += 0.5 * overlap * ny

    def rasterize_obj(self, o: Obj):
        H, W = self.H, self.W
        shell = np.zeros((H, W), dtype=bool)
        core = np.zeros((H, W), dtype=bool)

        if o.kind in ("square", "rect"):
            x0 = int(round(o.x))
            y0 = int(round(o.y))
            x1 = min(W, x0 + o.w)
            y1 = min(H, y0 + o.h)
            if x0 < 0 or y0 < 0 or x0 >= W or y0 >= H:
                return shell, core

            shell[y0:y1, x0] = True
            shell[y0:y1, x1 - 1] = True
            shell[y0, x0:x1] = True
            shell[y1 - 1, x0:x1] = True

            if (x1 - x0) >= 3 and (y1 - y0) >= 3:
                core[y0 + 1:y1 - 1, x0 + 1:x1 - 1] = True

        else:
            cx, cy, r = o.x, o.y, o.r
            yy, xx = np.mgrid[0:H, 0:W]
            dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            filled = dist <= (r + 0.25)
            inner = dist <= (max(0.0, r - 1.0) + 0.25)
            shell = filled & (~inner)
            core = inner

        return shell, core

    def render_visible(self):
        H, W = self.H, self.W
        grid = np.zeros((H, W), dtype=int)
        for o in sorted(self.objs, key=lambda x: x.z):
            shell, core = self.rasterize_obj(o)
            grid[core] = o.core_code
            grid[shell] = o.shell_code
        return grid.ravel()

    def step_physics(self):
        self.t += 1
        for o in self.objs:
            o.x += o.vx
            o.y += o.vy
            self._bounce_walls(o)
        self._elastic_collisions()
        for o in self.objs:
            self._bounce_walls(o)



@dataclass
class StepOut:
    action: Action
    pred: np.ndarray
    mismatch: np.ndarray
    seen: np.ndarray
    unknown: np.ndarray
    fidx: np.ndarray


# ============================================================
# Agent: NUPCA3 wrapper for grid observations
# ============================================================


class NUPCA3GridAgent:
    def __init__(self, *, H=HEIGHT, W=WIDTH, seed=2):
        self.H = int(H)
        self.W = int(W)
        self.D = self.H * self.W

        self.cfg = AgentConfig(
            D=self.D,
            B=self.D,
            fovea_blocks_per_step=BUDGET,
            fovea_log_every=0,
            fovea_shape="circle",
            # Strong routing so the disk centers on salient nonzero regions.
            fovea_routing_weight=6.0,
            fovea_routing_ema=0.6,
            # Keep the footprint contiguous (no extra probe blocks outside the disk).
            motion_probe_blocks=0,
            # Geometry metadata so the agent can enforce a circular fovea over a
            # potentially non-square grid (e.g., 32×64, motion_probe_blocks=0, fovea_shape="circle", fovea_routing_weight=6.0, fovea_routing_ema=0.6). The harness still passes
            # the full frame; the agent's fovea policy decides what is observed.
            grid_width=self.W,
            grid_height=self.H,
            grid_channels=1,
            grid_base_dim=self.D,
        )
        self.agent = NUPCA3Agent(self.cfg)
        self.agent.reset(seed=int(seed))
        self.underlay_mask = np.zeros((self.H, self.W), dtype=bool)
        self.logger = logging.getLogger("nupca3_grid_harness")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.FileHandler("output.txt", mode="w", encoding="utf-8")
            formatter = logging.Formatter("%(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.propagate = False

    def _log_step(self, *, action, trace, obs_vec, pred_display, mismatch, idx):
        state = self.agent.state
        learn_cache = getattr(state, "learn_cache", None)
        working_set = getattr(learn_cache, "A_t", None) if learn_cache is not None else None
        active_nodes = list(getattr(working_set, "active", [])) if working_set is not None else []
        active_weights = dict(getattr(working_set, "weights", {})) if working_set is not None else {}
        residual_stats = getattr(state, "residual_stats", {}) or {}

        node_info = []
        nodes = getattr(getattr(state, "library", None), "nodes", {}) or {}
        for node_id in active_nodes:
            node = nodes.get(node_id)
            if node is None:
                node_info.append({"node_id": int(node_id), "status": "missing"})
                continue
            footprint = int(getattr(node, "footprint", -1))
            footprint_stats = residual_stats.get(footprint)
            footprint_mean_abs = None
            if footprint_stats is not None:
                mean_ema = np.asarray(getattr(footprint_stats, "mean_ema", np.zeros(0)), dtype=float)
                if mean_ema.size:
                    footprint_mean_abs = float(np.mean(np.abs(mean_ema)))
            node_info.append(
                {
                    "node_id": int(node_id),
                    "weight": float(active_weights.get(node_id, 0.0)),
                    "footprint": footprint,
                    "footprint_mean_abs_residual": footprint_mean_abs,
                    "parents": sorted(int(p) for p in getattr(node, "parents", set())),
                    "children": sorted(int(c) for c in getattr(node, "children", set())),
                    "is_anchor": bool(getattr(node, "is_anchor", False)),
                }
            )
        learning_candidates = trace.get("learning_candidates", {}) if isinstance(trace, dict) else {}
        permit_param_info = trace.get("permit_param_info", {}) if isinstance(trace, dict) else {}
        cand_count = int(learning_candidates.get("candidates", 0)) if isinstance(learning_candidates, dict) else 0
        clamped_count = int(learning_candidates.get("clamped", 0)) if isinstance(learning_candidates, dict) else 0
        updated_count = int(permit_param_info.get("updated", 0)) if isinstance(permit_param_info, dict) else 0

        learning_active = updated_count > 0
        if not bool(trace.get("permit_param", False)) if isinstance(trace, dict) else False:
            learning_reason = "permit_closed"
        elif cand_count <= 0:
            learning_reason = "no_candidate_nodes"
        elif updated_count <= 0:
            learning_reason = "all_clamped_or_lr_zero"
        else:
            learning_reason = "updated_nodes"

        log_payload = {
            "t": int(trace.get("t", getattr(state, "t", 0))) if isinstance(trace, dict) else int(getattr(state, "t", 0)),
            "action": int(action.value),
            "rest": bool(trace.get("rest", False)) if isinstance(trace, dict) else False,
            "permit_param": bool(trace.get("permit_param", False)) if isinstance(trace, dict) else False,
            "errors": {
                "mismatch_mean": float(np.mean(mismatch)),
                "mismatch_max": float(np.max(mismatch)),
                "prior_obs_mae": float(trace.get("prior_obs_mae", float("nan"))) if isinstance(trace, dict) else float("nan"),
                "posterior_obs_mae": float(trace.get("posterior_obs_mae", float("nan"))) if isinstance(trace, dict) else float("nan"),
            },
            "dag_constellation": {
                "active_count": int(len(active_nodes)),
                "active_nodes": node_info,
            },
            "abstraction": {
                "edits_processed": int(trace.get("edits_processed", 0)) if isinstance(trace, dict) else 0,
                "rest_queue_len": int(trace.get("rest_queue_len", 0)) if isinstance(trace, dict) else 0,
                "rest_cycles_needed": int(trace.get("rest_cycles_needed", 0)) if isinstance(trace, dict) else 0,
            },
            "learning": {
                "active": learning_active,
                "reason": learning_reason,
                "candidates": learning_candidates,
                "candidate_count": cand_count,
                "clamped_count": clamped_count,
                "updated_count": updated_count,
                "theta_learn_eff": float(permit_param_info.get("theta_learn_eff", float("nan"))) if isinstance(permit_param_info, dict) else float("nan"),
            },
            "constellation_error": {
                "block_residual_mean": float(
                    np.mean(getattr(state.fovea, "block_residual", np.zeros(0)))
                )
                if getattr(state.fovea, "block_residual", np.zeros(0)).size
                else 0.0,
                "block_residual_max": float(
                    np.max(getattr(state.fovea, "block_residual", np.zeros(0)))
                )
                if getattr(state.fovea, "block_residual", np.zeros(0)).size
                else 0.0,
            },
            "observation": {
                "obs_count": int(len(idx)),
                "seen_count": int(len(idx)),
            },
            "prediction": {
                "pred_nonzero": int(np.count_nonzero(pred_display)),
            },
        }
        self.logger.info(json.dumps(log_payload, sort_keys=True))

    def step(self, obs_full: np.ndarray):
        obs_vec = np.asarray(obs_full, dtype=float).reshape(-1)
        if obs_vec.size != self.D:
            obs_vec = np.resize(obs_vec, (self.D,))

        # Axiom -1: the harness does not choose what to observe.
        # The harness asks the agent to precompute its *own* fovea selection, then
        # samples exactly those dims. This preserves strict fixed-budget semantics
        # while keeping all decisions inside the agent.
        blocks = self.agent.prepare_fovea_selection(periph_full=obs_vec)
        O_req = make_observation_set(blocks, self.cfg)
        dims = sorted(int(k) for k in O_req if 0 <= int(k) < self.D)
        x_partial = {int(k): float(obs_vec[int(k)]) for k in dims}
        env_obs = EnvObs(
            x_partial=x_partial,
            opp=0.0,
            danger=0.0,
            # Full frame is provided as a transient routing signal only; it is not
            # treated as observed unless selected by O_req.
            x_full=obs_vec,
            periph_full=obs_vec,
            allow_full_state=False,
        )
        action, trace = self.agent.step(env_obs)

        # What the agent actually treated as observed this step.
        obs_dims = sorted(
            int(d)
            for d in (getattr(self.agent.state.buffer, "observed_dims", set()) or set())
            if 0 <= int(d) < self.D
        )
        idx = np.asarray(obs_dims, dtype=int)

        pred_vec = np.asarray(getattr(self.agent.state.buffer, "x_prior", np.zeros(self.D)), dtype=float)
        if pred_vec.size != self.D:
            pred_vec = np.resize(pred_vec, (self.D,))
        pred_display = np.rint(pred_vec).astype(int)

        seen = np.zeros(self.D, dtype=int)
        if idx.size:
            seen[idx] = obs_vec[idx].astype(int)

        unknown = np.ones((self.H, self.W), dtype=bool)
        if idx.size:
            ys = idx // self.W
            xs = idx % self.W
            unknown[ys, xs] = False

        mismatch = (obs_vec.astype(int) != pred_display).astype(float)
        self._log_step(
            action=action,
            trace=trace,
            obs_vec=obs_vec,
            pred_display=pred_display,
            mismatch=mismatch,
            idx=idx,
        )
        return StepOut(
            action=action,
            pred=pred_display,
            mismatch=mismatch,
            seen=seen,
            unknown=unknown,
            fidx=idx,
        )


# ============================================================
# UI
# ============================================================

def render_dashboard(t, H, W, env_vec, pred_vec, seen_vec, unknown_mask, fovea_idx, mismatch,
                     fovea_blocks, underlay_mask, n_shapes, action_cmd):
    mean_d = float(np.mean(mismatch))
    max_d = float(np.max(mismatch))

    header = Text()
    header.append(f"t={t:6d}  ", style="bold")
    header.append(f"env_shapes={n_shapes:2d}/{MAX_SHAPES}  ", style="magenta")
    header.append(f"fovea_blocks={fovea_blocks:2d}  ", style="magenta")
    header.append(f"meanΔ={mean_d:.4f}  maxΔ={max_d:.1f}  ", style="yellow")
    header.append(f"cmd={action_cmd}", style="cyan")
            
    env_panel = Panel(grid_text(env_vec, H, W, underlay_mask=underlay_mask), title="ENV", border_style="white")
    pred_panel = Panel(grid_text(pred_vec, H, W), title="PRED", border_style="white")
    seen_panel = Panel(
        grid_text(seen_vec, H, W, unknown_mask=unknown_mask, highlight_idx=fovea_idx, unknown_as_blank=True),
        title="SEEN",
        border_style="white",
    )
    cols = Columns([env_panel, pred_panel, seen_panel], equal=True, expand=True)
    return Group(header, cols)


# ============================================================
# Run
# ============================================================


class KeyPoller:
    """Cross-platform single-key polling (no Enter).

    - On Unix, uses cbreak + select.
    - On Windows, uses msvcrt.
    """

    def __init__(self) -> None:
        self._fd = None
        self._old = None

    def __enter__(self):
        if msvcrt is None and termios is not None and tty is not None and sys.stdin.isatty():
            self._fd = sys.stdin.fileno()
            self._old = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._fd is not None and self._old is not None and termios is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)
        self._fd = None
        self._old = None
        return False

    def poll(self) -> str | None:
        if msvcrt is not None:
            try:
                if msvcrt.kbhit():
                    ch = msvcrt.getwch()
                    return ch
            except Exception:
                return None
            return None

        if termios is None or tty is None or not sys.stdin.isatty():
            return None
        try:
            r, _, _ = select.select([sys.stdin], [], [], 0)
            if r:
                return sys.stdin.read(1)
        except Exception:
            return None
        return None


if __name__ == "__main__":
    env = SemanticOcclusionEnv(H=HEIGHT, W=WIDTH, seed=1)
    agent = NUPCA3GridAgent(H=HEIGHT, W=WIDTH, seed=2)

    t = 0
    obs = env.render_visible()
    step_out = agent.step(obs)
    env.apply_action(step_out.action)

    PERSIST_PATH = Path("agent_state.pkl")

    with KeyPoller() as keys, Live(screen=True, refresh_per_second=FPS) as live:
        try:
            while True:
                t += 1

                key = keys.poll()
                if key in ("q", "Q"):
                    persist_state(agent.agent, PERSIST_PATH)
                    break

                env.step_physics()
                obs = env.render_visible()

                step_out = agent.step(obs)
                env.apply_action(step_out.action)

                live.update(
                    render_dashboard(
                        t=t,
                        H=HEIGHT,
                        W=WIDTH,
                        env_vec=obs,
                        pred_vec=step_out.pred,
                        seen_vec=step_out.seen,
                        unknown_mask=step_out.unknown,
                        fovea_idx=step_out.fidx,
                        mismatch=step_out.mismatch,
                        fovea_blocks=len(getattr(agent.agent.state.fovea, "current_blocks", set())),
                        underlay_mask=agent.underlay_mask,
                        n_shapes=len(env.objs),
                        action_cmd=str(getattr(step_out.action, "command", "")),
                    )
                )
        except KeyboardInterrupt:
            persist_state(agent.agent, PERSIST_PATH)
