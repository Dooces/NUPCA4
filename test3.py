import json
import logging
import numpy as np
from dataclasses import dataclass

from rich.live import Live
from rich.panel import Panel
from rich.columns import Columns
from rich.text import Text
from rich.console import Group

from nupca3.agent import NUPCA3Agent
from nupca3.config import AgentConfig
from nupca3.types import EnvObs

# ============================================================
# OPTIONS (edit these)
# ============================================================
HEIGHT = 32
WIDTH = 32
FPS = 120

MAX_SHAPES = 8
MIN_SHAPES = 0

STREAK_STEPS = 90
ADD_DELTA_THRESHOLD = 0.02

# High-streak threshold (requested): if mean_delta stays ABOVE this for STREAK_STEPS -> remove a shape
HIGH_STREAK_THRESHOLD = 0.10

# Which shape-types are "elastic" (participate in object-object collisions).
# Green squares + magenta circles are elastic; cyan rectangles are non-elastic occluders (foreground).
ELASTIC_KINDS = {"square", "circle"}

# Fovea budget (scaled for 32x64)
BUDGET = 512
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
    Curriculum:
      - if mean_delta < ADD_DELTA_THRESHOLD for STREAK_STEPS -> add a shape (until MAX_SHAPES)
      - if mean_delta > HIGH_STREAK_THRESHOLD for STREAK_STEPS -> remove a shape (down to MIN_SHAPES)
    """
    def __init__(self, H=HEIGHT, W=WIDTH, seed=1):
        self.H = int(H)
        self.W = int(W)
        self.rng = np.random.default_rng(seed)
        self.t = 0
        self.objs: list[Obj] = []
        self.low_streak = 0
        self.high_streak = 0

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

    def update_curriculum(self, mean_delta: float):
        if mean_delta < ADD_DELTA_THRESHOLD:
            self.low_streak += 1
        else:
            self.low_streak = max(0, self.low_streak - 2)

        if mean_delta > HIGH_STREAK_THRESHOLD:
            self.high_streak += 1
        else:
            self.high_streak = max(0, self.high_streak - 2)

        if self.low_streak >= STREAK_STEPS and len(self.objs) < MAX_SHAPES:
            self._spawn_obj()
            self.low_streak = 0
            self.high_streak = 0

        if self.high_streak >= STREAK_STEPS and len(self.objs) > MIN_SHAPES:
            self._remove_one()
            self.low_streak = 0
            self.high_streak = 0


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
            allow_selected_blocks_override=True,
        )
        self.agent = NUPCA3Agent(self.cfg)
        self.agent.reset(seed=int(seed))
        self.underlay_mask = np.zeros((self.H, self.W), dtype=bool)
        self.pred_prev = np.zeros(self.D, dtype=int)
        self.seen_dims: set[int] = set()
        self.rng = np.random.default_rng(seed)
        self.logger = logging.getLogger("nupca3_grid_harness")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.FileHandler("output.txt", mode="w", encoding="utf-8")
            formatter = logging.Formatter("%(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.propagate = False

    def _choose_fovea(self, mismatch):
        budget = min(BUDGET, self.D)
        if float(np.max(mismatch)) < 1e-12:
            return set(
                self.rng.choice(np.arange(self.D), size=budget, replace=False).tolist()
            )

        k_main = max(0, budget - PROBE_K)
        main_idx = np.argsort(mismatch)[-k_main:] if k_main > 0 else np.array([], dtype=int)

        remaining = np.setdiff1d(np.arange(self.D), main_idx, assume_unique=False)
        k_probe = min(PROBE_K, remaining.size)
        probe_idx = (
            self.rng.choice(remaining, size=k_probe, replace=False)
            if k_probe > 0
            else np.array([], dtype=int)
        )

        return set(np.concatenate([main_idx, probe_idx]).tolist())

    def _log_step(
        self,
        *,
        action,
        trace,
        obs_vec,
        pred_display,
        mismatch,
        idx,
        seen_vec,
        seen_dims,
    ):
        state = self.agent.state
        learn_cache = getattr(state, "learn_cache", None)
        working_set = getattr(learn_cache, "A_t", None) if learn_cache is not None else None
        active_nodes = list(getattr(working_set, "active", [])) if working_set is not None else []
        active_weights = dict(getattr(working_set, "weights", {})) if working_set is not None else {}

        node_info = []
        nodes = getattr(getattr(state, "library", None), "nodes", {}) or {}
        for node_id in active_nodes:
            node = nodes.get(node_id)
            if node is None:
                node_info.append({"node_id": int(node_id), "status": "missing"})
                continue
            node_info.append(
                {
                    "node_id": int(node_id),
                    "weight": float(active_weights.get(node_id, 0.0)),
                    "footprint": int(getattr(node, "footprint", -1)),
                    "parents": sorted(int(p) for p in getattr(node, "parents", set())),
                    "children": sorted(int(c) for c in getattr(node, "children", set())),
                    "is_anchor": bool(getattr(node, "is_anchor", False)),
                }
            )

        learning_candidates = trace.get("learning_candidates", {}) if isinstance(trace, dict) else {}
        learning_active = bool(learning_candidates)
        learning_reason = "learning_candidates_present" if learning_active else "no_learning_candidates"

        def _row_bounds(rows: list[int]) -> dict[str, int | None]:
            if not rows:
                return {"min": None, "max": None}
            return {"min": int(rows[0]), "max": int(rows[-1])}

        obs_rows = sorted(
            int(dim) // self.W
            for dim in idx
            if 0 <= int(dim) < self.D
        )
        seen_dims_list = sorted(int(dim) for dim in seen_dims if 0 <= int(dim) < self.D)
        seen_rows = sorted(dim // self.W for dim in seen_dims_list)
        preview_limit = 8
        seen_preview = []
        for dim in seen_dims_list[:preview_limit]:
            if 0 <= dim < seen_vec.size:
                val = int(seen_vec[int(dim)])
            else:
                val = None
            seen_preview.append({"dim": int(dim), "value": val})

        observation_payload = {
            "obs_count": int(len(idx)),
            "seen_count": int(len(idx)),
            "seen_total": int(len(seen_dims_list)),
            "seen_fraction": float(len(seen_dims_list)) / float(max(1, self.D)),
            "obs_rows": _row_bounds(obs_rows),
            "seen_rows": _row_bounds(seen_rows),
            "seen_preview": seen_preview,
        }

        log_payload = {
            "t": int(trace.get("t", getattr(state, "t", 0))) if isinstance(trace, dict) else int(getattr(state, "t", 0)),
            "action": int(action),
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
            },
            "observation": observation_payload,
            "prediction": {
                "pred_nonzero": int(np.count_nonzero(pred_display)),
            },
        }
        self.logger.info(json.dumps(log_payload, sort_keys=True))

    def step(self, obs_full: np.ndarray):
        obs_vec = np.asarray(obs_full, dtype=float).reshape(-1)
        if obs_vec.size != self.D:
            obs_vec = np.resize(obs_vec, (self.D,))

        mismatch_prev = (obs_vec.astype(int) != self.pred_prev).astype(float)
        O = self._choose_fovea(mismatch_prev)
        idx = np.fromiter(O, dtype=int)
        x_partial = {int(i): float(obs_vec[int(i)]) for i in idx}
        env_obs = EnvObs(x_partial=x_partial, opp=0.0, danger=0.0, selected_blocks=tuple(idx.tolist()))
        action, trace = self.agent.step(env_obs)

        pred_vec = np.asarray(getattr(self.agent.state.buffer, "x_prior", np.zeros(self.D)), dtype=float)
        if pred_vec.size != self.D:
            pred_vec = np.resize(pred_vec, (self.D,))
        pred_display = np.rint(pred_vec).astype(int)

        valid_idx = [int(dim) for dim in idx if 0 <= int(dim) < self.D]
        self.seen_dims.update(valid_idx)

        seen_raw = np.asarray(getattr(self.agent.state.buffer, "x_last", np.zeros(self.D)), dtype=float)
        if seen_raw.size != self.D:
            seen_raw = np.resize(seen_raw, (self.D,))
        seen = np.rint(seen_raw).astype(int)

        seen_indices = sorted(self.seen_dims)
        unknown = np.ones((self.H, self.W), dtype=bool)
        if seen_indices:
            arr = np.asarray(seen_indices, dtype=int)
            ys = arr // self.W
            xs = arr % self.W
            unknown[ys, xs] = False

        mismatch = (obs_vec.astype(int) != pred_display).astype(float)
        self.pred_prev = pred_display.copy()
        self._log_step(
            action=action,
            trace=trace,
            obs_vec=obs_vec,
            pred_display=pred_display,
            mismatch=mismatch,
            idx=idx,
            seen_vec=seen,
            seen_dims=seen_indices,
        )
        return pred_display, mismatch, seen, unknown, idx


# ============================================================
# UI
# ============================================================

def render_dashboard(t, H, W, env_vec, pred_vec, seen_vec, unknown_mask, fovea_idx, mismatch,
                     tracks_alive, underlay_mask, n_shapes, low_streak, high_streak):
    mean_d = float(np.mean(mismatch))
    max_d = float(np.max(mismatch))

    header = Text()
    header.append(f"t={t:6d}  ", style="bold")
    header.append(f"env_shapes={n_shapes:2d}/{MAX_SHAPES}  ", style="magenta")
    header.append(f"agent_tracks={tracks_alive:2d}  ", style="magenta")
    header.append(f"meanΔ={mean_d:.4f}  maxΔ={max_d:.1f}  ", style="yellow")
    header.append(f"low={low_streak:3d}/{STREAK_STEPS}  ", style="green")
    header.append(f"high={high_streak:3d}/{STREAK_STEPS}  ", style="red")
    header.append(f"high_thr={HIGH_STREAK_THRESHOLD:.2f}", style="red")

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

if __name__ == "__main__":
    env = SemanticOcclusionEnv(H=HEIGHT, W=WIDTH, seed=1)
    agent = NUPCA3GridAgent(H=HEIGHT, W=WIDTH, seed=2)

    t = 0
    obs = env.render_visible()
    pred, mismatch, seen, unk, fidx = agent.step(obs)
    mean_delta = float(np.mean(mismatch))

    with Live(screen=True, refresh_per_second=FPS) as live:
        while True:
            t += 1

            env.step_physics()
            obs = env.render_visible()

            pred, mismatch, seen, unk, fidx = agent.step(obs)
            mean_delta = float(np.mean(mismatch))

            env.update_curriculum(mean_delta)

            live.update(
                render_dashboard(
                    t=t,
                    H=HEIGHT,
                    W=WIDTH,
                    env_vec=obs,
                    pred_vec=pred,
                    seen_vec=seen,
                    unknown_mask=unk,
                    fovea_idx=fidx,
                    mismatch=mismatch,
                    tracks_alive=len(getattr(agent.agent.state.fovea, "current_blocks", set())),
                    underlay_mask=agent.underlay_mask,
                    n_shapes=len(env.objs),
                    low_streak=env.low_streak,
                    high_streak=env.high_streak,
                )
            )
