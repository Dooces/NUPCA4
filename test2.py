import numpy as np
from dataclasses import dataclass

from rich.live import Live
from rich.panel import Panel
from rich.columns import Columns
from rich.text import Text
from rich.console import Group

# ============================================================
# OPTIONS (edit these)
# ============================================================
HEIGHT = 32
WIDTH = 64            # 2x as wide as tall
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
# Agent: semantic persistence via tracks (simple component tracker)
# ============================================================

@dataclass
class Track:
    tid: int
    cx: float
    cy: float
    vx: float
    vy: float
    z: float
    patch: np.ndarray
    ph: int
    pw: int
    last_seen: int


class SemanticTrackerAgent:
    def __init__(self, H=HEIGHT, W=WIDTH, budget=BUDGET, probe_k=PROBE_K, max_lost=MAX_LOST, seed=2):
        self.H = int(H)
        self.W = int(W)
        self.D = self.H * self.W
        self.budget = int(budget)
        self.probe_k = int(probe_k)
        self.max_lost = int(max_lost)
        self.rng = np.random.default_rng(seed)

        self.t = 0
        self.next_tid = 1
        self.tracks: dict[int, Track] = {}
        self.pred_visible = np.zeros(self.D, dtype=int)
        self.underlay_mask = np.zeros((self.H, self.W), dtype=bool)

    def _connected_components(self, occ: np.ndarray):
        H, W = self.H, self.W
        visited = np.zeros_like(occ, dtype=bool)
        comps = []
        for y in range(H):
            for x in range(W):
                if not occ[y, x] or visited[y, x]:
                    continue
                stack = [(y, x)]
                visited[y, x] = True
                ys, xs = [], []
                while stack:
                    cy, cx = stack.pop()
                    ys.append(cy); xs.append(cx)
                    if cy > 0 and occ[cy - 1, cx] and not visited[cy - 1, cx]:
                        visited[cy - 1, cx] = True; stack.append((cy - 1, cx))
                    if cy < H - 1 and occ[cy + 1, cx] and not visited[cy + 1, cx]:
                        visited[cy + 1, cx] = True; stack.append((cy + 1, cx))
                    if cx > 0 and occ[cy, cx - 1] and not visited[cy, cx - 1]:
                        visited[cy, cx - 1] = True; stack.append((cy, cx - 1))
                    if cx < W - 1 and occ[cy, cx + 1] and not visited[cy, cx + 1]:
                        visited[cy, cx + 1] = True; stack.append((cy, cx + 1))
                ys = np.array(ys, dtype=int)
                xs = np.array(xs, dtype=int)
                y0, y1 = int(ys.min()), int(ys.max())
                x0, x1 = int(xs.min()), int(xs.max())
                comps.append((y0, y1, x0, x1))
        return comps

    def _match_tracks(self, detections):
        tids = list(self.tracks.keys())
        if not tids or not detections:
            return {}, set(range(len(detections))), set(tids)

        cost = np.zeros((len(tids), len(detections)), dtype=float)
        for ti, tid in enumerate(tids):
            tr = self.tracks[tid]
            for di, det in enumerate(detections):
                dx = det["cx"] - tr.cx
                dy = det["cy"] - tr.cy
                cost[ti, di] = dx * dx + dy * dy

        matches = {}
        assigned_t = set()
        assigned_d = set()

        while True:
            ti, di = np.unravel_index(np.argmin(cost), cost.shape)
            if not np.isfinite(cost[ti, di]):
                break
            if cost[ti, di] > 100.0:  # 10px gate
                cost[ti, di] = np.inf
                continue
            tid = tids[ti]
            matches[tid] = di
            assigned_t.add(tid)
            assigned_d.add(di)
            cost[ti, :] = np.inf
            cost[:, di] = np.inf

        return matches, (set(range(len(detections))) - assigned_d), (set(tids) - assigned_t)

    def _predict_tracks(self):
        H, W = self.H, self.W
        for tr in self.tracks.values():
            tr.cx += tr.vx
            tr.cy += tr.vy
            if tr.cx < 0:
                tr.cx = 0; tr.vx = abs(tr.vx)
            if tr.cx > W - 1:
                tr.cx = W - 1; tr.vx = -abs(tr.vx)
            if tr.cy < 0:
                tr.cy = 0; tr.vy = abs(tr.vy)
            if tr.cy > H - 1:
                tr.cy = H - 1; tr.vy = -abs(tr.vy)

    def _render_pred_visible_and_underlay(self):
        H, W = self.H, self.W
        vis = np.zeros((H, W), dtype=int)
        items = sorted(self.tracks.values(), key=lambda tr: tr.z)

        occ_stack = []
        for tr in items:
            ph, pw = tr.ph, tr.pw
            y0 = int(round(tr.cy - ph / 2))
            x0 = int(round(tr.cx - pw / 2))
            y1 = y0 + ph
            x1 = x0 + pw

            yy0 = max(0, y0); xx0 = max(0, x0)
            yy1 = min(H, y1); xx1 = min(W, x1)
            if yy0 >= yy1 or xx0 >= xx1:
                occ_stack.append(None)
                continue

            patch = tr.patch[(yy0 - y0):(yy1 - y0), (xx0 - x0):(xx1 - x0)]
            occ = np.zeros((H, W), dtype=bool)
            occ[yy0:yy1, xx0:xx1] = (patch != 0)
            occ_stack.append((yy0, yy1, xx0, xx1, patch, occ))

        for pack in occ_stack:
            if pack is None:
                continue
            yy0, yy1, xx0, xx1, patch, _occ = pack
            region = vis[yy0:yy1, xx0:xx1]
            mask = (patch != 0)
            region[mask] = patch[mask]
            vis[yy0:yy1, xx0:xx1] = region

        under = np.zeros((H, W), dtype=bool)
        covered = np.zeros((H, W), dtype=bool)
        for pack in reversed(occ_stack):
            if pack is None:
                continue
            _yy0, _yy1, _xx0, _xx1, _patch, occ = pack
            under |= (occ & covered)
            covered |= occ

        self.pred_visible = vis.ravel()
        self.underlay_mask = under

    def _choose_fovea(self, mismatch):
        D = self.D
        if float(np.max(mismatch)) < 1e-12:
            return set(self.rng.choice(np.arange(D), size=self.budget, replace=False).tolist())

        k_main = max(0, self.budget - self.probe_k)
        main_idx = np.argsort(mismatch)[-k_main:] if k_main > 0 else np.array([], dtype=int)

        remaining = np.setdiff1d(np.arange(D), main_idx, assume_unique=False)
        k_probe = min(self.probe_k, remaining.size)
        probe_idx = self.rng.choice(remaining, size=k_probe, replace=False) if k_probe > 0 else np.array([], dtype=int)

        return set(np.concatenate([main_idx, probe_idx]).tolist())

    def step(self, obs_full: np.ndarray):
        self.t += 1
        self._predict_tracks()
        self._render_pred_visible_and_underlay()
        pred_current = self.pred_visible.copy()

        mismatch = (obs_full != pred_current).astype(float)

        O = self._choose_fovea(mismatch)
        idx = np.fromiter(O, dtype=int)

        seen = np.zeros_like(obs_full)
        seen[idx] = obs_full[idx]

        unknown = np.ones((self.H, self.W), dtype=bool)
        ys = idx // self.W
        xs = idx % self.W
        unknown[ys, xs] = False

        # update tracks from visible observation
        obs_grid = obs_full.reshape(self.H, self.W)
        occ = (obs_grid != 0)
        bboxes = self._connected_components(occ)

        detections = []
        for (y0, y1, x0, x1) in bboxes:
            patch = obs_grid[y0:y1 + 1, x0:x1 + 1].copy()
            cy = 0.5 * (y0 + y1)
            cx = 0.5 * (x0 + x1)
            detections.append({"cx": float(cx), "cy": float(cy), "patch": patch, "ph": patch.shape[0], "pw": patch.shape[1]})

        matches, unmatched_dets, _unmatched_tracks = self._match_tracks(detections)

        for tid, di in matches.items():
            tr = self.tracks[tid]
            det = detections[di]
            new_cx, new_cy = det["cx"], det["cy"]
            tr.vx = 0.6 * tr.vx + 0.4 * (new_cx - tr.cx)
            tr.vy = 0.6 * tr.vy + 0.4 * (new_cy - tr.cy)
            tr.cx, tr.cy = new_cx, new_cy
            tr.patch = det["patch"]
            tr.ph, tr.pw = det["ph"], det["pw"]
            tr.last_seen = self.t

            # heuristic z: rect codes => occluder
            if np.any((tr.patch == 3) | (tr.patch == 4)):
                tr.z = 100.0
            else:
                tr.z = min(tr.z, 20.0)

        for di in unmatched_dets:
            det = detections[di]
            tid = self.next_tid
            self.next_tid += 1

            patch = det["patch"]
            is_rect = bool(np.any((patch == 3) | (patch == 4)))
            z = 100.0 if is_rect else float(self.rng.uniform(0.0, 10.0))

            self.tracks[tid] = Track(
                tid=tid, cx=det["cx"], cy=det["cy"], vx=0.0, vy=0.0,
                z=z, patch=patch, ph=det["ph"], pw=det["pw"], last_seen=self.t
            )

        for tid in list(self.tracks.keys()):
            if (self.t - self.tracks[tid].last_seen) > self.max_lost:
                del self.tracks[tid]

        self._render_pred_visible_and_underlay()
        return pred_current, mismatch, seen, unknown, idx


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
    agent = SemanticTrackerAgent(H=HEIGHT, W=WIDTH, budget=BUDGET, probe_k=PROBE_K, max_lost=MAX_LOST, seed=2)

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
                    tracks_alive=len(agent.tracks),
                    underlay_mask=agent.underlay_mask,
                    n_shapes=len(env.objs),
                    low_streak=env.low_streak,
                    high_streak=env.high_streak,
                )
            )
