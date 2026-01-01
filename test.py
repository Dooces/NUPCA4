#!/usr/bin/env python3
"""test.py (NUPCA5 / v5-only harness)

Non-negotiables:
- No legacy modes.
- No pickle persistence.
- World tick is decoupled from agent compute (agent runs in a separate process).
- Full observation (all 400 cells) is published every tick.
- Overwrite semantics: if the agent lags, it consumes the latest observation; no backlog.

Controls:
  s : save state (NPZ)
  q : quit (saves)
  r : reset agent (keep library)
  c : cold-reset agent (clear library)

The environment here is a discrete 20x20 grid with simple polyomino templates.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import struct
import sys
import time
from dataclasses import asdict, dataclass, replace as dc_replace
from multiprocessing.shared_memory import SharedMemory
from typing import Dict, List, Optional, Tuple

import numpy as np
from rich import box
from rich.console import Console
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text


# ----------------------------
# Harness parameters
# ----------------------------

UPDATE_DELAY_MS = 1000  # world tick pacing. 0 = as fast as possible.

STATE_PATH = "agent_state.npz"
VAL_MAX = 4.0


# ----------------------------
# Terminal key polling (nonblocking, cross-platform)
# ----------------------------

if os.name == "nt":
    import msvcrt  # type: ignore

    class KeyPoller:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def poll(self) -> Optional[str]:
            if msvcrt.kbhit():
                ch = msvcrt.getch()
                try:
                    return ch.decode("utf-8", errors="ignore")
                except Exception:
                    return None
            return None
else:
    import select
    import termios
    import tty

    class KeyPoller:
        def __enter__(self):
            self.fd = sys.stdin.fileno()
            self.old = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
            return self

        def __exit__(self, exc_type, exc, tb):
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)

        def poll(self) -> Optional[str]:
            if select.select([sys.stdin], [], [], 0)[0]:
                return sys.stdin.read(1)
            return None


# ----------------------------
# Environment (20x20 discrete grid)
# ----------------------------

@dataclass
class Shape:
    cells: List[Tuple[int, int]]
    x: int
    y: int
    dx: int
    dy: int
    color_id: int


class GridWorld:
    def __init__(self, grid_size: int = 20):
        self.grid_size = int(grid_size)
        self.shapes: List[Shape] = []
        self.t = 0

        self.spawn_p = 0.22
        self.max_shapes = 8

        # None => horizontal bar
        self.templates = [None, "vbar", "square", "L"]

    def _make_hbar(self) -> List[Tuple[int, int]]:
        L = random.randint(2, 5)
        return [(0, i) for i in range(L)]

    def _make_vbar(self) -> List[Tuple[int, int]]:
        L = random.randint(2, 5)
        return [(i, 0) for i in range(L)]

    def _make_square(self) -> List[Tuple[int, int]]:
        return [(0, 0), (0, 1), (1, 0), (1, 1)]

    def _make_L(self) -> List[Tuple[int, int]]:
        return [(0, 0), (1, 0), (2, 0), (2, 1)]

    def spawn_shape(self) -> None:
        if len(self.shapes) >= self.max_shapes:
            return

        typ = random.choice(self.templates)
        if typ is None:
            cells = self._make_hbar()
        elif typ == "vbar":
            cells = self._make_vbar()
        elif typ == "square":
            cells = self._make_square()
        else:
            cells = self._make_L()

        max_r = max(dr for dr, _ in cells)
        max_c = max(dc for _, dc in cells)

        x = random.randint(0, max(0, self.grid_size - (max_c + 1)))
        y = 0

        dx = random.choice([-1, 0, 1])
        dy = 1
        color_id = random.randint(1, 4)
        self.shapes.append(Shape(cells=cells, x=x, y=y, dx=dx, dy=dy, color_id=color_id))

    def _would_hit_wall(self, sh: Shape, x_next: int) -> bool:
        for dr, dc in sh.cells:
            c = x_next + dc
            if c < 0 or c >= self.grid_size:
                return True
        return False

    def update(self) -> None:
        self.t += 1
        if random.random() < self.spawn_p or not self.shapes:
            self.spawn_shape()

        new_shapes: List[Shape] = []
        for sh in self.shapes:
            x_next = sh.x + sh.dx
            if sh.dx != 0 and self._would_hit_wall(sh, x_next):
                sh.dx = -sh.dx
                x_next = sh.x + sh.dx
                if self._would_hit_wall(sh, x_next):
                    x_next = sh.x

            sh.x = x_next
            sh.y += sh.dy

            if any(0 <= sh.y + dr < self.grid_size for dr, _ in sh.cells):
                new_shapes.append(sh)

        self.shapes = new_shapes

    def get_current_grid(self) -> np.ndarray:
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        for sh in self.shapes:
            for dr, dc in sh.cells:
                r = sh.y + dr
                c = sh.x + dc
                if 0 <= r < self.grid_size and 0 <= c < self.grid_size:
                    grid[r, c] = int(sh.color_id)
        return grid


# ----------------------------
# UI helpers
# ----------------------------

COLOR_STYLE = {0: "grey50", 1: "red", 2: "green", 3: "blue", 4: "yellow"}
DIFF_STYLE = {".": "grey50", "X": "red"}


def grid_to_text(grid: np.ndarray) -> Text:
    grid = np.asarray(grid, dtype=np.int32)
    side_h, side_w = grid.shape
    t = Text()
    for r in range(side_h):
        for c in range(side_w):
            iv = int(grid[r, c])
            ch = "." if iv == 0 else "#"
            t.append(ch, style=COLOR_STYLE.get(iv, "white"))
        t.append("\n")
    return t


def diff_to_text(pred: np.ndarray, actual: np.ndarray) -> Text:
    pred = np.asarray(pred, dtype=np.int32)
    actual = np.asarray(actual, dtype=np.int32)
    h, w = actual.shape
    t = Text()
    for r in range(h):
        for c in range(w):
            ch = "X" if int(pred[r, c]) != int(actual[r, c]) else "."
            t.append(ch, style=DIFF_STYLE[ch])
        t.append("\n")
    return t


def build_layout(
    env_text: Text,
    pred_text: Text,
    diff_text: Text,
    status_text: Text,
    *,
    pane_w: int = 24,
    pane_h: int = 22,
    gap_w: int = 1,
) -> Layout:
    layout = Layout()
    total_w = 3 * pane_w + 2 * gap_w

    top = Layout(name="top", size=pane_h)
    top.split_row(
        Layout(name="padL", ratio=1),
        Layout(name="env", size=pane_w),
        Layout(name="gap1", size=gap_w),
        Layout(name="pred", size=pane_w),
        Layout(name="gap2", size=gap_w),
        Layout(name="diff", size=pane_w),
        Layout(name="padR", ratio=1),
    )

    bottom = Layout(name="bottom", ratio=1)
    bottom.split_row(
        Layout(name="padL2", ratio=1),
        Layout(name="status", size=total_w),
        Layout(name="padR2", ratio=1),
    )

    layout.split_column(top, bottom)

    top["env"].update(Panel(env_text, title="ENV", padding=(0, 0), box=box.SQUARE))
    top["pred"].update(Panel(pred_text, title="PRED (agent may lag)", padding=(0, 0), box=box.SQUARE))
    top["diff"].update(Panel(diff_text, title="DIFF", padding=(0, 0), box=box.SQUARE))
    bottom["status"].update(Panel(status_text, title="NUPCA5", padding=(0, 1), box=box.SQUARE))

    blank = Text("")
    top["padL"].update(blank)
    top["gap1"].update(blank)
    top["gap2"].update(blank)
    top["padR"].update(blank)
    bottom["padL2"].update(blank)
    bottom["padR2"].update(blank)

    return layout


# ----------------------------
# Periphery permutation
# ----------------------------

def build_periphery_permutation(side: int = 20, ring: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """Permute row-major flattening so periphery occupies the last indices."""
    side = int(side)
    ring = max(1, int(ring))
    N = side * side

    is_periph = np.zeros(N, dtype=bool)
    for r in range(side):
        for c in range(side):
            if r < ring or r >= side - ring or c < ring or c >= side - ring:
                is_periph[r * side + c] = True

    core = [i for i in range(N) if not is_periph[i]]
    periph = [i for i in range(N) if is_periph[i]]
    perm = np.array(core + periph, dtype=np.int32)
    inv = np.empty_like(perm)
    inv[perm] = np.arange(N, dtype=np.int32)
    return perm, inv


# -----------------------------
# NPZ persistence helpers (v5)
# -----------------------------

def _np_bytes(s: str) -> np.ndarray:
    b = s.encode("utf-8")
    return np.frombuffer(b, dtype=np.uint8)


def _bytes_to_str(a: np.ndarray) -> str:
    a = np.asarray(a, dtype=np.uint8).reshape(-1)
    return bytes(a.tobytes()).decode("utf-8")


def _pack_ragged_int(rows: List[Tuple[int, ...]]) -> Tuple[np.ndarray, np.ndarray]:
    if not rows:
        return np.zeros(1, dtype=np.int32), np.zeros(0, dtype=np.int32)
    ptr = np.zeros(len(rows) + 1, dtype=np.int32)
    out: List[int] = []
    n = 0
    for i, r in enumerate(rows):
        rr = [int(x) for x in r]
        out.extend(rr)
        n += len(rr)
        ptr[i + 1] = n
    idx = np.asarray(out, dtype=np.int32)
    return ptr, idx


def _unpack_ragged_int(ptr: np.ndarray, idx: np.ndarray, i: int) -> np.ndarray:
    ptr = np.asarray(ptr, dtype=np.int32).reshape(-1)
    idx = np.asarray(idx, dtype=np.int32).reshape(-1)
    i = int(i)
    if i < 0 or i + 1 >= ptr.size:
        return np.zeros(0, dtype=np.int32)
    a = int(ptr[i])
    b = int(ptr[i + 1])
    a = max(0, a)
    b = max(a, b)
    b = min(b, idx.size)
    return idx[a:b].copy()


def save_nupca5_state_npz(path: str, *, agent, cfg, perm: np.ndarray, periph_ring: int) -> None:
    """Persist v5 agent state without pickles."""
    from nupca3.memory.sig_index import PackedSigIndex

    st = agent.state

    cfg_json = json.dumps(asdict(cfg), sort_keys=True)

    packed_lib = st.library.pack()
    lib_arrays = {f"lib.{k}": np.asarray(v) for k, v in packed_lib.as_npz_dict().items()}

    sig_index = getattr(st.library, "sig_index", None)
    has_sig_index = int(isinstance(sig_index, PackedSigIndex))
    if has_sig_index:
        si = sig_index
        si_buckets = np.asarray(si.buckets, dtype=np.int32)
        si_salts = np.asarray(si.salts, dtype=np.uint64)
        si_err = (
            np.asarray(getattr(si, "_err_cache", None), dtype=np.float32)
            if getattr(si, "_err_cache", None) is not None
            else np.zeros((0, 0), dtype=np.float32)
        )
        si_cfg = np.array(
            [
                int(si.tables),
                int(si.bucket_bits),
                int(si.bucket_cap),
                int(si.cfg.seed),
                int(si.cfg.err_bins),
                int(si.cfg.eviction_bin),
            ],
            dtype=np.int32,
        )
    else:
        si_buckets = np.zeros((0, 0, 0), dtype=np.int32)
        si_salts = np.zeros(0, dtype=np.uint64)
        si_err = np.zeros((0, 0), dtype=np.float32)
        si_cfg = np.zeros(0, dtype=np.int32)

    node_ids = [int(x) for x in packed_lib.node_ids.tolist()]
    rows: List[Tuple[int, ...]] = []
    for nid in node_ids:
        n = st.library.nodes.get(int(nid))
        blocks = tuple(int(b) for b in getattr(n, "sig_index_blocks", tuple())) if n is not None else tuple()
        rows.append(blocks)
    sig_blocks_ptr, sig_blocks_idx = _pack_ragged_int(rows)

    margins = np.array(
        [
            float(st.margins.m_E),
            float(st.margins.m_D),
            float(st.margins.m_L),
            float(st.margins.m_C),
            float(st.margins.m_S),
        ],
        dtype=np.float32,
    )

    stress = np.array(
        [
            float(st.stress.s_E),
            float(st.stress.s_D),
            float(st.stress.s_L),
            float(st.stress.s_C),
            float(st.stress.s_S),
            float(getattr(st.stress, "s_int_need", 0.0)),
            float(getattr(st.stress, "s_ext_th", 0.0)),
        ],
        dtype=np.float32,
    )

    obs_dims = np.array(sorted(int(d) for d in (getattr(st.buffer, "observed_dims", set()) or set())), dtype=np.int32)

    pv = list(getattr(st, "pending_validation", []) or [])
    pv_node = np.array([int(getattr(r, "node_id", -1)) for r in pv], dtype=np.int32)
    pv_hbin = np.array([int(getattr(r, "h_bin", -1)) for r in pv], dtype=np.int32)
    pv_t = np.array([int(getattr(r, "t_emit", -1)) for r in pv], dtype=np.int32)
    pv_dist = np.array([int(getattr(r, "dist", 0)) for r in pv], dtype=np.int32)
    pv_err = np.array([float(getattr(r, "err", 0.0)) for r in pv], dtype=np.float32)

    f = st.fovea

    def _arr(x, dt):
        return np.asarray(x, dtype=dt).reshape(-1)

    np.savez_compressed(
        path,
        version=_np_bytes("nupca5_state_npz_v1"),
        cfg_json=_np_bytes(cfg_json),
        perm=np.asarray(perm, dtype=np.int32),
        periph_ring=np.array([int(periph_ring)], dtype=np.int32),

        t=np.array([int(getattr(st, "t", 0))], dtype=np.int64),
        E=np.array([float(getattr(st, "E", 0.0))], dtype=np.float32),
        D=np.array([float(getattr(st, "D", 0.0))], dtype=np.float32),
        drift_P=np.array([float(getattr(st, "drift_P", 0.0))], dtype=np.float32),
        arousal=np.array([float(getattr(st, "arousal", 0.0))], dtype=np.float32),
        arousal_prev=np.array([float(getattr(st, "arousal_prev", 0.0))], dtype=np.float32),
        b_cons=np.array([float(getattr(st, "b_cons", 0.0))], dtype=np.float32),
        last_sig64=np.array([int(getattr(st, "last_sig64", 0) or 0)], dtype=np.uint64),

        margins=margins,
        stress=stress,

        base_mu=np.asarray(getattr(st.baselines, "mu", np.zeros(0, dtype=np.float32)), dtype=np.float32),
        base_var_fast=np.asarray(getattr(st.baselines, "var_fast", np.zeros(0, dtype=np.float32)), dtype=np.float32),
        base_var_slow=np.asarray(getattr(st.baselines, "var_slow", np.zeros(0, dtype=np.float32)), dtype=np.float32),
        base_tilde_prev=np.asarray(
            getattr(st.baselines, "tilde_prev", np.zeros(0, dtype=np.float32))
            if getattr(st.baselines, "tilde_prev", None) is not None
            else np.zeros(0, dtype=np.float32),
            dtype=np.float32,
        ),
        base_last_struct_edit_t=np.array([int(getattr(st.baselines, "last_struct_edit_t", -10**9))], dtype=np.int64),

        macro_rest=np.array([int(bool(getattr(st.macro, "rest", False)))], dtype=np.int8),
        macro_T_since=np.array([int(getattr(st.macro, "T_since", 0))], dtype=np.int64),
        macro_T_rest=np.array([int(getattr(st.macro, "T_rest", 0))], dtype=np.int64),
        macro_P_rest=np.array([float(getattr(st.macro, "P_rest", 0.0))], dtype=np.float32),
        macro_rest_cooldown=np.array([int(getattr(st.macro, "rest_cooldown", 0))], dtype=np.int64),
        macro_rest_zero_processed_streak=np.array([int(getattr(st.macro, "rest_zero_processed_streak", 0))], dtype=np.int64),

        fovea_block_residual=_arr(getattr(f, "block_residual", np.zeros(0)), np.float32),
        fovea_block_age=_arr(getattr(f, "block_age", np.zeros(0)), np.float32),
        fovea_block_uncertainty=_arr(getattr(f, "block_uncertainty", np.zeros(0)), np.float32),
        fovea_block_costs=_arr(getattr(f, "block_costs", np.zeros(0)), np.float32),
        fovea_routing_scores=_arr(getattr(f, "routing_scores", np.zeros(0)), np.float32),
        fovea_block_disagreement=_arr(getattr(f, "block_disagreement", np.zeros(0)), np.float32),
        fovea_block_innovation=_arr(getattr(f, "block_innovation", np.zeros(0)), np.float32),

        obs_dims=obs_dims,

        pv_node=pv_node,
        pv_hbin=pv_hbin,
        pv_t=pv_t,
        pv_dist=pv_dist,
        pv_err=pv_err,

        # Packed library arrays
        **lib_arrays,

        sig_blocks_ptr=sig_blocks_ptr,
        sig_blocks_idx=sig_blocks_idx,

        has_sig_index=np.array([has_sig_index], dtype=np.int8),
        si_cfg=si_cfg,
        si_buckets=si_buckets,
        si_salts=si_salts,
        si_err=si_err,
    )


def load_nupca5_state_npz(path: str):
    """Load (cfg, library, perm, periph_ring, npz) from NPZ.

    v5-only / strict loader:
    - requires all expected arrays/keys to be present
    - no backward-compat defaults
    """
    from nupca3.config import AgentConfig
    from nupca3.types import PackedExpertLibrary, unpack_expert_library
    from nupca3.memory.sig_index import PackedSigIndex, SigIndexCfg

    npz = np.load(path, allow_pickle=False)
    if "cfg_json" not in npz:
        raise ValueError("missing cfg_json")

    cfg_dict = json.loads(_bytes_to_str(npz["cfg_json"]))
    cfg = AgentConfig(**cfg_dict)

    perm = np.asarray(npz["perm"], dtype=np.int32)
    periph_ring = int(np.asarray(npz["periph_ring"], dtype=np.int32).reshape(-1)[0])

    lib_entries = {
        key[len("lib."):]: np.asarray(npz[key])
        for key in npz.files
        if key.startswith("lib.")
    }
    packed = PackedExpertLibrary.from_npz_dict(lib_entries)
    lib = unpack_expert_library(packed)

    # Restore per-node sig_index_blocks (aligned to packed.node_ids ordering)
    sig_ptr = np.asarray(npz["sig_blocks_ptr"], dtype=np.int32)
    sig_idx = np.asarray(npz["sig_blocks_idx"], dtype=np.int32)
    node_ids = packed.node_ids
    for i in range(int(node_ids.size)):
        nid = int(node_ids[i])
        blocks = tuple(int(x) for x in _unpack_ragged_int(sig_ptr, sig_idx, i).tolist())
        n = lib.nodes.get(nid)
        if n is not None:
            n.sig_index_blocks = blocks

    has_sig = int(np.asarray(npz["has_sig_index"], dtype=np.int8).reshape(-1)[0])
    if has_sig:
        si_cfg = np.asarray(npz["si_cfg"], dtype=np.int32).reshape(-1)
        if si_cfg.size != 6:
            raise ValueError(f"si_cfg wrong length: {si_cfg.size}")
        tables = int(si_cfg[0])
        bucket_bits = int(si_cfg[1])
        bucket_cap = int(si_cfg[2])
        seed = int(si_cfg[3])
        err_bins = int(si_cfg[4])
        eviction_bin = int(si_cfg[5])

        si = PackedSigIndex(
            SigIndexCfg(
                tables=tables,
                bucket_bits=bucket_bits,
                bucket_cap=bucket_cap,
                seed=seed,
                err_bins=err_bins,
                eviction_bin=eviction_bin,
            )
        )
        si.buckets[...] = np.asarray(npz["si_buckets"], dtype=np.int32)
        si.salts[...] = np.asarray(npz["si_salts"], dtype=np.uint64)
        err = np.asarray(npz["si_err"], dtype=np.float32)
        if err.size:
            si._err_cache = err.copy()
        lib.sig_index = si

    return cfg, lib, perm, periph_ring, npz


# -----------------------------
# Status string generation
# -----------------------------

def make_status_text_v5(
    *,
    agent_state,
    cfg,
    trace: Dict,
    autosave_in_s: float,
    periph_ring: int,
    env_t: int,
    agent_processed_env_t: int,
) -> Text:
    t = Text()

    st = agent_state
    lib = getattr(st, "library", None)
    nodes = getattr(lib, "nodes", {}) if lib is not None else {}
    n_nodes = int(len(nodes) if nodes is not None else 0)
    active = getattr(st, "active_set", set()) or set()
    active_n = int(len(active))
    last_sig = getattr(st, "last_sig64", None)
    last_sig_s = "None" if last_sig is None else f"0x{int(last_sig):016x}"

    t_step = int(getattr(st, "t", 0))

    t.append(
        f"t={t_step}  env_t={env_t}  agent_env_t={agent_processed_env_t}  nodes={n_nodes}  |A|={active_n}  sig64={last_sig_s}\n",
        style="white",
    )

    h = int(trace.get("horizon", 0))
    b_enc = float(trace.get("b_enc", float("nan")))
    b_roll = float(trace.get("b_roll", float("nan")))
    xC = float(trace.get("x_C", float("nan")))
    t.append(f"h={h}  b_enc={b_enc:.3f}  b_roll={b_roll:.3f}  x_C={xC:.3f}\n", style="white")

    rest_t = bool(trace.get("rest_t", False))
    rest_perm = bool(trace.get("rest_permitted_t", False))
    demand = bool(trace.get("demand_t", False))
    intr = bool(trace.get("interrupt_t", False))
    permit_param = bool(trace.get("permit_param", False))
    permit_struct = bool(trace.get("permit_struct", False))
    t.append(
        f"REST={rest_t} perm={rest_perm} demand={demand} intr={intr}  learn_param={permit_param} learn_struct={permit_struct}\n",
        style="white",
    )

    prior_mae = float(trace.get("prior_obs_mae", float("nan")))
    post_mae = float(trace.get("posterior_obs_mae", float("nan")))
    tr_src = str(trace.get("transport_source", ""))
    tr_delta = trace.get("transport_delta", (0, 0))
    tr_conf = float(trace.get("transport_confidence", 0.0))
    t.append(
        f"mae(obs|prior)={prior_mae:.4f}  mae(obs|post)={post_mae:.4f}  transport={tr_src} delta={tuple(tr_delta)} conf={tr_conf:.2f}\n",
        style="white",
    )

    peq = float(trace.get("prior_eq_frac", float("nan")))
    qeq = float(trace.get("post_eq_frac", float("nan")))
    t.append(f"eq(prior,obs)={peq:.3f}  eq(post,obs)={qeq:.3f}  (same-tick, full-state)\n", style="white")

    U_n = int(trace.get("nonempty_union_n", 0) or 0)
    eqU = float(trace.get("prior_eq_union", float("nan")))
    maeU = float(trace.get("prior_obs_mae_union", float("nan")))
    f1 = float(trace.get("occ_f1", float("nan")))
    col = float(trace.get("color_eq_on_occ", float("nan")))
    wmae = float(trace.get("prior_obs_wmae_empty001", float("nan")))
    t.append(
        f"U(nonempty union)={U_n:3d}  eq|U={eqU:.3f}  mae|U={maeU:.3f}  occF1={f1:.3f}  color@occ={col:.3f}  wMAE(e=0.01)={wmae:.3f}\n",
        style="white",
    )

    ar = float(trace.get("arousal", float("nan")))
    s_need = float(trace.get("s_int_need", 0.0))
    s_th = float(trace.get("s_ext_th", 0.0))
    mE = float(trace.get("mE", 0.0))
    mD = float(trace.get("mD", 0.0))
    mC = float(trace.get("mC", 0.0))
    t.append(
        f"arousal={ar:.3f}  s_need={s_need:.3f}  s_th={s_th:.3f}  mE={mE:.3f} mD={mD:.3f} mC={mC:.3f}\n",
        style="white",
    )

    n_fine = int(trace.get("n_fine_blocks_selected", 0))
    n_periph = int(trace.get("n_periph_blocks_selected", 0))
    periph_on = bool(trace.get("peripheral_bg_active", False))
    periph_conf = float(trace.get("peripheral_confidence", 0.0))
    t.append(
        f"fovea_blocks={n_fine} periph_blocks={n_periph} periph_on={periph_on} periph_conf={periph_conf:.2f} periph_ring={periph_ring}\n",
        style="white",
    )

    t.append(
        f"autosave_in={autosave_in_s:5.1f}s  (keys: s save, q quit+save, r reset, c cold-reset)\n",
        style="grey70",
    )
    return t


# -----------------------------
# Shared-memory atomic helpers
# -----------------------------

def _atomic_write_in(shm_in_buf, *, seq: int, env_t: int, obs_f32: np.ndarray, in_hdr: int) -> None:
    """Write obs then commit seq last (prevents torn reads)."""
    shm_in_buf[in_hdr : in_hdr + obs_f32.nbytes] = obs_f32.tobytes(order="C")
    struct.pack_into("<Q", shm_in_buf, 8, int(env_t))
    struct.pack_into("<Q", shm_in_buf, 0, int(seq))


def _atomic_try_read_in(shm_in_buf, *, last_seq: int, obs_n: int, in_hdr: int) -> Optional[Tuple[int, int, np.ndarray]]:
    """Read a stable snapshot (seq, env_t, obs) or None if no new data."""
    seq1 = int(struct.unpack_from("<Q", shm_in_buf, 0)[0])
    if seq1 == 0 or seq1 == int(last_seq):
        return None
    env_t = int(struct.unpack_from("<Q", shm_in_buf, 8)[0])
    obs = np.frombuffer(shm_in_buf, dtype=np.float32, count=obs_n, offset=in_hdr).copy()
    seq2 = int(struct.unpack_from("<Q", shm_in_buf, 0)[0])
    if seq1 != seq2:
        return None
    return seq1, env_t, obs


def _atomic_write_out(
    shm_out_buf,
    *,
    seq: int,
    agent_t: int,
    env_t: int,
    last_sig64: int,
    x_prior_f32: np.ndarray,
    status_bytes: bytes,
    out_hdr: int,
    status_max: int,
) -> None:
    """Write payload then commit seq last (prevents torn reads)."""
    obs_bytes = x_prior_f32.nbytes
    shm_out_buf[out_hdr : out_hdr + obs_bytes] = x_prior_f32.tobytes(order="C")
    struct.pack_into("<I", shm_out_buf, out_hdr + obs_bytes, int(len(status_bytes)))
    start = out_hdr + obs_bytes + 4
    shm_out_buf[start : start + status_max] = b"\x00" * status_max
    shm_out_buf[start : start + len(status_bytes)] = status_bytes[:status_max]
    struct.pack_into("<QQQ", shm_out_buf, 8, int(agent_t), int(env_t), int(last_sig64))
    struct.pack_into("<Q", shm_out_buf, 0, int(seq))


def _atomic_try_read_out(
    shm_out_buf,
    *,
    last_seq: int,
    obs_n: int,
    out_hdr: int,
    status_max: int,
) -> Optional[Tuple[int, int, int, int, np.ndarray, str]]:
    """Read a stable snapshot (out_seq, agent_t, env_t, last_sig64, x_prior, status_str) or None."""
    seq1 = int(struct.unpack_from("<Q", shm_out_buf, 0)[0])
    if seq1 == 0 or seq1 == int(last_seq):
        return None
    agent_t, env_t, last_sig = struct.unpack_from("<QQQ", shm_out_buf, 8)
    agent_t = int(agent_t)
    env_t = int(env_t)
    last_sig = int(last_sig)
    x_prior = np.frombuffer(shm_out_buf, dtype=np.float32, count=obs_n, offset=out_hdr).copy()
    obs_bytes = obs_n * 4
    slen = int(struct.unpack_from("<I", shm_out_buf, out_hdr + obs_bytes)[0])
    start = out_hdr + obs_bytes + 4
    slen = max(0, min(int(slen), int(status_max)))
    sb = bytes(shm_out_buf[start : start + slen])
    status_plain = sb.decode("utf-8", errors="replace") if sb else ""
    seq2 = int(struct.unpack_from("<Q", shm_out_buf, 0)[0])
    if seq1 != seq2:
        return None
    return seq1, agent_t, env_t, last_sig, x_prior, status_plain


# -----------------------------
# Agent worker process (top-level for spawn safety)
# -----------------------------

def agent_worker(
    shm_in_name: str,
    shm_out_name: str,
    conn,
    cfg_json_str: str,
    state_path_inner: str,
    perm_arr: np.ndarray,
    periph_ring_inner: int,
    loaded_flag: bool,
    *,
    obs_n: int,
    in_hdr: int,
    out_hdr: int,
    status_max: int,
) -> None:
    import time as _time
    import numpy as _np
    from multiprocessing.shared_memory import SharedMemory as _SharedMemory

    from nupca3.agent import NUPCA3Agent
    from nupca3.config import AgentConfig
    from nupca3.types import EnvObs, PendingValidationRecord

    cfg_dict = json.loads(cfg_json_str)
    cfg_w = AgentConfig(**cfg_dict)

    agent = NUPCA3Agent(cfg_w)

    if loaded_flag:
        try:
            cfg2, lib2, _, _, npz2 = load_nupca5_state_npz(state_path_inner)
            if int(cfg2.D) == int(cfg_w.D) and int(cfg2.B) == int(cfg_w.B):
                agent.cfg = cfg2
                cfg_w = cfg2
            agent.state.library = lib2

            st = agent.state
            st.t = int(_np.asarray(npz2["t"]).reshape(-1)[0])
            st.E = float(_np.asarray(npz2["E"]).reshape(-1)[0])
            st.D = float(_np.asarray(npz2["D"]).reshape(-1)[0])
            st.drift_P = float(_np.asarray(npz2["drift_P"]).reshape(-1)[0])
            st.arousal = float(_np.asarray(npz2["arousal"]).reshape(-1)[0])
            st.arousal_prev = float(_np.asarray(npz2.get("arousal_prev", _np.array([0], dtype=_np.float32))).reshape(-1)[0])
            st.b_cons = float(_np.asarray(npz2.get("b_cons", _np.array([0], dtype=_np.float32))).reshape(-1)[0])
            st.last_sig64 = int(_np.asarray(npz2.get("last_sig64", _np.array([0], dtype=_np.uint64))).reshape(-1)[0])

            m = _np.asarray(npz2["margins"], dtype=_np.float32).reshape(-1)
            st.margins.m_E, st.margins.m_D, st.margins.m_L, st.margins.m_C, st.margins.m_S = map(float, m[:5])

            s = _np.asarray(npz2["stress"], dtype=_np.float32).reshape(-1)
            st.stress.s_E, st.stress.s_D, st.stress.s_L, st.stress.s_C, st.stress.s_S = map(float, s[:5])
            if s.size >= 7:
                st.stress.s_int_need = float(s[5])
                st.stress.s_ext_th = float(s[6])

            st.baselines.mu = _np.asarray(npz2["base_mu"], dtype=_np.float32).copy()
            st.baselines.var_fast = _np.asarray(npz2["base_var_fast"], dtype=_np.float32).copy()
            st.baselines.var_slow = _np.asarray(npz2["base_var_slow"], dtype=_np.float32).copy()
            bp = _np.asarray(npz2.get("base_tilde_prev", _np.zeros(0, dtype=_np.float32)), dtype=_np.float32).reshape(-1)
            st.baselines.tilde_prev = bp.copy() if bp.size else None
            st.baselines.last_struct_edit_t = int(_np.asarray(npz2.get("base_last_struct_edit_t", _np.array([-10**9], dtype=_np.int64))).reshape(-1)[0])

            st.macro.rest = bool(int(_np.asarray(npz2.get("macro_rest", _np.array([0], dtype=_np.int8))).reshape(-1)[0]))
            st.macro.T_since = int(_np.asarray(npz2.get("macro_T_since", _np.array([0], dtype=_np.int64))).reshape(-1)[0])
            st.macro.T_rest = int(_np.asarray(npz2.get("macro_T_rest", _np.array([0], dtype=_np.int64))).reshape(-1)[0])
            st.macro.P_rest = float(_np.asarray(npz2.get("macro_P_rest", _np.array([0], dtype=_np.float32))).reshape(-1)[0])
            st.macro.rest_cooldown = int(_np.asarray(npz2.get("macro_rest_cooldown", _np.array([0], dtype=_np.int64))).reshape(-1)[0])
            st.macro.rest_zero_processed_streak = int(_np.asarray(npz2.get("macro_rest_zero_processed_streak", _np.array([0], dtype=_np.int64))).reshape(-1)[0])

            f = st.fovea
            f.block_residual = _np.asarray(npz2.get("fovea_block_residual", _np.zeros(0, dtype=_np.float32)), dtype=_np.float32).copy()
            f.block_age = _np.asarray(npz2.get("fovea_block_age", _np.zeros(0, dtype=_np.float32)), dtype=_np.float32).copy()
            f.block_uncertainty = _np.asarray(npz2.get("fovea_block_uncertainty", _np.zeros(0, dtype=_np.float32)), dtype=_np.float32).copy()
            f.block_costs = _np.asarray(npz2.get("fovea_block_costs", _np.zeros(0, dtype=_np.float32)), dtype=_np.float32).copy()
            f.routing_scores = _np.asarray(npz2.get("fovea_routing_scores", _np.zeros(0, dtype=_np.float32)), dtype=_np.float32).copy()
            f.block_disagreement = _np.asarray(npz2.get("fovea_block_disagreement", _np.zeros(0, dtype=_np.float32)), dtype=_np.float32).copy()
            f.block_innovation = _np.asarray(npz2.get("fovea_block_innovation", _np.zeros(0, dtype=_np.float32)), dtype=_np.float32).copy()

            obs_dims = _np.asarray(npz2.get("obs_dims", _np.zeros(0, dtype=_np.int32)), dtype=_np.int32).reshape(-1)
            st.buffer.observed_dims = set(int(x) for x in obs_dims.tolist())

            pv_node = _np.asarray(npz2.get("pv_node", _np.zeros(0, dtype=_np.int32)), dtype=_np.int32).reshape(-1)
            pv_hbin = _np.asarray(npz2.get("pv_hbin", _np.zeros(0, dtype=_np.int32)), dtype=_np.int32).reshape(-1)
            pv_t = _np.asarray(npz2.get("pv_t", _np.zeros(0, dtype=_np.int32)), dtype=_np.int32).reshape(-1)
            pv_dist = _np.asarray(npz2.get("pv_dist", _np.zeros(0, dtype=_np.int32)), dtype=_np.int32).reshape(-1)
            pv_err = _np.asarray(npz2.get("pv_err", _np.zeros(0, dtype=_np.float32)), dtype=_np.float32).reshape(-1)
            st.pending_validation.clear()
            for i in range(int(pv_node.size)):
                st.pending_validation.append(
                    PendingValidationRecord(
                        node_id=int(pv_node[i]),
                        h_bin=int(pv_hbin[i]),
                        t_emit=int(pv_t[i]),
                        dist=int(pv_dist[i]),
                        err=float(pv_err[i]),
                    )
                )
        except Exception:
            pass

    shm_in_w = _SharedMemory(name=shm_in_name)
    shm_out_w = _SharedMemory(name=shm_out_name)

    last_in_seq = 0
    last_trace: Dict = {}

    try:
        while True:
            while conn.poll():
                msg = conn.recv()
                cmd = msg.get("cmd") if isinstance(msg, dict) else None

                if cmd == "quit":
                    if msg.get("save", False):
                        save_nupca5_state_npz(
                            state_path_inner,
                            agent=agent,
                            cfg=cfg_w,
                            perm=perm_arr,
                            periph_ring=periph_ring_inner,
                        )
                    conn.send({"cmd": "quit", "ok": True})
                    return

                if cmd == "save":
                    try:
                        save_nupca5_state_npz(
                            state_path_inner,
                            agent=agent,
                            cfg=cfg_w,
                            perm=perm_arr,
                            periph_ring=periph_ring_inner,
                        )
                        conn.send({"cmd": "save", "ok": True, "t": int(getattr(agent.state, "t", 0))})
                    except Exception as e:
                        conn.send({"cmd": "save", "ok": False, "err": str(e)})

                if cmd == "reset":
                    try:
                        agent.reset(clear_memory=bool(msg.get("clear_memory", False)))
                        conn.send({"cmd": "reset", "ok": True})
                    except Exception as e:
                        conn.send({"cmd": "reset", "ok": False, "err": str(e)})

            snap = _atomic_try_read_in(shm_in_w.buf, last_seq=last_in_seq, obs_n=obs_n, in_hdr=in_hdr)
            if snap is None:
                _time.sleep(0.001)
                continue

            in_seq, env_t, obs = snap
            last_in_seq = int(in_seq)

            x_partial = {i: float(obs[i]) for i in range(int(obs.size))}
            o = EnvObs(
                x_partial=x_partial,
                x_full=obs,
                periph_full=obs,
                allow_full_state=True,
                selected_blocks=tuple(range(int(cfg_w.B))),
            )

            _action, trace = agent.step(o)
            last_trace = trace if isinstance(trace, dict) else {}

            st = agent.state
            x_prior = _np.asarray(getattr(st.buffer, "x_prior", _np.zeros(obs_n, dtype=_np.float32)), dtype=_np.float32).reshape(-1)
            if x_prior.size != obs_n:
                x_prior = _np.resize(x_prior, (obs_n,)).astype(_np.float32, copy=False)

            x_post = _np.asarray(getattr(st.buffer, "x_last", _np.zeros(obs_n, dtype=_np.float32)), dtype=_np.float32).reshape(-1)
            if x_post.size != obs_n:
                x_post = _np.resize(x_post, (obs_n,)).astype(_np.float32, copy=False)

            try:
                prior_mae_full = float(_np.mean(_np.abs(x_prior - obs)))
                post_mae_full = float(_np.mean(_np.abs(x_post - obs)))

                pred_i = _np.rint(_np.clip(x_prior, 0.0, 4.0)).astype(_np.int32, copy=False)
                post_i = _np.rint(_np.clip(x_post, 0.0, 4.0)).astype(_np.int32, copy=False)
                obs_i = _np.rint(_np.clip(obs, 0.0, 4.0)).astype(_np.int32, copy=False)

                prior_eq = float(_np.mean(pred_i == obs_i))
                post_eq = float(_np.mean(post_i == obs_i))

                # Sparsity-robust diagnostics:
                # U := union of non-empty cells in either pred or obs (in int space).
                obs_occ = (obs_i != 0)
                pred_occ = (pred_i != 0)
                U = (obs_occ | pred_occ)
                U_n = int(_np.count_nonzero(U))

                if U_n > 0:
                    prior_eq_U = float(_np.mean((pred_i[U] == obs_i[U])))
                    prior_mae_U = float(_np.mean(_np.abs(x_prior[U] - obs[U])))
                else:
                    prior_eq_U = 1.0
                    prior_mae_U = 0.0

                # Occupancy F1 on non-empty detection.
                tp = int(_np.count_nonzero(pred_occ & obs_occ))
                fp = int(_np.count_nonzero(pred_occ & (~obs_occ)))
                fn = int(_np.count_nonzero((~pred_occ) & obs_occ))
                prec = (tp / (tp + fp)) if (tp + fp) > 0 else (1.0 if (tp + fn) == 0 else 0.0)
                rec = (tp / (tp + fn)) if (tp + fn) > 0 else 1.0
                f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

                # Color accuracy on truly-occupied cells.
                occ_n = int(_np.count_nonzero(obs_occ))
                if occ_n > 0:
                    color_eq = float(_np.mean(pred_i[obs_occ] == obs_i[obs_occ]))
                else:
                    color_eq = 1.0

                # Weighted MAE that down-weights empty/empty matches.
                # Empty weight fixed to 0.01 here (move to cfg later if desired).
                w = _np.where(U, 1.0, 0.01).astype(_np.float32, copy=False)
                w_mae = float(_np.sum(w * _np.abs(x_prior - obs)) / max(1e-12, float(_np.sum(w))))

                last_trace["prior_obs_mae"] = prior_mae_full
                last_trace["posterior_obs_mae"] = post_mae_full
                last_trace["prior_eq_frac"] = prior_eq
                last_trace["post_eq_frac"] = post_eq

                last_trace["nonempty_union_n"] = U_n
                last_trace["prior_eq_union"] = prior_eq_U
                last_trace["prior_obs_mae_union"] = prior_mae_U
                last_trace["occ_precision"] = float(prec)
                last_trace["occ_recall"] = float(rec)
                last_trace["occ_f1"] = float(f1)
                last_trace["color_eq_on_occ"] = float(color_eq)
                last_trace["prior_obs_wmae_empty001"] = float(w_mae)
            except Exception:
                pass

            try:
                txt = make_status_text_v5(
                    agent_state=st,
                    cfg=cfg_w,
                    trace=last_trace,
                    autosave_in_s=0.0,
                    periph_ring=periph_ring_inner,
                    env_t=env_t,
                    agent_processed_env_t=env_t,
                )
                status_plain = txt.plain
            except Exception:
                status_plain = "status unavailable"

            status_bytes = status_plain.encode("utf-8")[:status_max]

            _atomic_write_out(
                shm_out_w.buf,
                seq=int(in_seq),
                agent_t=int(getattr(st, "t", 0)),
                env_t=int(env_t),
                last_sig64=int(getattr(st, "last_sig64", 0) or 0),
                x_prior_f32=x_prior,
                status_bytes=status_bytes,
                out_hdr=out_hdr,
                status_max=status_max,
            )

            _time.sleep(0.001)

    finally:
        try:
            shm_in_w.close()
        except Exception:
            pass
        try:
            shm_out_w.close()
        except Exception:
            pass


# -----------------------------
# Decoupled harness
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--state-path", type=str, default=STATE_PATH)
    parser.add_argument("--autosave-every-s", type=float, default=30.0)
    parser.add_argument("--periph-ring", type=int, default=4)
    parser.add_argument("--grid-size", type=int, default=20)
    args = parser.parse_args()

    import multiprocessing as mp

    from nupca3.config import default_config

    side = int(args.grid_size)
    D = side * side

    state_path = str(args.state_path)
    periph_ring = int(args.periph_ring)

    perm, inv_perm = build_periphery_permutation(side=side, ring=periph_ring)

    loaded = False
    cfg = default_config()
    cfg = dc_replace(cfg, D=D, B=D, grid_width=side, grid_height=side, grid_channels=1, fovea_blocks_per_step=D)
    cfg.validate()

    if os.path.exists(state_path) and state_path.endswith(".npz"):
        try:
            cfg_loaded, _lib_loaded, perm_loaded, ring_loaded, _ = load_nupca5_state_npz(state_path)
            if int(cfg_loaded.D) == D and int(cfg_loaded.B) == D:
                cfg = cfg_loaded
                perm = perm_loaded.astype(np.int32, copy=False)
                inv_perm = np.argsort(perm)
                periph_ring = int(ring_loaded)
                loaded = True
        except Exception as e:
            print(f"State load failed ({state_path}): {e}")

    world = GridWorld(grid_size=side)

    OBS_N = D
    OBS_BYTES = OBS_N * 4
    IN_HDR = 16
    OUT_HDR = 32
    STATUS_MAX = 4096

    shm_in = SharedMemory(create=True, size=IN_HDR + OBS_BYTES)
    shm_out = SharedMemory(create=True, size=OUT_HDR + OBS_BYTES + 4 + STATUS_MAX)
    shm_in.buf[:] = b"\x00" * (IN_HDR + OBS_BYTES)
    shm_out.buf[:] = b"\x00" * (OUT_HDR + OBS_BYTES + 4 + STATUS_MAX)

    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=True)

    cfg_json = json.dumps(asdict(cfg), sort_keys=True)

    proc = ctx.Process(
        target=agent_worker,
        args=(shm_in.name, shm_out.name, child_conn, cfg_json, state_path, perm.copy(), periph_ring, loaded),
        kwargs=dict(obs_n=OBS_N, in_hdr=IN_HDR, out_hdr=OUT_HDR, status_max=STATUS_MAX),
        daemon=True,
    )
    proc.start()

    env_seq = 0
    last_out_seq = 0
    x_prior_perm = np.zeros(D, dtype=np.float32)
    status_plain = ""

    autosave_every_s = float(args.autosave_every_s)
    last_autosave_wall = time.time()

    console = Console()

    env_grid = world.get_current_grid()
    env_text = grid_to_text(env_grid)
    pred_text = grid_to_text(np.zeros_like(env_grid))
    diff_text = diff_to_text(np.zeros_like(env_grid), env_grid)
    status_text = Text("starting...", style="white")
    layout = build_layout(env_text, pred_text, diff_text, status_text, pane_w=24, pane_h=22)

    try:
        with KeyPoller() as kp, Live(layout, console=console, refresh_per_second=30, screen=True):
            while True:
                if not proc.is_alive():
                    raise RuntimeError("agent worker died")

                world.update()
                env_grid = world.get_current_grid()
                env_flat = env_grid.reshape(-1).astype(np.float32, copy=False)
                env_flat_perm = env_flat[perm]

                env_seq += 1
                _atomic_write_in(shm_in.buf, seq=env_seq, env_t=world.t, obs_f32=env_flat_perm, in_hdr=IN_HDR)

                while parent_conn.poll():
                    _ = parent_conn.recv()

                out = _atomic_try_read_out(
                    shm_out.buf,
                    last_seq=last_out_seq,
                    obs_n=OBS_N,
                    out_hdr=OUT_HDR,
                    status_max=STATUS_MAX,
                )
                if out is not None:
                    out_seq, _agent_t, _proc_env_t, _last_sig, x_prior_perm, status_plain = out
                    last_out_seq = int(out_seq)

                env_show = env_flat_perm[inv_perm].reshape(side, side).astype(np.int32, copy=False)
                pred_show = x_prior_perm[inv_perm].reshape(side, side)
                pred_int = np.rint(np.clip(pred_show, 0, 4)).astype(np.int32)

                env_text = grid_to_text(env_show)
                pred_text = grid_to_text(pred_int)
                diff_text = diff_to_text(pred_int, env_show)

                autosave_in = max(0.0, autosave_every_s - (time.time() - last_autosave_wall)) if autosave_every_s > 0 else 0.0

                if status_plain:
                    patched_lines = []
                    for ln in status_plain.splitlines():
                        s = ln.strip()
                        if s.startswith("autosave_in="):
                            suffix = ""
                            if "  (keys:" in ln:
                                suffix = ln[ln.find("  (keys:") :]
                            patched_lines.append(f"autosave_in={autosave_in:5.1f}s{suffix}")
                        else:
                            patched_lines.append(ln)
                    status_text = Text("\n".join(patched_lines))
                else:
                    status_text = Text(f"t={world.t}  (waiting for agent)\nautosave_in={autosave_in:5.1f}s")

                layout["top"]["env"].update(Panel(env_text, title=f"ENV t={world.t}", padding=(0, 0), box=box.SQUARE))
                layout["top"]["pred"].update(Panel(pred_text, title="PRED (agent may lag)", padding=(0, 0), box=box.SQUARE))
                layout["top"]["diff"].update(Panel(diff_text, title="DIFF", padding=(0, 0), box=box.SQUARE))
                layout["bottom"]["status"].update(Panel(status_text, title="NUPCA5", padding=(0, 1), box=box.SQUARE))

                k = kp.poll()
                if k:
                    kk = k.lower()
                    if kk == "q":
                        parent_conn.send({"cmd": "quit", "save": True})
                        break
                    if kk == "s":
                        parent_conn.send({"cmd": "save"})
                    if kk == "r":
                        parent_conn.send({"cmd": "reset", "clear_memory": False})
                    if kk == "c":
                        parent_conn.send({"cmd": "reset", "clear_memory": True})

                now = time.time()
                if autosave_every_s > 0 and (now - last_autosave_wall) >= autosave_every_s:
                    last_autosave_wall = now
                    parent_conn.send({"cmd": "save"})

                if UPDATE_DELAY_MS > 0:
                    time.sleep(UPDATE_DELAY_MS / 1000.0)

    finally:
        try:
            parent_conn.send({"cmd": "quit", "save": True})
        except Exception:
            pass
        try:
            proc.join(timeout=1.0)
        except Exception:
            pass
        try:
            shm_in.close()
            shm_in.unlink()
        except Exception:
            pass
        try:
            shm_out.close()
            shm_out.unlink()
        except Exception:
            pass


if __name__ == "__main__":
    main()
