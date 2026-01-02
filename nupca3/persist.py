"""nupca3/persist.py

NUPCA5 checkpoint persistence (NPZ only).

Rules enforced:
  - No `pickle`.
  - The expert library is persisted only via `ExpertLibrary.pack()`.
  - Checkpoints are a single `.npz` loaded with `allow_pickle=False`.

Intentionally NOT persisted (v5 durability boundary):
  - ObservationBuffer contents (dense x_last/x_prior/observed_dims).
  - Macrostate Q_struct (REST structural queue).
  - Pending validation queue / prediction refs (must be rebuilt online).
  - Large diagnostic dicts/logs on AgentState.
"""

from __future__ import annotations

import json
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, Tuple
from collections import deque

import numpy as np
from .memory.pred_store import PredStore
from .memory.trace_cache import init_trace_cache

from .config import AgentConfig
from .types import (
    AgentState,
    Margins,
    Stress,
    Baselines,
    MacrostateVars,
    FoveaState,
    ObservationBuffer,
    PackedExpertLibrary,
    ExpertLibrary,
)

from .geometry.block_spec import build_block_specs, BlockView
from .geometry.fovea import build_blocks_from_cfg
from .incumbents import rebuild_incumbents_by_block

_META_KEY = "__meta_json_utf8__"
_SCHEMA = "nupca5-checkpoint-1"


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, (str, bool)) or obj is None:
        return obj
    if isinstance(obj, (int, float)):
        return obj
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if is_dataclass(obj):
        return {k: _json_safe(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, set):
        return sorted([_json_safe(v) for v in obj])
    if isinstance(obj, np.ndarray):
        # Never inline arrays into JSON meta. Caller must store arrays in the NPZ.
        return {"__ndarray__": True, "shape": list(obj.shape), "dtype": str(obj.dtype)}
    raise TypeError(f"Non-JSON-safe type in checkpoint meta: {type(obj)!r}")


def _require_npz(path: Path) -> Path:
    p = Path(path)
    if p.suffix.lower() != ".npz":
        raise ValueError("NUPCA5 checkpoint must be a single .npz file")
    return p


def _dim2block_from_blocks(blocks: list[list[int]], D: int) -> np.ndarray:
    """Build dim->block map consistent with the actual block partition."""
    D = int(D)
    m = np.full(D, -1, dtype=np.int32)
    for bi, dims in enumerate(blocks):
        for d in dims:
            dd = int(d)
            if 0 <= dd < D:
                m[dd] = int(bi)
    return m


def save_checkpoint(path: Path, cfg: AgentConfig, state: AgentState) -> None:
    p = _require_npz(path)
    if not isinstance(cfg, AgentConfig):
        raise TypeError("cfg must be AgentConfig")
    if not isinstance(state, AgentState):
        raise TypeError("state must be AgentState")

    arrays: Dict[str, np.ndarray] = {}

    # ---------------------------
    # Durable library (packed)
    # ---------------------------
    packed = state.library.pack()
    lib_scalars: Dict[str, Any] = {}
    for f in fields(PackedExpertLibrary):
        v = getattr(packed, f.name)
        if isinstance(v, np.ndarray):
            arrays[f"lib.{f.name}"] = np.asarray(v)
        else:
            lib_scalars[f.name] = _json_safe(v)

    # ---------------------------
    # Optional packed sig_index cache (error cache only; buckets can be rebuilt)
    # ---------------------------
    sig_meta: Dict[str, Any] = {"present": False, "err_cache_present": False}
    si = getattr(state.library, "sig_index", None)
    if si is not None:
        # Duck-typing to avoid pickling / object-graph persistence.
        err_cache = getattr(si, "_err_cache", None)
        if err_cache is not None:
            arrays["sig_index.err_cache"] = np.asarray(err_cache, dtype=np.float32)
            sig_meta["err_cache_present"] = True
        # Back-compat: if buckets/salts exist, store them too.
        if hasattr(si, "buckets") and hasattr(si, "salts"):
            arrays["sig_index.buckets"] = np.asarray(si.buckets, dtype=np.int32)
            arrays["sig_index.salts"] = np.asarray(si.salts, dtype=np.uint64)
        sig_meta["present"] = True

    # ---------------------------
    # Small state needed to resume
    # ---------------------------
    arrays["state.baselines.mu"] = np.asarray(state.baselines.mu)
    arrays["state.baselines.var_fast"] = np.asarray(state.baselines.var_fast)
    arrays["state.baselines.var_slow"] = np.asarray(state.baselines.var_slow)
    tilde_prev = state.baselines.tilde_prev
    if tilde_prev is not None:
        arrays["state.baselines.tilde_prev"] = np.asarray(tilde_prev)
        tilde_prev_present = True
    else:
        tilde_prev_present = False

    arrays["state.fovea.block_residual"] = np.asarray(state.fovea.block_residual)
    arrays["state.fovea.block_age"] = np.asarray(state.fovea.block_age)
    arrays["state.fovea.block_uncertainty"] = np.asarray(state.fovea.block_uncertainty)
    arrays["state.fovea.block_costs"] = np.asarray(state.fovea.block_costs)
    arrays["state.fovea.routing_scores"] = np.asarray(state.fovea.routing_scores)
    arrays["state.fovea.block_disagreement"] = np.asarray(state.fovea.block_disagreement)
    arrays["state.fovea.block_innovation"] = np.asarray(state.fovea.block_innovation)
    arrays["state.fovea.block_periph_demand"] = np.asarray(state.fovea.block_periph_demand)
    arrays["state.fovea.block_confidence"] = np.asarray(state.fovea.block_confidence)

    arrays["state.sig_prev_counts"] = np.asarray(state.sig_prev_counts, dtype=np.int16)
    arrays["state.sig_prev_hist"] = np.asarray(state.sig_prev_hist, dtype=np.uint16)
    arrays["state.P_nov_state"] = np.asarray(getattr(state, "P_nov_state", np.zeros(0, dtype=float)), dtype=float)
    arrays["state.U_prev_state"] = np.asarray(getattr(state, "U_prev_state", np.zeros(0, dtype=float)), dtype=float)

    # NOTE: pending_validation is intentionally NOT persisted in v5.

    meta = {
        "schema": _SCHEMA,
        "cfg": _json_safe(asdict(cfg)),
        "packed_lib_scalars": lib_scalars,
        "sig_index": sig_meta,
        "state": {
            "t_w": int(state.t_w),
            "k_op": int(state.k_op),
            "wall_ms": int(state.wall_ms),
            "E": float(state.E),
            "D": float(state.D),
            "drift_P": float(state.drift_P),
            "arousal": float(state.arousal),
            "b_cons": float(getattr(state, "b_cons", 0.0)),
            "margins": _json_safe(state.margins),
            "stress": _json_safe(state.stress),
            "macro": {
                "rest": bool(state.macro.rest),
                "T_since": int(getattr(state.macro, "T_since", 0)),
                "T_rest": int(getattr(state.macro, "T_rest", 0)),
                "P_rest": float(getattr(state.macro, "P_rest", 0.0)),
                "rest_cooldown": int(getattr(state.macro, "rest_cooldown", 0)),
                "rest_zero_processed_streak": int(getattr(state.macro, "rest_zero_processed_streak", 0)),
            },
            "baselines": {
                "last_struct_edit_t": int(getattr(state.baselines, "last_struct_edit_t", -10**9)),
                "tilde_prev_present": bool(tilde_prev_present),
            },
            "fovea": {
                "routing_last_t": int(getattr(state.fovea, "routing_last_t", -1)),
                "coverage_cursor": int(getattr(state.fovea, "coverage_cursor", 0)),
                "current_blocks": sorted(int(b) for b in (state.fovea.current_blocks or set())),
            },
            "sig": {
                "last_sig64": None
                if state.last_sig64 is None
                else int(state.last_sig64) & ((1 << 64) - 1),
            },
            "value_of_compute": {
                "value_of_compute": float(getattr(state, "value_of_compute", 0.0)),
                "hazard_pressure": float(getattr(state, "hazard_pressure", 0.0)),
                "novelty_pressure": float(getattr(state, "novelty_pressure", 0.0)),
            },
            "prev": {
                "rest_permitted_prev": bool(getattr(state, "rest_permitted_prev", True)),
                "demand_prev": bool(getattr(state, "demand_prev", False)),
                "interrupt_prev": bool(getattr(state, "interrupt_prev", False)),
                "arousal_prev": float(getattr(state, "arousal_prev", 0.0)),
                "s_int_need_prev": float(getattr(state, "s_int_need_prev", 0.0)),
                "s_ext_th_prev": float(getattr(state, "s_ext_th_prev", 0.0)),
                "x_C_prev": float(getattr(state, "x_C_prev", 0.0)),
                "rawE_prev": float(getattr(state, "rawE_prev", 0.0)),
                "rawD_prev": float(getattr(state, "rawD_prev", 0.0)),
                "c_d_prev": float(getattr(state, "c_d_prev", 1.0)),
            },
        },
    }

    meta_json = json.dumps(meta, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    arrays[_META_KEY] = np.frombuffer(meta_json, dtype=np.uint8)

    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, **arrays)


def load_checkpoint(path: Path) -> Tuple[AgentConfig, AgentState]:
    p = _require_npz(path)
    with np.load(p, allow_pickle=False) as z:
        if _META_KEY not in z:
            raise ValueError("Not a NUPCA5 checkpoint: missing meta key")
        meta = json.loads(bytes(z[_META_KEY].astype(np.uint8)).decode("utf-8"))

        if meta.get("schema") != _SCHEMA:
            raise ValueError(f"Unsupported checkpoint schema: {meta.get('schema')!r}")

        cfg = AgentConfig(**meta["cfg"])  # type: ignore[arg-type]

        # Rebuild packed library
        lib_kwargs: Dict[str, Any] = {}
        lib_scalars = meta.get("packed_lib_scalars", {})
        for f in fields(PackedExpertLibrary):
            key = f"lib.{f.name}"
            if key in z:
                lib_kwargs[f.name] = np.asarray(z[key])
            else:
                if f.name not in lib_scalars:
                    raise ValueError(f"Checkpoint missing PackedExpertLibrary field: {f.name}")
                lib_kwargs[f.name] = lib_scalars[f.name]
        packed = PackedExpertLibrary(**lib_kwargs)  # type: ignore[arg-type]

        base_lib = ExpertLibrary.unpack(packed)

        from .memory.library import V5ExpertLibrary

        lib = V5ExpertLibrary(
            nodes=base_lib.nodes,
            anchors=base_lib.anchors,
            footprint_index=base_lib.footprint_index,
            next_node_id=int(getattr(base_lib, "next_node_id", 0)),
            revision=int(getattr(base_lib, "revision", 0)),
        )

        # Restore/build sig_index (v5 requires an index; buckets are rebuilt deterministically).
        from .memory.sig_index import PackedSigIndex

        si = PackedSigIndex.from_cfg_obj(cfg)
        # Load error cache first so overflow replacement uses vetted priority.
        if "sig_index.err_cache" in z:
            ec = np.asarray(z["sig_index.err_cache"], dtype=np.float32)
            if ec.ndim != 2:
                raise ValueError("sig_index.err_cache must be a 2D array")
            setattr(si, "_err_cache", ec.copy())
        lib.sig_index = si

        # Rebuild small state
        st = meta["state"]

        margins = Margins(**st["margins"])
        stress = Stress(**st["stress"])

        mu = np.asarray(z["state.baselines.mu"]).copy()
        var_fast = np.asarray(z["state.baselines.var_fast"]).copy()
        var_slow = np.asarray(z["state.baselines.var_slow"]).copy()
        tilde_prev = None
        if bool(st["baselines"].get("tilde_prev_present", False)):
            if "state.baselines.tilde_prev" not in z:
                raise ValueError("Checkpoint indicates tilde_prev present but array is missing")
            tilde_prev = np.asarray(z["state.baselines.tilde_prev"]).copy()
        baselines = Baselines(
            mu=mu,
            var_fast=var_fast,
            var_slow=var_slow,
            tilde_prev=tilde_prev,
            last_struct_edit_t=int(st["baselines"].get("last_struct_edit_t", -10**9)),
        )

        macro_d = st["macro"]
        macro = MacrostateVars(
            rest=bool(macro_d.get("rest", False)),
            Q_struct=[],
            T_since=int(macro_d.get("T_since", 0)),
            T_rest=int(macro_d.get("T_rest", 0)),
            P_rest=float(macro_d.get("P_rest", 0.0)),
            rest_cooldown=int(macro_d.get("rest_cooldown", 0)),
            rest_zero_processed_streak=int(macro_d.get("rest_zero_processed_streak", 0)),
        )

        fov = FoveaState(
            block_residual=np.asarray(z["state.fovea.block_residual"]).copy(),
            block_age=np.asarray(z["state.fovea.block_age"]).copy(),
            block_uncertainty=np.asarray(z["state.fovea.block_uncertainty"]).copy(),
            block_costs=np.asarray(z["state.fovea.block_costs"]).copy(),
            routing_scores=np.asarray(z["state.fovea.routing_scores"]).copy(),
            block_disagreement=np.asarray(z["state.fovea.block_disagreement"]).copy(),
            block_innovation=np.asarray(z["state.fovea.block_innovation"]).copy(),
            block_periph_demand=np.asarray(z["state.fovea.block_periph_demand"]).copy(),
            block_confidence=np.asarray(z["state.fovea.block_confidence"]).copy(),
            routing_last_t=int(st["fovea"].get("routing_last_t", -1)),
            current_blocks=set(int(b) for b in (st["fovea"].get("current_blocks", []) or [])),
            coverage_cursor=int(st["fovea"].get("coverage_cursor", 0)),
        )

        # ObservationBuffer is intentionally cleared on load.
        D = int(getattr(cfg, "D", 0))
        buf = ObservationBuffer(
            x_last=np.zeros(D, dtype=float),
            x_prior=np.zeros(D, dtype=float),
            observed_dims=set(),
        )

        state = AgentState(
            t_w=int(st["t_w"]),
            k_op=int(st.get("k_op", 0)),
            wall_ms=int(st.get("wall_ms", 0)),
            E=float(st["E"]),
            D=float(st["D"]),
            drift_P=float(st["drift_P"]),
            margins=margins,
            stress=stress,
            arousal=float(st["arousal"]),
            baselines=baselines,
            macro=macro,
            fovea=fov,
            buffer=buf,
            library=lib,
            b_cons=float(st.get("b_cons", 0.0)),
        )

        B = int(getattr(cfg, "B", 0))
        if "state.P_nov_state" in z:
            state.P_nov_state = np.asarray(z["state.P_nov_state"], dtype=float).copy()
        else:
            state.P_nov_state = np.zeros(B, dtype=float)
        if "state.U_prev_state" in z:
            state.U_prev_state = np.asarray(z["state.U_prev_state"], dtype=float).copy()
        else:
            state.U_prev_state = np.zeros(B, dtype=float)
        adviser_meta = st.get("value_of_compute", {}) or {}
        state.value_of_compute = float(adviser_meta.get("value_of_compute", 0.0))
        state.hazard_pressure = float(adviser_meta.get("hazard_pressure", 0.0))
        state.novelty_pressure = float(adviser_meta.get("novelty_pressure", 0.0))

        # Restore v5 sig state
        state.last_sig64 = st.get("sig", {}).get("last_sig64", None)
        state.sig_prev_counts = np.asarray(z["state.sig_prev_counts"], dtype=np.int16).copy()
        state.sig_prev_hist = np.asarray(z["state.sig_prev_hist"], dtype=np.uint16).copy()

        # Restore prev scalars
        prev = st.get("prev", {}) or {}
        state.rest_permitted_prev = bool(prev.get("rest_permitted_prev", True))
        state.demand_prev = bool(prev.get("demand_prev", False))
        state.interrupt_prev = bool(prev.get("interrupt_prev", False))
        state.arousal_prev = float(prev.get("arousal_prev", 0.0))
        state.s_int_need_prev = float(prev.get("s_int_need_prev", 0.0))
        state.s_ext_th_prev = float(prev.get("s_ext_th_prev", 0.0))
        state.x_C_prev = float(prev.get("x_C_prev", 0.0))
        state.rawE_prev = float(prev.get("rawE_prev", 0.0))
        state.rawD_prev = float(prev.get("rawD_prev", 0.0))
        state.c_d_prev = float(prev.get("c_d_prev", 1.0))

        # Pending validation queue is intentionally cleared on load in v5.
        state.pending_validation = deque()

        # Rebuild geometry + incumbents indices (derived state).
        blocks = build_blocks_from_cfg(cfg)
        if blocks and len(blocks) != int(getattr(cfg, "B", len(blocks))):
            raise ValueError("Checkpoint cfg.B does not match block partition")
        block_specs = build_block_specs(blocks, cost_fn=lambda dims: float(max(1, len(dims))))
        state.blocks = [dims.copy() for dims in blocks]
        state.block_specs = block_specs
        state.block_view = BlockView(block_specs)
        state.incumbents_by_block = rebuild_incumbents_by_block(state.library, state.blocks)
        state.incumbents_revision = int(getattr(state.library, "revision", 0))

        # Ensure v5 lifecycle removal works: compute and store node.sig_index_blocks.
        from .memory.library import register_unit_in_sig_index

        D0 = int(getattr(cfg, "D", 0))
        dim2block = _dim2block_from_blocks(state.blocks, D0) if D0 > 0 else None
        setattr(state.library, "_sig_dim2block", dim2block)

        # Rebuild buckets deterministically from loaded nodes.
        si = getattr(state.library, "sig_index", None)
        if si is None:
            from .memory.sig_index import PackedSigIndex

            state.library.sig_index = PackedSigIndex.from_cfg_obj(cfg)
            si = state.library.sig_index

        # Clear buckets if the structure exists.
        if hasattr(si, "buckets"):
            try:
                si.buckets.fill(-1)
            except Exception:
                pass

        for nid in sorted(int(k) for k in state.library.nodes.keys()):
            node = state.library.nodes[int(nid)]
            addr = int(getattr(node, "unit_sig64", 0)) & ((1 << 64) - 1)
            if addr == 0:
                raise ValueError(f"NUPCA5: loaded node {nid} has unit_sig64=0 (invalid stored address)")
            register_unit_in_sig_index(state.library, node, dim2block=dim2block)

        state.pred_store = PredStore(capacity=int(cfg.pred_store_capacity))
        state.budget_degradation_level = getattr(state, "budget_degradation_level", 0)
        state.budget_degradation_history = tuple(getattr(state, "budget_degradation_history", ()))
        state.budget_hat_max = float(getattr(state, "budget_hat_max", cfg.B_rt))
        state.thread_pinned_units = set(getattr(state, "thread_pinned_units", set()))
        state.trace_cache = init_trace_cache(cfg)
        return cfg, state
