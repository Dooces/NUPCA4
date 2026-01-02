"""nupca3/step_pipeline/core.py

Core step pipeline state machine built from helper modules.
"""

from __future__ import annotations

import heapq
from collections import deque
from dataclasses import asdict
import math
from typing import Any, Dict, Tuple, Set, List, Deque

import numpy as np

from ..config import AgentConfig
from ..control.budget import compute_budget_and_horizon
from ..control.governor import create_budget_meter
from ..control.intake import decide_intake_and_compute
from ..control.commitment import commit_gate, select_action
from ..control.edit_control import freeze_predicate, permit_param_updates
from ..diagnostics.metrics import compute_feel_proxy
from ..dynamics.margin_dynamics import HardState, step_hard_dynamics
from ..edits.proposals import propose_structural_edits
from ..edits.rest_processor import RestProcessingResult, process_struct_queue
from ..geometry.binding import (build_binding_maps, select_best_binding,
                                 select_best_binding_by_fit)
from ..geometry.fovea import dims_for_block, make_observation_set, update_fovea_tracking
from ..geometry.streams import (apply_transport, compute_transport_shift,
                                extract_coarse)
from ..memory.completion import complete
from ..memory.expert import sgd_update
from ..memory.fusion import fuse_predictions
from ..memory.rollout import rollout_and_confidence
from ..memory.sig64 import Sig64Meta, compute_sig64_from_obs
from ..memory.salience import (
    SalienceResult,
    compute_activations,
    compute_salience,
    compute_temperature,
    get_stress_signals,
    infer_node_band_level,
)
from ..memory.working_set import select_working_set
from ..memory.pred_store import emit_pred_snapshot, process_due_validations
from ..memory.trace_cache import cache_observation
from ..state.baselines import (commit_tilde_prev, normalize_margins,
                               update_baselines)
from ..state.macrostate import evolve_macrostate, rest_permitted
from ..state.margins import compute_arousal, compute_stress
from ..state.stability import update_stability_metrics
from ..types import (AgentState, Action, EnvObs, FootprintResidualStats,
                     LearningCache, Margins, PersistentResidualState,
                     PlanningThread, StepTrace, TransitionRecord)

from .fovea import FoveaSignals, apply_signals_and_select
from .learning import (_build_training_mask, _derive_margins,
                       _feature_probe_vectors, LearningProcessor)
from .logging import _dbg
from .observations import (
    _cfg_D,
    _coarse_summary,
    _compute_peripheral_gist,
    _compute_peripheral_metrics,
    _filter_cue_to_Oreq,
    _peripheral_dim_set,
    _prior_obs_mae,
    _support_window_union,
    _update_context_register,
    _update_context_tags,
    _update_observed_history,
    compute_block_uncertainty,
)
from .transport import (_compute_transport_disagreement_blocks,
                        _select_transport_delta,
                        _update_transport_learning_state,
                        synthesize_action)
from .worlds import _build_world_hypotheses, _compute_block_signals


learning_processor = LearningProcessor()


def _copy_array(arr: np.ndarray | None) -> np.ndarray:
    """Return a safe copy of an array-like or an empty array when missing."""
    if arr is None:
        return np.zeros(0, dtype=float)
    return np.asarray(arr).copy()


def _serialize_pending_validation(queue: Deque | None) -> Tuple[Tuple[int, int, int, int, float], ...]:
    if queue is None:
        return tuple()
    return tuple(
        (
            int(getattr(item, "node_id", 0)),
            int(getattr(item, "h_bin", 0)),
            int(getattr(item, "t_emit", 0)),
            int(getattr(item, "dist", 0)),
            float(getattr(item, "err", 0.0)),
        )
        for item in queue
    )


def _serialize_learning_candidates(stats: Dict[str, Any] | None) -> Tuple[Tuple[str, Any], ...]:
    if not stats:
        return tuple()
    entries: list[Tuple[str, Any]] = []
    for key in sorted(stats):
        value = stats.get(key)
        if isinstance(value, dict):
            entries.append((key, tuple(sorted((k, v) for k, v in value.items()))))
        else:
            entries.append((key, value))
    return tuple(entries)


def _snapshot_planning_forbidden(state: AgentState) -> Dict[str, Any]:
    buffer = getattr(state, "buffer", None)
    fovea = getattr(state, "fovea", None)
    return {
        "buffer_x_last": _copy_array(getattr(buffer, "x_last", None)),
        "buffer_x_prior": _copy_array(getattr(buffer, "x_prior", None)),
        "buffer_observed_dims": frozenset(getattr(buffer, "observed_dims", set()) or set()),
        "fovea_block_residual": _copy_array(getattr(fovea, "block_residual", None)),
        "fovea_block_age": _copy_array(getattr(fovea, "block_age", None)),
        "fovea_block_uncertainty": _copy_array(getattr(fovea, "block_uncertainty", None)),
        "fovea_block_disagreement": _copy_array(getattr(fovea, "block_disagreement", None)),
        "fovea_block_innovation": _copy_array(getattr(fovea, "block_innovation", None)),
        "fovea_block_periph_demand": _copy_array(getattr(fovea, "block_periph_demand", None)),
        "fovea_block_confidence": _copy_array(getattr(fovea, "block_confidence", None)),
        "coverage_expert_debt": tuple(sorted((int(k), int(v)) for k, v in (getattr(state, "coverage_expert_debt", {}) or {}).items())),
        "coverage_band_debt": tuple(sorted((int(k), int(v)) for k, v in (getattr(state, "coverage_band_debt", {}) or {}).items())),
        "coverage_debt_last_seen": tuple(sorted((int(k), int(v)) for k, v in (getattr(state, "coverage_debt_last_seen", {}) or {}).items())),
        "pending_validation": _serialize_pending_validation(getattr(state, "pending_validation", None)),
        "learning_candidates_prev": _serialize_learning_candidates(getattr(state, "learning_candidates_prev", None)),
    }


def _assert_planning_fence(before: Dict[str, Any], after: Dict[str, Any]) -> None:
    for key, before_val in before.items():
        after_val = after.get(key)
        if isinstance(before_val, np.ndarray):
            if not isinstance(after_val, np.ndarray) or before_val.shape != after_val.shape or not np.array_equal(before_val, after_val):
                raise AssertionError(f"Planning micro-steps mutated {key}")
        else:
            if before_val != after_val:
                raise AssertionError(f"Planning micro-steps mutated {key}")


def _ensure_planning_threads(state: AgentState, cfg: AgentConfig) -> None:
    threads = getattr(state, "planning_threads", {}) or {}
    if threads:
        return
    cap = max(1, int(getattr(cfg, "planning_threads_max", 1)))
    focus_blocks = tuple(getattr(state, "planning_target_blocks") or ())
    t_w = int(getattr(state, "t_w", 0))
    state.planning_threads = {
        tid: PlanningThread(
            thread_id=tid,
            focus_blocks=focus_blocks,
            focus_object=None,
            timeline=tuple(),
            status="idle",
            plan_state={},
            last_progress_t_w=t_w,
        )
        for tid in range(cap)
    }


def _prepare_planning_stage(state: AgentState, cfg: AgentConfig) -> Dict[int, PlanningThread]:
    _ensure_planning_threads(state, cfg)
    threads = getattr(state, "planning_threads", {}) or {}
    focus_blocks = tuple(getattr(state, "planning_target_blocks") or ())
    stage: Dict[int, PlanningThread] = {}
    for tid, thread in threads.items():
        stage_focus = focus_blocks if focus_blocks else tuple(thread.focus_blocks)
        stage[tid] = PlanningThread(
            thread_id=thread.thread_id,
            focus_blocks=stage_focus,
            focus_object=thread.focus_object,
            timeline=tuple(thread.timeline),
            status=thread.status,
            plan_state=dict(thread.plan_state),
            last_progress_t_w=thread.last_progress_t_w,
        )
    state.planning_thread_stage = stage
    return stage


def _build_planning_scheduler(stage: Dict[int, PlanningThread]) -> list[tuple[int, int]]:
    heap: list[tuple[int, int]] = []
    for tid, thread in stage.items():
        heap.append((thread.last_progress_t_w, tid))
    heapq.heapify(heap)
    return heap


def _advance_planning_thread_stage(thread: PlanningThread, state: AgentState) -> None:
    thread.status = "running"
    current_t_w = int(getattr(state, "t_w", 0))
    thread.timeline = tuple(list(thread.timeline) + [current_t_w])
    thread.plan_state["micro_steps"] = int(thread.plan_state.get("micro_steps", 0)) + 1
    thread.last_progress_t_w = current_t_w


def _commit_planning_stage(state: AgentState) -> None:
    stage = getattr(state, "planning_thread_stage", {}) or {}
    if not stage:
        state.planning_thread_stage = {}
        return
    committed: Dict[int, PlanningThread] = {}
    for tid, thread in stage.items():
        committed[tid] = PlanningThread(
            thread_id=thread.thread_id,
            focus_blocks=tuple(thread.focus_blocks),
            focus_object=thread.focus_object,
            timeline=tuple(thread.timeline),
            status=thread.status,
            plan_state=dict(thread.plan_state),
            last_progress_t_w=thread.last_progress_t_w,
        )
    state.planning_threads = committed
    state.planning_thread_stage = {}


def _compute_planning_micro_steps(state: AgentState, cfg: AgentConfig) -> int:
    max_steps = int(max(0, getattr(cfg, "planning_micro_steps_per_tick", 0)))
    if max_steps <= 0:
        return 0
    budget = float(getattr(state, "planning_budget", 0.0))
    if budget <= 0.0:
        return 0
    steps = min(max_steps, max(1, int(math.ceil(budget))))
    return steps


def _run_planning_micro_steps(state: AgentState, cfg: AgentConfig) -> None:
    if not getattr(state, "g_contemplate", False):
        state.planning_thread_stage = {}
        return
    before = _snapshot_planning_forbidden(state)
    stage = _prepare_planning_stage(state, cfg)
    steps = _compute_planning_micro_steps(state, cfg)
    if steps > 0 and stage:
        scheduler = _build_planning_scheduler(stage)
        for _ in range(steps):
            if not scheduler:
                break
            _, tid = heapq.heappop(scheduler)
            thread_stage = stage.get(tid)
            if thread_stage is None:
                continue
            _advance_planning_thread_stage(thread_stage, state)
            heapq.heappush(scheduler, (thread_stage.last_progress_t_w, tid))
            state.k_op += 1
    _commit_planning_stage(state)
    state.k_op = 0
    after = _snapshot_planning_forbidden(state)
    _assert_planning_fence(before, after)


def _ensure_block_array(arr: np.ndarray | None, B: int) -> np.ndarray:
    data = np.asarray(arr, dtype=float).reshape(-1) if arr is not None else np.zeros(0, dtype=float)
    if data.size != B:
        data = np.resize(data, (B,))
    return data


def _update_value_of_compute(
    state: AgentState,
    cfg: AgentConfig,
    innovation: np.ndarray,
    disagreement: np.ndarray,
    periph_demand: np.ndarray,
    abs_error: np.ndarray,
) -> Tuple[float, float, float]:
    B = max(0, int(getattr(cfg, "B", 0)))
    if B <= 0:
        state.value_of_compute = 0.0
        state.hazard_pressure = 0.0
        state.novelty_pressure = 0.0
        return 0.0, 0.0, 0.0

    innovation_arr = _ensure_block_array(innovation, B)
    disagreement_arr = _ensure_block_array(disagreement, B)
    periph_arr = _ensure_block_array(periph_demand, B)
    u_prev = _ensure_block_array(getattr(state, "U_prev_state", np.zeros(0, dtype=float)), B)
    periph_weight = float(getattr(cfg, "value_of_compute_periph_weight", 0.1))
    current_u = np.clip(innovation_arr + periph_weight * periph_arr, 0.0, None)
    delta_u = np.abs(current_u - u_prev)
    P_nov_state = _ensure_block_array(getattr(state, "P_nov_state", np.zeros(0, dtype=float)), B)
    decay = float(getattr(cfg, "value_of_compute_novelty_decay", 0.8))
    decay = max(0.0, min(1.0, decay))
    P_nov_state = decay * P_nov_state + (1.0 - decay) * delta_u
    state.P_nov_state = P_nov_state.copy()
    state.U_prev_state = current_u.copy()
    novelty = float(np.mean(P_nov_state)) if P_nov_state.size else 0.0
    hazard = float(np.max(disagreement_arr)) if disagreement_arr.size else 0.0
    expected_error = float(np.mean(np.abs(abs_error))) if abs_error.size else 0.0

    alpha = float(getattr(cfg, "value_of_compute_alpha", 1.0))
    beta = float(getattr(cfg, "value_of_compute_beta", 0.5))
    gamma = float(getattr(cfg, "value_of_compute_gamma", 0.5))
    value = alpha * hazard + beta * novelty + gamma * expected_error

    state.value_of_compute = max(0.0, float(value))
    state.hazard_pressure = hazard
    state.novelty_pressure = novelty
    return hazard, novelty, state.value_of_compute


def _update_coverage_debts_bounded(state: AgentState, cfg: AgentConfig, tracked_ids: Set[int]) -> None:
    """Fixed-cost coverage debt maintenance (v5).

    v5 fixed-budget semantics forbid per-step full scans over the durable
    library. Coverage bookkeeping must therefore be maintained over a bounded
    *tracked* set (typically the salience candidate set + active set).

    This function updates:
      - state.coverage_expert_debt[nid] (time-since-active counter)
      - state.coverage_band_debt[level] (time-since-band-coverage counter)
      - state.node_band_levels[nid] (cached band level, lazily inferred)

    It also prunes the tracked dictionaries to a bounded size.
    """

    library = getattr(state, "library", None)
    nodes = getattr(library, "nodes", None) if library is not None else None
    if not nodes:
        state.coverage_expert_debt = {}
        state.coverage_band_debt = {}
        state.node_band_levels = {}
        return

    # Bound the amount of bookkeeping we maintain.
    cap = int(getattr(cfg, "coverage_debt_cap", getattr(cfg, "salience_max_candidates", getattr(cfg, "max_candidates", 256))))
    cap = max(32, cap)

    active_set = {int(n) for n in getattr(state, "active_set", set()) or set()}
    tracked = {int(n) for n in (tracked_ids or set())}
    if not tracked:
        # Minimal invariants: keep empty debts.
        state.coverage_expert_debt = {}
        state.coverage_band_debt = {}
        # Keep node_band_levels as-is (salience maintains lazily).
        return

    # Intersect with existing nodes to avoid growing on stale ids.
    tracked = {nid for nid in tracked if nid in nodes}
    if not tracked:
        state.coverage_expert_debt = {}
        state.coverage_band_debt = {}
        return

    expert_debt: Dict[int, int] = dict(getattr(state, "coverage_expert_debt", {}) or {})
    band_debt: Dict[int, int] = dict(getattr(state, "coverage_band_debt", {}) or {})
    node_levels: Dict[int, int] = dict(getattr(state, "node_band_levels", {}) or {})
    last_seen: Dict[int, int] = dict(getattr(state, "coverage_debt_last_seen", {}) or {})

    t_now = int(getattr(state, "t_w", 0))

    # Lazily infer band levels only for tracked nodes.
    for nid in tracked:
        if nid not in node_levels:
            node = nodes.get(nid)
            if node is None:
                continue
            node_levels[nid] = int(infer_node_band_level(node, cfg))

    # Expert debts: increment for tracked-but-inactive, reset for active.
    debt_max = int(getattr(cfg, "coverage_debt_max", 10_000))
    debt_max = max(1, debt_max)
    for nid in tracked:
        if nid in active_set:
            expert_debt[nid] = 0
        else:
            expert_debt[nid] = min(debt_max, int(expert_debt.get(nid, 0)) + 1)
        last_seen[nid] = t_now

    # Optional innovation discount, applied only to tracked nodes (fixed cost).
    innovation_weight = float(getattr(cfg, "fovea_innovation_weight", 0.0))
    if innovation_weight > 0.0:
        innovation = np.asarray(
            getattr(state.fovea, "block_innovation", np.zeros(int(getattr(cfg, "B", 0)) or 0)),
            dtype=float,
        ).reshape(-1)
        if innovation.size:
            for nid in tracked:
                if nid in active_set:
                    continue
                node = nodes.get(nid)
                if node is None:
                    continue
                phi = int(getattr(node, "footprint", getattr(node, "block_id", -1)))
                if 0 <= phi < innovation.size and float(innovation[phi]) > 0.0:
                    expert_debt[nid] = max(0, int(expert_debt.get(nid, 0)) - 1)

    # Band debts: maintain only for levels present in tracked set.
    tracked_levels = {int(node_levels[nid]) for nid in tracked if nid in node_levels}
    active_levels = {int(node_levels[nid]) for nid in active_set if nid in node_levels}
    for level in tracked_levels:
        if level in active_levels:
            band_debt[level] = 0
        else:
            band_debt[level] = min(debt_max, int(band_debt.get(level, 0)) + 1)
    # Drop stale levels.
    for level in list(band_debt.keys()):
        if level not in tracked_levels:
            band_debt.pop(level, None)

    # Prune bookkeeping dictionaries to bounded size (LRU by last_seen).
    if len(last_seen) > cap:
        # Remove oldest entries first.
        overflow = len(last_seen) - cap
        for nid, _ts in sorted(last_seen.items(), key=lambda kv: kv[1])[:overflow]:
            last_seen.pop(nid, None)
            expert_debt.pop(nid, None)
            node_levels.pop(nid, None)

    state.coverage_expert_debt = expert_debt
    state.coverage_band_debt = band_debt
    state.node_band_levels = node_levels
    state.coverage_debt_last_seen = last_seen


def step_pipeline(state: AgentState, env_obs: EnvObs, cfg: AgentConfig) -> Tuple[Action, AgentState, Dict[str, Any]]:
    """
    Advance the agent by one timestep.

    Inputs:
      state: current AgentState
      env_obs: environment observation (sparse cue in env_obs.x_partial)
      cfg: AgentConfig

    Outputs:
      action: Action action selected for this timestep
      next_state: updated AgentState (mutated in place in this snapshot style)
      trace: dict diagnostics (runner expects a dict)
    """
    D = _cfg_D(state, cfg)
    ordering_markers: List[str] = []
    value_of_compute_prev = float(getattr(state, "value_of_compute", 0.0))
    tick_idx = int(getattr(state, "t_w", 0)) + 1
    _dbg('enter', state=state)
    _dbg(f'cfg.D={D}', state=state)
    periph_dims = _peripheral_dim_set(D, cfg)
    x_prev = np.asarray(getattr(state.buffer, "x_last", np.zeros(D)), dtype=float).reshape(-1)
    if x_prev.shape[0] != D:
        raise AssertionError(
            f"Invariant violation: buffer.x_last has shape {x_prev.shape}, expected D={D}."
        )
    x_prev_pre = x_prev.copy()
    prev_observed_dims = set(getattr(state.buffer, "observed_dims", set()) or set())
    state.scan_counter = 0
    state.k_op = 0
    meter = create_budget_meter(state, env_obs, cfg)
    state.budget_hat_max = float(meter.B_hat_max)

    transport_source = "buffer_infer"
    force_true_delta = bool(getattr(cfg, "transport_force_true_delta", False))
    env_shift: Tuple[int, int] | None = None
    use_env_grid = bool(getattr(cfg, "transport_debug_env_grid", False))
    state.grid_prev_mass = np.zeros(0, dtype=float)

    coarse_prev_snapshot = getattr(state, "coarse_prev", None)
    use_true_transport = bool(getattr(cfg, "transport_use_true_full", False))
    coarse_true = np.zeros(0, dtype=float)
    coarse_true_size = 0
    periph_full_vec = getattr(env_obs, "periph_full", None)
    if use_true_transport and periph_full_vec is not None:
        periph_arr = np.asarray(periph_full_vec, dtype=float).reshape(-1)
        if periph_arr.size < D:
            periph_arr = np.resize(periph_arr, (D,))
        elif periph_arr.size > D:
            periph_arr = periph_arr[:D]
        coarse_true = extract_coarse(periph_arr, cfg)
        coarse_true_size = coarse_true.size
    coarse_shift_hint = tuple(getattr(state, "coarse_shift", (0, 0)))
    if (
        use_true_transport
        and coarse_true_size > 0
        and coarse_prev_snapshot is not None
        and coarse_prev_snapshot.shape == coarse_true.shape
    ):
        coarse_shift_hint = compute_transport_shift(coarse_prev_snapshot, coarse_true, cfg)
    env_true_delta_hint = getattr(env_obs, "true_delta", None)
    if force_true_delta and env_true_delta_hint is not None:
        env_shift = tuple(int(v) for v in env_true_delta_hint)

    # -------------------------------------------------------------------------
    # A14.7: rest(t) from lagged predicates
    _dbg('A14.7 compute rest_t from lagged predicates', state=state)
    # -------------------------------------------------------------------------
    rest_t = bool(getattr(state, "rest_permitted_prev", True)
                  and getattr(state, "demand_prev", False)
                  and (not getattr(state, "interrupt_prev", False)))
    _dbg(
        f'rest_t={rest_t} rest_permitted_prev={getattr(state,"rest_permitted_prev",None)} '
        f'demand_prev={getattr(state,"demand_prev",None)} interrupt_prev={getattr(state,"interrupt_prev",None)}',
        state=state,
    )

    # -------------------------------------------------------------------------
    # A16.3: select fovea blocks for this step (uses t-1 tracking)
    # -------------------------------------------------------------------------
    pending_selection = getattr(state, "pending_fovea_selection", None)
    if pending_selection is None:
        periph_full = getattr(env_obs, "periph_full", None)
        pending_selection = apply_signals_and_select(
            state,
            cfg,
            periph_full=periph_full,
            prev_observed_dims=prev_observed_dims,
            value_of_compute=value_of_compute_prev,
        )
    blocks_t = pending_selection.get("blocks", []) or []
    forced_periph_blocks = pending_selection.get("forced_periph_blocks", [])
    motion_probe_blocks_used = pending_selection.get("motion_probe_blocks", [])
    state.pending_fovea_selection = None
    selected_blocks = tuple(getattr(env_obs, "selected_blocks", ()) or ())
    if selected_blocks and bool(getattr(cfg, "allow_selected_blocks_override", False)):
        blocks_t = [int(b) for b in selected_blocks]
    _dbg(f'A16.3 select_fovea -> n_blocks={len(blocks_t) if blocks_t is not None else 0}', state=state)
    state.fovea.current_blocks = set(int(b) for b in blocks_t)
    _dbg(f'A16.3 current_blocks={len(state.fovea.current_blocks)}', state=state)
    log_every = int(getattr(cfg, "fovea_log_every", 0))
    if log_every > 0 and (int(getattr(state, "t_w", 0)) % log_every == 0):
        _dbg(f'A16.3 blocks={list(blocks_t)}', state=state)
        _dbg(f'current_blocks={sorted(list(state.fovea.current_blocks))}', state=state)

    # Rolling visit counts per block over a sliding window.
    window = int(getattr(cfg, "fovea_visit_window", 256))
    if window > 0:
        visit_queue = getattr(state, "fovea_visit_window", None)
        visit_counts = getattr(state, "fovea_visit_counts", None)
        if visit_queue is None or visit_counts is None or len(getattr(visit_counts, "shape", [])) != 1:
            visit_queue = deque()
            visit_counts = np.zeros(int(cfg.B), dtype=int)
        if int(visit_counts.shape[0]) != int(cfg.B):
            visit_counts = np.resize(visit_counts, (int(cfg.B),))
            visit_counts = np.asarray(visit_counts, dtype=int)
        if len(visit_queue) >= window:
            oldest = visit_queue.popleft()
            for b in oldest:
                if 0 <= int(b) < int(cfg.B):
                    visit_counts[int(b)] -= 1
        current = [int(b) for b in blocks_t if 0 <= int(b) < int(cfg.B)]
        visit_queue.append(current)
        for b in current:
            visit_counts[int(b)] += 1
        state.fovea_visit_window = visit_queue
        state.fovea_visit_counts = visit_counts
        if log_every > 0 and (int(getattr(state, "t_w", 0)) % log_every == 0):
            if visit_counts.size:
                v_min = int(np.min(visit_counts))
                v_med = float(np.median(visit_counts))
                v_max = int(np.max(visit_counts))
                _dbg(
                    f'A16.2 visit_counts(window={window}): min={v_min} median={v_med:.1f} max={v_max}',
                    state=state,
                )

    # A16.5 requested observation set
    O_req = make_observation_set(blocks_t, cfg)
    _dbg(f'A16.5 make_observation_set -> |O_req|={len(O_req)}', state=state)
    forced_periph_dims: Set[int] = set()
    missing_periph_dims: List[int] = []
    periph_dims_present = 0
    if forced_periph_blocks:
        for b in forced_periph_blocks:
            forced_periph_dims.update(dims_for_block(b, cfg))
        if forced_periph_dims:
            missing_periph_dims = sorted(
                int(dim) for dim in forced_periph_dims if dim not in O_req
            )
            if missing_periph_dims:
                missing_head = missing_periph_dims[: min(8, len(missing_periph_dims))]
                _dbg(
                    f'A16.5 periph dims missing from O_req: count={len(missing_periph_dims)} '
                    f'head={missing_head}',
                    state=state,
                )
        periph_dims_present = int(len(forced_periph_dims & O_req))

    # Mask incoming sparse cue to O_req and bounds
    env_obs_dims = {
        int(k) for k in (getattr(env_obs, "x_partial", {}) or {}).keys() if 0 <= int(k) < D
    }
    cue_t = _filter_cue_to_Oreq(getattr(env_obs, "x_partial", {}) or {}, O_req, D)
    _dbg(f'cue_in|x_partial|={len(getattr(env_obs,"x_partial",{}) or {})}', state=state)
    O_t = set(cue_t.keys())
    _dbg(f'A16.5 cue_t filtered -> |O_t|={len(O_t)}', state=state)
    if env_obs_dims:
        env_min = min(env_obs_dims)
        env_max = max(env_obs_dims)
    else:
        env_min = None
        env_max = None
    if O_req:
        req_min = min(O_req)
        req_max = max(O_req)
    else:
        req_min = None
        req_max = None
    if O_t:
        used_min = min(O_t)
        used_max = max(O_t)
    else:
        used_min = None
        used_max = None
    _dbg(
        f'A16.5 obs_sets env_size={len(env_obs_dims)} env_min={env_min} env_max={env_max} '
        f'req_size={len(O_req)} req_min={req_min} req_max={req_max} '
        f'used_size={len(O_t)} used_min={used_min} used_max={used_max}',
        state=state,
    )

    if O_t:
        obs_idx = np.array(sorted(O_t), dtype=int)
        obs_vals = np.array([float(cue_t[int(i)]) for i in obs_idx], dtype=float)
    else:
        obs_idx = np.zeros(0, dtype=int)
        obs_vals = np.zeros(0, dtype=float)
    committed_obs_idx = obs_idx.copy()

    _update_observed_history(state, obs_idx, cfg, extra_dims=periph_dims)

    # -------------------------------------------------------------------------
    # NUPCA5: scan-proof sig64(t) from committed metadata + ephemeral periphery gist
    # (Used by PackedSigIndex retrieval; must not depend on library size.)
    # -------------------------------------------------------------------------
    gist_bins = int(cfg.sig_gist_bins)
    gist_bins = max(1, gist_bins)
    full_vec = periph_full_vec
    gist_u8 = np.zeros(0, dtype=np.uint8)
    if full_vec is not None and periph_dims:
        full_arr = np.asarray(full_vec, dtype=float).reshape(-1)
        if full_arr.size < D:
            full_arr = np.resize(full_arr, (D,))
        periph_idx = np.array(sorted(int(d) for d in periph_dims if 0 <= int(d) < D), dtype=int)
        if periph_idx.size:
            vals = full_arr[periph_idx]
            occ = (vals > 0.0).astype(np.uint8)
            if O_req:
                # Zero any periphery dims that are currently in the requested observation set.
                # (Avoids fovea leakage into the periphery gist.)
                for k, dim in enumerate(periph_idx):
                    if int(dim) in O_req:
                        occ[k] = 0
            P = int(occ.size)
            bins = min(gist_bins, P)
            chunk = (P + bins - 1) // bins
            padded = np.zeros(bins * chunk, dtype=np.uint8)
            padded[:P] = occ
            pooled = padded.reshape(bins, chunk).max(axis=1)
            gist_u8 = (pooled * np.uint8(255)).astype(np.uint8)
    prev_counts = getattr(state, "sig_prev_counts", None)
    prev_hist = getattr(state, "sig_prev_hist", None)
    prev_meta = None
    if prev_counts is not None and prev_hist is not None:
        pc = np.asarray(prev_counts, dtype=np.int16).reshape(-1)
        ph = np.asarray(prev_hist, dtype=np.uint16).reshape(-1)
        if pc.size and ph.size:
            prev_meta = Sig64Meta(counts=pc, hist=ph)
    sig64_t, meta_t = compute_sig64_from_obs(
        obs_vals,
        gist_u8,
        prev_meta=prev_meta,
        value_bins=int(cfg.sig_value_bins),
        vmax=float(cfg.sig_vmax),
        seed=int(cfg.sig_seed),
    )
    state.last_sig64 = int(sig64_t)
    state.sig_prev_counts = np.asarray(meta_t.counts, dtype=np.int16).copy()
    state.sig_prev_hist = np.asarray(meta_t.hist, dtype=np.uint16).copy()


    (
        shift,
        x_prev_post,
        transport_best_candidate,
        transport_runner_candidate,
        transport_margin_val,
        transport_confidence_prob,
        transport_prob_diff,
        transport_source_hint,
        transport_candidates_info,
        transport_null_evidence,
        transport_posterior_entropy,
        transport_score_spread,
        transport_tie_flag,
        transport_best_overlap,
    ) = _select_transport_delta(
        x_prev_pre,
        obs_idx,
        obs_vals,
        cfg,
        state,
        env_shift,
        coarse_shift_hint,
        true_delta=env_true_delta_hint,
        force_true_delta=force_true_delta,
    )
    ordering_markers.append("A13.transport")
    x_prev = x_prev_post
    candidate_shift = tuple(int(v) for v in shift)
    if transport_null_evidence:
        transport_confidence = 0.0
        transport_prob_diff = 0.0
        transport_margin_val = 0.0
    else:
        transport_confidence = transport_confidence_prob
    state.transport_confidence = float(transport_confidence)
    state.transport_margin = float(transport_margin_val)
    transport_source = transport_source_hint
    confidence_margin_threshold = float(getattr(cfg, "transport_confidence_margin", 0.25))
    if transport_prob_diff < confidence_margin_threshold:
        state.transport_disagreement_scores = _compute_transport_disagreement_blocks(
            transport_best_candidate,
            transport_runner_candidate,
            cfg,
        )
    else:
        state.transport_disagreement_scores = {}
    state.transport_disagreement_margin = float(transport_prob_diff)
    transport_score_margin = 0.0
    if transport_best_candidate is not None and transport_runner_candidate is not None:
        transport_score_margin = float(transport_best_candidate.score - transport_runner_candidate.score)
    hc_margin = float(getattr(cfg, "transport_high_confidence_margin", 0.05))
    hc_overlap = max(1, int(getattr(cfg, "transport_high_confidence_overlap", 2)))
    runner_exists = transport_runner_candidate is not None
    margin_ok = (not runner_exists) or (transport_score_margin >= hc_margin)
    first_step = int(getattr(state, "t_w", 0)) == 0
    base_confidence = (not transport_null_evidence) and margin_ok
    if first_step and obs_idx.size:
        base_confidence = True
    if first_step:
        transport_high_confidence = base_confidence
    else:
        transport_high_confidence = base_confidence and transport_best_overlap >= hc_overlap
    if not prev_observed_dims and int(getattr(state, "t_w", 0)) > 0:
        transport_high_confidence = False
    if not transport_high_confidence and first_step and bool(state.buffer.observed_dims):
        transport_high_confidence = True

    applied_shift = candidate_shift if transport_high_confidence else (0, 0)
    if applied_shift != candidate_shift:
        x_prev_post = apply_transport(x_prev_pre, applied_shift, cfg)
    x_prev = x_prev_post
    shift = applied_shift
    _update_transport_learning_state(
        state,
        cfg,
        transport_best_candidate,
        shift,
        env_true_delta_hint,
    )
    state.transport_last_delta = shift
    state.coarse_shift = shift
    transport_effect = float(np.mean(np.abs(x_prev_post - x_prev_pre))) if x_prev_pre.size else 0.0
    transport_applied_norm = float(transport_effect)

    if obs_idx.size:
        mae_pos_pre_transport = float(np.mean(np.abs(obs_vals - x_prev_pre[obs_idx])))
        mae_pos_post_transport = float("nan")
    else:
        mae_pos_pre_transport = float("nan")
        mae_pos_post_transport = float("nan")

    selected_blocks = tuple(getattr(env_obs, "selected_blocks", ()) or ())
    periph_blocks_cfg = max(0, int(getattr(cfg, "periph_blocks", 0)))
    B_cfg = max(0, int(getattr(cfg, "B", 0)))
    if periph_blocks_cfg > 0:
        periph_block_ids = tuple(range(max(0, B_cfg - periph_blocks_cfg), B_cfg))
    else:
        periph_block_ids = tuple()
    n_periph_blocks_selected = sum(1 for block in selected_blocks if block in periph_block_ids)
    n_fine_blocks_selected = max(0, len(selected_blocks) - n_periph_blocks_selected)
    periph_included = bool(n_periph_blocks_selected)

    pos_dims = getattr(env_obs, "pos_dims", None) or set()
    pos_idx = np.array(sorted({int(dim) for dim in pos_dims if 0 <= int(dim) < D}), dtype=int)
    pos_unobs_idx = np.zeros(0, dtype=int)
    pos_obs_mask = np.zeros(0, dtype=bool)
    if pos_idx.size:
        pos_obs_mask = np.isin(pos_idx, obs_idx) if obs_idx.size else np.zeros(pos_idx.shape, dtype=bool)
        pos_unobs_idx = pos_idx[~pos_obs_mask]
    true_vals = None
    true_vals_unobs = None
    mae_pos_unobs_pre_transport = float("nan")
    mae_pos_unobs_post_transport = float("nan")

    periph_selected = periph_included

    # -------------------------------------------------------------------------
    # A13 (perception): complete/clamp observed dims into prior
    # -------------------------------------------------------------------------
    x_t, Sigma_prior, prior_t = complete(
        cue_t,
        mode="perception",
        state=state,
        cfg=cfg,
        transport_shift=shift,
    )
    ordering_markers.append("A13.complete")
    _dbg('A13 complete(perception) begin', state=state)

    x_t = np.asarray(x_t, dtype=float).reshape(-1)
    prior_t = np.asarray(prior_t, dtype=float).reshape(-1)
    if x_t.shape[0] != D:
        raise AssertionError(
            f"Invariant violation: posterior x_t has shape {x_t.shape}, expected D={D}."
        )
    if prior_t.shape[0] != D:
        raise AssertionError(
            f"Invariant violation: prior_t has shape {prior_t.shape}, expected D={D}."
        )
    if obs_idx.size:
        mae_pos_post_transport = float(np.mean(np.abs(prior_t[obs_idx] - obs_vals)))
    else:
        mae_pos_post_transport = float("nan")
    _dbg('A13 complete(perception) end', state=state)

    validations_processed = process_due_validations(state, cfg, tick_idx, obs_idx, obs_vals)
    ordering_markers.append("A13.validation")
    _compute_peripheral_metrics(state, cfg, prior_t, env_obs, obs_idx, obs_vals, D, periph_dims)
    clamp_delta = x_t - prior_t
    not_obs_mask = np.ones(D, dtype=bool)
    if obs_idx.size:
        not_obs_mask[obs_idx] = False
    outside_idx = np.nonzero(not_obs_mask)[0]
    delta_outside_vals = clamp_delta[outside_idx] if outside_idx.size else np.zeros(0, dtype=float)
    delta_outside_O = float(np.mean(np.abs(delta_outside_vals))) if outside_idx.size else 0.0
    if O_t:
        obs_idx = np.array(sorted(O_t), dtype=int)
        mean_abs_clamp = float(np.mean(np.abs(clamp_delta[obs_idx]))) if obs_idx.size else 0.0
    else:
        mean_abs_clamp = 0.0
    posterior_obs_mae = _prior_obs_mae(obs_idx, obs_vals, x_t)
    innov_energy = float(np.mean(np.abs(clamp_delta))) if clamp_delta.size else 0.0
    innovation_mean_abs = innov_energy

    mae_pos_prior = float("nan")
    mae_pos_prior_unobs = 0.0
    if pos_idx.size and true_vals is not None:
        prior_vals = np.asarray(prior_t[pos_idx], dtype=float)
        diff_prior = prior_vals - true_vals
        finite_prior = np.isfinite(diff_prior)
        if finite_prior.any():
            mae_pos_prior = float(np.mean(np.abs(diff_prior[finite_prior])))
        if pos_unobs_idx.size and true_vals_unobs is not None:
            prior_unobs_vals = np.asarray(prior_t[pos_unobs_idx], dtype=float)
            diff_prior_unobs = prior_unobs_vals - true_vals_unobs
            finite_unobs = np.isfinite(diff_prior_unobs)
            if finite_unobs.any():
                mae_pos_prior_unobs = float(np.mean(np.abs(diff_prior_unobs[finite_unobs])))

    periph_missing_count = int(len(missing_periph_dims))
    _dbg(
        f'A13 transport_diag delta={shift} transport_mae_pre={mae_pos_pre_transport:.6f} '
        f'transport_mae_post={mae_pos_post_transport:.6f} mae_pos_prior={mae_pos_prior:.6f} '
        f'mae_pos_prior_unobs={mae_pos_prior_unobs:.6f} mae_pos_unobs_pre={mae_pos_unobs_pre_transport:.6f} '
        f'mae_pos_unobs_post={mae_pos_unobs_post_transport:.6f} trans_norm={transport_applied_norm:.6f} '
        f'transport_effect={transport_effect:.6f} transport_confidence={state.transport_confidence:.6f} '
        f'transport_margin={state.transport_margin:.6f} '
        f'transport_source={transport_source} periph_dims_missing_count={periph_missing_count} '
        f'periph_selected={periph_selected} transport_candidates={len(transport_candidates_info)}',
        state=state,
    )

    _dbg(
        f'A13 clamp_stats: obs_dims={len(O_t)} mean_abs_delta={mean_abs_clamp:.6f} '
        f'delta_outside={delta_outside_O:.6f} mae_pos_prior={mae_pos_prior:.6f} '
        f'mae_pos_prior_unobs={mae_pos_prior_unobs:.6f}',
        state=state,
    )
    Sigp_diag = np.diag(Sigma_prior).copy() if np.asarray(Sigma_prior).ndim == 2 else np.asarray(Sigma_prior, dtype=float).reshape(-1)
    if Sigp_diag.shape[0] != D:
        Sigp_diag = np.resize(Sigp_diag, (D,))
    finite_mask = np.isfinite(Sigp_diag)
    if np.any(finite_mask):
        _dbg(
            f'A7 prior_sigma_diag: mean={float(np.mean(Sigp_diag[finite_mask])):.6f} '
            f'min={float(np.min(Sigp_diag[finite_mask])):.6f} max={float(np.max(Sigp_diag[finite_mask])):.6f}',
            state=state,
        )

    # -------------------------------------------------------------------------
    # Prediction error on observed dims for A16.2 tracking and A17 diagnostics
    # -------------------------------------------------------------------------
    error_vec = np.zeros(D, dtype=float)
    if obs_idx.size:
        error_vec[obs_idx] = obs_vals - prior_t[obs_idx]
    abs_error = np.abs(error_vec)
    mean_delta = float(np.mean(abs_error[obs_idx])) if obs_idx.size else 0.0
    learning_processor.update_streaks(mean_delta)
    state.low_streak = learning_processor.low_streak
    state.high_streak = learning_processor.high_streak
    if obs_idx.size:
        active_thresh = float(getattr(cfg, "train_active_threshold", 0.0))
        active_obs_mask = np.abs(obs_vals) > active_thresh
        active_obs_count = int(np.sum(active_obs_mask))
        active_obs_err = float(np.mean(abs_error[obs_idx][active_obs_mask])) if active_obs_count else 0.0
        _dbg(
            f'A16.2 active_obs: count={active_obs_count} mean_abs_err={active_obs_err:.6f}',
            state=state,
        )
        prior_obs_mae = float(np.mean(abs_error[obs_idx]))
    else:
        prior_obs_mae = float("nan")

    worlds = _build_world_hypotheses(
        state,
        cfg,
        D,
        cue_t,
        obs_idx,
        obs_vals,
        transport_candidates_info,
        transport_best_candidate,
        prior_t,
        x_t,
        Sigp_diag,
    )
    disagreement, innovation, periph_demand = _compute_block_signals(state, cfg, worlds, D)
    pending_signals = getattr(state, "pending_fovea_signals", None)
    if pending_signals is None:
        pending_signals = FoveaSignals()
    pending_signals.block_disagreement = disagreement
    pending_signals.block_innovation = innovation
    pending_signals.block_periph_demand = periph_demand
    state.pending_fovea_signals = pending_signals
    finite_world_maes = [float(w.prior_mae) for w in worlds if np.isfinite(w.prior_mae)]
    if finite_world_maes:
        best_world_mae = float(min(finite_world_maes))
        expected_world_mae = float(
            sum(w.weight * float(w.prior_mae) for w in worlds if np.isfinite(w.prior_mae))
        )
    else:
        best_world_mae = float("nan")
        expected_world_mae = float("nan")
    weight_entropy = 0.0
    for world in worlds:
        w = float(world.weight)
        if w > 0.0 and np.isfinite(w):
            weight_entropy -= w * math.log(w)
    multi_world_summary = [
        {
            "delta": tuple(world.delta),
            "weight": float(world.weight),
            "prior_mae": float(world.prior_mae),
            "likelihood": float(world.likelihood),
            "score": float(world.metadata.get("score", 0.0)),
        }
        for world in worlds
    ]
    multi_world_count = len(worlds)
    support_window = _support_window_union(state)

    hazard, novelty, _value_of_compute = _update_value_of_compute(
        state,
        cfg,
        innovation=np.asarray(innovation, dtype=float).reshape(-1),
        disagreement=np.asarray(disagreement, dtype=float).reshape(-1),
        periph_demand=np.asarray(periph_demand, dtype=float).reshape(-1),
        abs_error=abs_error,
    )

    # Update observation buffer (dense estimate and observed dims)
    state.buffer.x_prior = prior_t.copy()
    state.buffer.x_last = x_t.copy()
    _dbg(f'BUFFER update x_last; D={D}', state=state)
    coarse_buffer = extract_coarse(state.buffer.x_last, cfg)
    state.buffer.observed_dims = set(int(k) for k in O_t if 0 <= int(k) < D)
    _dbg(f'BUFFER observed_dims={len(state.buffer.observed_dims)}', state=state)

    coarse_prev = getattr(state, "coarse_prev", None)
    coarse_prev_norm, coarse_prev_nonzero, coarse_prev_head = _coarse_summary(coarse_prev)
    use_true_source = use_true_transport and coarse_true_size > 0
    coarse_curr = coarse_true if use_true_source else coarse_buffer
    if use_true_source:
        transport_source = "debug_env"
    coarse_curr_norm, coarse_curr_nonzero, coarse_curr_head = _coarse_summary(coarse_curr)

    if "coarse_prev" in state.__dataclass_fields__:
        if (
            coarse_curr.size > 0
            and coarse_prev is not None
            and coarse_prev.shape == coarse_curr.shape
        ):
            coarse_shift = compute_transport_shift(coarse_prev, coarse_curr, cfg)
            if coarse_shift == (0, 0) and not np.allclose(coarse_prev, coarse_curr):
                delta = float(np.linalg.norm(coarse_curr - coarse_prev))
                nonzero_prev = int(np.count_nonzero(coarse_prev))
                nonzero_curr = int(np.count_nonzero(coarse_curr))
                _dbg(
                    f'A13 transport_no_shift delta={delta:.6f} prev_nz={nonzero_prev} curr_nz={nonzero_curr}',
                    state=state,
                )
        state.coarse_prev = coarse_curr.copy() if coarse_curr.size else np.zeros(0, dtype=float)

    # v5 fixed-budget: do NOT scan the full durable library each step.
    # Band levels are inferred lazily for bounded candidate sets (see salience),
    # and coverage bookkeeping is maintained over bounded tracked sets.
    if not bool(cfg.sig_disable_context_register):
        gist_vec = _compute_peripheral_gist(x_prev, cfg)
        _update_context_register(state, gist_vec, cfg)

    def _should_skip_salience(state: AgentState, cfg: AgentConfig) -> bool:
        if not getattr(state, "scores_prev", None):
            return False
        prev_learning = getattr(state, "learning_candidates_prev", {}) or {}
        prev_candidates = int(prev_learning.get("candidates", 0))
        if prev_candidates > 0:
            return False
        if int(getattr(state, "proposals_prev", 0)) > 0:
            return False
        return True

    # -------------------------------------------------------------------------
    # A5 salience (uses lagged stress + lagged scores)
    # -------------------------------------------------------------------------
    stress_signals_lagged = get_stress_signals(state)
    skip_salience = _should_skip_salience(state, cfg)
    if skip_salience:
        prev_scores = dict(getattr(state, "scores_prev", {}) or {})
        temperature, s_play = compute_temperature(stress_signals_lagged, cfg)
        activations = compute_activations(prev_scores, temperature, cfg)
        sal = SalienceResult(
            scores=prev_scores,
            activations=activations,
            temperature=temperature,
            s_play=s_play,
        )
        state.salience_candidate_ids = set(prev_scores.keys())
        state.salience_num_nodes_scored = len(prev_scores)
        state.salience_candidate_count_raw = len(prev_scores)
        state.salience_candidate_limit = len(prev_scores)
        state.salience_candidates_truncated = False
    else:
        sal = compute_salience(
            state=state,
            stress=stress_signals_lagged,
            scores_prev=getattr(state, "scores_prev", None),
            cfg=cfg,
            observed_dims=state.buffer.observed_dims,
        )
    _dbg('A5 compute_salience', state=state)

    salience_cost = max(1.0, float(len(getattr(sal, "activations", {}) or {})))
    meter.spend(salience_cost, "salience")

    # -------------------------------------------------------------------------
    # A4/A5 working set selection
    # -------------------------------------------------------------------------
    A_t = select_working_set(state, salience=sal.activations, cfg=cfg, budget_meter=meter)
    _dbg('A4/A5 select_working_set', state=state)
    state.active_set = set(int(nid) for nid in getattr(A_t, "active", []) or [])
    ws_cost = max(1.0, float(len(getattr(state, "active_set", set()) or set())))
    meter.spend(ws_cost, "working_set")

    # v5 scan-proof retrieval: PackedSigIndex updates belong to unit lifecycle
    # (creation/eviction) and outcome-vetted validation, not "insert every
    # activation". The online step loop must not mutate buckets based on usage.

    if not bool(cfg.sig_disable_context_register):
        _update_context_tags(state, cfg)
    tracked_ids = set(getattr(state, "salience_candidate_ids", set()) or set())
    tracked_ids.update(state.active_set or set())
    _update_coverage_debts_bounded(state, cfg, tracked_ids)

    L_eff = float(getattr(A_t, "effective_load", 0.0))
    _dbg(f'L_eff={L_eff:.3f} active_set={len(getattr(state,"active_set",[]) or [])}', state=state)

    # Optional binding/equivariance: select a transform per active node.
    if bool(getattr(cfg, "binding_enabled", False)):
        side = int(getattr(cfg, "grid_side", 0))
        channels = int(getattr(cfg, "grid_channels", 0))
        base_dim = int(getattr(cfg, "grid_base_dim", 0) or D)
        if side > 0 and channels > 0:
            cache = getattr(state, "_binding_cache", {})
            cache_key = (side, channels, base_dim, int(getattr(cfg, "binding_shift_radius", 1)), bool(getattr(cfg, "binding_rotations", True)))
            maps = cache.get(cache_key)
            if maps is None:
                rots = [0, 90, 180, 270] if bool(getattr(cfg, "binding_rotations", True)) else [0]
                maps = build_binding_maps(
                    D=D,
                    side=side,
                    channels=channels,
                    base_dim=base_dim,
                    shift_radius=int(getattr(cfg, "binding_shift_radius", 1)),
                    rotations=rots,
                )
                cache[cache_key] = maps
                state._binding_cache = cache
            observed_dims = set(state.buffer.observed_dims)
            for nid in state.active_set:
                node = state.library.nodes.get(int(nid))
                if node is None:
                    continue
                binding = select_best_binding_by_fit(
                    mask=getattr(node, "mask", None),
                    W=getattr(node, "W", np.zeros((D, D))),
                    b=getattr(node, "b", np.zeros(D)),
                    input_mask=getattr(node, "input_mask", None),
                    x_prev=x_prev,
                    cue_t=cue_t,
                    maps=maps,
                )
                if binding is None:
                    binding = select_best_binding(mask=getattr(node, "mask", None), observed_dims=observed_dims, maps=maps)
                if binding is not None:
                    setattr(node, "binding_map", binding)

    # -------------------------------------------------------------------------
    # A5/A12 activity logging for structural proposals
    # -------------------------------------------------------------------------
    activation_log = getattr(state, "activation_log", {})
    activation_max = int(getattr(cfg, "activation_log_max", 200))
    for nid, a_j in (getattr(A_t, "weights", {}) or {}).items():
        log = list(activation_log.get(int(nid), []))
        log.append((int(getattr(state, "t_w", 0)), float(a_j)))
        if len(log) > activation_max:
            log = log[-activation_max:]
        activation_log[int(nid)] = log
    state.activation_log = activation_log

    # Track last active step for incumbents (A12.3 PRUNE)
    for nid in state.active_set:
        node = state.library.nodes.get(int(nid))
        if node is not None:
            node.last_active_step = int(getattr(state, "t_w", 0))

    # -------------------------------------------------------------------------
    # A16.2: update fovea tracking after applying observation at t
    # (kept after A4.3 retrieval so retrieval keys to t-1 greedy_cov stats)
    # -------------------------------------------------------------------------
    ages_before = np.asarray(getattr(state.fovea, "block_age", []), dtype=float).copy()
    update_fovea_tracking(
        state.fovea,
        state.buffer,
        cfg,
        abs_error=abs_error,
        observed_dims=state.buffer.observed_dims,
    )
    ages_after = np.asarray(getattr(state.fovea, "block_age", []), dtype=float)
    if ages_before.size and ages_after.size:
        delta_age_mean = float(np.mean(ages_after) - np.mean(ages_before))
        _dbg(f'A16.2 age_delta_mean={delta_age_mean:.3f} rest_t={rest_t}', state=state)
        resids = np.asarray(getattr(state.fovea, "block_residual", []), dtype=float)
        if resids.size:
            age_min = float(np.min(ages_after))
            age_max = float(np.max(ages_after))
            age_mean = float(np.mean(ages_after))
            resid_min = float(np.min(resids))
            resid_max = float(np.max(resids))
            resid_mean = float(np.mean(resids))
            top_age = np.argsort(-ages_after)[: min(5, ages_after.size)]
            top_resid = np.argsort(-resids)[: min(5, resids.size)]
            _dbg(
                'A16.2 coverage_stats: '
                f'age_mean={age_mean:.3f} age_min={age_min:.3f} age_max={age_max:.3f} '
                f'resid_mean={resid_mean:.3f} resid_min={resid_min:.3f} resid_max={resid_max:.3f} '
                f'top_age={list(top_age)} top_resid={list(top_resid)}',
                state=state,
            )
            alpha_cov = float(getattr(cfg, "alpha_cov", 0.10))
            score = resids + alpha_cov * np.log1p(np.maximum(0.0, ages_after))
            top_score = np.argsort(-score)[: min(3, score.size)]
            top_terms = [
                (
                    int(b),
                    float(resids[b]),
                    float(ages_after[b]),
                    float(alpha_cov * np.log1p(max(0.0, ages_after[b]))),
                    float(score[b]),
                )
                for b in top_score
            ]
            _dbg(
                f'A16.2 score_terms top3=(b,resid,age,age_term,score)={top_terms}',
                state=state,
            )
    _dbg('A16.2 update_fovea_tracking', state=state)

    # -------------------------------------------------------------------------
    # A12.4 persistent residuals + residual stats + transition logging
    # -------------------------------------------------------------------------
    persistent = getattr(state, "persistent_residuals", {})
    residual_stats = getattr(state, "residual_stats", {})
    observed_transitions = getattr(state, "observed_transitions", {})

    beta_R = float(getattr(cfg, "beta_R", 0.10))
    split_beta = float(getattr(cfg, "split_stats_beta", 0.10))
    trans_max = int(getattr(cfg, "transition_log_max", 128))

    for block_id, dims in enumerate(getattr(state, "blocks", []) or []):
        block_dims = set(int(d) for d in dims)
        obs_in_block = block_dims & set(state.buffer.observed_dims)
        if not obs_in_block:
            continue

        idx = np.array(sorted(obs_in_block), dtype=int)
        resid_block = float(np.mean(np.abs(error_vec[idx]))) if idx.size else 0.0

        rstate = persistent.get(int(block_id))
        if rstate is None:
            rstate = PersistentResidualState()
        rstate.value = (1.0 - beta_R) * float(getattr(rstate, "value", 0.0)) + beta_R * resid_block
        rstate.coverage_visits = int(getattr(rstate, "coverage_visits", 0)) + 1
        rstate.last_update_step = int(getattr(state, "t_w", 0))
        persistent[int(block_id)] = rstate

        stats = residual_stats.get(int(block_id))
        if stats is None:
            stats = FootprintResidualStats(dims=sorted(list(block_dims)))
        stats.update(error_vec, beta=split_beta)
        residual_stats[int(block_id)] = stats

        log = list(observed_transitions.get(int(block_id), []))
        dims_tuple = tuple(int(k) for k in idx)
        x_tau_vals = x_prev[idx].copy()
        x_tau_plus_1_vals = x_t[idx].copy()
        log.append(
            TransitionRecord(
                tau=int(getattr(state, "t_w", 0)),
                dims=dims_tuple,

                x_tau_block=x_tau_vals,
                x_tau_plus_1_block=x_tau_plus_1_vals,
                observed_dims_tau_plus_1=set(int(k) for k in idx),
            )
        )
        if len(log) > trans_max:
            log = log[-trans_max:]
        observed_transitions[int(block_id)] = log

    state.persistent_residuals = persistent
    state.residual_stats = residual_stats
    state.observed_transitions = observed_transitions

    # -------------------------------------------------------------------------
    # A3.3 stability metrics plumbing (low-dimensional only)
    # -------------------------------------------------------------------------
    probe_vec, feature_vec = _feature_probe_vectors(
        state=state,
        obs=env_obs,
        abs_error=abs_error,
        observed_dims=state.buffer.observed_dims,
    )
    _dbg('A3.3 feature/probe vectors', state=state)
    update_stability_metrics(state, cfg, probe_vec=probe_vec, feature_vec=feature_vec)
    _dbg('A3.3 update_stability_metrics', state=state)

    # -------------------------------------------------------------------------
    # A7.3: one-step fusion prediction x̂(t+1|t)
    # -------------------------------------------------------------------------
    yhat_tp1, Sigma_tp1 = fuse_predictions(
        state.library,
        A_t,
        state.buffer,
        set(state.buffer.observed_dims),
        cfg,
    )
    _dbg('A7.3 fuse_predictions', state=state)

    yhat_tp1 = np.asarray(yhat_tp1, dtype=float).reshape(-1)
    if yhat_tp1.shape[0] != D:
        yhat_tp1 = np.resize(yhat_tp1, (D,))

    Sigma_tp1 = np.asarray(Sigma_tp1, dtype=float)
    if Sigma_tp1.ndim == 2 and Sigma_tp1.shape[0] == Sigma_tp1.shape[1]:
        sigma_tp1_diag = np.diag(Sigma_tp1).copy()
    else:
        sigma_tp1_diag = np.asarray(Sigma_tp1, dtype=float).reshape(-1)
        if sigma_tp1_diag.shape[0] != D:
            sigma_tp1_diag = np.resize(sigma_tp1_diag, (D,))

    # A13 prediction uses the same completion operator (no cue overwrite).
    yhat_tp1, Sigma_tp1_pred, _prior_pred = complete(
        None,
        mode="prediction",
        state=state,
        cfg=cfg,
        predicted_prior_t=yhat_tp1,
        predicted_sigma_diag=sigma_tp1_diag,
    )
    yhat_tp1 = np.asarray(yhat_tp1, dtype=float).reshape(-1)
    Sigma_tp1_pred = np.asarray(Sigma_tp1_pred, dtype=float)
    if Sigma_tp1_pred.ndim == 2 and Sigma_tp1_pred.shape[0] == Sigma_tp1_pred.shape[1]:
        sigma_tp1_diag = np.diag(Sigma_tp1_pred).copy()

    block_uncertainty = compute_block_uncertainty(sigma_tp1_diag, cfg)
    if block_uncertainty is not None:
        pending_signals = getattr(state, "pending_fovea_signals", None)
        if pending_signals is None:
            pending_signals = FoveaSignals()
        pending_signals.block_uncertainty = block_uncertainty
        state.pending_fovea_signals = pending_signals

    # -------------------------------------------------------------------------
    # REST structural processing (A12/A14): REST-only, queue ownership preserved
    # -------------------------------------------------------------------------
    edits_processed_t = 0
    b_cons_t = 0.0
    rest_res = RestProcessingResult()

    if rest_t:
        _dbg('REST branch', state=state)
        # Some modules gate on state.is_rest (macro.rest). Make best-effort to
        # reflect rest(t) before calling REST processors.
        try:
            if hasattr(state, "macro") and hasattr(state.macro, "rest"):
                state.macro.rest = True  # type: ignore[attr-defined]
        except Exception:
            pass

        rest_res = process_struct_queue(
            state,
            cfg,
            queue=list(state.q_struct),
            max_edits=int(getattr(cfg, "max_edits_per_rest_step", 32)),
        )
        _dbg('REST process_struct_queue begin', state=state)
        edits_processed_t = int(getattr(rest_res, "proposals_processed", 0))
        b_cons_t = float(getattr(rest_res, "total_consolidation_cost", 0.0))
        _dbg(f'REST processed={edits_processed_t} b_cons_t={b_cons_t:.3f}', state=state)

    queue_len = int(len(getattr(state, "q_struct", []) or []))
    rest_permit_struct = bool(getattr(rest_res, "permit_struct", False))
    rest_actionable = bool(queue_len > 0)
    rest_actionable_reason = ""
    if rest_t and not rest_actionable:
        rest_actionable_reason = "queue_empty" if queue_len == 0 else "no_work"

    # -------------------------------------------------------------------------
    # A6 budget and horizon
    # -------------------------------------------------------------------------
    budget = compute_budget_and_horizon(
        rest=rest_t,
        cfg=cfg,
        L_eff=L_eff,
        L_eff_roll=float(getattr(A_t, "rollout_load", L_eff)),
        L_eff_anc=float(getattr(A_t, "anchor_load", 0.0)),
        b_cons=b_cons_t,
    )
    _dbg('A6 compute_budget_and_horizon', state=state)

    # -------------------------------------------------------------------------
    # A7.4 rollout + confidence (provides c list for A8.2)
    # -------------------------------------------------------------------------
    rollout = rollout_and_confidence(
        x0=x_t,
        x_hat_1=yhat_tp1,
        Sigma_1=np.diag(sigma_tp1_diag),
        h=int(budget.h),
        cfg=cfg,
    )
    _dbg('A7.4 rollout_and_confidence', state=state)
    c_vals = np.asarray(list(getattr(rollout, "c", []) or []), dtype=float)
    if c_vals.size:
        _dbg(
            f'A7.4 c_stats: mean={float(np.mean(c_vals)):.6f} '
            f'min={float(np.min(c_vals)):.6f} max={float(np.max(c_vals)):.6f}',
            state=state,
        )
    rho_vals = np.asarray(list(getattr(rollout, "rho", []) or []), dtype=float)
    H_vals = np.asarray(list(getattr(rollout, "H", []) or []), dtype=float)
    c_qual_vals = np.asarray(list(getattr(rollout, "c_qual", []) or []), dtype=float)
    c_cov_vals = np.asarray(list(getattr(rollout, "c_cov", []) or []), dtype=float)
    if rho_vals.size and H_vals.size:
        _dbg(
            f'A7.4 cov_stats: rho_mean={float(np.mean(rho_vals)):.6f} '
            f'H_cov_mean={float(np.mean(H_vals)):.6f} '
            f'c_qual_mean={float(np.mean(c_qual_vals)):.6f} '
            f'c_cov_mean={float(np.mean(c_cov_vals)):.6f}',
            state=state,
        )

    rollout_cost = max(1.0, float(int(budget.h)) + float(len(getattr(rollout, "c", []) or [])))
    meter.spend(rollout_cost, "rollout")

    # -------------------------------------------------------------------------
    # A8 commitment gate and action selection
    # -------------------------------------------------------------------------
    commit_t = commit_gate(rest=rest_t, h=int(budget.h), c=list(getattr(rollout, "c", [])), cfg=cfg)
    _dbg(f'A8 commit_gate with h={int(budget.h)} c_len={len(list(getattr(rollout,"c",[]) or []))}', state=state)
    action = synthesize_action(state)
    _dbg(f'A13.10 synthesize_action -> action={action} commit={bool(commit_t)}', state=state)

    # -------------------------------------------------------------------------
    # A15 hard dynamics update of (E, D, drift_P)
    # -------------------------------------------------------------------------
    hard_prev = HardState(E=float(state.E), D=float(state.D), drift_P=float(state.drift_P))
    hard_t = step_hard_dynamics(prev=hard_prev, rest=rest_t, commit=commit_t, L_eff=L_eff, cfg=cfg)
    _dbg('A15 step_hard_dynamics', state=state)

    state.E = float(hard_t.E)
    state.D = float(hard_t.D)
    state.drift_P = float(hard_t.drift_P)

    # -------------------------------------------------------------------------
    # Margins (A0.1 / A2) + baselines (A3.1) + arousal (A0.2–A0.4)
    # -------------------------------------------------------------------------
    margins_t, rawE_t, rawD_t, _rawS = _derive_margins(
        E=state.E,
        D=state.D,
        drift_P=state.drift_P,
        opp=float(getattr(env_obs, "opp", 0.0)),
        x_C=float(budget.x_C),
        cfg=cfg,
    )
    _dbg('A0/A2 derive margins', state=state)
    _dbg(
        f'A0/A2 raw_headrooms: E={state.E:.3f} D={state.D:.3f} rawE={rawE_t:.3f} rawD={rawD_t:.3f}',
        state=state,
    )

    baselines_t = update_baselines(baselines=state.baselines, margins=margins_t, cfg=cfg)
    _dbg('A3.1 update_baselines', state=state)

    # Feel proxy (A17) uses sigma prior (Σ_global(t) diag) and H_d at latency floor.
    if Sigma_prior is None:
        sigma_prior_diag = np.ones(D, dtype=float)
    else:
        Sigp = np.asarray(Sigma_prior, dtype=float)
        if Sigp.ndim == 2 and Sigp.shape[0] == Sigp.shape[1]:
            sigma_prior_diag = np.diag(Sigp).copy()
        else:
            sigma_prior_diag = np.asarray(Sigp, dtype=float).reshape(-1)
        if sigma_prior_diag.shape[0] != D:
            sigma_prior_diag = np.resize(sigma_prior_diag, (D,))

    d_floor = int(getattr(cfg, "d_latency_floor", 1))
    if d_floor <= 0:
        d_floor = 1
    H_d = 0.0
    if len(getattr(rollout, "H", [])) >= d_floor:
        H_d = float(rollout.H[d_floor - 1])

    feel = compute_feel_proxy(
        observed_dims=state.buffer.observed_dims,
        error_vec=error_vec,
        sigma_global_diag=sigma_prior_diag,
        L_eff=L_eff,
        H_d=H_d,
        sigma_floor=float(getattr(cfg, "sigma_floor_diag", 1e-2)),
    )
    _dbg('A17 compute_feel_proxy', state=state)

    # Arousal uses pred_error magnitude proxy; use q_res (A17.1) as a scalar proxy.
    arousal_prev = float(getattr(state, "arousal_prev", getattr(state, "arousal", 0.0)))
    s_inst, s_ar = compute_arousal(
        arousal_prev=arousal_prev,
        margins=margins_t,
        baselines=baselines_t,
        pred_error=float(getattr(feel, "q_res_raw", 0.0)),
        cfg=cfg,
    )
    _dbg('A0.2 compute_arousal', state=state)

    # Persist tilde_prev for A0.2 delta computation (no other module does this yet)
    tilde, delta_tilde = normalize_margins(margins=margins_t, baselines=baselines_t, cfg=cfg)
    baselines_t = commit_tilde_prev(baselines_t, tilde=tilde)
    mE, mD, mL, mC, mS = [float(x) for x in tilde]
    dE, dD, dL, dC, dS = [float(x) for x in delta_tilde]
    w_L = float(getattr(cfg, "w_L", getattr(cfg, "w_L_ar", 1.0)))
    w_C = float(getattr(cfg, "w_C", getattr(cfg, "w_C_ar", 1.0)))
    w_S = float(getattr(cfg, "w_S", getattr(cfg, "w_S_ar", 1.0)))
    w_delta = float(getattr(cfg, "w_delta", getattr(cfg, "w_delta_ar", 1.0)))
    w_E = float(getattr(cfg, "w_E", getattr(cfg, "w_E_ar", 0.0)))
    term_L = w_L * abs(mL)
    term_C = w_C * abs(mC)
    term_S = w_S * abs(mS)
    term_delta = w_delta * (abs(dE) + abs(dD) + abs(dL) + abs(dC) + abs(dS))
    term_E = w_E * abs(float(getattr(feel, "q_res", 0.0)))
    A_raw = term_L + term_C + term_S + term_delta + term_E
    _dbg(
        'A0.2 arousal_terms: '
        f'L={term_L:.3f} C={term_C:.3f} S={term_S:.3f} '
        f'delta={term_delta:.3f} E={term_E:.3f} A_raw={A_raw:.3f}',
        state=state,
    )

    # -------------------------------------------------------------------------
    # Stress (A0.3) from hard observables + exogenous threat
    # -------------------------------------------------------------------------
    s_ext_th_t = float(getattr(env_obs, "danger", 0.0))
    stress_t = compute_stress(E=state.E, D=state.D, drift_P=state.drift_P, s_ext_th=s_ext_th_t, cfg=cfg)
    _dbg('A0.3 compute_stress', state=state)

    # -------------------------------------------------------------------------
    # A10 learning gates (freeze uses lagged stress; permit_param uses lagged slack/headrooms)
    # -------------------------------------------------------------------------
    freeze_t = freeze_predicate(stress_lagged=state.stress, cfg=cfg)
    _dbg('A10 freeze_predicate (uses lagged stress)', state=state)
    chi_th = float(getattr(cfg, "chi_th", 0.90))
    s_ext_th_lagged = float(getattr(state.stress, "s_ext_th", 0.0))
    _dbg(
        f'A10 freeze_lagged: s_ext_th={s_ext_th_lagged:.3f} chi_th={chi_th:.3f} freeze={freeze_t}',
        state=state,
    )

    use_current = bool(getattr(cfg, "gates_use_current", False))
    if use_current:
        x_C_lagged = float(budget.x_C)
        arousal_lagged = float(s_ar)
        rawE_lagged = float(rawE_t)
        rawD_lagged = float(rawD_t)
    else:
        x_C_lagged = float(getattr(state, "x_C_prev", 0.0))
        arousal_lagged = float(getattr(state, "arousal_prev", arousal_prev))
        rawE_lagged = float(getattr(state, "rawE_prev", rawE_t))
        rawD_lagged = float(getattr(state, "rawD_prev", rawD_t))

    theta_learn = float(getattr(cfg, "theta_learn", 0.10))
    permit_param_t = permit_param_updates(
        rest_t=rest_t,
        freeze_t=freeze_t,
        x_C_lagged=x_C_lagged,
        arousal_lagged=arousal_lagged,
        rawE_lagged=rawE_lagged,
        rawD_lagged=rawD_lagged,
        cfg=cfg,
    )
    _dbg('A10 permit_param_updates', state=state)
    tau_C = float(getattr(cfg, "tau_C_edit", 0.0))
    tau_E = float(getattr(cfg, "tau_E_edit", 0.0))
    tau_D = float(getattr(cfg, "tau_D_edit", 0.0))
    theta_panic = float(getattr(cfg, "theta_ar_panic", 0.95))
    _dbg(
        'A10 permit_lagged: '
        f'rest_t={rest_t} freeze={freeze_t} '
        f'x_C={x_C_lagged:.3f} tau_C={tau_C:.3f} '
        f'arousal={arousal_lagged:.3f} theta_panic={theta_panic:.3f} '
        f'rawE={rawE_lagged:.3f} tau_E={tau_E:.3f} '
        f'rawD={rawD_lagged:.3f} tau_D={tau_D:.3f} '
        f'permit={permit_param_t}',
        state=state,
    )
    _dbg(
        f'A10 gates lagged: freeze={freeze_t} x_C={x_C_lagged:.3f} arousal={arousal_lagged:.3f} rawE={rawE_lagged:.3f} rawD={rawD_lagged:.3f}',
        state=state,
    )

    # -------------------------------------------------------------------------
    # A10.3 responsibility-gated parameter learning (observed footprint only)
    # -------------------------------------------------------------------------
    learning_candidates_info = None
    permit_param_info = {
        "theta_learn": theta_learn,
        "permit": bool(permit_param_t),
        "candidate_count": 0,
        "clamped": 0,
        "updated": 0,
        "transport_high_confidence": bool(transport_high_confidence),
        "transport_score_margin": float(transport_score_margin),
        "transport_best_overlap": int(transport_best_overlap),
        "transport_high_confidence_margin_threshold": float(hc_margin),
        "transport_high_confidence_overlap_threshold": int(hc_overlap),
    }

    # Conservative clamp when transport evidence is weak: we still *evaluate*
    # candidates for visibility, but we tighten the update threshold so only
    # very low-error samples update parameters.
    theta_learn_low_conf_scale = 0.25
    theta_learn_eff = float(theta_learn) if bool(transport_high_confidence) else float(theta_learn) * float(theta_learn_low_conf_scale)
    permit_param_info["theta_learn_eff"] = float(theta_learn_eff)
    permit_param_info["theta_learn_low_conf_scale"] = float(theta_learn_low_conf_scale)

    if permit_param_t:
        lr = float(getattr(cfg, "lr_expert", 0.0))
        sigma_ema = float(getattr(cfg, "sigma_ema", 0.01))
        observed_dims = set(state.buffer.observed_dims)
        if not observed_dims:
            observed_dims = set()

        updated_nodes: list[int] = []
        candidate_nodes = 0
        clamped_candidates = 0
        err_j_vals: list[float] = []
        candidate_samples: list[Dict[str, Any]] = []
        sample_cap = int(min(8, max(1, getattr(cfg, "fovea_blocks_per_step", 16))))
        for node_id in getattr(A_t, "active", []) or []:
            node = state.library.nodes.get(int(node_id))
            if node is None:
                continue

            mask = np.asarray(getattr(node, "mask", np.zeros(D)), dtype=float).reshape(-1)
            if mask.shape[0] != D:
                mask = np.resize(mask, (D,))

            if not observed_dims:
                continue

            obs_mask = (mask > 0.5)
            obs_mask &= np.isin(np.arange(D), list(observed_dims))
            if not np.any(obs_mask):
                continue

            candidate_nodes += 1
            obs_idx = np.where(obs_mask)[0]
            err_j = float(np.mean(np.abs(error_vec[obs_idx]))) if obs_idx.size else float("inf")
            err_j_vals.append(err_j)
            clamped = err_j > float(theta_learn_eff)
            if clamped:
                clamped_candidates += 1
            if len(candidate_samples) < sample_cap:
                candidate_samples.append(
                    {
                        "node": int(node_id),
                        "footprint": int(getattr(node, "footprint", -1)),
                        "err": err_j,
                        "obs_dims": int(obs_idx.size),
                        "clamped": clamped,
                    }
                )

            if not clamped:
                sgd_update(
                    node,
                    x_t=x_prev,
                    y_target=x_t,
                    out_mask=_build_training_mask(
                        obs_mask=obs_mask,
                        x_obs=x_t,
                        cfg=cfg,
                    ),
                    lr=lr,
                    sigma_ema=sigma_ema,
                )
                updated_nodes.append(int(node_id))
        if err_j_vals:
            err_min = float(np.min(err_j_vals))
            err_mean = float(np.mean(err_j_vals))
            err_max = float(np.max(err_j_vals))
        else:
            err_min = err_mean = err_max = float("nan")
        learning_candidates_info = {
            "candidates": candidate_nodes,
            "clamped": clamped_candidates,
            "err_min": err_min,
            "err_mean": err_mean,
            "err_max": err_max,
            "samples": candidate_samples,
        }
        permit_param_info.update(
            {
                "candidate_count": candidate_nodes,
                "clamped": clamped_candidates,
                "updated": len(updated_nodes),
            }
        )
        learning_cost = max(1.0, float(candidate_nodes))
        meter.spend(learning_cost, "learning")
        _dbg(
            f'A10.3 learn_gate: candidates={candidate_nodes} clamped={clamped_candidates} '
            f'err_j[min/mean/max]={err_min:.6f}/{err_mean:.6f}/{err_max:.6f} '
            f'theta_learn_eff={float(theta_learn_eff):.6f} transport_high_conf={bool(transport_high_confidence)}',
            state=state,
        )
        if not bool(transport_high_confidence):
            _dbg(
                'A10.3 learn_gate: low transport confidence -> tightened clamp '
                f'margin={transport_score_margin:.6f} overlap={int(transport_best_overlap)} prev_obs={len(prev_observed_dims)}',
                state=state,
            )
        _dbg(f'A10.3 sgd_updates={len(updated_nodes)} nodes={updated_nodes}', state=state)

    _dbg(
        f'A10 permit_param_stats candidate_count={permit_param_info["candidate_count"]} '
        f'clamped_count={permit_param_info["clamped"]} updated={permit_param_info["updated"]} '
        f'permit={permit_param_info["permit"]}',
        state=state,
    )
    state.learning_candidates_prev = learning_candidates_info if learning_candidates_info is not None else {"candidates": 0}

    # -------------------------------------------------------------------------
    # A14 macrostate evolution (queue ownership in macrostate.py)
    # -------------------------------------------------------------------------
    # Generate structural proposals during OPERATING only (A14.2).
    proposals_t: List[Any] = []
    if not rest_t:
        proposals_t = list(propose_structural_edits(state, cfg))
    state.proposals_prev = int(len(proposals_t))

    # Coverage debt from A16 block ages (use current ages at t).
    ages = np.asarray(getattr(state.fovea, "block_age", []), dtype=float)
    log1p_ages = np.log1p(np.maximum(0.0, ages)) if ages.size else np.zeros(0, dtype=float)
    coverage_debt = float(np.sum(log1p_ages)) if ages.size else 0.0
    coverage_debt_prev = float(getattr(state, "coverage_debt_prev", coverage_debt))
    _dbg(f'A16.2 coverage_debt_delta={coverage_debt - coverage_debt_prev:.3f}', state=state)
    if ages.size:
        _dbg(
            'A16.2 coverage_debt_terms: '
            f'sum_log1p={coverage_debt:.3f} max_log1p={float(np.max(log1p_ages)):.3f} '
            f'max_age={float(np.max(ages)):.3f}',
            state=state,
        )

    _dbg(
        'A14 inputs: '
        f's_int_need={float(getattr(stress_t, "s_int_need", 0.0)):.3f} '
        f's_ext_th={float(getattr(stress_t, "s_ext_th", 0.0)):.3f} '
        f'mE={float(getattr(margins_t, "m_E", 0.0)):.3f} '
        f'mD={float(getattr(margins_t, "m_D", 0.0)):.3f} '
        f'mL={float(getattr(margins_t, "m_L", 0.0)):.3f} '
        f'mC={float(getattr(margins_t, "m_C", 0.0)):.3f} '
        f'mS={float(getattr(margins_t, "m_S", 0.0)):.3f} '
        f'coverage_debt={coverage_debt:.3f} '
        f'proposals={len(proposals_t)} edits_processed={int(edits_processed_t)}',
        state=state,
    )
    macro_t, demand_t, interrupt_t, P_eff_t = evolve_macrostate(
        prev=state.macro,
        rest_t=rest_t,
        proposals_t=proposals_t,
        edits_processed_t=edits_processed_t,
        stress_t=stress_t,
        margins_t=margins_t,
        coverage_debt=coverage_debt,
        rest_actionable=rest_actionable,
        cfg=cfg,
    )
    _dbg('A14 evolve_macrostate', state=state)
    P_rest_t = float(getattr(macro_t, "P_rest", 0.0))
    P_wake = float(coverage_debt)
    _dbg(f'A14 pressures: coverage_debt={coverage_debt:.3f} P_wake={P_wake:.3f} P_rest={P_rest_t:.3f} P_rest_eff={P_eff_t:.3f}', state=state)
    # A14.6: rest_permitted(t) from actual predicate (not from a missing field)
    rest_perm_t, rest_perm_reason = rest_permitted(stress_t, coverage_debt, cfg, arousal=s_ar)
    # Require stability windows before REST permission (A3.3-driven guard).
    W = int(getattr(cfg, "W", 50))
    if len(getattr(state, "probe_window", [])) < W or len(getattr(state, "feature_window", [])) < W:
        rest_perm_t = False
    # REST requires work: gate entry using same condition as continuation.
    if len(getattr(state, "q_struct", []) or []) == 0 and float(b_cons_t) == 0.0:
        rest_perm_t = False
        _dbg('A14 REST entry gated: no work in queue and no maintenance debt', state=state)
    if rest_t and not rest_actionable:
        rest_perm_t = False
        _dbg(
            f'A14 rest_actionable guard: reason={rest_actionable_reason} '
            f'queue_len={queue_len} permit_struct={rest_permit_struct}',
            state=state,
        )
    _dbg(f'A14.6 rest_permitted -> {rest_perm_t}', state=state)
    _dbg(
        'A14 macro_vars: '
        f'rest={bool(getattr(macro_t, "rest", False))} '
        f'T_since={int(getattr(macro_t, "T_since", 0))} '
        f'T_rest={int(getattr(macro_t, "T_rest", 0))} '
        f'Q_struct_len={int(len(getattr(macro_t, "Q_struct", []) or []))} '
        f'rest_permitted={bool(rest_perm_t)} '
        f'rest_reason={rest_perm_reason} '
        f'demand={bool(demand_t)} interrupt={bool(interrupt_t)} '
        f'rest_actionable={rest_actionable} rest_actionable_reason={rest_actionable_reason} '
        f'rest_zero_streak={int(getattr(macro_t, "rest_zero_processed_streak", 0))} '
        f'rest_cooldown={int(getattr(macro_t, "rest_cooldown", 0))}',
        state=state,
    )

    # -------------------------------------------------------------------------
    # Learning cache for next step’s completion prior (A13) and A17 diag use
    # -------------------------------------------------------------------------
    state.learn_cache = LearningCache(
        x_t=x_t.copy(),
        yhat_tp1=yhat_tp1.copy(),
        sigma_tp1_diag=np.asarray(sigma_tp1_diag, dtype=float).copy(),
        A_t=A_t,
        permit_param_t=bool(permit_param_t),
        rest_t=bool(rest_t),
    )

    emit_pred_snapshot(
        state,
        cfg,
        tick=tick_idx,
        obs_idx=committed_obs_idx,
        prior=prior_t,
        blocks=blocks_t,
        active_set=getattr(state, "active_set", set()),
        sig64=getattr(state, "last_sig64", None),
    )

    decision = decide_intake_and_compute(
        rest=rest_t,
        stress=stress_t,
        budget_slack=float(getattr(budget, "x_C", 0.0)),
        cfg=cfg,
        P_haz=float(getattr(state, "hazard_pressure", 0.0)),
        P_nov=float(getattr(state, "novelty_pressure", 0.0)),
        problem_bound=None,
        value_of_compute=float(getattr(state, "value_of_compute", 0.0)),
    )
    state.g_contemplate = decision.g_contemplate
    state.focus_mode = decision.focus_mode
    state.planning_target_blocks = decision.O_t_target
    state.planning_budget = float(decision.planning_budget)
    _run_planning_micro_steps(state, cfg)

    trace_cache_stored = False
    if not rest_t:
        trace_cache_stored = cache_observation(
            state,
            cfg,
            tick=tick_idx,
            blocks=blocks_t,
            obs_idx=committed_obs_idx,
            obs_vals=obs_vals,
            sig64=getattr(state, "last_sig64", None),
        )

    # -------------------------------------------------------------------------
    # Commit updated state fields and lagged values
    # -------------------------------------------------------------------------
    state.t_w = int(getattr(state, "t_w", 0)) + 1
    _dbg(f'commit state.t_w -> {state.t_w}', state=state)

    state.macro = macro_t
    state.margins = margins_t
    state.baselines = baselines_t
    state.stress = stress_t
    state.arousal = float(s_ar)

    # Lagged predicates/signals for t+1
    state.rest_permitted_prev = bool(rest_perm_t)
    # If REST has no work (no queue, no maintenance debt), force exit next step.
    maint_debt = float(b_cons_t)
    if rest_t and len(getattr(macro_t, "Q_struct", []) or []) == 0 and maint_debt == 0.0:
        _dbg(
            'A14 REST requires work: forcing exit next step '
            f'(Q_struct_len=0 maint_debt={maint_debt:.3f})',
            state=state,
        )
        demand_t = False
        state.rest_permitted_prev = False
    if rest_t and not rest_actionable:
        _dbg(
            'A14 REST actionable guard: forcing exit next step '
            f'reason={rest_actionable_reason} queue_len={queue_len} permit_struct={rest_permit_struct}',
            state=state,
        )
        demand_t = False
        state.rest_permitted_prev = False
    state.demand_prev = bool(demand_t)
    state.interrupt_prev = bool(interrupt_t)

    state.s_int_need_prev = float(getattr(stress_t, "s_int_need", 0.0))
    state.s_ext_th_prev = float(getattr(stress_t, "s_ext_th", 0.0))

    state.arousal_prev = float(s_ar)
    state.scores_prev = dict(getattr(sal, "scores", {}) or {})

    state.x_C_prev = float(budget.x_C)
    state.rawE_prev = float(rawE_t)
    state.rawD_prev = float(rawD_t)
    state.coverage_debt_prev = float(coverage_debt)
    state.budget_degradation_level = meter.degrade_level
    state.budget_degradation_history = tuple(meter.degrade_history)
    state.budget_hat_max = float(meter.B_hat_max)

    # Lagged rollout confidence at latency floor (A8.2 timing discipline)
    c_list = list(getattr(rollout, "c", []) or [])
    if len(c_list) >= d_floor:
        state.c_d_prev = float(c_list[d_floor - 1])
    elif c_list:
        state.c_d_prev = float(c_list[-1])
    else:
        state.c_d_prev = 0.0

    # Consolidation cost channel (A6.2) (store for visibility)
    state.b_cons = float(b_cons_t)

    rest_queue_len_next = int(len(getattr(macro_t, "Q_struct", []) or []))
    max_edits_per_rest = max(1, int(getattr(cfg, "max_edits_per_rest_step", 32)))
    if rest_queue_len_next > 0:
        rest_cycles_needed = int((rest_queue_len_next + max_edits_per_rest - 1) // max_edits_per_rest)
    else:
        rest_cycles_needed = 0

    # -------------------------------------------------------------------------
    # Trace (runner expects dict)
    # -------------------------------------------------------------------------
    trace = StepTrace(
        t=int(state.t_w),
        rest=bool(rest_t),
        h=int(budget.h),
        commit=bool(commit_t),
        x_C=float(budget.x_C),
        b_enc=float(budget.b_enc),
        b_roll=float(budget.b_roll),
        b_cons=float(budget.b_cons),
        L_eff=float(L_eff),
        arousal=float(s_ar),
        feel={
            "q_res": float(getattr(feel, "q_res", 0.0)),
            "q_maint": float(getattr(feel, "q_maint", 0.0)),
            "q_unc": float(getattr(feel, "q_unc", 0.0)),
        },
        permit_param=bool(permit_param_t),
        freeze=bool(freeze_t),
    )

    cache = getattr(state, "trace_cache", None)
    trace_cache_entries = int(cache.size if cache is not None else 0)
    trace_cache_blocks = int(cache.block_count if cache is not None else 0)
    trace_cache_mass = int(cache.cue_mass if cache is not None else 0)

    library_nodes = getattr(getattr(state, "library", None), "nodes", {})
    library_size = int(len(library_nodes))
    trace_dict = asdict(trace)
    # Extra diagnostics (dict-only; does not affect StepTrace typing)
    trace_dict.update(
        {
            "P_rest_eff": float(P_eff_t),
            "P_rest": float(P_rest_t),
            "P_wake": float(P_wake),
            "coverage_debt": float(coverage_debt),
            "coverage_debt_delta": float(coverage_debt - coverage_debt_prev),
            "maint_debt": float(maint_debt),
            "low_streak": getattr(state, "low_streak", 0),
            "high_streak": getattr(state, "high_streak", 0),
            "Q_struct_len": int(len(getattr(macro_t, "Q_struct", []) or [])),
            "observed_dims": int(len(state.buffer.observed_dims)),
            "obs_env_size": int(len(env_obs_dims)),
            "obs_env_min": int(env_min) if env_min is not None else None,
            "obs_env_max": int(env_max) if env_max is not None else None,
            "obs_req_size": int(len(O_req)),
            "obs_req_min": int(req_min) if req_min is not None else None,
            "obs_req_max": int(req_max) if req_max is not None else None,
            "obs_used_size": int(len(O_t)),
            "obs_used_min": int(used_min) if used_min is not None else None,
            "obs_used_max": int(used_max) if used_max is not None else None,
            "obs_filtered_count": int(len(O_req) - len(O_t)),
            "validations_processed": int(validations_processed),
            "salience_library_size": library_size,
            "salience_candidate_count": int(len(getattr(state, "salience_candidate_ids", set()))),
            "salience_nodes_scored": int(getattr(state, "salience_num_nodes_scored", 0)),
            "salience_candidate_ratio": float(
                len(getattr(state, "salience_candidate_ids", set())) / max(1, library_size)
            )
            if library_size > 0
            else 0.0,
            "salience_debug_exhaustive": bool(getattr(cfg, "salience_debug_exhaustive", False)),
            "salience_candidate_limit": int(getattr(state, "salience_candidate_limit", 0)),
            "salience_candidate_count_raw": int(getattr(state, "salience_candidate_count_raw", 0)),
            "salience_candidate_truncated": bool(getattr(state, "salience_candidates_truncated", False)),
            "salience_skipped": bool(skip_salience),
            "periph_full_provided": bool(periph_full_vec is not None),
            "use_true_transport": bool(use_true_transport),
            "transport_debug_env_grid": bool(use_env_grid),
            "edits_processed": int(edits_processed_t),
            "rest_permitted_t": bool(rest_perm_t),
            "rest_unsafe_reason": str(rest_perm_reason),
            "demand_t": bool(demand_t),
            "interrupt_t": bool(interrupt_t),
            "rest_actionable": bool(rest_actionable),
            "rest_queue_len": queue_len,
            "rest_permit_struct": rest_permit_struct,
            "rest_cooldown": int(getattr(macro_t, "rest_cooldown", 0)),
            "rest_cycles_needed": int(rest_cycles_needed),
            "rest_zero_processed_streak": int(getattr(macro_t, "rest_zero_processed_streak", 0)),
            "s_int_need": float(getattr(stress_t, "s_int_need", 0.0)),
            "s_ext_th": float(getattr(stress_t, "s_ext_th", 0.0)),
            "mE": float(getattr(margins_t, "m_E", 0.0)),
            "mD": float(getattr(margins_t, "m_D", 0.0)),
            "mL": float(getattr(margins_t, "m_L", 0.0)),
            "mC": float(getattr(margins_t, "m_C", 0.0)),
            "mS": float(getattr(margins_t, "m_S", 0.0)),
            "permit_struct": bool(getattr(rest_res, "permit_struct", False)),
            "permit_struct_reason": str(getattr(rest_res, "permit_struct_reason", "")),
            "transport_delta": tuple(shift),
            "transport_mae_pre": float(mae_pos_pre_transport),
            "transport_mae_post": float(mae_pos_post_transport),
            "transport_applied_norm": float(transport_applied_norm),
            "transport_source": transport_source,
            "transport_effect": float(transport_effect),
            "transport_confidence": float(state.transport_confidence),
            "transport_margin": float(state.transport_margin),
            "transport_candidate_count": int(len(transport_candidates_info)),
            "transport_score_margin": float(transport_score_margin),
            "candidate_score_spread": float(transport_score_spread),
            "overlap_count_best": int(transport_best_overlap),
            "posterior_entropy": float(transport_posterior_entropy),
            "tie_flag": bool(transport_tie_flag),
            "null_chosen_due_to_low_evidence": bool(transport_null_evidence),
            "motion_probe_blocks_used": int(motion_probe_blocks_used),
            "transport_high_confidence": bool(transport_high_confidence),
            "transport_ascii_mismatch": int(transport_best_candidate.ascii_mismatch) if transport_best_candidate is not None else 0,
            "delta_outside_O": float(delta_outside_O),
            "innovation_mean_abs": float(innovation_mean_abs),
            "innov_energy": float(innov_energy),
            "prior_obs_mae": float(prior_obs_mae),
            "posterior_obs_mae": float(posterior_obs_mae),
            "multi_world_count": int(multi_world_count),
            "multi_world_best_prior_mae": float(best_world_mae),
            "multi_world_expected_prior_mae": float(expected_world_mae),
            "multi_world_weight_entropy": float(weight_entropy),
            "multi_world_summary": multi_world_summary,
            "support_window_size": int(len(getattr(state, "observed_history", []))),
            "support_window_union_size": int(len(support_window)),
            "peripheral_confidence": float(np.clip(getattr(state, "peripheral_confidence", 0.0), 0.0, 1.0)),
            "peripheral_residual": float(np.nan_to_num(getattr(state, "peripheral_residual", float("nan")))),
            "peripheral_prior_size": int(getattr(state, "peripheral_prior", np.zeros(0)).size),
            "peripheral_obs_size": int(getattr(state, "peripheral_obs", np.zeros(0)).size),
            "peripheral_bg_dim_count": int(len(periph_dims)),
            "peripheral_bg_active": bool(periph_dims),
            "block_disagreement_mean": float(
                np.nanmean(getattr(state.fovea, "block_disagreement", np.zeros(0))) if getattr(state.fovea, "block_disagreement", np.zeros(0)).size else 0.0
            ),
            "block_innovation_mean": float(
                np.nanmean(getattr(state.fovea, "block_innovation", np.zeros(0))) if getattr(state.fovea, "block_innovation", np.zeros(0)).size else 0.0
            ),
            "block_periph_demand_mean": float(
                np.nanmean(getattr(state.fovea, "block_periph_demand", np.zeros(0))) if getattr(state.fovea, "block_periph_demand", np.zeros(0)).size else 0.0
            ),
            "mean_abs_clamp": float(mean_abs_clamp),
            "mae_pos_prior": float(mae_pos_prior),
            "mae_pos_prior_unobs": float(mae_pos_prior_unobs),
            "mae_pos_unobs_pre": float(mae_pos_unobs_pre_transport),
            "mae_pos_unobs_post": float(mae_pos_unobs_post_transport),
            "coarse_prev_norm": float(coarse_prev_norm),
            "coarse_curr_norm": float(coarse_curr_norm),
            "coarse_prev_nonzero": int(coarse_prev_nonzero),
            "coarse_curr_nonzero": int(coarse_curr_nonzero),
            "coarse_prev_head": tuple(coarse_prev_head),
            "coarse_curr_head": tuple(coarse_curr_head),
            "periph_block_ids": tuple(int(b) for b in forced_periph_blocks),
            "periph_dims_forced": int(len(forced_periph_dims)),
            "periph_dims_in_req": int(periph_dims_present),
            "periph_dims_missing_count": int(len(missing_periph_dims)),
            "periph_dims_missing_head": tuple(
                missing_periph_dims[: min(8, len(missing_periph_dims))]
            ),
            "n_fine_blocks_selected": int(n_fine_blocks_selected),
            "n_periph_blocks_selected": int(n_periph_blocks_selected),
            "periph_included": bool(periph_included),
            "probe_var": float(state.probe_var) if getattr(state, "probe_var", None) is not None else float("nan"),
            "feature_var": float(state.feature_var) if getattr(state, "feature_var", None) is not None else float("nan"),
            "arousal": float(getattr(state, "arousal", 0.0)),
            "arousal_prev": float(getattr(state, "arousal_prev", 0.0)),
            "last_struct_edit_t": int(getattr(state.baselines, "last_struct_edit_t", -10**9)),
            "W_window": int(getattr(cfg, "W", 50)),
            "learning_candidates": learning_candidates_info if learning_candidates_info is not None else {},
            "permit_param_info": permit_param_info,
            "budget_B_rt": float(meter.B_rt),
            "budget_B_max": float(meter.B_max),
            "budget_B_plan": float(meter.B_plan),
            "budget_B_use": float(meter.B_use),
            "budget_limit": float(meter.limit()),
            "budget_degrade_level": int(meter.degrade_level),
            "budget_degrade_history": tuple(meter.degrade_history),
            "trace_cache_entries": trace_cache_entries,
            "trace_cache_blocks": trace_cache_blocks,
            "trace_cache_cue_mass": trace_cache_mass,
            "trace_cache_stored": bool(trace_cache_stored),
            "value_of_compute": float(getattr(state, "value_of_compute", 0.0)),
            "value_of_compute_prev": float(value_of_compute_prev),
            "hazard_pressure": float(getattr(state, "hazard_pressure", 0.0)),
            "novelty_pressure": float(getattr(state, "novelty_pressure", 0.0)),
            "g_contemplate": bool(getattr(state, "g_contemplate", False)),
            "planning_focus_mode": str(getattr(state, "focus_mode", "operate")),
            "planning_budget": float(getattr(state, "planning_budget", 0.0)),
            "planning_target_blocks": tuple(getattr(state, "planning_target_blocks") or ()),
        }
    )
    trace_dict["ordering_markers"] = tuple(ordering_markers)

    _dbg('returning (action, state, trace_dict)', state=state)
    return action, state, trace_dict


__all__ = ["step_pipeline"]
