"""Step loop for the harness runner."""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Tuple

import time
import numpy as np

from nupca3.agent import NUPCA3Agent
from nupca3.config import AgentConfig
from nupca3.geometry.fovea import make_observation_set, select_fovea, update_fovea_routing_scores
from nupca3.step_pipeline import (
    _enforce_motion_probe_blocks,
    _enforce_peripheral_blocks,
    _peripheral_block_ids,
    _select_motion_probe_blocks,
)
from nupca3.step_pipeline.v5_kernel import step_v5_kernel
from nupca3.types import EnvObs

from .logging import (
    log_active_metrics,
    log_coverage_summary,
    log_diff_and_trace,
    log_emit_debug,
    log_emit_header,
    log_env_state,
    log_learning_info,
    log_obs_previews,
    log_occlusion_metrics,
    log_permit_summary,
    log_sparse_metrics,
    log_transport_check,
    log_transport_summary,
    log_v5023_diagnostics,
)
from .render import _block_grid_bounds, _occupancy_array, _print_visualization
from .utils import build_partial_obs
from .worlds import LinearARWorld, LinearSquareWorld, MovingColorShapeWorld


def run_steps(
    *,
    agent: NUPCA3Agent,
    cfg: AgentConfig,
    world: str,
    steps: int,
    seed: int,
    B_int: int,
    D_agent: int,
    world_dim: int,
    base_dim: int,
    periph_dim: int,
    periph_blocks: int,
    periph_bg_full: bool,
    periph_dims_set: set[int],
    periph_bins: int,
    coverage_cap_G: int,
    coverage_log_every: int,
    diagnose_coverage: bool,
    rest_test_period: int,
    rest_test_length: int,
    periph_test: bool,
    transport_test: bool,
    transport_force_true_delta: bool,
    obs_budget_mode: str,
    coverage_debt_target: float,
    dense_world: bool,
    k_fixed: int,
    k_min: int,
    k_max: int,
    min_fovea_blocks: int,
    scan_steps: int,
    warm_steps: int,
    warm_fovea_blocks: int,
    pred_only_start: int,
    pred_only_len: int,
    occlude_start: int,
    occlude_len: int,
    occlude_period: int,
    debug_full_state: bool,
    force_selected_blocks: bool,
    side: int,
    n_colors: int,
    n_shapes: int,
    vis_n_colors: int,
    vis_n_shapes: int,
    log_every: int,
    visualize_steps: int,
    block_size_agent: int,
    moving: MovingColorShapeWorld | None,
    square: LinearSquareWorld | None,
    linear_world: LinearARWorld | None,
) -> Dict[str, float]:
    periph_ids = _peripheral_block_ids(cfg) if periph_blocks > 0 else []
    coverage_diag_enabled = bool(diagnose_coverage) and world == "square"
    coverage_steps_total = 0
    coverage_hits = 0
    coverage_square_blocks_seen = set()
    coverage_covered_blocks_seen = set()
    coverage_dim_to_block: Dict[int, int] = {}
    if coverage_diag_enabled:
        blocks_partition = getattr(agent.state, "blocks", []) or []
        for block_id, dims in enumerate(blocks_partition):
            for dim in dims:
                coverage_dim_to_block[int(dim)] = block_id
    coverage_log_every = int(coverage_log_every)
    coverage_log_every = coverage_log_every if coverage_log_every > 0 else 0
    rest_test_period = max(0, int(rest_test_period))
    rest_test_length = max(0, int(rest_test_length))
    periph_test_active = bool(periph_test and periph_blocks > 0)
    transport_test_active = bool(transport_test)
    rest_test_forced_steps = 0
    rest_test_edits_processed = 0
    periph_missing_steps = 0
    periph_present_steps = 0
    transport_test_total = 0
    transport_test_matches = 0
    mae_obs = []
    mae_unobs = []
    mae_unobs_predonly = []
    mae_pos = []
    mae_pos_obs = []
    mae_pos_unobs = []
    mae_pos_predonly = []
    corr_obs = []
    corr_unobs = []
    corr_unobs_predonly = []
    pos_frac = []
    mae_zero = []
    same_block_err = []
    diff_block_err = []
    block_changes = 0
    block_steps = 0
    prev_active_block = None
    prev_active_block_step = None
    obs_active_hits = 0
    obs_active_steps = 0
    first_seen_err = []
    reappear_err = []
    was_occluded = False
    step_log_limit = 25
    permit_param_true = 0
    permit_param_total = 0
    diff_zero_match = 0
    diff_zero_mismatch = 0
    diff_nonzero_match = 0
    diff_nonzero_mismatch = 0

    for step_idx in range(int(steps)):
        rest_permitted_prev = bool(getattr(agent.state, "rest_permitted_prev", False))
        demand_prev = bool(getattr(agent.state, "demand_prev", False))
        interrupt_prev = bool(getattr(agent.state, "interrupt_prev", False))
        s_int_need_prev = float(getattr(agent.state, "s_int_need_prev", 0.0))
        s_ext_th_prev = float(getattr(agent.state, "s_ext_th_prev", 0.0))
        x_C_prev = float(getattr(agent.state, "x_C_prev", 0.0))
        rawE_prev = float(getattr(agent.state, "rawE_prev", 0.0))
        rawD_prev = float(getattr(agent.state, "rawD_prev", 0.0))
        coverage_debt_prev_val = float(getattr(agent.state, "coverage_debt_prev", 0.0))
        b_cons_prev = float(getattr(agent.state, "b_cons", 0.0))
        active_idx = np.array([], dtype=int)
        obs_active_idx = np.array([], dtype=int)
        active_blocks = set()
        obs_active_count = 0
        force_rest = False
        if rest_test_period > 0 and rest_test_length > 0:
            cycle_phase = step_idx % rest_test_period
            force_rest = cycle_phase < rest_test_length
        if force_rest:
            agent.state.rest_permitted_prev = True
            agent.state.demand_prev = True
            agent.state.interrupt_prev = False
        # A16.3: select fovea blocks from current agent state (t-1 stats).
        k_eff = int(k_fixed)
        if str(obs_budget_mode).lower() == "coverage":
            debt = float(getattr(agent.state, "coverage_debt_prev", 0.0))
            denom = float(coverage_debt_target) if coverage_debt_target > 0 else float(max(1, B_int))
            ratio = 0.0 if denom <= 0 else min(1.0, max(0.0, debt / denom))
            k_eff = int(round(k_min + (k_max - k_min) * ratio))
            k_eff = max(1, min(int(B_int), k_eff))
        if step_idx < int(scan_steps):
            k_eff = int(B_int)
        elif warm_steps > 0 and step_idx < (int(scan_steps) + int(warm_steps)):
            k_eff = int(warm_fovea_blocks) if int(warm_fovea_blocks) > 0 else int(k_fixed)
            k_eff = max(1, min(int(B_int), k_eff))
        if int(min_fovea_blocks) > 0:
            k_eff = max(k_eff, min(int(B_int), int(min_fovea_blocks)))
        cfg_step = replace(agent.cfg, fovea_blocks_per_step=int(k_eff))
        update_fovea_routing_scores(agent.state.fovea, agent.state.buffer.x_last, cfg_step, t=step_idx)
        fovea_state = agent.state.fovea
        ages_snapshot = np.asarray(getattr(fovea_state, "block_age", []), dtype=float)
        resid_snapshot = np.asarray(getattr(fovea_state, "block_residual", []), dtype=float)
        top_k = int(min(int(k_eff), int(B_int))) if int(B_int) > 0 else 0
        if top_k > 0 and ages_snapshot.size:
            top_age_blocks = [int(i) for i in np.argsort(-ages_snapshot)[:top_k]]
        else:
            top_age_blocks = []
        if top_k > 0 and resid_snapshot.size:
            top_resid_blocks = [int(i) for i in np.argsort(-resid_snapshot)[:top_k]]
        else:
            top_resid_blocks = []
        blocks = select_fovea(agent.state.fovea, cfg_step)
        ages_snapshot = np.asarray(getattr(agent.state.fovea, "block_age", []), dtype=float)
        if int(coverage_cap_G) > 0 and ages_snapshot.size:
            mandatory_blocks = [
                int(b)
                for b in range(int(B_int))
                if float(ages_snapshot[b]) >= float(coverage_cap_G)
            ]
            if mandatory_blocks and not set(mandatory_blocks).intersection(set(blocks or [])):
                mandatory_blocks = sorted(
                    mandatory_blocks, key=lambda b: float(ages_snapshot[b]), reverse=True
                )
                blocks = mandatory_blocks[: max(1, int(k_eff))]
        periph_candidates = list(range(max(0, int(B_int) - int(periph_blocks)), int(B_int)))
        blocks, forced_periph_blocks = _enforce_peripheral_blocks(
            blocks,
            cfg_step,
            periph_candidates,
        )
        prev_observed_dims = set(int(k) for k in getattr(agent.state.buffer, "observed_dims", set()) or set())
        motion_probe_budget = max(0, int(getattr(cfg_step, "motion_probe_blocks", 0)))
        if int(k_eff) <= 1:
            motion_probe_budget = 0
        motion_probe_blocks = _select_motion_probe_blocks(prev_observed_dims, cfg_step, motion_probe_budget)
        blocks, motion_probe_blocks_used = _enforce_motion_probe_blocks(blocks, cfg_step, motion_probe_blocks)
        selected_blocks_tuple = tuple(int(b) for b in blocks)
        max_block_id = len(getattr(agent.state, "blocks", []) or [])
        if any(int(b) < 0 or int(b) >= max_block_id for b in blocks):
            raise AssertionError(
                f"Invariant violation: fovea block id out of range (blocks={blocks}, B={max_block_id})."
            )
        if visualize_steps and step_idx < visualize_steps:
            ages_preview = ages_snapshot[: min(8, ages_snapshot.size)].tolist() if ages_snapshot.size else []
            resid_preview = resid_snapshot[: min(8, resid_snapshot.size)].tolist() if resid_snapshot.size else []
            print(
                f"[fovea debug] step={step_idx} ages_head={ages_preview} "
                f"resid_head={resid_preview} blocks={blocks}"
            )
        if periph_test_active:
            periph_present = all(int(b) in blocks for b in periph_ids)
            if periph_present:
                periph_present_steps += 1
            else:
                periph_missing_steps += 1
            print(
                f"[periph check] step={step_idx} n_blocks={len(blocks)} "
                f"periph_present={periph_present} missing={periph_missing_steps}"
            )
        obs_dims = sorted(make_observation_set(blocks, cfg))
        pred_only = pred_only_len > 0 and pred_only_start <= step_idx < (pred_only_start + pred_only_len)
        occluding = False
        if occlude_len > 0 and occlude_period > 0 and step_idx >= occlude_start:
            if ((step_idx - occlude_start) % occlude_period) < occlude_len:
                occluding = True
        if pred_only:
            obs_dims = []
        if occluding:
            obs_dims = []
        if visualize_steps and step_idx < visualize_steps:
            obs_head = obs_dims[: min(16, len(obs_dims))]
            obs_tail = obs_dims[-min(16, len(obs_dims)) :] if obs_dims else []
            print(
                f"[obs debug] step={step_idx} k_eff={k_eff} blocks={blocks} "
                f"obs_count={len(obs_dims)} obs_head={obs_head} obs_tail={obs_tail}"
            )
            periph_bg_dims = len(periph_dims_set) if periph_bg_full else 0
            periph_bg_cost = int(periph_blocks) if periph_bg_full else 0
            print(
                f"[obs policy] step={step_idx} override_selected_blocks={force_selected_blocks} "
                f"fovea_blocks_per_step={k_eff} selected_blocks={blocks} "
                f"O_t={len(obs_dims)} periph_uses_full={periph_bg_full and periph_bg_dims > 0} "
                f"periph_bg_dims={periph_bg_dims} periph_bg_cost={periph_bg_cost}"
            )
            block_partitions = getattr(agent.state, "blocks", []) or []
            channels = max(1, int(n_colors + n_shapes)) if world in {"moving", "square"} else 1
            for block_id in blocks:
                if int(block_id) < 0 or int(block_id) >= len(block_partitions):
                    print(f"[obs debug] block_id={block_id} bounds=invalid")
                    continue
                bounds = _block_grid_bounds(
                    [int(dim) for dim in block_partitions[int(block_id)]],
                    side=side,
                    channels=channels,
                    base_dim=int(base_dim),
                )
                if bounds is None:
                    print(f"[obs debug] block_id={block_id} bounds=periph_or_empty")
                else:
                    r0, r1, c0, c1 = bounds
                    print(f"[obs debug] block_id={block_id} bounds=r{r0}:{r1} c{c0}:{c1}")

        # Environment evolves, then we reveal only the selected dims.
        true_delta: Tuple[int, int] = (0, 0)
        true_env_vec = np.zeros(0, dtype=float)
        true_env_dim = int(world_dim)
        if world == "moving":
            env_state_full = moving.step()
            true_env_vec = env_state_full.copy()
            true_env_dim = int(moving.D)
            base_x = env_state_full
            true_delta = moving.last_move()
            if int(base_dim) != int(world_dim):
                if int(base_dim) > int(world_dim):
                    base_x = np.pad(base_x, (0, int(base_dim) - int(world_dim)), mode="constant")
                else:
                    base_x = base_x[: int(base_dim)]
            periph = moving.encode_peripheral() if periph_blocks > 0 else np.zeros(0, dtype=float)
            if periph_blocks > 0:
                pad = periph_dim - periph.shape[0]
                if pad < 0:
                    periph = periph[:periph_dim]
                elif pad > 0:
                    periph = np.pad(periph, (0, pad), mode="constant")
            full_x = np.concatenate([base_x, periph]) if periph_blocks > 0 else base_x
        elif world == "square":
            env_state_full = square.step()
            true_env_vec = env_state_full.copy()
            true_env_dim = int(square.D)
            base_x = env_state_full
            true_delta = square.last_move()
            if int(base_dim) != int(world_dim):
                if int(base_dim) > int(world_dim):
                    base_x = np.pad(base_x, (0, int(base_dim) - int(world_dim)), mode="constant")
                else:
                    base_x = base_x[: int(base_dim)]
            periph = square.encode_peripheral() if periph_blocks > 0 else np.zeros(0, dtype=float)
            if periph_blocks > 0:
                pad = periph_dim - periph.shape[0]
                if pad < 0:
                    periph = periph[:periph_dim]
                elif pad > 0:
                    periph = np.pad(periph, (0, pad), mode="constant")
            full_x = np.concatenate([base_x, periph]) if periph_blocks > 0 else base_x
        else:
            env_state_full = linear_world.step()
            true_env_vec = env_state_full.copy()
            true_env_dim = int(linear_world.D)
            base_x = env_state_full
            full_x = base_x
        pos_dims = set(int(idx) for idx in np.where(full_x > 0.0)[0])
        periph_full = None
        if periph_bg_full and periph_dims_set:
            periph_full = np.zeros(int(D_agent), dtype=float)
            for dim in periph_dims_set:
                periph_full[int(dim)] = float(full_x[int(dim)])
        env_tick = step_idx + 1
        wall_ms = int(time.perf_counter() * 1000)
        obs = EnvObs(
            x_partial=build_partial_obs(full_x, obs_dims),
            opp=0.0,
            danger=0.0,
            periph_full=periph_full,
            true_delta=true_delta,
            pos_dims=pos_dims,
            selected_blocks=selected_blocks_tuple if force_selected_blocks else tuple(),
            t_w=env_tick,
            wall_ms=wall_ms,
        )

        buffer_prev = agent.state.buffer.x_last.copy()

        # Pre-step prior for evaluation (A16.2 residual definition).
        prior = getattr(agent.state.learn_cache, "yhat_tp1", None)
        prior_arr: np.ndarray | None = None
        if prior is not None:
            prior_arr = np.asarray(prior, dtype=float).reshape(-1)
        action, next_state, trace = step_v5_kernel(agent.state, obs, cfg_step)
        agent.state = next_state
        obs_dims_req = list(obs_dims)
        obs_dims_actual = sorted(int(dim) for dim in agent.state.buffer.observed_dims)
        obs_req_set = set(obs_dims_req)
        obs_act_set = set(obs_dims_actual)
        if obs_req_set != obs_act_set:
            req_head = sorted(obs_req_set)[:8]
            act_head = sorted(obs_act_set)[:8]
            print(
                f"[obs mismatch] step={step_idx} req_count={len(obs_req_set)} "
                f"used_count={len(obs_act_set)} req_head={req_head} used_head={act_head}"
            )
        if dense_world and not pred_only and not occluding:
            if obs_act_set:
                obs_min = min(obs_act_set)
                obs_max = max(obs_act_set)
            else:
                obs_min = None
                obs_max = None
            full_obs_expected = len(obs_req_set) == int(D_agent)
            if full_obs_expected and len(obs_act_set) != int(D_agent):
                raise AssertionError(
                    f"Invariant violation: dense-world obs size={len(obs_act_set)} "
                    f"expected D_world={D_agent}."
                )
            if full_obs_expected and (obs_min != 0 or obs_max != int(D_agent) - 1):
                raise AssertionError(
                    f"Invariant violation: dense-world obs bounds min={obs_min} max={obs_max} "
                    f"expected [0, {int(D_agent) - 1}]."
                )
        if log_every > 0 and (step_idx % log_every == 0 or log_every == 1):
            obs_min = min(obs_act_set) if obs_act_set else None
            obs_max = max(obs_act_set) if obs_act_set else None
            print(
                f"[obs summary] step={step_idx} env_obs_count={len(obs_req_set)} "
                f"used_obs_count={len(obs_act_set)} used_min={obs_min} used_max={obs_max} "
                f"req_min={min(obs_req_set) if obs_req_set else None} "
                f"req_max={max(obs_req_set) if obs_req_set else None} "
                f"trace_env_full={trace.get('env_full_provided')} "
                f"trace_use_true_transport={trace.get('use_true_transport')}"
            )
        if force_rest:
            rest_test_forced_steps += 1
            rest_test_edits_processed += int(trace.get("edits_processed", 0))
            agent.state.rest_permitted_prev = True
            agent.state.demand_prev = True
            agent.state.interrupt_prev = False
        if coverage_diag_enabled:
            square_blocks = {
                coverage_dim_to_block.get(int(dim))
                for dim in np.where(full_x > 0.0)[0]
                if coverage_dim_to_block.get(int(dim)) is not None
            }
            square_blocks.discard(None)
            library_nodes = getattr(agent.state.library, "nodes", {}) or {}
            active_set_ids = set(getattr(agent.state, "active_set", set()))
            covered_blocks = set()
            for nid in active_set_ids:
                node = library_nodes.get(int(nid))
                if node is None:
                    continue
                footprint = int(getattr(node, "footprint", -1))
                if footprint >= 0:
                    covered_blocks.add(footprint)
            coverage_steps_total += 1
            if square_blocks:
                coverage_square_blocks_seen.update(square_blocks)
            coverage_covered_blocks_seen.update(covered_blocks)
            hit = bool(square_blocks & covered_blocks)
            coverage_hits += int(hit)
            if coverage_log_every > 0 and (step_idx % coverage_log_every == 0 or coverage_log_every == 1):
                print(
                    f"[coverage] step={trace['t']} square_blocks={sorted(square_blocks)} "
                    f"covered_blocks={sorted(covered_blocks)} hit={int(hit)}"
                )
        permit_param_total += 1
        if bool(trace.get("permit_param", False)):
            permit_param_true += 1

        if prior_arr is not None:
            active_mask = full_x > 0.0
            active_idx = np.where(active_mask)[0]
            active_blocks = set()
            if active_idx.size:
                active_blocks = set(int(i) // block_size_agent for i in active_idx)
            obs_active_idx = np.array([i for i in obs_dims_actual if active_mask[i]], dtype=int)
            obs_active_count = int(obs_active_idx.size)
            if obs_active_count:
                obs_active_hits += 1
            if active_idx.size:
                obs_active_steps += 1
            if np.any(active_mask):
                pos_frac.append(float(np.mean(active_mask)))
                mae_zero.append(float(np.mean(np.abs(full_x))))
                err_pos = np.abs(prior_arr[active_mask] - full_x[active_mask])
                mae_pos.append(float(np.mean(err_pos)))
                if pred_only:
                    mae_pos_predonly.append(float(np.mean(err_pos)))
                if not occluding and not first_seen_err:
                    first_seen_err.append(float(np.mean(err_pos)))
                if was_occluded and not occluding:
                    reappear_err.append(float(np.mean(err_pos)))
                if active_idx.size:
                    active_blocks = np.array(
                        [int(i) // block_size_agent for i in active_idx], dtype=int
                    )
                    active_block_now = int(np.bincount(active_blocks).argmax())
                    if prev_active_block is not None:
                        block_steps += 1
                        if active_block_now != prev_active_block:
                            block_changes += 1
                            diff_block_err.append(float(np.mean(err_pos)))
                        else:
                            same_block_err.append(float(np.mean(err_pos)))
                    prev_active_block = active_block_now
                    prev_active_block_step = step_idx
            if obs_dims_actual:
                err = np.abs(prior_arr[obs_dims_actual] - full_x[obs_dims_actual])
                mae_obs.append(float(np.mean(err)))
                if len(obs_dims_actual) > 1:
                    obs_pred = prior_arr[obs_dims_actual]
                    obs_true = full_x[obs_dims_actual]
                    if np.std(obs_pred) > 0 and np.std(obs_true) > 0:
                        corr_obs.append(float(np.corrcoef(obs_pred, obs_true)[0, 1]))
            if prior_arr.shape[0] == full_x.shape[0]:
                mask = np.ones(prior_arr.shape[0], dtype=bool)
                if obs_dims_actual:
                    mask[np.asarray(obs_dims_actual, dtype=int)] = False
                if np.any(mask):
                    err_unobs = np.abs(prior_arr[mask] - full_x[mask])
                    mae_unobs.append(float(np.mean(err_unobs)))
                    if pred_only:
                        mae_unobs_predonly.append(float(np.mean(err_unobs)))
                    if np.sum(mask) > 1:
                        unobs_pred = prior_arr[mask]
                        unobs_true = full_x[mask]
                        if np.std(unobs_pred) > 0 and np.std(unobs_true) > 0:
                            corr_unobs.append(float(np.corrcoef(unobs_pred, unobs_true)[0, 1]))
                            if pred_only:
                                corr_unobs_predonly.append(float(np.corrcoef(unobs_pred, unobs_true)[0, 1]))
                if np.any(active_mask):
                    if obs_dims_actual:
                        obs_mask = np.zeros_like(active_mask, dtype=bool)
                        obs_mask[np.asarray(obs_dims_actual, dtype=int)] = True
                    else:
                        obs_mask = np.zeros_like(active_mask, dtype=bool)
                    pos_obs_mask = active_mask & obs_mask
                    pos_unobs_mask = active_mask & ~obs_mask
                    if np.any(pos_obs_mask):
                        err_pos_obs = np.abs(prior_arr[pos_obs_mask] - full_x[pos_obs_mask])
                        mae_pos_obs.append(float(np.mean(err_pos_obs)))
                    if np.any(pos_unobs_mask):
                        err_pos_unobs = np.abs(prior_arr[pos_unobs_mask] - full_x[pos_unobs_mask])
                        mae_pos_unobs.append(float(np.mean(err_pos_unobs)))
        perc = build_partial_obs(full_x, obs_dims_actual)
        pred_vec = prior_arr if prior_arr is not None else agent.state.buffer.x_last
        pred_vec = np.asarray(pred_vec, dtype=float).reshape(-1)
        pred_source = "prior" if prior_arr is not None else "buffer_last"
        occ_env = _occupancy_array(
            full_x[:int(base_dim)],
            side=side,
            base_dim=int(base_dim),
            n_colors=vis_n_colors,
            n_shapes=vis_n_shapes,
        )
        occ_pred = _occupancy_array(
            pred_vec[:int(base_dim)],
            side=side,
            base_dim=int(base_dim),
            n_colors=vis_n_colors,
            n_shapes=vis_n_shapes,
        )
        diff_mask = np.asarray(occ_env, dtype=bool) != np.asarray(occ_pred, dtype=bool)
        diff_count = int(np.sum(diff_mask)) if diff_mask.size else 0
        trace["diff_count"] = diff_count
        preds = prior_arr.tolist() if prior_arr is not None else None
        log_diff_and_trace(
            step_idx=step_idx,
            diff_count=diff_count,
            pred_source=pred_source,
            full_x=full_x,
            perc=perc,
            preds=preds,
        )
        if visualize_steps and step_idx < visualize_steps:
            obs_set = {int(dim) for dim in obs_dims_actual if 0 <= int(dim) < int(D_agent)}
            transport_delta = tuple(trace.get("transport_delta", (0, 0)))
            _print_visualization(
                step_idx=step_idx,
                true_delta=true_delta,
                transport_delta=transport_delta,
                env_vec=full_x,
                obs_dims=obs_set,
                prev_vec=buffer_prev,
                pred_vec=pred_vec,
                side=side,
                n_colors=vis_n_colors,
                n_shapes=vis_n_shapes,
                base_dim=int(base_dim),
                trace=trace,
                true_env_vec=true_env_vec,
                true_env_dim=true_env_dim,
            )
        was_occluded = bool(occluding)

        block_change_rate = float(block_changes) / float(block_steps) if block_steps > 0 else 0.0
        if world == "square":
            env_pos = (int(square.x), int(square.y))
            log_env_state(
                world=world,
                step_idx=step_idx,
                env_pos=env_pos,
                true_delta=true_delta,
                block_change_rate=block_change_rate,
                coverage_debt=trace["coverage_debt"],
                forced_rest=force_rest,
            )
        elif world == "moving":
            env_pos = (int(moving.x), int(moving.y))
            log_env_state(
                world=world,
                step_idx=step_idx,
                env_pos=env_pos,
                true_delta=true_delta,
                block_change_rate=block_change_rate,
                coverage_debt=trace["coverage_debt"],
                forced_rest=force_rest,
                color=int(moving.color),
                shape=int(moving.shape),
            )

        if transport_test_active:
            objects_present = bool(pos_dims)
            tdelta = tuple(trace.get("transport_delta", (0, 0)))
            match = tdelta == tuple(true_delta)
            if objects_present:
                transport_test_total += 1
                if match:
                    transport_test_matches += 1
                if diff_count == 0 and match:
                    diff_zero_match += 1
                elif diff_count == 0 and not match:
                    diff_zero_mismatch += 1
                elif diff_count != 0 and match:
                    diff_nonzero_match += 1
                else:
                    diff_nonzero_mismatch += 1
                log_transport_check(
                    step_idx=step_idx,
                    true_delta=true_delta,
                    tdelta=tdelta,
                    match=match,
                    coarse_prev_norm=trace.get("coarse_prev_norm", 0.0),
                    coarse_curr_norm=trace.get("coarse_curr_norm", 0.0),
                    periph_block_ids=trace.get("periph_block_ids", ()),
                    periph_dims_in_req=trace.get("periph_dims_in_req", 0),
                    periph_dims_missing_count=trace.get("periph_dims_missing_count", 0),
                    periph_dims_missing_head=trace.get("periph_dims_missing_head", ()),
                    coarse_prev_head=trace.get("coarse_prev_head", ()),
                    coarse_curr_head=trace.get("coarse_curr_head", ()),
                )

        emit = step_idx < step_log_limit or (int(log_every) > 0 and step_idx % int(log_every) == 0)
        if emit:
            log_emit_header(
                D_agent=D_agent,
                seed=seed,
                trace=trace,
                force_rest=force_rest,
                rest_permitted_prev=rest_permitted_prev,
                demand_prev=demand_prev,
                interrupt_prev=interrupt_prev,
                s_int_need_prev=s_int_need_prev,
                s_ext_th_prev=s_ext_th_prev,
                x_C_prev=x_C_prev,
                rawE_prev=rawE_prev,
                rawD_prev=rawD_prev,
                coverage_debt_prev_val=coverage_debt_prev_val,
                b_cons_prev=b_cons_prev,
            )
            log_learning_info(D_agent, seed, trace)
            log_permit_summary(D_agent, seed, trace)
            log_emit_debug(D_agent=D_agent, seed=seed, pred_only=pred_only, occluding=occluding, trace=trace)
        if emit and step_idx < step_log_limit:
            log_obs_previews(
                D_agent=D_agent,
                seed=seed,
                obs_dims_actual=obs_dims_actual,
                active_idx=active_idx,
                obs_active_idx=obs_active_idx,
                obs_active_count=obs_active_count,
                blocks=blocks,
                k_eff=k_eff,
                top_age_blocks=top_age_blocks,
                top_resid_blocks=top_resid_blocks,
            )
        active_count = int(active_idx.size) if active_idx.size else 0
        obs_active_rate = float(obs_active_hits) / float(obs_active_steps) if obs_active_steps else 0.0
        obs_active_blocks = sorted({int(i) // block_size_agent for i in obs_active_idx})
        active_blocks_list = sorted(list(active_blocks))
        if len(active_blocks_list) > 8:
            active_blocks_list = active_blocks_list[:8]
        if len(obs_active_blocks) > 8:
            obs_active_blocks = obs_active_blocks[:8]
        if emit:
            top_age_hits = len(set(blocks) & set(top_age_blocks)) if top_age_blocks else 0
            top_resid_hits = len(set(blocks) & set(top_resid_blocks)) if top_resid_blocks else 0
            full_obs = int(len(obs_dims_actual) >= int(D_agent) and k_eff >= int(B_int))
            cov_debt = float(trace.get("coverage_debt", 0.0))
            cov_violation = int(full_obs and cov_debt > 1e-6)
            log_active_metrics(
                D_agent=D_agent,
                seed=seed,
                active_count=active_count,
                obs_active_count=obs_active_count,
                obs_active_rate=obs_active_rate,
                active_blocks_list=active_blocks_list,
                obs_active_blocks=obs_active_blocks,
                top_age_hits=top_age_hits,
                top_resid_hits=top_resid_hits,
                top_k=top_k,
                full_obs=full_obs,
                cov_debt=cov_debt,
                cov_violation=cov_violation,
            )
        mae_pos_avg = float(np.mean(mae_pos)) if mae_pos else 0.0
        mae_pos_unobs_avg = float(np.mean(mae_pos_unobs)) if mae_pos_unobs else 0.0
        pos_frac_avg = float(np.mean(pos_frac)) if pos_frac else 0.0
        mae_zero_avg = float(np.mean(mae_zero)) if mae_zero else 0.0
        change_rate = float(block_changes) / float(block_steps) if block_steps > 0 else 0.0
        same_block_avg = float(np.mean(same_block_err)) if same_block_err else 0.0
        diff_block_avg = float(np.mean(diff_block_err)) if diff_block_err else 0.0
        if emit:
            permit_rate = float(permit_param_true) / float(permit_param_total) if permit_param_total else 0.0
            log_sparse_metrics(
                D_agent=D_agent,
                seed=seed,
                mae_pos_avg=mae_pos_avg,
                mae_pos_unobs_avg=mae_pos_unobs_avg,
                pos_frac_avg=pos_frac_avg,
                mae_zero_avg=mae_zero_avg,
                change_rate=change_rate,
                same_block_avg=same_block_avg,
                diff_block_avg=diff_block_avg,
                permit_rate=permit_rate,
                permit_param_true=permit_param_true,
            )
        first_avg = float(np.mean(first_seen_err)) if first_seen_err else 0.0
        reap_avg = float(np.mean(reappear_err)) if reappear_err else 0.0
        ratio = (reap_avg / first_avg) if first_avg > 0 else 0.0
        if emit:
            log_occlusion_metrics(
                D_agent=D_agent,
                seed=seed,
                first_avg=first_avg,
                reap_avg=reap_avg,
                ratio=ratio,
            )
            fovea_state = getattr(agent.state, "fovea", None)
            block_resid_arr = np.asarray(getattr(fovea_state, "block_residual", np.zeros(0)), dtype=float).reshape(-1)
            block_age_arr = np.asarray(getattr(fovea_state, "block_age", np.zeros(0)), dtype=float).reshape(-1)
            periph_demand_arr = np.asarray(
                getattr(fovea_state, "block_periph_demand", np.zeros(0)), dtype=float
            ).reshape(-1)
            block_stats = {
                "count": int(block_resid_arr.size),
                "resid_mean": float(np.nanmean(block_resid_arr)) if block_resid_arr.size else 0.0,
                "resid_max": float(np.nanmax(block_resid_arr)) if block_resid_arr.size else 0.0,
                "age_mean": float(np.nanmean(block_age_arr)) if block_age_arr.size else 0.0,
                "age_max": float(np.nanmax(block_age_arr)) if block_age_arr.size else 0.0,
                "periph_demand_mean": float(np.nanmean(periph_demand_arr)) if periph_demand_arr.size else 0.0,
            }
            context_register = getattr(agent.state, "context_register", np.zeros(0))
            purge_health = {
                "context_register": int(np.asarray(context_register).size),
                "observed_history": int(len(getattr(agent.state, "observed_history", []))),
                "trace_cache_entries": int(trace.get("trace_cache_entries", 0)),
                "trace_cache_blocks": int(trace.get("trace_cache_blocks", 0)),
                "trace_cache_cue_mass": int(trace.get("trace_cache_cue_mass", 0)),
                "support_window": int(trace.get("support_window_size", 0)),
                "support_union": int(trace.get("support_window_union_size", 0)),
            }
            lib = getattr(agent.state, "library", None)
            sig_index = getattr(lib, "sig_index", None) if lib is not None else None
            err_cache = getattr(sig_index, "_err_cache", None) if sig_index is not None else None
            if isinstance(err_cache, np.ndarray):
                err_shape = tuple(err_cache.shape)
                err_nan = bool(np.isnan(err_cache).any())
                err_present = bool(err_cache.size)
            else:
                err_shape = (0, 0)
                err_nan = False
                err_present = False
            buckets = getattr(sig_index, "buckets", []) if sig_index is not None else []
            populated_blocks = tuple(len(tbl) for tbl in buckets) if buckets else tuple()
            index_health = {
                "tables": int(getattr(sig_index, "tables", 0)) if sig_index is not None else 0,
                "bucket_bits": int(getattr(sig_index, "bucket_bits", 0)) if sig_index is not None else 0,
                "bucket_cap": int(getattr(sig_index, "bucket_cap", 0)) if sig_index is not None else 0,
                "populated_blocks": populated_blocks,
                "err_cache_shape": err_shape,
                "err_cache_nan": err_nan,
                "err_cache_present": err_present,
            }
            log_v5023_diagnostics(
                D_agent=D_agent,
                seed=seed,
                block_stats=block_stats,
                purge_health=purge_health,
                index_health=index_health,
            )

    coverage_hit_rate = float(coverage_hits) / float(coverage_steps_total) if coverage_steps_total else 0.0
    if coverage_diag_enabled:
        log_coverage_summary(
            coverage_steps_total=coverage_steps_total,
            coverage_hit_rate=coverage_hit_rate,
            coverage_square_blocks_seen=coverage_square_blocks_seen,
            coverage_covered_blocks_seen=coverage_covered_blocks_seen,
        )
    if transport_test_active:
        log_transport_summary(
            diff_zero_match=diff_zero_match,
            diff_zero_mismatch=diff_zero_mismatch,
            diff_nonzero_match=diff_nonzero_match,
            diff_nonzero_mismatch=diff_nonzero_mismatch,
        )

    return {
        "mae_obs": float(np.mean(mae_obs)) if mae_obs else 0.0,
        "mae_unobs": float(np.mean(mae_unobs)) if mae_unobs else 0.0,
        "corr_obs": float(np.mean(corr_obs)) if corr_obs else 0.0,
        "corr_unobs": float(np.mean(corr_unobs)) if corr_unobs else 0.0,
        "mae_unobs_predonly": float(np.mean(mae_unobs_predonly)) if mae_unobs_predonly else 0.0,
        "corr_unobs_predonly": float(np.mean(corr_unobs_predonly)) if corr_unobs_predonly else 0.0,
        "mae_pos": float(np.mean(mae_pos)) if mae_pos else 0.0,
        "mae_pos_obs": float(np.mean(mae_pos_obs)) if mae_pos_obs else 0.0,
        "mae_pos_unobs": float(np.mean(mae_pos_unobs)) if mae_pos_unobs else 0.0,
        "mae_pos_predonly": float(np.mean(mae_pos_predonly)) if mae_pos_predonly else 0.0,
        "pos_frac": float(np.mean(pos_frac)) if pos_frac else 0.0,
        "mae_zero": float(np.mean(mae_zero)) if mae_zero else 0.0,
        "block_change_rate": float(block_changes) / float(block_steps) if block_steps > 0 else 0.0,
        "mae_same_block": float(np.mean(same_block_err)) if same_block_err else 0.0,
        "mae_diff_block": float(np.mean(diff_block_err)) if diff_block_err else 0.0,
        "binding_enabled": bool(cfg.binding_enabled),
        "binding_shift_radius": int(getattr(cfg, "binding_shift_radius", 0)),
        "binding_rotations": bool(getattr(cfg, "binding_rotations", False)),
        "periph_blocks": int(periph_blocks),
        "periph_bins": int(periph_bins),
        "first_seen_err": float(np.mean(first_seen_err)) if first_seen_err else 0.0,
        "reappear_err": float(np.mean(reappear_err)) if reappear_err else 0.0,
        "steps": int(steps),
        "coverage_hit_rate": coverage_hit_rate,
        "coverage_steps": int(coverage_steps_total),
        "rest_test_forced_steps": int(rest_test_forced_steps),
        "rest_test_edits_processed": int(rest_test_edits_processed),
        "periph_test_active": bool(periph_test_active),
        "periph_missing_steps": int(periph_missing_steps),
        "periph_present_steps": int(periph_present_steps),
        "transport_test_active": bool(transport_test_active),
        "transport_test_total": int(transport_test_total),
        "transport_test_matches": int(transport_test_matches),
    }
