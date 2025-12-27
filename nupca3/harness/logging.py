"""Logging helpers for the harness run loop."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np


def log_diff_and_trace(
    *,
    step_idx: int,
    diff_count: int,
    pred_source: str,
    full_x: np.ndarray,
    perc: Dict[int, float],
    preds: List[float] | None,
) -> None:
    print(f"[diff check] step={step_idx} diff_count={diff_count} pred_source={pred_source}")
    print(
        f"[TRACE step={step_idx}] EXACT_ENV={full_x.tolist()} "
        f"EXACT_AGENT_PERCEIVES={perc} "
        f"EXACT_AGENT_PREDICTS={preds}"
    )


def log_env_state(
    *,
    world: str,
    step_idx: int,
    env_pos: Tuple[int, int],
    true_delta: Tuple[int, int],
    block_change_rate: float,
    coverage_debt: float,
    forced_rest: bool,
    color: int | None = None,
    shape: int | None = None,
) -> None:
    if world == "square":
        print(
            f"[env square] step={step_idx} pos={env_pos} true_delta={true_delta} "
            f"block_change_rate={block_change_rate:.6f} coverage_debt={coverage_debt:.6f} "
            f"forced_rest={forced_rest}"
        )
    elif world == "moving":
        print(
            f"[env moving] step={step_idx} pos={env_pos} true_delta={true_delta} "
            f"color={int(color or 0)} shape={int(shape or 0)} "
            f"block_change_rate={block_change_rate:.6f} coverage_debt={coverage_debt:.6f} "
            f"forced_rest={forced_rest}"
        )


def log_transport_check(
    *,
    step_idx: int,
    true_delta: Tuple[int, int],
    tdelta: Tuple[int, int],
    match: bool,
    coarse_prev_norm: float,
    coarse_curr_norm: float,
    periph_block_ids: Sequence[int],
    periph_dims_in_req: int,
    periph_dims_missing_count: int,
    periph_dims_missing_head: Sequence[int],
    coarse_prev_head: Sequence[float],
    coarse_curr_head: Sequence[float],
) -> None:
    print(
        f"[transport check] step={step_idx} true_delta={true_delta} "
        f"transport_delta={tdelta} match={match}"
    )
    if match:
        return
    print(
        f"[transport diag] step={step_idx} coarse_prev_norm={coarse_prev_norm:.3f} "
        f"coarse_curr_norm={coarse_curr_norm:.3f} "
        f"periph_block_ids={periph_block_ids} periph_dims_in_req={periph_dims_in_req} "
        f"periph_dims_missing_count={periph_dims_missing_count} "
        f"periph_dims_missing_head={periph_dims_missing_head}"
    )
    print(
        f"[transport diag] step={step_idx} coarse_prev_head={coarse_prev_head} "
        f"coarse_curr_head={coarse_curr_head}"
    )


def log_emit_header(
    *,
    D_agent: int,
    seed: int,
    trace: Dict[str, Any],
    force_rest: bool,
    rest_permitted_prev: bool,
    demand_prev: bool,
    interrupt_prev: bool,
    s_int_need_prev: float,
    s_ext_th_prev: float,
    x_C_prev: float,
    rawE_prev: float,
    rawD_prev: float,
    coverage_debt_prev_val: float,
    b_cons_prev: float,
) -> None:
    print(
        f"[D{D_agent} seed{seed}] step={trace['t']} rest={trace['rest']} forced_rest={force_rest} "
        f"rest_permitted_prev={rest_permitted_prev} rest_permitted_t={trace['rest_permitted_t']} "
        f"demand_prev={demand_prev} demand_t={trace['demand_t']} "
        f"interrupt_prev={interrupt_prev} interrupt_t={trace['interrupt_t']} "
        f"coverage_debt={trace['coverage_debt']:.6f} coverage_debt_prev={coverage_debt_prev_val:.6f} "
        f"coverage_debt_delta={trace['coverage_debt_delta']:.6f} "
        f"s_int_need_prev={s_int_need_prev:.3f} s_int_need={trace['s_int_need']:.3f} "
        f"s_ext_th_prev={s_ext_th_prev:.3f} s_ext_th={trace['s_ext_th']:.3f} "
        f"mE={trace['mE']:.3f} mD={trace['mD']:.3f} mL={trace['mL']:.3f} mC={trace['mC']:.6f} "
        f"mS={trace['mS']:.3f} P_rest={trace['P_rest']:.3f} P_rest_eff={trace['P_rest_eff']:.3f} "
        f"P_wake={trace['P_wake']:.3f} maint_debt={trace['maint_debt']:.3f} "
        f"b_cons_prev={b_cons_prev:.3f} Q_struct_len={trace['Q_struct_len']} "
        f"last_struct_edit_t={trace.get('last_struct_edit_t', -999999)} "
        f"W={trace.get('W_window', 50)} "
        f"observed_dims={trace['observed_dims']} edits_processed={trace['edits_processed']} "
        f"permit_param={trace['permit_param']} x_C_prev={x_C_prev:.3f} "
        f"rawE_prev={rawE_prev:.3f} rawD_prev={rawD_prev:.3f} "
        f"arousal={trace.get('arousal', 0.0):.3f} "
        f"arousal_prev={trace.get('arousal_prev', 0.0):.3f} "
        f"permit_struct={trace.get('permit_struct', False)} "
        f"permit_reason={trace.get('permit_struct_reason','')} "
        f"probe_var={trace.get('probe_var', float('nan')):.6f} "
        f"feature_var={trace.get('feature_var', float('nan')):.6f}"
    )


def log_learning_info(D_agent: int, seed: int, trace: Dict[str, Any]) -> None:
    learning_info = trace.get("learning_candidates", {}) or {}
    if not learning_info:
        return
    samples = learning_info.get("samples", [])
    sample_str = ", ".join(
        f"n{int(s['node'])}@b{int(s['footprint'])} err={s['err']:.3f} clamped={s['clamped']}"
        for s in samples
    )
    print(
        f"[D{D_agent} seed{seed}] learning_info candidates={learning_info.get('candidates',0)} "
        f"clamped={learning_info.get('clamped',0)} err_max={learning_info.get('err_max',float('nan')):.6f} "
        f"samples=[{sample_str}]"
    )


def log_permit_summary(D_agent: int, seed: int, trace: Dict[str, Any]) -> None:
    permit_meta = trace.get("permit_param_info", {}) or {}
    print(
        f"[D{D_agent} seed{seed}] permit_param_summary theta_learn={permit_meta.get('theta_learn',0.0):.3f} "
        f"permit={permit_meta.get('permit',False)} "
        f"cand={permit_meta.get('candidate_count',0)} clamped={permit_meta.get('clamped',0)} "
        f"updated={permit_meta.get('updated',0)}"
    )


def log_emit_debug(
    *,
    D_agent: int,
    seed: int,
    pred_only: bool,
    occluding: bool,
    trace: Dict[str, Any],
) -> None:
    if pred_only:
        print(f"[D{D_agent} seed{seed}] pred_only_step={trace['t']} obs_dims=0")
    if occluding:
        print(f"[D{D_agent} seed{seed}] occluding_step={trace['t']} obs_dims=0")


def log_obs_previews(
    *,
    D_agent: int,
    seed: int,
    obs_dims_actual: Sequence[int],
    active_idx: np.ndarray,
    obs_active_idx: np.ndarray,
    obs_active_count: int,
    blocks: Sequence[int],
    k_eff: int,
    top_age_blocks: Sequence[int],
    top_resid_blocks: Sequence[int],
) -> None:
    obs_preview = obs_dims_actual[: min(32, len(obs_dims_actual))]
    active_preview = active_idx[: min(32, active_idx.size)].tolist() if active_idx.size else []
    obs_active_preview = obs_active_idx[: min(32, obs_active_idx.size)].tolist() if obs_active_idx.size else []
    print(f"[D{D_agent} seed{seed}] obs_dims_count={len(obs_dims_actual)} obs_dims_head={obs_preview}")
    print(f"[D{D_agent} seed{seed}] active_dims_count={int(active_idx.size)} active_dims_head={active_preview}")
    print(f"[D{D_agent} seed{seed}] obs_active_count={obs_active_count} obs_active_head={obs_active_preview}")
    blocks_preview = blocks[: min(16, len(blocks))]
    print(
        f"[D{D_agent} seed{seed}] fovea_blocks_count={len(blocks)} "
        f"fovea_blocks_head={blocks_preview} k_eff={k_eff}"
    )
    top_age_preview = top_age_blocks[: min(16, len(top_age_blocks))]
    top_resid_preview = top_resid_blocks[: min(16, len(top_resid_blocks))]
    print(
        f"[D{D_agent} seed{seed}] fovea_top_age_head={top_age_preview} "
        f"fovea_top_resid_head={top_resid_preview}"
    )


def log_active_metrics(
    *,
    D_agent: int,
    seed: int,
    active_count: int,
    obs_active_count: int,
    obs_active_rate: float,
    active_blocks_list: Sequence[int],
    obs_active_blocks: Sequence[int],
    top_age_hits: int,
    top_resid_hits: int,
    top_k: int,
    full_obs: int,
    cov_debt: float,
    cov_violation: int,
) -> None:
    print(
        f"[D{D_agent} seed{seed}] active_obs_metrics active_count={active_count} "
        f"obs_active_count={obs_active_count} obs_active_rate={obs_active_rate:.3f} "
        f"active_blocks={list(active_blocks_list)} obs_active_blocks={list(obs_active_blocks)}"
    )
    print(
        f"[D{D_agent} seed{seed}] fovea_overlap top_age_hits={top_age_hits} "
        f"top_resid_hits={top_resid_hits} top_k={top_k}"
    )
    print(
        f"[D{D_agent} seed{seed}] coverage_check full_obs={full_obs} "
        f"coverage_debt={cov_debt:.6f} coverage_debt_full_obs_violation={cov_violation}"
    )


def log_sparse_metrics(
    *,
    D_agent: int,
    seed: int,
    mae_pos_avg: float,
    mae_pos_unobs_avg: float,
    pos_frac_avg: float,
    mae_zero_avg: float,
    change_rate: float,
    same_block_avg: float,
    diff_block_avg: float,
    permit_rate: float,
    permit_param_true: int,
) -> None:
    print(
        f"[D{D_agent} seed{seed}] sparse_metrics mae_pos={mae_pos_avg:.6f} "
        f"mae_pos_unobs={mae_pos_unobs_avg:.6f} "
        f"pos_frac={pos_frac_avg:.6f} mae_zero={mae_zero_avg:.6f}"
    )
    print(
        f"[D{D_agent} seed{seed}] block_metrics change_rate={change_rate:.6f} "
        f"mae_same_block={same_block_avg:.6f} mae_diff_block={diff_block_avg:.6f}"
    )
    print(
        f"[D{D_agent} seed{seed}] permit_param_rate={permit_rate:.6f} "
        f"permit_param_true={permit_param_true}"
    )


def log_occlusion_metrics(
    *,
    D_agent: int,
    seed: int,
    first_avg: float,
    reap_avg: float,
    ratio: float,
) -> None:
    print(
        f"[D{D_agent} seed{seed}] occlusion_metrics first_err={first_avg:.6f} "
        f"reappear_err={reap_avg:.6f} ratio={ratio:.6f}"
    )


def log_coverage_summary(
    *,
    coverage_steps_total: int,
    coverage_hit_rate: float,
    coverage_square_blocks_seen: Iterable[int],
    coverage_covered_blocks_seen: Iterable[int],
) -> None:
    print(
        f"[coverage summary] steps={coverage_steps_total} hit_rate={coverage_hit_rate:.6f} "
        f"square_blocks_seen={sorted(coverage_square_blocks_seen)} "
        f"covered_blocks_seen={sorted(coverage_covered_blocks_seen)}"
    )


def log_transport_summary(
    *,
    diff_zero_match: int,
    diff_zero_mismatch: int,
    diff_nonzero_match: int,
    diff_nonzero_mismatch: int,
) -> None:
    print(
        "[summary diff/transport] "
        f"diff0_match={diff_zero_match} diff0_mismatch={diff_zero_mismatch} "
        f"diffnz_match={diff_nonzero_match} diffnz_mismatch={diff_nonzero_mismatch}"
    )
