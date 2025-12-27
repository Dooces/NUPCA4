"""Harness runner for NUPCA3."""

from __future__ import annotations

import math
from typing import Dict, List

import numpy as np

from nupca3.agent import NUPCA3Agent
from nupca3.config import AgentConfig
from .render import _check_block_partition
from .steps import run_steps
from .worlds import LinearARWorld, LinearSquareWorld, MovingColorShapeWorld
from .utils import build_partial_obs


def run_task(
    *,
    D: int,
    B: int,
    steps: int,
    seed: int,
    world: str,
    side: int,
    n_colors: int,
    n_shapes: int,
    square_small: int,
    square_big: int,
    pattern_period: int,
    dx: int,
    dy: int,
    p_color_shift: float,
    p_shape_shift: float,
    obs_budget: float,
    obs_cost: float,
    obs_budget_mode: str,
    obs_budget_min: float,
    obs_budget_max: float,
    coverage_debt_target: float,
    pred_only_start: int,
    pred_only_len: int,
    dense_world: bool,
    dense_sigma: float,
    fovea_residual_only: bool,
    binding_enabled: bool,
    binding_shift_radius: int,
    binding_rotations: bool,
    periph_blocks: int,
    periph_bins: int,
    periph_bg_full: bool,
    object_size: int,
    alpha_cov: float,
    coverage_cap_G: int,
    fovea_residual_ema: float,
    fovea_use_age: bool,
    fovea_age_min_inc: float,
    fovea_age_resid_scale: float,
    fovea_age_resid_thresh: float,
    fovea_routing_weight: float,
    fovea_routing_ema: float,
    occlude_start: int,
    occlude_len: int,
    occlude_period: int,
    working_set_linger_steps: int,
    transport_span_blocks: int,
    min_fovea_blocks: int,
    train_active_only: bool,
    train_active_threshold: float,
    train_weight_by_value: bool,
    train_value_power: float,
    lr_expert: float,
    sigma_ema: float,
    theta_learn: float,
    theta_ar_rest: float,
    nu_max: float,
    xi_max: float,
    stability_window: int,
    theta_ar: float,
    kappa_ar: float,
    scan_steps: int,
    warm_steps: int,
    warm_fovea_blocks: int,
    log_every: int,
    n_max: int,
    l_work_max: float,
    force_block_anchors: bool,
    diagnose_coverage: bool,
    coverage_log_every: int,
    rest_test_period: int,
    rest_test_length: int,
    periph_test: bool,
    transport_test: bool,
    transport_force_true_delta: bool,
    debug_full_state: bool,
    force_selected_blocks: bool,
    visualize_steps: int = 0,
) -> Dict[str, float]:
    D_cli = int(D)
    B_int = max(1, int(B))
    if D_cli <= 0:
        raise ValueError("D must be > 0")
    obs_cost_val = max(float(obs_cost), 1e-9)
    periph_blocks = max(0, int(periph_blocks))
    periph_bg_full = bool(periph_bg_full)
    debug_full_state = bool(debug_full_state)
    force_selected_blocks = bool(force_selected_blocks)
    moving = None
    square = None
    linear_world = None
    world_dim = D_cli
    if world == "moving":
        moving = MovingColorShapeWorld(
            side=side,
            n_colors=n_colors,
            n_shapes=n_shapes,
            seed=seed,
            p_color_shift=p_color_shift,
            p_shape_shift=p_shape_shift,
            periph_bins=periph_bins,
            object_size=object_size,
        )
        world_dim = moving.D
    elif world == "square":
        square = LinearSquareWorld(
            side=side,
            seed=seed,
            square_small=square_small,
            square_big=square_big,
            pattern_period=pattern_period,
            dx=dx,
            dy=dy,
            periph_bins=periph_bins,
        )
        world_dim = square.D
    else:
        linear_world = LinearARWorld(D=int(D_cli), seed=int(seed))
        world_dim = linear_world.D

    D_agent = int(world_dim) if dense_world else D_cli
    avg_block_size = float(D_agent) / float(B_int) if B_int > 0 else float(D_agent)
    avg_block_size = max(1.0, avg_block_size)
    budget_units = float(obs_budget) / float(obs_cost_val) if obs_cost_val > 0 else 0.0

    def _budget_to_blocks(budget: float) -> int:
        if budget <= 0.0:
            return 1
        dims = float(budget) / obs_cost_val
        ratio = float(dims) / avg_block_size
        ratio = max(0.0, ratio)
        blocks = int(math.ceil(max(1.0, ratio)))
        return max(1, min(B_int, blocks))

    k_fixed = _budget_to_blocks(float(obs_budget))
    k_min = _budget_to_blocks(float(obs_budget_min)) if obs_budget_min > 0 else k_fixed
    k_max = _budget_to_blocks(float(obs_budget_max)) if obs_budget_max > 0 else k_fixed
    if k_min > k_max:
        k_max = k_min
    if int(min_fovea_blocks) > 0:
        k_fixed = max(k_fixed, min(int(B), int(min_fovea_blocks)))
    print(
        f"[BUDGET] obs_budget={obs_budget} obs_cost={obs_cost_val} budget_units={budget_units:.3f} "
        f"avg_block_size={avg_block_size:.3f} blocks_selected={k_fixed}"
    )
    step_idx = 0
    obs_dims: List[int] = []
    transport_span_effective = int(transport_span_blocks)
    if transport_span_effective <= 0 and int(k_fixed) >= B_int:
        transport_span_effective = B_int
    train_weight_by_value_effective = bool(train_weight_by_value)
    if not train_weight_by_value_effective and world == "square" and int(k_fixed) >= B_int:
        train_weight_by_value_effective = True
    theta_learn_effective = float(theta_learn)
    if world == "square" and int(k_fixed) >= B_int and theta_learn_effective < 0.5:
        theta_learn_effective = 0.5
    block_size_agent = int(D_agent) // int(B_int)
    if block_size_agent * B_int != int(D_agent):
        raise ValueError("D must be divisible by B")
    periph_dim = int(periph_blocks) * block_size_agent
    if periph_dim > int(D_agent):
        raise ValueError("peripheral encoding requires more dims than available")
    base_dim = int(D_agent) - periph_dim
    periph_dims_set = set(range(base_dim, int(D_agent))) if periph_blocks > 0 else set()

    if world == "moving":
        cfg = AgentConfig(
            D=int(D_agent),
            B=int(B_int),
            fovea_blocks_per_step=int(k_fixed),
            fovea_residual_only=bool(fovea_residual_only),
            alpha_cov=float(alpha_cov),
            coverage_cap_G=int(coverage_cap_G),
            fovea_residual_ema=float(fovea_residual_ema),
            fovea_use_age=bool(fovea_use_age),
            fovea_age_min_inc=float(fovea_age_min_inc),
            fovea_age_resid_scale=float(fovea_age_resid_scale),
            fovea_age_resid_thresh=float(fovea_age_resid_thresh),
            working_set_linger_steps=int(working_set_linger_steps),
            transport_span_blocks=int(transport_span_effective),
            train_active_only=bool(train_active_only),
            train_active_threshold=float(train_active_threshold),
            train_weight_by_value=train_weight_by_value_effective,
            train_value_power=float(train_value_power),
            lr_expert=float(lr_expert),
            sigma_ema=float(sigma_ema),
            theta_learn=float(theta_learn_effective),
            theta_ar_rest=float(theta_ar_rest),
            nu_max=float(nu_max),
            xi_max=float(xi_max),
            W=int(stability_window),
            theta_ar=float(theta_ar),
            kappa_ar=float(kappa_ar),
            N_max=int(n_max),
            L_work_max=float(l_work_max),
            force_block_anchors=bool(force_block_anchors),
            grid_side=int(side),
            grid_channels=int(n_colors + n_shapes),
            grid_color_channels=int(n_colors),
            grid_shape_channels=int(n_shapes),
            grid_base_dim=int(base_dim),
            periph_bins=int(periph_bins),
            periph_blocks=int(periph_blocks),
            periph_channels=int(n_colors + n_shapes),
            transport_use_true_full=bool(transport_test),
            transport_force_true_delta=bool(transport_force_true_delta),
            fovea_routing_weight=float(fovea_routing_weight),
            fovea_routing_ema=float(fovea_routing_ema),
            allow_selected_blocks_override=bool(force_selected_blocks),
        )
        agent = NUPCA3Agent(cfg)
        moving.reset()
        vis_n_colors = int(n_colors)
        vis_n_shapes = int(n_shapes)
    elif world == "square":
        cfg = AgentConfig(
            D=int(D_agent),
            B=int(B_int),
            fovea_blocks_per_step=int(k_fixed),
            fovea_residual_only=bool(fovea_residual_only),
            alpha_cov=float(alpha_cov),
            coverage_cap_G=int(coverage_cap_G),
            fovea_residual_ema=float(fovea_residual_ema),
            fovea_use_age=bool(fovea_use_age),
            fovea_age_min_inc=float(fovea_age_min_inc),
            fovea_age_resid_scale=float(fovea_age_resid_scale),
            fovea_age_resid_thresh=float(fovea_age_resid_thresh),
            working_set_linger_steps=int(working_set_linger_steps),
            transport_span_blocks=int(transport_span_effective),
            train_active_only=bool(train_active_only),
            train_active_threshold=float(train_active_threshold),
            train_weight_by_value=train_weight_by_value_effective,
            train_value_power=float(train_value_power),
            lr_expert=float(lr_expert),
            sigma_ema=float(sigma_ema),
            theta_learn=float(theta_learn_effective),
            theta_ar_rest=float(theta_ar_rest),
            nu_max=float(nu_max),
            xi_max=float(xi_max),
            W=int(stability_window),
            theta_ar=float(theta_ar),
            kappa_ar=float(kappa_ar),
            N_max=int(n_max),
            L_work_max=float(l_work_max),
            force_block_anchors=bool(force_block_anchors),
            grid_side=int(side),
            grid_channels=1,
            grid_color_channels=1,
            grid_shape_channels=0,
            grid_base_dim=int(base_dim),
            periph_bins=int(periph_bins),
            periph_blocks=int(periph_blocks),
            periph_channels=1,
            transport_use_true_full=bool(transport_test),
            transport_force_true_delta=bool(transport_force_true_delta),
            fovea_routing_weight=float(fovea_routing_weight),
            fovea_routing_ema=float(fovea_routing_ema),
            allow_selected_blocks_override=bool(force_selected_blocks),
        )
        agent = NUPCA3Agent(cfg)
        square.reset()
        vis_n_colors = 1
        vis_n_shapes = 0
    else:
        cfg = AgentConfig(
            D=int(D_agent),
            B=int(B_int),
            fovea_blocks_per_step=int(k_fixed),
            fovea_residual_only=bool(fovea_residual_only),
            alpha_cov=float(alpha_cov),
            coverage_cap_G=int(coverage_cap_G),
            fovea_residual_ema=float(fovea_residual_ema),
            fovea_use_age=bool(fovea_use_age),
            fovea_age_min_inc=float(fovea_age_min_inc),
            fovea_age_resid_scale=float(fovea_age_resid_scale),
            fovea_age_resid_thresh=float(fovea_age_resid_thresh),
            working_set_linger_steps=int(working_set_linger_steps),
            transport_span_blocks=int(transport_span_effective),
            train_active_only=bool(train_active_only),
            train_active_threshold=float(train_active_threshold),
            train_weight_by_value=train_weight_by_value_effective,
            train_value_power=float(train_value_power),
            lr_expert=float(lr_expert),
            sigma_ema=float(sigma_ema),
            theta_learn=float(theta_learn_effective),
            theta_ar_rest=float(theta_ar_rest),
            nu_max=float(nu_max),
            xi_max=float(xi_max),
            W=int(stability_window),
            theta_ar=float(theta_ar),
            kappa_ar=float(kappa_ar),
            N_max=int(n_max),
            L_work_max=float(l_work_max),
            force_block_anchors=bool(force_block_anchors),
            binding_enabled=bool(binding_enabled),
            binding_shift_radius=int(binding_shift_radius),
            binding_rotations=bool(binding_rotations),
            periph_bins=int(periph_bins),
            periph_blocks=int(periph_blocks),
            periph_channels=1,
            transport_use_true_full=bool(transport_test),
            transport_force_true_delta=bool(transport_force_true_delta),
            fovea_routing_weight=float(fovea_routing_weight),
            fovea_routing_ema=float(fovea_routing_ema),
            allow_selected_blocks_override=bool(force_selected_blocks),
        )
        agent = NUPCA3Agent(cfg)
        linear_world.reset()
        vis_n_colors = 0
        vis_n_shapes = 0
    D_source = "world" if dense_world else "cli"
    print(
        f"[INIT] world_created=True world={world} D_world={world_dim} "
        f"D_agent={D_agent} D_source={D_source} B={B_int}"
    )
    buffer_last = np.asarray(agent.state.buffer.x_last, dtype=float).reshape(-1)
    buffer_prior = np.asarray(agent.state.buffer.x_prior, dtype=float).reshape(-1)
    if buffer_last.shape[0] != int(D_agent) or buffer_prior.shape[0] != int(D_agent):
        raise AssertionError(
            "Invariant violation: buffer sizes changed after agent init "
            f"(x_last={buffer_last.shape}, x_prior={buffer_prior.shape}, D_agent={D_agent})."
        )
    blocks_partition = getattr(agent.state, "blocks", []) or []
    blocks_ok, blocks_msg = _check_block_partition(blocks_partition, int(D_agent))
    print(f"[INIT] block_partition_ok={blocks_ok} detail={blocks_msg}")
    if not blocks_ok:
        raise AssertionError(f"Invariant violation: block partition invalid ({blocks_msg}).")
    if blocks_partition:
        block_sizes = [len(b) for b in blocks_partition]
        print(
            f"[INIT] block_sizes count={len(block_sizes)} min={min(block_sizes)} "
            f"max={max(block_sizes)} sum={sum(block_sizes)}"
        )
    print("[INIT] agent_constructed=True (buffers/weights stable after init)")
    return run_steps(
        agent=agent,
        cfg=cfg,
        world=world,
        steps=steps,
        seed=seed,
        B_int=B_int,
        D_agent=D_agent,
        world_dim=world_dim,
        base_dim=base_dim,
        periph_dim=periph_dim,
        periph_blocks=periph_blocks,
        periph_bg_full=periph_bg_full,
        periph_dims_set=periph_dims_set,
        periph_bins=periph_bins,
        coverage_cap_G=coverage_cap_G,
        coverage_log_every=coverage_log_every,
        diagnose_coverage=diagnose_coverage,
        rest_test_period=rest_test_period,
        rest_test_length=rest_test_length,
        periph_test=periph_test,
        transport_test=transport_test,
        transport_force_true_delta=transport_force_true_delta,
        obs_budget_mode=obs_budget_mode,
        coverage_debt_target=coverage_debt_target,
        dense_world=dense_world,
        k_fixed=k_fixed,
        k_min=k_min,
        k_max=k_max,
        min_fovea_blocks=min_fovea_blocks,
        scan_steps=scan_steps,
        warm_steps=warm_steps,
        warm_fovea_blocks=warm_fovea_blocks,
        pred_only_start=pred_only_start,
        pred_only_len=pred_only_len,
        occlude_start=occlude_start,
        occlude_len=occlude_len,
        occlude_period=occlude_period,
        debug_full_state=debug_full_state,
        force_selected_blocks=force_selected_blocks,
        side=side,
        n_colors=n_colors,
        n_shapes=n_shapes,
        vis_n_colors=vis_n_colors,
        vis_n_shapes=vis_n_shapes,
        log_every=log_every,
        visualize_steps=visualize_steps,
        block_size_agent=block_size_agent,
        moving=moving,
        square=square,
        linear_world=linear_world,
    )
