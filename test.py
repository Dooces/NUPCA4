#!/usr/bin/env python3
"""NUPCA3 axioms harness entrypoint."""

from __future__ import annotations

import argparse

from nupca3.harness.runner import run_task


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world", choices=["linear", "moving", "square"], default="linear")
    parser.add_argument("--D", type=int, default=16)
    parser.add_argument("--B", type=int, default=4)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--obs-budget", type=float, default=2.0)
    parser.add_argument("--obs-cost", type=float, default=1.0)
    parser.add_argument("--obs-budget-mode", choices=["fixed", "coverage"], default="fixed")
    parser.add_argument("--obs-budget-min", type=float, default=0.0)
    parser.add_argument("--obs-budget-max", type=float, default=0.0)
    parser.add_argument("--coverage-debt-target", type=float, default=0.0)
    parser.add_argument("--pred-only-start", type=int, default=-1)
    parser.add_argument("--pred-only-len", type=int, default=0)
    parser.add_argument("--dense-world", action="store_true")
    parser.add_argument("--dense-sigma", type=float, default=1.5)
    parser.add_argument("--fovea-residual-only", action="store_true")
    parser.add_argument("--binding-enabled", action="store_true")
    parser.add_argument("--binding-shift-radius", type=int, default=1)
    parser.add_argument("--binding-rotations", action="store_true")
    parser.add_argument("--periph-blocks", type=int, default=0)
    parser.add_argument("--periph-bins", type=int, default=2)
    parser.add_argument("--periph-bg-full", action="store_true")
    parser.add_argument("--object-size", type=int, default=3)
    parser.add_argument("--rest-test-period", type=int, default=0)
    parser.add_argument("--rest-test-length", type=int, default=0)
    parser.add_argument("--periph-test", action="store_true")
    parser.add_argument("--transport-test", action="store_true")
    parser.add_argument("--transport-force-true-delta", action="store_true")
    parser.add_argument("--debug-full-state", action="store_true")
    parser.add_argument("--force-selected-blocks", action="store_true")
    parser.add_argument("--alpha-cov", type=float, default=0.10)
    parser.add_argument("--coverage-cap-G", type=int, default=50)
    parser.add_argument("--fovea-residual-ema", type=float, default=0.10)
    parser.add_argument("--fovea-use-age", dest="fovea_use_age", action="store_true", default=True)
    parser.add_argument("--no-fovea-use-age", dest="fovea_use_age", action="store_false")
    parser.add_argument("--fovea-age-min-inc", type=float, default=0.05)
    parser.add_argument("--fovea-age-resid-scale", type=float, default=0.05)
    parser.add_argument("--fovea-age-resid-thresh", type=float, default=0.01)
    parser.add_argument("--fovea-routing-weight", type=float, default=1.0)
    parser.add_argument("--fovea-routing-ema", type=float, default=0.0)
    parser.add_argument("--occlude-start", type=int, default=-1)
    parser.add_argument("--occlude-len", type=int, default=0)
    parser.add_argument("--occlude-period", type=int, default=0)
    parser.add_argument("--working-set-linger-steps", type=int, default=0)
    parser.add_argument("--transport-span-blocks", type=int, default=0)
    parser.add_argument("--min-fovea-blocks", type=int, default=0)
    parser.add_argument("--train-active-only", action="store_true")
    parser.add_argument("--train-active-threshold", type=float, default=0.0)
    parser.add_argument("--train-weight-by-value", action="store_true")
    parser.add_argument("--train-value-power", type=float, default=1.0)
    parser.add_argument("--theta-learn", type=float, default=0.02)
    parser.add_argument("--lr-expert", type=float, default=0.01)
    parser.add_argument("--sigma-ema", type=float, default=0.01)
    parser.add_argument("--theta-ar-rest", type=float, default=1.0)
    parser.add_argument("--theta-ar", type=float, default=0.5)
    parser.add_argument("--kappa-ar", type=float, default=0.2)
    parser.add_argument("--nu-max", type=float, default=1.0)
    parser.add_argument("--xi-max", type=float, default=12.0)
    parser.add_argument("--stability-window", type=int, default=50)
    parser.add_argument("--scan-steps", type=int, default=0)
    parser.add_argument("--warm-steps", type=int, default=0)
    parser.add_argument("--warm-fovea-blocks", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--visualize-steps", type=int, default=0)
    parser.add_argument("--n-max", type=int, default=256)
    parser.add_argument("--l-work-max", type=float, default=48.0)
    parser.add_argument("--force-block-anchors", action="store_true")
    parser.add_argument("--diagnose-coverage", action="store_true")
    parser.add_argument("--coverage-log-every", type=int, default=0)
    parser.add_argument("--side", type=int, default=8)
    parser.add_argument("--n-colors", type=int, default=4)
    parser.add_argument("--n-shapes", type=int, default=4)
    parser.add_argument("--p-color-shift", type=float, default=0.05)
    parser.add_argument("--p-shape-shift", type=float, default=0.05)
    parser.add_argument("--square-small", type=int, default=2)
    parser.add_argument("--square-big", type=int, default=3)
    parser.add_argument("--pattern-period", type=int, default=20)
    parser.add_argument("--dx", type=int, default=1)
    parser.add_argument("--dy", type=int, default=0)
    args = parser.parse_args()

    summary = run_task(
        D=args.D,
        B=args.B,
        steps=args.steps,
        seed=args.seed,
        world=args.world,
        side=args.side,
        n_colors=args.n_colors,
        n_shapes=args.n_shapes,
        square_small=args.square_small,
        square_big=args.square_big,
        pattern_period=args.pattern_period,
        dx=args.dx,
        dy=args.dy,
        p_color_shift=args.p_color_shift,
        p_shape_shift=args.p_shape_shift,
        obs_budget=args.obs_budget,
        obs_cost=args.obs_cost,
        obs_budget_mode=args.obs_budget_mode,
        obs_budget_min=args.obs_budget_min,
        obs_budget_max=args.obs_budget_max,
        coverage_debt_target=args.coverage_debt_target,
        pred_only_start=args.pred_only_start,
        pred_only_len=args.pred_only_len,
        dense_world=args.dense_world,
        dense_sigma=args.dense_sigma,
        fovea_residual_only=args.fovea_residual_only,
        binding_enabled=args.binding_enabled,
        binding_shift_radius=args.binding_shift_radius,
        binding_rotations=args.binding_rotations,
        periph_blocks=args.periph_blocks,
        periph_bins=args.periph_bins,
        periph_bg_full=args.periph_bg_full,
        object_size=args.object_size,
        alpha_cov=args.alpha_cov,
        coverage_cap_G=args.coverage_cap_G,
        fovea_residual_ema=args.fovea_residual_ema,
        fovea_use_age=args.fovea_use_age,
        fovea_age_min_inc=args.fovea_age_min_inc,
        fovea_age_resid_scale=args.fovea_age_resid_scale,
        fovea_age_resid_thresh=args.fovea_age_resid_thresh,
        fovea_routing_weight=args.fovea_routing_weight,
        fovea_routing_ema=args.fovea_routing_ema,
        occlude_start=args.occlude_start,
        occlude_len=args.occlude_len,
        occlude_period=args.occlude_period,
        working_set_linger_steps=args.working_set_linger_steps,
        transport_span_blocks=args.transport_span_blocks,
        min_fovea_blocks=args.min_fovea_blocks,
        train_active_only=args.train_active_only,
        train_active_threshold=args.train_active_threshold,
        train_weight_by_value=args.train_weight_by_value,
        train_value_power=args.train_value_power,
        lr_expert=args.lr_expert,
        sigma_ema=args.sigma_ema,
        theta_learn=args.theta_learn,
        theta_ar_rest=args.theta_ar_rest,
        nu_max=args.nu_max,
        xi_max=args.xi_max,
        stability_window=args.stability_window,
        theta_ar=args.theta_ar,
        kappa_ar=args.kappa_ar,
        scan_steps=args.scan_steps,
        warm_steps=args.warm_steps,
        warm_fovea_blocks=args.warm_fovea_blocks,
        log_every=args.log_every,
        n_max=args.n_max,
        l_work_max=args.l_work_max,
        force_block_anchors=args.force_block_anchors,
        diagnose_coverage=args.diagnose_coverage,
        coverage_log_every=args.coverage_log_every,
        rest_test_period=args.rest_test_period,
        rest_test_length=args.rest_test_length,
        periph_test=args.periph_test,
        transport_test=args.transport_test,
        transport_force_true_delta=args.transport_force_true_delta,
        debug_full_state=args.debug_full_state,
        force_selected_blocks=args.force_selected_blocks,
        visualize_steps=args.visualize_steps,
    )
    print("[SUMMARY]", vars(args), summary)


if __name__ == "__main__":
    main()
