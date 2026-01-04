"""nupca3/config.py

NUPCA configuration (v5).

This module defines :class:`AgentConfig`, a frozen dataclass that centralizes
all numeric and boolean hyperparameters referenced across the codebase.

v5 requirements enforced here
-----------------------------
- Core paths must not depend on implicit config fallbacks (no getattr-based
  defaulting for required fields).
- NUPCA5 scan-proof signature retrieval parameters (sig_*)
  are first-class fields with strict validation.
- Context/gist used for sig64 is ephemeral; gist-derived state must not be persisted.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace as dc_replace
from typing import Any, Dict, Tuple


@dataclass(frozen=True)
class AgentConfig:
    """
    Immutable configuration for a NUPCA agent (v5).

    Inputs
    ------
    This dataclass is typically constructed either:
      - directly (e.g., AgentConfig(D=..., B=..., ...)), or
      - via :func:`default_config` and then :meth:`replace`.

    Outputs
    -------
    An immutable configuration object whose attributes are read by all modules.
    """

    # =========================================================================
    # A16 — Observation geometry / receptive-field geometry
    # =========================================================================
    # Total state dimensionality (DoF) (A16.1).
    # Reference default aligns with the common “16×16 pixels, 4 channels”
    # harness used in this repo family: D = 256 blocks × 4 dims/block.
    D: int = 1024

    # Number of DoF-aligned blocks (A16.1). Typical default: one block per pixel.
    B: int = 256

    # Metadata cap on block domain size for sig64 (B_MAX). 0 -> use B.
    B_max: int = 0

    # Fovea budget per step (A16.3): number of blocks to observe.
    fovea_blocks_per_step: int = 64

    # Metadata cap on selected blocks for sig64 (F_MAX). 0 -> use fovea_blocks_per_step.
    F_max: int = 0

    # Hard coverage cap G (A16.4): if age(b,t-1) >= G, block b must be observed.
    coverage_cap_G: int = 50

    # Starting age for each block; use < G so the coverage emergency path waits for real data.
    initial_block_age: int = 0

    # Stale persistence TTL in steps for unobserved blocks; 0 disables expiry.
    stale_block_ttl: int = 20

    # Removed legacy sticky retention; not allowed in v5.

    # Coverage regularizer for greedy_cov fovea selection (A16).
    alpha_cov: float = 0.10

    # Coverage debt boosts for expert and abstraction strata (Fix 1).
    alpha_cov_exp: float = 0.05
    alpha_cov_band: float = 0.05
    # Coverage debt limits and caps used by debt trackers.
    coverage_debt_cap: int = 64
    coverage_debt_max: int = 10_000
    coverage_debt_thresh: float = 1e6

    # Diagnostic mode: if True, select fovea blocks by residual only (ignores age).
    fovea_residual_only: bool = False

    # EMA rate for block residual tracking (A16.2).
    fovea_residual_ema: float = 0.10

    # Debug logging cadence for fovea block IDs (0 disables logging).
    fovea_log_every: int = 1

    # Sliding window size for fovea visit count logging.
    fovea_visit_window: int = 256

    # Relevance-weighted age increments (implementation detail).
    fovea_age_min_inc: float = 0.05
    fovea_age_resid_scale: float = 0.05
    fovea_age_resid_thresh: float = 0.01
    fovea_use_age: bool = True

    # Uncertainty-driven coverage bonus (Fix 3).
    fovea_uncertainty_weight: float = 0.0
    fovea_uncertainty_default: float = 1.0

    # Multi-world disagreement/inovation routing contributions into fovea budget.
    fovea_disagreement_weight: float = 0.0
    fovea_innovation_weight: float = 0.0
    fovea_periph_demand_weight: float = 0.0
    fovea_confidence_weight: float = 0.10
    fovea_confidence_beta_up: float = 0.50
    fovea_confidence_beta_down: float = 0.01
    # Allow env-provided selected_blocks to override fovea selection.
    allow_selected_blocks_override: bool = False

    # Trace cache (A16.8) bounds (operating-only cue history).
    trace_cache_max_entries: int = 16
    trace_cache_max_cues_per_entry: int = 12
    trace_cache_block_cap: int = 32

    # Contemplation intake gating (A0.5) + planning-only budget reuse.
    contemplate_force: bool = False
    contemplate_hazard_threshold: float = 1.0
    contemplate_novelty_threshold: float = 1.0
    contemplate_novelty_slack_frac: float = 0.5
    contemplate_budget_slack_frac: float = 0.5
    contemplate_budget_reuse_frac: float = 0.5
    contemplate_anchor_blocks: Tuple[int, ...] = (0,)

    # Value-of-compute advisor (A0.BUDGET / A16.4)
    value_of_compute_alpha: float = 1.0
    value_of_compute_beta: float = 0.5
    value_of_compute_gamma: float = 0.5
    value_of_compute_periph_weight: float = 0.1
    value_of_compute_novelty_decay: float = 0.8
    value_of_compute_budget_scale: float = 0.5
    value_of_compute_planning_scale: float = 0.5
    value_of_compute_stage2_scale: float = 0.5
    value_of_compute_candidate_scale: float = 0.5

    # Planning-thread scheduler (A0.6 time-sliced micro-steps)
    planning_threads_max: int = 2
    planning_micro_steps_per_tick: int = 4

    # Allowed transport span (blocks) for expert input masks.
    transport_span_blocks: int = 0

    # Transport diagnostics/controls: use full state when True (debug/test only).
    transport_use_true_full: bool = False
    # Maximum cell offset (L1) explored when aligning grid mass for transport.
    transport_search_radius: int = 1
    # Enable transport derived from the environment grid (debug only).
    transport_debug_env_grid: bool = False
    # Belief filtering over candidate deltas
    transport_belief_decay: float = 0.5
    transport_inertia_weight: float = 0.0
    # Margin required between top candidates to commit with confidence (probability difference)
    transport_confidence_margin: float = 0.25
    # Additional routing bonus for blocks that disambiguate candidate shifts
    transport_disambiguation_weight: float = 1.0

    # When True, seed the candidate set with the environment-provided delta and force selection (debug).
    transport_force_true_delta: bool = False

    # High-confidence gating thresholds for pseudo-label acquisition.
    transport_high_confidence_margin: float = 0.05
    transport_high_confidence_overlap: int = 2

    # Minimum overlap (|I(d)|) required for a candidate to be informative.
    transport_min_overlap: int = 1
    # Score penalty applied when overlap is small (higher => penalize more).
    transport_overlap_penalty: float = 0.0
    # Score bonus applied for larger overlaps to prefer richer evidence.
    transport_overlap_bonus: float = 0.0
    # Evidence margin τ below which we default to zero transport.
    transport_evidence_margin: float = 0.02
    # Minimum absolute signal on observed dims to treat as usable evidence.
    transport_signal_floor: float = 1e-5
    # Penalty applied per ASCII occupancy mismatch during transport scoring.
    transport_ascii_penalty: float = 0.1
    # Probability threshold for treating multiple candidates as tied (for zero-bias ordering).
    transport_tie_probability_threshold: float = 1e-4
    # Score threshold below which candidates are treated as uninformative.
    transport_uninformative_score: float = -1e6
    # Score margin threshold used to flag ties (diagnostics only).
    transport_tie_threshold: float = 1e-4

    # Extended transform search: rotations + learned shift corrections
    transport_rotation_enabled: bool = False
    transport_rotation_steps: Tuple[int, ...] = (0,)
    transport_offset_history_size: int = 0
    transport_offset_radius: int = 1
    transport_bias_decay: float = 0.9
    transport_bias_weight: float = 0.0
    transport_bias_increment: float = 0.1
    transport_bias_max_entries: int = 4

    coverage_score_tol: float = 1e-5
    coverage_score_threshold: float = -1e-6
    coverage_cursor_step: int = 1

    # Multi-world hypothesis tracking for transport ambiguity (Phase 3).
    multi_world_K: int = 1
    multi_world_lambda: float = 1.0
    multi_world_support_window: int = 4
    multi_world_merge_eps: float = 1e-3

    # Binding/equivariance (optional; implementation detail).
    binding_enabled: bool = False
    binding_shift_radius: int = 1
    binding_rotations: bool = True
    # Explicit grid dimensions (no legacy aliases).
    # When provided (and when B matches grid_width*grid_height for 1-channel
    # grids), geometry-aware fovea selection can enforce a circular receptive
    # field over a rectangular plane.
    grid_width: int = 0
    grid_height: int = 0
    grid_channels: int = 0
    grid_color_channels: int = 0
    grid_shape_channels: int = 0
    grid_base_dim: int = 0

    # Objective shaping for sparse signals (training only).
    train_active_only: bool = False
    train_active_threshold: float = 0.0
    train_weight_by_value: bool = False
    train_value_power: float = 1.0

    # Working-set linger window (steps) for recently active non-anchors.
    working_set_linger_steps: int = 0



    # =========================================================================
    # NUPCA5 — Scan-proof signature retrieval (A4.3′)
    # =========================================================================

    # Deterministic seed for sig64 and sig_index salts.
    sig_seed: int = 0

    # Ephemeral periphery gist bins for sig64 (used at decision time only).
    sig_gist_bins: int = 8

    # NUPCA5: prohibit persisting gist/context state across steps.
    sig_disable_context_register: bool = True

    # Deferred validation of sig_index priorities.
    sig_enable_validation: bool = True
    sig_err_ema_beta: float = 0.10


    # Committed metadata quantization (sig64).
    sig_value_bins: int = 8
    sig_vmax: float = 4.0

    # Packed signature index parameters (sig_index).
    sig_min_evidence: int = 2
    sig_df_stop_frac: float = 0.2
    sig_sketch_K: int = 0
    sig_tables: int = 4
    sig_bucket_bits: int = 10
    sig_bucket_cap: int = 8

    # Candidate cap for sig_index query (bounded).
    sig_query_cand_cap: int = 64

    # Stage-2 rerank weighting of deferred-validation error (working_set stage-2).
    sig_stage2_alpha_err: float = 0.25

    # Outcome-vetted priority cache driving bucket overflow replacement.
    sig_enable_err_cache: bool = True
    sig_err_bins: int = 3
    sig_err_init: float = 1e6
    sig_eviction_bin: int = 2

    # Pending prediction storage capacity (PREDVAL_MAX).
    pred_store_capacity: int = 64

    # =========================================================================
    # A3 — Stability / introversion thresholds used by permit_struct
    # =========================================================================
    nu_max: float = 0.02
    xi_max: float = 0.10
    W: int = 50

    # =========================================================================
    # A7 — Precision / uncertainty handling
    # =========================================================================
    sigma_floor: float = 1e-3
    sigma_ema: float = 0.01
    # Initial per-dimension uncertainty for newly created nodes (untrained).
    # Use +inf to indicate "no coverage" in fusion until the node has updated at least once.
    sigma_init_untrained: float = float("inf")

    # A7.4 rollout uncertainty propagation / confidence mapping
    rollout_eta_proc: float = 0.01
    rollout_mu_H: float = 0.10
    rollout_sigma_H: float = 0.05
    # Rollout confidence (Option A): c = exp(-alpha * H_cov) * rho^beta
    rollout_c_alpha: float = 1.0
    rollout_c_beta: float = 1.0

    # =========================================================================
    # A6 — Compute budget & horizon scalars
    # =========================================================================
    # Real-time compute budget per step (A6.1). Reference value from v1.5b.
    B_rt: float = 260.0

    # Optional wallclock-based tick budget (converted to same units via compute_units_per_ms).
    tick_budget_ms: float = 0.0
    compute_units_per_ms: float = 1.0

    # Encoding base cost b_enc,0 (A6.2). Reference value from v1.5b.
    b_enc_base: float = 3.2

    # Rollout per-step cost b_roll,0 (A6.2). Reference value from v1.5b.
    b_roll_base: float = 0.85

    # Anchor overhead coefficient b_anc,0 (A6.2).
    b_anc_base: float = 0.0

    # Consolidation bookkeeping base cost.
    b_cons_base: float = 0.0

    # Small epsilon to avoid divide-by-zero in budget terms (A6.2).
    eps_budget: float = 1e-6

    # Hard cap on horizon to keep rollout computation bounded.
    h_max: int = 32



    # =========================================================================
    # A10/A11 — Learning gates and safety clamps
    # =========================================================================
    enable_learning: bool = True
    lr_expert: float = 1e-2

    # ---------------------------------------------------------------------
    # Gate timing discipline
    # ---------------------------------------------------------------------
    # When False (default), safety/learning gates use lagged t-1 signals (A10.2).
    # When True, gates use same-step values (useful for debugging).
    gates_use_current: bool = False

    # Baseline update rates (A3); required by state.baselines
    baseline_alpha_fast: float = 0.01
    baseline_alpha_slow: float = 0.001
    baseline_var_floor: float = 1.0

    # Commitment gate parameters (A8); required by control.commitment
    d_latency_floor: int = 1
    theta_act: float = 0.5

    # Diagnostics feel proxy weights (A17); required by diagnostics.metrics
    feel_eta: float = 1.0
    feel_xi: float = 1.0

    # Salience temperature controls (A5); memory.salience falls back if absent
    tau_base: float = 1.0
    tau_min: float = 0.5
    tau_max: float = 5.0
    tau_a: float = 0.5
    # Debug hook: score all nodes when True (override candidate universe).
    salience_debug_exhaustive: bool = False

    # Peripheral channel (optional; used for routing bias in fovea selection)
    periph_bins: int = 0
    periph_blocks: int = 0
    periph_channels: int = 1
    # Spatial fovea geometry constraint. "circle" enforces a single disk-like
    # observation footprint in grid-aware environments.
    fovea_shape: str = "circle"
    fovea_routing_weight: float = 0.0
    fovea_routing_ema: float = 0.0
    motion_probe_blocks: int = 2

    # Salience / context control (A5: biasing expert scores and gist tracking).
    alpha_pi: float = 0.4
    alpha_deg: float = 0.2
    alpha_ctx: float = 0.6  # unused legacy hook; kept as scalar field only.
    alpha_ctx_relevance: float = 0.4
    alpha_ctx_gist: float = 0.1
    beta_context: float = 0.1
    beta_context_node: float = 0.1

    # Log sizes / statistics update rates (operational bounds)
    activation_log_max: int = 200
    transition_log_max: int = 128
    split_stats_beta: float = 0.1

    # Learn gate threshold (A10/A11).
    theta_learn: float = 0.15

    # Baseline normalization safety (A3.2).
    baseline_var_floor: float = 0.0
    baseline_var_floor_C: float = 0.01
    baseline_z_clip: float = 6.0

    # ---------------------------------------------------------------------
    # A10 — Edit control thresholds (v1.5b symbols)
    # ---------------------------------------------------------------------
    # Freeze threshold χ^{th} on external threat (A10.1).
    chi_th: float = 0.90

    # Compute slack threshold τ_C^{edit} (A10.2).
    tau_C_edit: float = 0.0

    # Raw headroom thresholds τ_E^{edit}, τ_D^{edit} (A10.2).
    # These apply to rawE(t)=E(t)-E_min and rawD(t)=D_max-D(t).
    tau_E_edit: float = 0.0
    tau_D_edit: float = 0.0

    # Arousal threshold above which learning is suppressed during REST (A10).
    theta_ar_rest: float = 0.80

    # Panic threshold used to block risky operations when arousal is extreme (A10).
    theta_ar_panic: float = 0.95

    # =========================================================================
    # A9 — Spawn / structural edit pressure (queueing into REST)
    # =========================================================================
    beta_R: float = 0.05
    K: float = 32.0
    theta_spawn: float = 0.20

    # =========================================================================
    # A12 — Acceptance criteria (MDL vs improvement)
    # =========================================================================
    mdl_beta: float = 0.10
    epsilon_merge: float = 1e-3
    eps_baseline: float = 0.1

    # A12 MDL deltas used as rough complexity penalties for different edits.
    delta_L_MDL_merge: float = 0.50
    delta_L_MDL_spawn: float = 0.80

    # A4.4 / A12.3 anti-aliasing threshold (structural edits).
    # In the axiom table this appears as θ_alias.
    theta_alias: float = 0.95

    # =========================================================================
    # A14 — Macrostate (REST / WAKE) dynamics: rest pressure parameters
    # Naming: the implementation uses a P_rest_* prefix (see macrostate.py).
    # =========================================================================
    # Exponential decay factor (γ_rest).
    P_rest_gamma: float = 0.10

    # Baseline drift (δ_base).
    P_rest_delta_base: float = 0.01

    # Need-driven drift gain (δ_need).
    P_rest_delta_need: float = 0.03

    # Need weights α_E, α_D, α_S.
    P_rest_alpha_E: float = 1.0
    P_rest_alpha_D: float = 1.0
    P_rest_alpha_S: float = 1.0

    # Demand hysteresis thresholds θ_enter, θ_exit.
    P_rest_theta_demand_enter: float = 0.60
    P_rest_theta_demand_exit: float = 0.40

    # Structural queue gates Θ_Q^{on,off}.
    P_rest_Theta_Q_on: float = 0.50
    P_rest_Theta_Q_off: float = 0.30

    # Maximum contiguous durations T_max^{wake}, T_max^{rest}.
    P_rest_Tmax_wake: int = 4000
    P_rest_Tmax_rest: int = 800

    # Minimum contiguous REST duration (prevents thrashing).
    P_rest_Tmin_rest: int = 10

    # REST hysteresis leak.
    P_rest_lambda_hyst: float = 0.5

    # Freeze threshold on rest pressure.
    P_rest_theta_freeze: float = 0.95

    # Margin-based hard triggers θ_E^rest, θ_D^rest, θ_S^rest.
    P_rest_theta_E_rest: float = 0.0
    P_rest_theta_D_rest: float = 0.0
    P_rest_theta_S_rest: float = 0.0

    # Safety/interrupt thresholds θ_safe^{th}, θ_interrupt^{th}.
    P_rest_theta_safe_th: float = 0.20
    P_rest_theta_interrupt_th: float = 0.90
    # Minimum rest cycles before demand can re-enter REST.
    rest_min_cycles: int = 1
    # Latch steps to skip REST after a zero-processed cycle.
    rest_cooldown_steps: int = 0

    # Maximum structural edits to process per REST step (A12/A14).
    # This bounds consolidation work per REST tick; actual compute slack is
    # still governed by A6 via x_C and downstream acceptance criteria.
    max_edits_per_rest_step: int = 32


    # =========================================================================
    # A15 — Margins, stress, arousal dynamics
    # =========================================================================
    # Initial hard observables (E, D) for A15 dynamics.
    E_init: float = 1.0
    D_init: float = 0.0
    tau_E: float = 2000.0
    tau_D: float = 2000.0
    tau_S: float = 2000.0
    tau_E_need: float = 2000.0
    tau_D_need: float = 2000.0
    tau_S_need: float = 2000.0

    kappa_E: float = 1.0
    kappa_D: float = 1.0
    kappa_S: float = 1.0
    kappa_E_need: float = 1.0
    kappa_D_need: float = 1.0
    kappa_S_need: float = 1.0

    k_rest_E: float = 0.05
    k_rest_D: float = 0.05
    k_rest_S: float = 0.05

    # Arousal scoring weights (w_*^{ar}).
    w_E_ar: float = 1.0
    w_D_ar: float = 1.0
    w_S_ar: float = 1.0
    w_deltam_ar: float = 1.0
    w_Epred_ar: float = 1.0
    w_L: float = 1.0
    w_C: float = 1.0
    w_S: float = 1.0
    w_delta: float = 1.0
    w_E: float = 0.0

    # Arousal logistic parameters (θ_ar, κ_ar).
    theta_ar: float = 0.50
    kappa_ar: float = 0.20
    theta_a: float = 0.50

    # Arousal leaky dynamics (τ_rise, τ_decay).
    tau_rise: float = 50.0
    tau_decay: float = 500.0

    # Exploration/“openness” coupling β_open and need sharpening β_sharp.
    beta_open: float = 0.10
    beta_sharp: float = 2.0

    # Diagonal σ floor used when forming precision-weighted diagnostics.
    sigma_floor_diag: float = 1e-2

    # Legacy gain/bias/decay knobs (kept because some modules may still use them).
    arousal_gain: float = 1.0
    arousal_bias: float = 0.0
    arousal_decay: float = 0.99

    # =========================================================================
    # Implementation capacity / queue limits (not new axioms; operational bounds)
    # =========================================================================
    N_max: int = 256
    L_work_max: float = 48.0
    K_max: int = 0
    C_cand_max: int = 0
    max_candidates: int = 32
    max_retrieval_candidates: int = 64
    salience_max_candidates: int = 64
    salience_explore_budget: int = 2
    max_experts_per_k: int = 2

    max_queue: int = 256
    rest_queue_trigger: float = 0.80
    force_block_anchors: bool = False

    # REST sampling / replay knobs (OPERATING ↔ REST schedules).
    rest_epochs_per_rest: int = 3
    rest_rollouts_per_epoch: int = 16
    rest_rollout_horizon: int = 8
    rest_rollout_k: int = 2

    # Structural proposal throttling (used by REST processors).
    max_proposals_per_step: int = 8

    # MDL-like costing helpers used by acceptance and pruning heuristics.
    expert_base_cost: float = 1.0
    expert_dim_cost: float = 0.05
    expert_cost_include_inputs: bool = False

    # =========================================================================
    # Methods
    # =========================================================================

    def validate(self) -> None:
        """Validate invariants implied by the axioms and implementation.

        Raises ValueError if any invariant is violated.
        """

        # Geometry
        if not isinstance(self.D, int) or self.D <= 0:
            raise ValueError("D must be a positive int")
        if not isinstance(self.B, int) or self.B <= 0:
            raise ValueError("B must be a positive int")
        if self.B > self.D:
            raise ValueError("B must be <= D")
        B_cap = int(self.B_max) if int(self.B_max) > 0 else int(self.B)
        if B_cap <= 0:
            raise ValueError("B_max must resolve to a positive int")
        if B_cap < self.B:
            raise ValueError("B_max must be >= B")
        if self.fovea_blocks_per_step <= 0:
            raise ValueError("fovea_blocks_per_step must be > 0")
        F_cap = int(self.F_max) if int(self.F_max) > 0 else int(self.fovea_blocks_per_step)
        if F_cap <= 0:
            raise ValueError("F_max must resolve to a positive int")
        if F_cap < self.fovea_blocks_per_step:
            raise ValueError("F_max must be >= fovea_blocks_per_step")
        if self.coverage_cap_G <= 0:
            raise ValueError("coverage_cap_G must be > 0")
        if self.initial_block_age < 0:
            raise ValueError("initial_block_age must be >= 0")
        if self.stale_block_ttl < 0:
            raise ValueError("stale_block_ttl must be >= 0")

        # Uncertainty / EMA
        if not (self.sigma_floor > 0.0):
            raise ValueError("sigma_floor must be > 0")
        if not (0.0 < self.sigma_ema <= 1.0):
            raise ValueError("sigma_ema must be in (0, 1]")
        if self.baseline_var_floor < 0.0 or self.baseline_var_floor_C < 0.0:
            raise ValueError("baseline_var_floor/baseline_var_floor_C must be >= 0")
        if self.baseline_z_clip <= 0.0:
            raise ValueError("baseline_z_clip must be > 0")

        # Budgets
        if self.B_rt < 0.0:
            raise ValueError("B_rt must be >= 0")
        if self.b_enc_base < 0.0:
            raise ValueError("b_enc_base must be >= 0")
        if self.b_roll_base < 0.0:
            raise ValueError("b_roll_base must be >= 0")
        if self.b_anc_base < 0.0:
            raise ValueError("b_anc_base must be >= 0")
        if self.b_cons_base < 0.0:
            raise ValueError("b_cons_base must be >= 0")
        if self.eps_budget <= 0.0:
            raise ValueError("eps_budget must be > 0")
        if self.h_max <= 0:
            raise ValueError("h_max must be > 0")
        if self.tick_budget_ms < 0.0:
            raise ValueError("tick_budget_ms must be >= 0")
        if self.compute_units_per_ms <= 0.0:
            raise ValueError("compute_units_per_ms must be > 0")

        # Time constants
        if self.tau_E <= 0.0 or self.tau_D <= 0.0 or self.tau_S <= 0.0:
            raise ValueError("tau_E/tau_D/tau_S must be > 0")
        if self.tau_rise <= 0.0 or self.tau_decay <= 0.0:
            raise ValueError("tau_rise and tau_decay must be > 0")

        # Coverage / fovea tuning
        if self.alpha_cov < 0.0 or self.alpha_cov_exp < 0.0 or self.alpha_cov_band < 0.0:
            raise ValueError("alpha_cov/alpha_cov_exp/alpha_cov_band must be >= 0")
        if self.fovea_residual_ema < 0.0 or self.fovea_residual_ema > 1.0:
            raise ValueError("fovea_residual_ema must be in [0, 1]")
        if self.fovea_visit_window <= 0:
            raise ValueError("fovea_visit_window must be > 0")
        if self.fovea_age_min_inc < 0.0:
            raise ValueError("fovea_age_min_inc must be >= 0")
        if self.fovea_age_resid_scale < 0.0:
            raise ValueError("fovea_age_resid_scale must be >= 0")
        if self.fovea_age_resid_thresh < 0.0:
            raise ValueError("fovea_age_resid_thresh must be >= 0")
        if self.fovea_uncertainty_weight < 0.0:
            raise ValueError("fovea_uncertainty_weight must be >= 0")
        if self.fovea_uncertainty_default < 0.0:
            raise ValueError("fovea_uncertainty_default must be >= 0")
        if self.trace_cache_max_entries <= 0:
            raise ValueError("trace_cache_max_entries must be > 0")
        if self.trace_cache_max_cues_per_entry <= 0:
            raise ValueError("trace_cache_max_cues_per_entry must be > 0")
        if self.trace_cache_block_cap <= 0:
            raise ValueError("trace_cache_block_cap must be > 0")

        # Transport
        if self.transport_search_radius < 0:
            raise ValueError("transport_search_radius must be >= 0")
        if not (0.0 < self.transport_belief_decay < 1.0):
            raise ValueError("transport_belief_decay must be in (0, 1)")
        if self.transport_inertia_weight < 0.0:
            raise ValueError("transport_inertia_weight must be >= 0")
        if not (0.0 <= self.transport_confidence_margin <= 1.0):
            raise ValueError("transport_confidence_margin must be in [0, 1]")
        if self.transport_disambiguation_weight < 0.0:
            raise ValueError("transport_disambiguation_weight must be >= 0")
        if self.transport_signal_floor < 0.0:
            raise ValueError("transport_signal_floor must be >= 0")

        # Context tracking
        if not (0.0 <= self.beta_context <= 1.0):
            raise ValueError("beta_context must be in [0, 1]")
        if not (0.0 <= self.beta_context_node <= 1.0):
            raise ValueError("beta_context_node must be in [0, 1]")

        # Grid metadata consistency (only when explicitly specified)
        if self.grid_color_channels < 0 or self.grid_shape_channels < 0:
            raise ValueError("grid_color_channels and grid_shape_channels must be >= 0")
        if (self.grid_color_channels + self.grid_shape_channels) > 0:
            if (self.grid_color_channels + self.grid_shape_channels) != self.grid_channels:
                raise ValueError(
                    "grid_color_channels + grid_shape_channels must equal grid_channels when specified"
                )

        # Anti-aliasing thresholds
        if not (0.0 <= self.theta_alias <= 1.0):
            raise ValueError("theta_alias must be in [0, 1]")

        # NUPCA5 signature retrieval invariants
        if self.sig_seed < 0:
            raise ValueError("sig_seed must be >= 0")
        if self.sig_gist_bins < 1:
            raise ValueError("sig_gist_bins must be >= 1")
        if not self.sig_disable_context_register:
            raise ValueError("sig_disable_context_register must be True under v5")
        if not self.sig_enable_validation:
            raise ValueError("sig_enable_validation must be True under v5")
        if not (0.0 < self.sig_err_ema_beta <= 1.0):
            raise ValueError("sig_err_ema_beta must be in (0, 1]")
        if self.sig_value_bins < 1:
            raise ValueError("sig_value_bins must be >= 1")
        if self.sig_vmax <= 0.0:
            raise ValueError("sig_vmax must be > 0")
        if self.sig_tables < 1:
            raise ValueError("sig_tables must be >= 1")
        if self.sig_bucket_bits < 1:
            raise ValueError("sig_bucket_bits must be >= 1")
        if self.sig_bucket_cap < 1:
            raise ValueError("sig_bucket_cap must be >= 1")
        if self.sig_query_cand_cap < 1:
            raise ValueError("sig_query_cand_cap must be >= 1")
        if self.sig_stage2_alpha_err < 0.0:
            raise ValueError("sig_stage2_alpha_err must be >= 0")
        if self.pred_store_capacity < 1:
            raise ValueError("pred_store_capacity must be >= 1")
        if not self.sig_enable_err_cache:
            raise ValueError("sig_enable_err_cache must be True under v5")
        if self.sig_err_bins < 1:
            raise ValueError("sig_err_bins must be >= 1")
        if self.sig_err_init < 0.0:
            raise ValueError("sig_err_init must be >= 0")
        if not (0 <= self.sig_eviction_bin < self.sig_err_bins):
            raise ValueError("sig_eviction_bin out of range")
        cand_cap = int(self.C_cand_max) if int(self.C_cand_max) > 0 else int(self.sig_query_cand_cap)
        if cand_cap < 1:
            raise ValueError("C_cand_max or sig_query_cand_cap must be >= 1")
        k_cap = int(self.K_max) if int(self.K_max) > 0 else int(self.max_retrieval_candidates)
        if k_cap < 1:
            raise ValueError("K_max or max_retrieval_candidates must be >= 1")
        if self.max_retrieval_candidates <= 0:
            raise ValueError("max_retrieval_candidates must be > 0")

        # Operational bounds
        if self.N_max <= 0:
            raise ValueError("N_max must be > 0")
        if k_cap > int(self.N_max):
            raise ValueError("K_max must be <= N_max")
        if self.max_queue <= 0:
            raise ValueError("max_queue must be > 0")
        if self.max_candidates <= 0:
            raise ValueError("max_candidates must be > 0")
        if self.max_proposals_per_step < 0:
            raise ValueError("max_proposals_per_step must be >= 0")
        if self.value_of_compute_beta < 0.0:
            raise ValueError("value_of_compute_beta must be >= 0")
        if self.value_of_compute_gamma < 0.0:
            raise ValueError("value_of_compute_gamma must be >= 0")
        if not (0.0 <= self.value_of_compute_novelty_decay <= 1.0):
            raise ValueError("value_of_compute_novelty_decay must be in [0,1]")
        if self.value_of_compute_budget_scale < 0.0:
            raise ValueError("value_of_compute_budget_scale must be >= 0")
        if self.value_of_compute_planning_scale < 0.0:
            raise ValueError("value_of_compute_planning_scale must be >= 0")
        if self.value_of_compute_stage2_scale < 0.0:
            raise ValueError("value_of_compute_stage2_scale must be >= 0")
        if self.value_of_compute_candidate_scale < 0.0:
            raise ValueError("value_of_compute_candidate_scale must be >= 0")
        if self.planning_threads_max <= 0:
            raise ValueError("planning_threads_max must be > 0")
        if self.planning_micro_steps_per_tick < 0:
            raise ValueError("planning_micro_steps_per_tick must be >= 0")

    def replace(self, **overrides: Any) -> "AgentConfig":
        """
        Create a modified copy of this config (immutable update).

        Inputs
        ------
        overrides:
            Keyword arguments mapping field names to new values.

        Outputs
        -------
        AgentConfig:
            A new config instance with the specified overrides applied.
        """
        return dc_replace(self, **overrides)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert this configuration to a plain Python dictionary.

        Inputs
        ------
        None.

        Outputs
        -------
        Dict[str, Any]:
            A JSON-serializable (where values are simple types) dict of all fields.
        """
        return asdict(self)


def default_config() -> AgentConfig:
    """
    Return the default v5 AgentConfig.

    Inputs
    ------
    None.

    Outputs
    -------
    AgentConfig:
        A config instance populated with the defaults defined in this module.

    Notes
    -----
    The returned config is valid for most toy/sanity simulations and is intended
    as the baseline against which overrides are applied.
    """
    cfg = AgentConfig()
    cfg.validate()
    return cfg
