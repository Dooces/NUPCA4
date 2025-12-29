"""
nupca3/config.py

NUPCA3 Agent configuration (v1.5b).

What this file does
-------------------
This module defines :class:`AgentConfig`, an immutable (frozen) dataclass that
collects *all* numeric hyperparameters referenced by the NUPCA3 implementation.

The NUPCA3 codebase is axiom-driven (A0–A16 in the v1.5b spec). In that style,
implementation modules read constants from config rather than hard-coding them.
This file exists to:
  - centralize those constants,
  - provide axiom-aligned defaults (matching the “reference configuration”
    table in the axiom document where available),
  - make missing/invalid parameters fail fast via validate().

Axioms this file supports (by providing parameters)
---------------------------------------------------
- A6  (Compute budget + horizon scalars): B_rt, b_enc_base, b_roll_base, b_cons_base
- A7  (Precision/uncertainty floors + EMA): sigma_floor, sigma_ema
- A9/A11 (Spawn pressure + responsibility selection): beta_R, K, theta_spawn, theta_learn
- A10 (Learning gating + panic/arousal gating): theta_ar_rest, theta_ar_panic
- A12 (Acceptance + MDL tradeoff): mdl_beta, epsilon_merge, delta_L_MDL_*, theta_alias
- A14 (Macrostate dynamics / rest pressure): P_rest_* parameters
- A15 (Margins + arousal dynamics): tau_*, kappa_*, k_rest_*, w_*_ar, theta_ar, kappa_ar,
      tau_rise, tau_decay, beta_open
- A16 (Observation geometry / fovea): D, B, fovea_blocks_per_step, sticky_k, alpha_cov
- Implementation bookkeeping constraints (capacity / queue / candidates):
  N_max, max_candidates, max_experts_per_k, max_queue, max_proposals_per_step, etc.

Notes on defaults
-----------------
Where v1.5b provides explicit “reference configuration” numbers, those are used.
Where v1.5b defines a symbol but does not specify a numeric reference value,
a reasonable default is chosen so the code can run. Those defaults are explicitly
marked with ``#ITOOKASHORTCUT`` and include a brief justification.

This file is deliberately *not* a “framework” configuration system; it’s a plain
dataclass with clear, auditable defaults aligned to the axioms.


[AXIOM_CLARIFICATION_ADDENDUM — Representation & Naming]

- Terminology: identifiers like "Expert" in this codebase refer to NUPCA3 **abstraction/resonance nodes** (a "constellation"), not conventional Mixture-of-Experts "experts" or router-based MoE.

- Representation boundary (clarified intent of v1.5b): the completion/fusion operator (A7) is defined over an **encoded, multi-resolution abstraction vector** \(x(t)\). Raw pixels may exist only in a transient observation buffer for the current step; **raw pixel values must never be inserted into long-term storage** (library/cold storage) and must not persist across REST boundaries.

- Decomposition intuition: each node is an operator that *factors out* a predictable/resonant component on its footprint, leaving residual structure for other nodes (or for REST-time proposal) to capture. This is the intended "FFT-like" interpretation of masks/constellations.
- Naming collision warning (v1.5b): symbols like \(\tau_E,\tau_D,\tau_S\) in A0.3 are **need thresholds** (sigmoid offsets), not time constants. Time constants use explicit names (e.g., `tau_rise`, `tau_decay`, `tau_C_edit`).
- Compatibility aliases: some modules historically referenced `use_current_gates`; the canonical field is `gates_use_current` (A10 lag discipline debugging knob). `AgentConfig.use_current_gates` is provided as a read-only alias.

"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace as dc_replace
from typing import Any, Dict, Tuple


@dataclass(frozen=True)
class AgentConfig:
    """
    Immutable configuration for a NUPCA3 agent.

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

    # Fovea budget per step (A16.3): number of blocks to observe.
    fovea_blocks_per_step: int = 64

    # Hard coverage cap G (A16.4): if age(b,t-1) >= G, block b must be observed.
    #ITOOKASHORTCUT: v1.5b defines G but does not pin a numeric reference.
    coverage_cap_G: int = 50

    # Starting age for each block; use < G so the coverage emergency path waits for real data.
    initial_block_age: int = 0

    # Implementation-only: sticky retention (NOT part of v1.5b A16).
    # Kept for backwards compatibility with earlier experiments, but unused by
    # the axiom-faithful selector in geometry.fovea.
    sticky_k: int = 0

    # Coverage regularizer for greedy_cov fovea selection (A16).
    alpha_cov: float = 0.10

    # Coverage debt boosts for expert and abstraction strata (Fix 1).
    alpha_cov_exp: float = 0.05
    alpha_cov_band: float = 0.05

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
    grid_side: int = 0
    # Optional explicit grid dimensions for non-square harnesses (metadata only).
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
    # A4 — Library initialization (implementation detail)
    # =========================================================================
    # If True, seed one per-block expert for each footprint block at init.
    # If False, only the lightweight global anchor is created and the library
    # grows solely via REST-time structural edits (A14/A12).
    library_seed_block_experts: bool = True

    # Anchor input width (kept tiny to avoid allocating a dense D×D anchor W).
    # The anchor uses a compact W with shape (D, library_anchor_inputs).
    library_anchor_inputs: int = 1

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

    # Encoding base cost b_enc,0 (A6.2). Reference value from v1.5b.
    b_enc_base: float = 3.2

    # Rollout per-step cost b_roll,0 (A6.2). Reference value from v1.5b.
    b_roll_base: float = 0.85

    # Anchor overhead coefficient b_anc,0 (A6.2).
    #ITOOKASHORTCUT: the v1.5b document defines b_anc,0 but does not provide a
    # reference numeric in the table. Default 0.0 keeps anchor overhead from
    # dominating until you calibrate it in simulation.
    b_anc_base: float = 0.0

    # Consolidation bookkeeping base cost.
    #ITOOKASHORTCUT: v1.5b discusses b_cons tracking, but the reference table
    # does not pin a numeric. Defaulting to 0.0 preserves behavior when
    # consolidation cost is tracked dynamically by REST processors.
    b_cons_base: float = 0.0

    # Small epsilon to avoid divide-by-zero in budget terms (A6.2).
    eps_budget: float = 1e-6

    # Hard cap on horizon to keep rollout computation bounded.
    #ITOOKASHORTCUT: v1.5b defines emergent h(t) but does not prescribe a strict cap.
    # A moderate default prevents pathological slowdowns in toy environments.
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
    alpha_ctx: float = 0.6  # legacy hook used by the simplified score interface.
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

    # A12 MDL deltas used as rough complexity penalties for different edits.
    delta_L_MDL_merge: float = 0.50
    delta_L_MDL_spawn: float = 0.80

    # A4.4 / A12.3 anti-aliasing threshold (structural edits).
    # In the axiom table this appears as θ_alias; some code also uses anti_alias_cos.
    theta_alias: float = 0.95
    anti_alias_cos: float = 0.95  # kept for backward compatibility with earlier modules

    # =========================================================================
    # A14 — Macrostate (REST / WAKE) dynamics: rest pressure parameters
    # Naming: the implementation uses a P_rest_* prefix (see macrostate.py).
    # =========================================================================
    # Exponential decay factor (γ_rest).
    P_rest_gamma: float = 0.10

    # Baseline drift (δ_base).
    #ITOOKASHORTCUT: A14 defines δ_base but v1.5b does not include a reference
    # numeric in the table. Small positive default yields slow rest-pressure creep.
    P_rest_delta_base: float = 0.01

    # Need-driven drift gain (δ_need).
    #ITOOKASHORTCUT: A14 defines δ_need but v1.5b does not include a reference
    # numeric in the table. Default chosen to make needs matter vs baseline drift.
    P_rest_delta_need: float = 0.03

    # Need weights α_E, α_D, α_S.
    P_rest_alpha_E: float = 1.0
    P_rest_alpha_D: float = 1.0
    P_rest_alpha_S: float = 1.0

    # Demand hysteresis thresholds θ_enter, θ_exit.
    #ITOOKASHORTCUT: Not specified numerically in v1.5b reference table.
    P_rest_theta_demand_enter: float = 0.60
    #ITOOKASHORTCUT: Not specified numerically in v1.5b reference table.
    P_rest_theta_demand_exit: float = 0.40

    # Structural queue gates Θ_Q^{on,off}.
    #ITOOKASHORTCUT: Not specified numerically in v1.5b reference table.
    P_rest_Theta_Q_on: float = 0.50
    #ITOOKASHORTCUT: Not specified numerically in v1.5b reference table.
    P_rest_Theta_Q_off: float = 0.30

    # Maximum contiguous durations T_max^{wake}, T_max^{rest}.
    #ITOOKASHORTCUT: Not specified numerically in v1.5b reference table.
    P_rest_Tmax_wake: int = 4000
    #ITOOKASHORTCUT: Not specified numerically in v1.5b reference table.
    P_rest_Tmax_rest: int = 800

    # Margin-based hard triggers θ_E^rest, θ_D^rest, θ_S^rest.
    #ITOOKASHORTCUT: Spec does not give a reference numeric; 0.0 means
    # “rest if margin goes negative” (a natural minimal choice).
    P_rest_theta_E_rest: float = 0.0
    #ITOOKASHORTCUT: See θ_E^rest note.
    P_rest_theta_D_rest: float = 0.0
    #ITOOKASHORTCUT: See θ_E^rest note.
    P_rest_theta_S_rest: float = 0.0

    # Safety/interrupt thresholds θ_safe^{th}, θ_interrupt^{th}.
    #ITOOKASHORTCUT: Not specified numerically in v1.5b reference table.
    P_rest_theta_safe_th: float = 0.20
    #ITOOKASHORTCUT: Not specified numerically in v1.5b reference table.
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
    tau_E: float = 2000.0
    tau_D: float = 2000.0
    tau_S: float = 2000.0

    kappa_E: float = 1.0
    kappa_D: float = 1.0
    kappa_S: float = 1.0

    k_rest_E: float = 0.05
    k_rest_D: float = 0.05
    k_rest_S: float = 0.05

    # Arousal scoring weights (w_*^{ar}).
    w_E_ar: float = 1.0
    w_D_ar: float = 1.0
    w_S_ar: float = 1.0
    w_deltam_ar: float = 1.0
    w_Epred_ar: float = 1.0

    # Arousal logistic parameters (θ_ar, κ_ar).
    theta_ar: float = 0.50
    kappa_ar: float = 0.20

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
    max_candidates: int = 32
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

    # =========================================================================
    # Alias properties (to support both “symbolic” and “prefixed” naming styles)
    # =========================================================================
    @property
    def gamma_rest(self) -> float:
        """
        Alias for A14 γ_rest.

        Inputs: none.
        Outputs: float (same value as P_rest_gamma).
        """
        return self.P_rest_gamma

    @property
    def delta_base(self) -> float:
        """
        Alias for A14 δ_base.

        Inputs: none.
        Outputs: float (same value as P_rest_delta_base).
        """
        return self.P_rest_delta_base

    @property
    def delta_need(self) -> float:
        """
        Alias for A14 δ_need.

        Inputs: none.
        Outputs: float (same value as P_rest_delta_need).
        """
        return self.P_rest_delta_need

    @property
    def theta_safe_th(self) -> float:
        """Alias for θ_safe^{th} used by macrostate (A14)."""
        return self.P_rest_theta_safe_th

    @property
    def theta_th(self) -> float:
        """Alias for θ_interrupt^{th} (external threat) used by macrostate (A14)."""
        return self.P_rest_theta_interrupt_th

    @property
    def theta_int(self) -> float:
        """Alias for θ_interrupt^{int} (arousal) used by macrostate (A14).

        #ITOOKASHORTCUT: The repo snapshot does not expose a separate config
        parameter for θ_interrupt^{int}. We map it to theta_ar_panic, which
        already represents an extreme-arousal safety limit.
        """
        return self.theta_ar_panic

    @property
    def theta_E_rest(self) -> float:
        """Alias for θ_E^{rest} used by macrostate (A14)."""
        return self.P_rest_theta_E_rest

    @property
    def theta_D_rest(self) -> float:
        """Alias for θ_D^{rest} used by macrostate (A14)."""
        return self.P_rest_theta_D_rest

    @property
    def theta_S_rest(self) -> float:
        """Alias for θ_S^{rest} used by macrostate (A14)."""
        return self.P_rest_theta_S_rest

    @property
    def theta_Trest_min(self) -> int:
        """Minimum REST duration (T_rest^{min}) used by macrostate (A14).

        #ITOOKASHORTCUT: v1.5b config in this repo contains only T_rest^{max}.
        We set a small default minimum to avoid immediate thrashing.
        """
        return int(getattr(self, 'P_rest_Tmin_rest', 10))

    @property
    def theta_rest(self) -> float:
        """Alias for θ_demand^{enter} used by macrostate demand hysteresis (A14.4)."""
        return self.P_rest_theta_demand_enter

    @property
    def theta_rest_low(self) -> float:
        """Alias for θ_demand^{exit} used by macrostate demand hysteresis (A14.4)."""
        return self.P_rest_theta_demand_exit

    @property
    def lambda_hyst(self) -> float:
        """Hysteresis leak λ_hyst used in rest pressure update (A14.4).

        #ITOOKASHORTCUT: config name not present in this repo snapshot.
        """
        return float(getattr(self, 'P_rest_lambda_hyst', 0.5))

    @property
    def theta_freeze(self) -> float:
        """Freeze threshold on rest pressure used by macrostate (A14.5).

        #ITOOKASHORTCUT: config name not present in this repo snapshot.
        """
        return float(getattr(self, 'P_rest_theta_freeze', 0.95))

    @property
    def use_current_gates(self) -> bool:
        """Compatibility alias for `gates_use_current`.

        v1.5b is defined with lagged gates; this flag exists only to allow
        debugging against same-step values in toy harnesses. The canonical
        field name is `gates_use_current`.
        """
        return bool(self.gates_use_current)

    # =========================================================================
    # Methods
    # =========================================================================
    def validate(self) -> None:
        """
        Validate basic invariants implied by the axioms and implementation.

        Inputs
        ------
        None (reads self).

        Outputs
        -------
        None. Raises ValueError/TypeError if an invariant is violated.

        What it checks (non-exhaustive)
        -------------------------------
        - Geometry: D and B must be positive and compatible.
        - EMA / floor: sigma_floor > 0 and sigma_ema in (0, 1].
        - Budgets: base costs and B_rt must be non-negative.
        - Time constants: taus must be positive.
        """
        if not isinstance(self.D, int) or self.D <= 0:
            raise ValueError("D must be a positive int.")
        if not isinstance(self.B, int) or self.B <= 0:
            raise ValueError("B must be a positive int.")
        # D need not be divisible by B; remainder is distributed across blocks.
        if self.B > self.D:
            raise ValueError("B must be <= D.")

        if not (self.sigma_floor > 0.0):
            raise ValueError("sigma_floor must be > 0.")
        if not (0.0 < self.sigma_ema <= 1.0):
            raise ValueError("sigma_ema must be in (0, 1].")

        for name in ("B_rt", "b_enc_base", "b_roll_base", "b_cons_base"):
            v = getattr(self, name)
            if v < 0.0:
                raise ValueError(f"{name} must be >= 0.")

        for name in ("tau_E", "tau_D", "tau_S", "tau_rise", "tau_decay"):
            v = getattr(self, name)
            if v <= 0.0:
                raise ValueError(f"{name} must be > 0.")

        if self.fovea_blocks_per_step <= 0:
            raise ValueError("fovea_blocks_per_step must be > 0.")
        if self.sticky_k < 0:
            raise ValueError("sticky_k must be >= 0.")
        if self.fovea_uncertainty_weight < 0.0:
            raise ValueError("fovea_uncertainty_weight must be >= 0.")
        if self.fovea_uncertainty_default < 0.0:
            raise ValueError("fovea_uncertainty_default must be >= 0.")
        if self.initial_block_age < 0:
            raise ValueError("initial_block_age must be >= 0.")
        for name in ("alpha_pi", "alpha_deg", "alpha_ctx", "alpha_ctx_relevance", "alpha_ctx_gist"):
            v = getattr(self, name)
            if v < 0.0:
                raise ValueError(f"{name} must be >= 0.")
        for name in ("alpha_cov", "alpha_cov_exp", "alpha_cov_band"):
            v = getattr(self, name)
            if v < 0.0:
                raise ValueError(f"{name} must be >= 0.")
        if self.grid_color_channels < 0 or self.grid_shape_channels < 0:
            raise ValueError("grid_color_channels and grid_shape_channels must be >= 0.")
        if (
            self.grid_color_channels + self.grid_shape_channels > 0
            and self.grid_color_channels + self.grid_shape_channels != self.grid_channels
        ):
            raise ValueError(
                "grid_color_channels + grid_shape_channels must equal grid_channels when specified."
            )
        if not (0.0 <= self.beta_context <= 1.0):
            raise ValueError("beta_context must be in [0, 1].")
        if not (0.0 <= self.beta_context_node <= 1.0):
            raise ValueError("beta_context_node must be in [0, 1].")
        if self.beta_sharp < 0.0:
            raise ValueError("beta_sharp must be >= 0.")
        if self.transport_search_radius < 0:
            raise ValueError("transport_search_radius must be >= 0.")
        if not (0.0 < self.transport_belief_decay < 1.0):
            raise ValueError("transport_belief_decay must be in (0, 1).")
        if self.transport_inertia_weight < 0.0:
            raise ValueError("transport_inertia_weight must be >= 0.")
        if not (0.0 <= self.transport_confidence_margin <= 1.0):
            raise ValueError("transport_confidence_margin must be in [0, 1].")
        if self.transport_disambiguation_weight < 0.0:
            raise ValueError("transport_disambiguation_weight must be >= 0.")

        if not (0.0 <= self.theta_alias <= 1.0):
            raise ValueError("theta_alias must be in [0, 1].")
        if not (0.0 <= self.anti_alias_cos <= 1.0):
            raise ValueError("anti_alias_cos must be in [0, 1].")

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
    Return the default v1.5b AgentConfig.

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