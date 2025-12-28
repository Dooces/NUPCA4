NUPCA3 AXIOMS — v1.6.3
(= v1.6.2 + a hard “Axiom (-1)” sovereignty contract + a corresponding clarification of A16 sensory ingress as “full in, selective commit”.)

=====================================================================
A(-1) — AGENT SOVEREIGNTY & ENVIRONMENT NON-DECISIONAL CONTRACT (ABSOLUTE)
=====================================================================

A(-1.1) World does physics only.
- The environment updates by physics: s(t+1) = F(s(t), a(t), noise).
- The environment makes NO decisions about attention, fovea, routing, masking, salience, learning, memory, edits, or “what the agent should see.”
- The environment may expose raw sensory input y_full(t) and domain observables (e.g., danger signals, opportunity signals) that are functions of physics/state only.

A(-1.2) The harness is not a cognitive module.
- The test harness is:
  (i) the environment simulator, and
  (ii) a display/logger for humans.
- The harness must not implement “peripheral vision,” “advisory routing,” “fovea planning,” “mask expansion,” “coverage scoring,” “candidate selection,” or any heuristic that affects what the agent attends to or learns.
- The harness may (and must) pass y_full(t) into the agent; it may apply a(t) to the environment.

A(-1.3) Every decision belongs to the agent.
- The agent alone decides:
  attention/fovea selection, peripheral prepass computation, working-set selection, retrieval, planning, edits, learning updates, action choice, and REST vs OPERATE mode.
- Any mechanism described as “advising where to look” is an INTERNAL agent computation.

A(-1.4) Fixed step tempo; no backpressure; missed input is lost.
- The environment advances at a fixed step interval Δt.
- The agent has a fixed per-step compute budget B_rt in the same time base.
- If the agent cannot process all incoming y_full(t) within the step, the unprocessed portion is gone. There is no environment-side buffering that “waits” for the agent.

A(-1.5) Variable fovea size is allowed; compute is the invariant.
- The fovea/attention footprint may vary per step.
- Total compute per step must still respect B_rt; larger sensory processing must reduce other compute (rollouts, edits, consolidation, etc.) according to the same budget accounting.

A(-1.6) Enforcement principle.
- Any code path that chooses, expands, edits, or biases the agent’s observation/attention on the environment/harness side is an axiomatic violation.

------------------------------------------------------------

=============================
A0 — STATE & TWO-CHANNEL STRESS
=============================

A0.1 Internal physiology state.
- Maintain margins v(t) = (m_E, m_D, m_L, m_C, m_S):
  m_E operational capacity headroom (restored by REST)
  m_D stability/damage headroom (restored by REST)
  m_L learning opportunity signal (domain-defined)
  m_C compute slack (budget headroom)
  m_S semantic integrity headroom (restored by REST)

A0.2 Fast stress: arousal s_ar(t).
- Arousal is a filtered (fast-rise / slow-decay) urgency signal driven by:
  deviations in (L,C,S), their changes, and predicted error terms.
- Arousal is lag-disciplined (used as t-1 modulators downstream).

A0.3 Slow stress split.
- Internal need/deficit s_need_int(t): homeostatic proximity to limits (E, D, S).
- External threat s_th_ext(t): safety/danger signal in [0,1] (domain-defined).
- Need and threat are distinct; they may gate different behaviors.

===========
A1 — VIABILITY
===========

A1.1 Viability set.
- Define a viability region V as linear constraints over v(t): A v(t) ≥ b.
- Leaving V is inadmissible (constitutional), not merely “suboptimal.”

=========================
A2 — MARGINS FROM OBSERVABLES
=========================

A2.1 Energy headroom.
- rawE(t) = E(t) − E_min; m_E(t) = rawE(t) / σ_E.

A2.2 Damage/stability headroom.
- rawD(t) = D_max − D(t); m_D(t) = rawD(t) / σ_D.

A2.3 Semantic integrity headroom.
- rawS(t) = S_max_stab − drift_P(t); m_S(t) derived from rawS(t).

A2.4 Learning opportunity margin.
- m_L derived from opp(t) (domain-defined “opportunity” observable).

A2.5 Compute slack margin (fixed budget doctrine).
- Compute slack x_C(t) = B_rt − [ b_enc(t) + (1−rest(t))*h(t)*b_roll(t) + rest(t)*b_cons(t) ].
- b_enc(t) must include sensory prepass + encoding/selection costs (including variable fovea size).
- Units must be consistent with Δt.

================================
A3 — BASELINES & NORMALIZATION
================================

A3.1 Stability predicate stable(t).
- Define stability using rolling windows and “no recent structural edits” constraints.

A3.2 Baseline updates when stable.
- Update running means/variances for normalized margin computation.

A3.3 Weber/scale normalization.
- Normalize L,C,S channels so arousal/need gates are not arbitrary hidden knobs.

========================
A4 — MEMORY SUBSTRATE (LIBRARY)
========================

A4.1 Node structure (“experts” are NOT MoE).
- Library node j contains: mask m_j, predictor params (W_j,b_j), uncertainty Σ_j, reliability π_j, compute cost L_j, DAG links.
- “Expert” means footprint-local predictive operator; not mixture gating semantics.

A4.2 Bounded working set.
- Active set A_t obeys hard caps: |A_t| ≤ N_max and Σ_{j∈A_t} L_j ≤ L_work_max.
- Anchors are force-included (must fit).

A4.3 Cold storage retrieval is keyed.
- Archived nodes are retrieved via block keys driven by coverage/residual state (see A16), not arbitrary embedding similarity.

A4.4 Anti-aliasing at write time.
- Do not store two incumbents indistinguishable under the footprint/cue basis.
- For same-footprint candidates: REPLACE only if strictly better on the same evidence; else REJECT.

===============================
A5 — SALIENCE & TEMPERATURE (ATTENTION)
===============================

A5.1 Salience score u_j(t).
- Combine reliability, DAG utility (e.g., outgoing degree), and contextual relevance.

A5.2 Temperature τ_eff(t) (need-sharpened; safe-play opened).
- Internal need sharpens selection (more deterministic focus).
- Safe arousal opens selection (more exploratory breadth).
- External threat suppresses “play opening.”

A5.3 Activation propensity a_j(t) from lagged salience and τ_eff(t).

A5.4 Active set selection.
- Choose nodes by (activation × reliability) per unit compute cost under L_work_max.
- Anchors included first; remainder greedily chosen.

A5.5 Effective engaged complexity L_eff(t) tracked for audit.

========================================
A6 — FIXED BUDGET, EMERGENT HORIZON
========================================

A6.1 Per-step compute budget is fixed.
- B_rt is constant per step (same Δt base); the agent allocates compute internally.

A6.2 Emergent horizon h(t).
- h(t) is determined by remaining slack and per-step rollout costs; it is not fixed.
- Increased sensory processing (b_enc) reduces h(t) automatically.

A6.3 No hidden compute.
- Any additional processing (sensory, logging, planning, edits) must be accounted in b_enc, b_roll, b_cons.

==========================
A7 — COMPLETION & FUSION
==========================

A7.1 Unified completion.
- Maintain an internal belief state x(t) (encoded abstraction).
- Combine predictions from active nodes via precision-weighted fusion.
- If coverage is absent for a dimension, uncertainty must reflect that (do not hallucinate certainty).

A7.2 Confidence is explicit.
- Confidence at rollout depth k is derived from predictive uncertainty (entropy/variance) and used downstream (A8/A9).

============================
A8 — LATENCY & COMMITMENT
============================

A8.1 Latency floor.
- Define minimum depth d = ceil(T_proc / Δt) required to justify commitment.

A8.2 Commitment gate.
- Commit only if:
  not in REST,
  horizon h(t) ≥ d,
  confidence at depth d (lagged) exceeds θ_act,
  and any safety conditions required by threat policy are satisfied.

A8.3 Safe fallback policy.
- If commit=0, execute π_safe (must be viability-preserving under A1).

==========================
A9 — PREDICTION OBJECTIVE
==========================

A9.1 Multi-step predictive loss.
- Train predictive accuracy at least at depth d; optionally beyond d up to h(t).

A9.2 Arousal-modulated horizon weighting.
- Lagged arousal modulates the weight placed on deeper horizons vs minimal competence.

A9.3 Threat-gated exploration/curiosity.
- Exploration pressure is suppressed under external threat.

====================
A10 — EDIT CONTROL
====================

A10.1 Freeze under external threat.
- freeze(t) depends only on lagged external threat; it is a hard safety lock.

A10.2 Permit parameter updates only with headroom and slack.
- permit_param(t) requires:
  OPERATING, not frozen,
  compute slack above τ_C_edit,
  arousal below panic cap,
  sufficient rawE/rawD headroom (lagged).

A10.3 Responsibility-gated updates.
- Update an expert only if it overlaps the committed observation set and its local error is below θ_learn.
- If not responsible, penalize reliability; do not corrupt weights off-context.

==========================
A11 — SEMANTIC INTEGRITY
==========================

A11.1 Semantic drift is tracked and bounded.
- drift_P(t) is a first-class observable; margin m_S reflects remaining headroom.

A11.2 Edits must not degrade semantic integrity.
- Structural changes must be evaluated against drift and predictive performance metrics.

=========================
A12 — EDIT ACCEPTANCE
=========================

A12.1 Only a small set of structural edits is allowed (typed edits).
- CREATE/SPAwn (new node), SPLIT, MERGE, REPLACE/PRUNE.

A12.2 Acceptance is evidence-based and audited.
- All edits must be evaluated on the same evidence window, with explicit before/after metrics.

A12.3 MERGE fix: per-domain non-worsening (replacement-consistent).
- Merge/replacement must not worsen error on any relevant domain slice beyond tolerance.

A12.4 Residual-driven invention.
- Persistent residual/coverage debt is the trigger for proposing new structure; not arbitrary growth.

==========================
A13 — UNIFIED STEP ORDERING
==========================

A13.1 Canonical per-step invariant (ordering is constitutional).
At step t:

1) Ingress: environment provides y_full(t) (ephemeral) and domain observables (danger, opp, etc.).
2) Mode: rest(t) is computed from lagged predicates (see A14).
3) Sensory prepass (agent-internal): compute peripheral gist p(t) from y_full(t) under cheap compute.
4) Attention commit (agent-internal): select observation/attention set O_t (blocks/dims) using A16.
5) Encode/commit: form the committed cue y(t)|_{O_t} and discard the rest of y_full(t).
6) Prior: compute predictive prior x_hat(t|t−1) from previous completed state and active library.
7) Complete: overwrite committed dims to produce completed x(t); fuse via A7.
8) Residual: define prediction error e_obs(t) only on committed dims O_t: e_obs = x(t)−x_hat on O_t.
9) OPERATING updates: apply A10/A9 learning rules if permitted.
10) REST updates: if in REST, run consolidation/edit evaluation only (no action execution except safe invariants).
11) Action: choose a(t) from plan if committed (A8), else π_safe.
12) Purge: y_full(t) must not persist beyond the step (see A16.5).

A13.2 Residual definition is prediction error on committed dims.
- Any “residual = Δx” shortcut is non-compliant.

=============================
A14 — MACROSTATES (REST vs OPERATING)
=============================

A14.1 REST is lag-disciplined.
- rest(t) = rest_permitted(t−1) * demand(t−1) * (1 − interrupt(t−1)).

A14.2 OPERATING vs REST roles.
- OPERATING: act + learn (if permitted).
- REST: consolidate/evaluate edits/restructure (no same-step action planning dependence on new updates).

A14.3 External threat can interrupt REST and freeze learning (A10.1).

=========================
A15 — MARGIN DYNAMICS
=========================

A15.1 Margins evolve with work and repair.
- OPERATING consumes E/C and may accumulate D/drift.
- REST restores E/D/S according to domain-defined recovery dynamics and consolidation cost.

A15.2 Dynamics must be explicit and auditable.
- No hidden “magic recovery”; all restoration/consumption terms are logged.

=========================================================
A16 — ATTENTION GEOMETRY & COVERAGE-DISCIPLINED FOVEATION (greedy_cov)
=========================================================

A16.0 Full ingress, selective commit (the A-1/A13 contract).
- The agent receives y_full(t) but may only COMMIT a subset O_t downstream.
- “Fovea” refers to the committed subset O_t, not an environment-provided mask.

A16.1 Block partition.
- Partition D into B disjoint blocks B_b.
- Footprints for non-anchor nodes lie within exactly one block (block-keyed memory).

A16.2 Per-block age and residual (on committed dims).
- age(b,t): steps since block b last belonged to O_t.
- residual(b,t): prediction error statistics computed when b is committed (using A13 residual definition).

A16.3 Coverage debt.
- Maintain a per-block debt that increases with age and/or residual and decreases when the block is committed and well-explained.

A16.4 greedy_cov selection.
- Select blocks for O_t to reduce total expected error and debt under the per-step compute budget.
- Variable |O_t| is allowed; its encoding cost must be charged to b_enc(t) (A2.5), reducing other compute.

A16.5 Peripheral adviser is internal (first-pass over incoming information).
- Compute a bounded peripheral gist p(t) from y_full(t) under cheap compute.
- Use p(t) only to bias greedy_cov (routing prior); it does not directly update x(t) outside O_t.

A16.6 Cold storage retrieval coupling.
- The same block scores used for O_t also drive retrieval of archived nodes keyed to those blocks (A4.3).

=================================================
A16.5 (REST RAW BUFFER SEMANTICS; NO POST-REST PIXELS)
=================================================

A16.5.1 Bounded rolling raw buffer is permitted only for short-horizon audit during REST.
- Buffer content must be foveated (committed dims + geometry), not full-field pixels.
- It exists to support reconstruction/audit and edit evaluation during REST.

A16.5.2 Purge requirement.
- Raw buffers must be purged before leaving REST.
- No pixels (raw or reconstructible full-field sensory) may appear in post-REST durable state or serialization.

A16.5.3 Explicit encoder boundary.
- x(t) is an encoded abstraction vector; completion/fusion operates only on x(t).
- Feeding raw pixels directly as x(t) is non-compliant unless explicitly declared as a temporary prototype mode.

=========================================
A17 — DERIVED DIAGNOSTICS (NO CONTROL AUTHORITY)
=========================================

A17.1 Diagnostics are logged observables only.
- Derived quantities (e.g., “feel proxy” q(t), coverage debt summaries, confidence traces) have zero direct control authority unless explicitly referenced by another axiom.

A17.2 Audit requirements.
- Log enough to verify:
  lag discipline,
  budget accounting,
  purge compliance,
  selection decisions (O_t),
  edit acceptance tests,
  and non-decisionality of the environment/harness.

END.
