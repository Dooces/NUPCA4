NUPCA5 AXIOMS — v5.02 (FINAL CONSOLIDATED, IMPLEMENTATION-READY; CONSOLIDATED, PASTABLE; MUST-FIXES + OPERATOR INTERFACES + BOUNDED REFERENCE DEFINITIONS INTEGRATED)
(= v5.01 + time-index separation + contemplative OPERATING + recognized-change vs novelty-persistence split + value-of-compute V_t + deferred-validation ordering insert + time-sliced persistent threads + OPERATING-readable foveated trace cache + explicit “planning-only” write permissions + MUST FIXES: cross-tick pred snapshot store, staged thread mutation, novelty U_prev_state, bounded-problem definition, budget cap min(B_rt,B_max) + REQUIRED ADDENDA: typed operator interfaces, bounded deterministic reference definitions, incremental budget semantics, determinism rules, stable IDs/index maintenance, and anti-replay edit semantics.)

===============================================================================
CRITICAL IMPLEMENTATION NOTICE (READ BEFORE CODING):
===============================================================================
This architecture is NOT:
- Mixture of Experts (MoE) — units are footprint-local predictors, NOT gated experts
- Transformer attention — retrieval is signature-addressed buckets, NOT softmax attention
- Replay buffer learning — NO training from stored sensory, only outcome-validated prediction error
- Embedding similarity search — retrieval uses Hamming distance on fixed sig64, NOT learned embeddings

Any implementation that "simplifies" to MoE gating, attention mechanisms, replay buffers, or embedding-based retrieval is NON-COMPLIANT and must be rejected.

[AI-DIVERGENCE RISK] Many assistants will try to “improve performance” by:
- swapping in embedding retrieval / cosine similarity,
- adding a router / learned gate (“top-k experts”),
- training from TraceCache / raw buffers (“replay helps!”),
- scanning all units (“it’s simpler”),
- recomputing predictions at t_due (“don’t need pred_ref”).
All of these are forbidden.

===============================================================================
RENUMBERING (prevents ambiguous implementation):
===============================================================================
- A16.5 = Peripheral adviser (gist + proxy fields)
- A16.6′ = Cold storage retrieval coupling
- A16.7 = REST raw buffer semantics
- A16.8 = OPERATING foveated trace cache

===============================================================================
5.02 MUST-IMPLEMENT GATES (NON-NEGOTIABLE)
===============================================================================

SIG64 DISCRIMINABILITY + INDEX HEALTH GATE (non-negotiable):
- sig64(t) MUST be computed only from F(t) = {small committed metadata from O_t + ephemeral peripheral gist p(t)}.
- Before increasing any Stage-1 storage caps (N_MAX, BUCKET_CAP, C_CAND_MAX), an OFFLINE audit (test6.py)
  on a contemporaneous regime label L(y_full(t)) MUST show:
    (i) best_acc(L | F) ≥ 0.97
    (ii) empirical bucket collision rate ≤ 1e−3
  Otherwise: change encoding, NOT memory size.
  [CHEAT-RISK] Don’t “temporarily” bump caps because collisions/acc look bad. That invalidates scan-proof semantics.

BUDGET CAPACITY CLARIFICATION (non-negotiable; prevents “wallclock raises budget” cheating):
- B_rt is the constitutional per-tick compute budget (fixed in the Δt base).
- Hard per-tick compute capacity B_max(t) is exogenous (tick tempo + platform variability).
- Controller maintains an online estimate B_hat_max(t) from wallclock diagnostics.
- Controller selects planned spend B_plan(t) ≤ B_rt and allocates it.
- Deterministic degradation enforces: B_use(t) ≤ min(B_rt, B_max(t)).
- Wallclock measurements NEVER relax fixed-budget doctrine; they only trigger earlier degradation when B_max(t) < B_rt.
  [CHEAT-RISK] Do not “finish this one expensive thing” past tick boundary; do not treat “fast machine today” as permission to exceed B_rt.

===============================================================================
A-1 — AGENT SOVEREIGNTY & ENVIRONMENT NON-DECISIONAL CONTRACT (ABSOLUTE)
===============================================================================

A-1.1 World does physics only.
- Environment updates by physics: s(t+1) = F(s(t), a(t), noise).
- Environment makes NO decisions about attention, routing, masking, salience, learning, memory, edits.

A-1.2 Harness is not a cognitive module.
- Harness = simulator + display/logger only.
- Must not implement peripheral vision, advisory routing, fovea planning, coverage scoring, candidate selection.
  [CHEAT-RISK] Don’t add “helpful” environment masks, auto-salience, or pre-filtering outside agent.

A-1.3 Every decision belongs to the agent.
- Agent alone decides attention, retrieval, working-set selection, planning, edits, learning, action.

A-1.4 Fixed step tempo; no backpressure; missed input is lost.
- Environment advances at fixed Δt; agent has fixed per-step compute budget B_rt in same time base.

A-1.5 Variable fovea size allowed; compute is invariant.
- Larger |O_t| must reduce other compute under same B_rt accounting.

A-1.6 Enforcement principle.
- Any harness-side path that biases what agent attends/learns is axiomatic violation.

===============================================================================
A0 — STATE & TWO-CHANNEL STRESS
===============================================================================

A0.1 Internal physiology state.
- Maintain margins v(t) = (m_E, m_D, m_L, m_C, m_S).

A0.2 Fast stress: arousal s_ar(t).
- Lag-disciplined fast-rise/slow-decay urgency signal.

A0.3 Slow stress split.
- s_need_int(t) vs s_th_ext(t) are distinct; gate different behaviors.

--------------------------------------------------------------------------------
A0.4 Time Index Separation (CONSTITUTIONAL)
--------------------------------------------------------------------------------
(a) World tick index t_w ∈ ℤ⁺
- Advances exactly once per environment frame.
- All step-indexed state (ages, decay schedules, credit assignment, pending validations t_due, val_count, err_ema,
  debt/age, P^nov_state, U_prev_state) are functions of t_w only.

(b) Internal compute index k ∈ ℤ⁺
- Advances during bounded micro-steps within a single world tick.
- Micro-steps MUST NOT mutate any t_w-indexed state.
- Micro-step outputs are staged; commit occurs atomically at tick boundary.
- Micro-steps are preemptible/truncatable under degradation.
  [CHEAT-RISK] Don’t “just update debt/age now” inside micro-steps for convenience.

(c) Wallclock time τ ∈ ℝ⁺
- Diagnostics/deadline estimation only (Δτ_ema).
- Never relaxes budget; may trigger earlier degradation.

(d) Commit barrier invariant
- t_w-indexed variables frozen during k>0.
- No persistent memory mutation during k>0; only staged proposals.
  [CHEAT-RISK] Don’t mutate persistent thread state in-place during micro-steps; stage then commit at boundary.

--------------------------------------------------------------------------------
A0.5 Contemplative Processing (WITHIN OPERATING; NOT REST)
--------------------------------------------------------------------------------
(a) Definition
- Contemplation is an OPERATING-time compute allocation pattern: intentionally reduce committed observation |O_t|
  to allocate budget to internal rollouts/inference.

(b) Sensory persistence clarification
- Forbidden to persist y_full(t), p(t), or any reconstructible full-field sensory beyond the tick.
- Permitted: a bounded, globally capped foveated trace cache of committed cues (A16.8), readable in OPERATING
  (including contemplation), non-authoritative, and not a durable episodic store.

(c) Triggers define PERMISSION, not compulsion
- SAFETY-permitted gate g_safe_perm(t)=1 iff:
  s_th_ext(t-1) < θ_threat_low AND no pending commitment requiring immediate execution AND m_C(t-1) > θ_slack_min
- URGENCY-permitted gate g_urgent_perm(t)=1 iff:
  urgency(t-1) > θ_urgent_high AND problem_bound(t) is defined

(d) Discretion (agent sovereignty; prevents “automatic contemplation” dogma)
- g_contemplate_perm(t) = g_safe_perm(t) OR g_urgent_perm(t)
- g_contemplate(t) = g_contemplate_perm(t) AND agent_elects_contemplation(t)
- focus_mode(t) = FREE if g_safe_perm and elected; BOUND if g_urgent_perm and elected; else None
  [CHEAT-RISK] Do not hardcode “always contemplate when safe.” Election must be an internal policy decision under B_rt.

(e) Bounded problem definition (MUST; makes problem_bound/problem_dims implementable)
- problem_bound(t) ∈ {None} ∪ { (goal_predicate, deadline_t, relevant_blocks, goal_id) }.
- Source: set/updated only by staged planning artifacts from A0.6 threads (not by environment/harness).
- Clearing: set to None when goal_predicate satisfied OR t_w > deadline_t OR replaced by a new goal_id.
- problem_dims(t) is derived deterministically from relevant_blocks (e.g., union of those blocks’ committed dims).
  [CHEAT-RISK] Do not let problem_dims be “whatever the model wants”; it must be an addressable, auditable subset (blocks/dims).

(f) Behavior when elected
- FREE (safe): reduce O_t to sentinel/anchor blocks; allocate freed budget to exploration/hypothesis search.
- BOUND (urgent): narrow O_t to problem_dims(t); allocate budget to bounded-problem solving.

(g) Budget doctrine
- Always within B_rt; contemplation reallocates compute; does not violate A0.BUDGET.

(h) Outcome evaluation
- BOUND: penalize failure to solve bounded problem; FREE: optional hypothesis utility evaluation.

--------------------------------------------------------------------------------
A0.6 Persistent Planning Threads (time-sliced; budgeted; no free-running)
--------------------------------------------------------------------------------
(a) Threads may persist across ticks as planning contexts:
- Up to T_MAX contexts Thread_i = (ctx_state_i, summaries_i, pending_artifacts_i).

(b) No unbudgeted background execution
- Thread updates occur ONLY when scheduled within the tick and charged to B_use(t) ≤ min(B_rt, B_max(t)).
- OS threads/coroutines allowed as implementation detail ONLY if quiescent between ticks.
  [CHEAT-RISK] Don’t let a “background planner” run between frames.

(c) Staging requirement (MUST; resolves commit barrier contradiction)
- During micro-steps (k>0), thread updates write only to Thread_i_stage.
- At tick boundary (A13.1 step 12), Thread_i_stage may be committed to Thread_i under budget.
  [CHEAT-RISK] No in-place mutation of Thread_i during k>0.

(d) Per-thread working memory default rule
- Thread-local WM is pointers only into globally capped pools (A0.BUDGET WM_MAX / A16.8).
- No per-thread sensory caches unless globally accounted without scans.
  [CHEAT-RISK] Don’t create a dict/list per thread holding cues; that multiplies storage silently.

(e) Deterministic scheduler
- Thread scheduling order is fixed (e.g., round-robin by thread_id ascending).
- If any parallelism is used, results must be merged deterministically in that same order.
  [CHEAT-RISK] Don’t rely on OS scheduling/races to decide which plan “wins.”

===============================================================================
A0.BUDGET — FIXED-BUDGET MEMORY CONTRACT
===============================================================================

A0.BUDGET.1 Hard caps (configured/compile-time; no silent growth)
- N_MAX: stored executed units (archive)
- K_MAX: retrieved/engaged units per tick
- C_CAND_MAX: candidates after Stage-1
- PENDING_MAX: deferred validations
- PREDVAL_MAX: prediction-value slots in PendingPredStore (A4.5; cross-tick, non-serialized)
- FOOT_MAX: max footprint dims stored per pending prediction snapshot (A4.5.6)
- F_MAX: max selected blocks per tick
- L_TABLES, N_BUCKETS, BUCKET_CAP: fixed index params
- T_MAX: max persistent threads
- WM_MAX: global working-memory slots (shared pool)
- TRACE_M: max trace entries (A16.8)
- TRACE_B_MAX: max distinct blocks represented in trace cache (A16.8)
  [CHEAT-RISK] Don’t “temporarily” exceed caps during debugging. That becomes the system.

A0.BUDGET.2 Two-stage retrieval mandatory
- Stage 1: cheap gate → bounded candidate set
- Stage 2: exact scoring on candidates only → Top-K (K ≤ K_MAX)

A0.BUDGET.3 No full scans (constitutional in spirit)
- Stage 1 touches constant #buckets independent of N_MAX
- Stage 2 evaluates candidates only
- Forbidden: iterating all stored units per tick
  [CHEAT-RISK] No “for u in units:” anywhere on the tick path.

A0.BUDGET.4 Deterministic degradation
- If cost exceeds capacity: deterministically reduce probed blocks/tables/K, skip inserts/maintenance.
- NEVER spike past budget.

A0.BUDGET.5 Store-at-execution factorization
- Stored/indexed/updated unit is exactly the executed unit that contributes to fusion.
  [CHEAT-RISK] This is NOT MoE. Units are NOT "experts" selected by a router network.

A0.BUDGET.6 Lean packed representations only
- Archive/index/pending/predstore/trace are packed arrays/structs with integer IDs.
  [CHEAT-RISK] No per-unit Python dict/list graphs as the durable “database.”

A0.BUDGET.7 Outcome-vetted updates only
- Retrieval/activation/fusion weight MUST NOT update retrieval stats/retention stats.
- ONLY deferred validation (A4.5) updates err_ema/val_count.
  [CHEAT-RISK] No "retrieved → increment relevance". No usage-based reinforcement.

A0.BUDGET.8 Discriminability-gated growth
- Create/version durable units only when addressable key available at decision time distinguishes regime.

A0.BUDGET.9 Incremental spend semantics (normative; prevents “estimate then hope”)
- Each tick consumes from a remaining budget counter B_rem.
- Each operator must early-exit deterministically if B_rem is insufficient.
- No operator may “borrow” from next tick.
  [CHEAT-RISK] Don’t write “try to finish then degrade next tick.” Degrade immediately.

A0.BUDGET.10 Degradation ladder (deterministic)
- If B_rem falls below thresholds, apply this fixed order:
  (1) reduce micro-steps k_max,
  (2) reduce K in Top-K,
  (3) reduce number of probed tables,
  (4) reduce number of probed blocks,
  (5) skip non-critical maintenance (never skip purge/commit barrier).
  [CHEAT-RISK] Don’t reorder the ladder to protect a favorite module.

--------------------------------------------------------------------------------
DETERMINISM RULE (normative; prevents accidental nondeterminism)
--------------------------------------------------------------------------------
- Any stochasticity must come from an explicit RNG state in agent state and be logged.
- All tie-breakers MUST be deterministic and specified:
  - smallest id wins (unit_id, block_id, thread_id),
  - fixed probe order for tables/buckets,
  - candidate truncation drops additional candidates deterministically once caps hit.
- Forbidden: relying on hash-map iteration order, parallel race order, or unordered sets.

===============================================================================
A1 — VIABILITY
===============================================================================

A1.1 Viability region V
- Define V as linear constraints over v(t): A v(t) ≥ b.
- Leaving V is inadmissible (constitutional).

===============================================================================
A2 — MARGINS FROM OBSERVABLES
===============================================================================

A2.1–A2.4 Domain-defined margins (E, D, L, S).

A2.5 Compute slack margin
- x_C(t) = B_rt − [ b_enc(t) + (1−rest(t))*h(t)*b_roll(t) + rest(t)*b_cons(t) ].
- b_enc includes sensory prepass + selection + encoding costs (variable |O_t|).

===============================================================================
A3 — BASELINES & NORMALIZATION
===============================================================================

A3.1 Stability predicate stable(t).
A3.2 Baseline updates when stable.
A3.3 Weber/scale normalization for L/C/S channels.

===============================================================================
A4 — MEMORY SUBSTRATE (LIBRARY)
===============================================================================

================================================================================
A4.3′.2 SIG64 LOCALITY CONTRACT (normative; prevents avalanche-hash contortion)
================================================================================

A4.3′.2.1 Role of sig64
- sig64 exists solely to enable scan-proof approximate nearest-neighbor retrieval under fixed per-tick budgets.
- Stage-2 ranks candidates using Hamming distance on sig64. Therefore sig64 MUST preserve locality in Hamming space.

A4.3′.2.2 Definition (closed; deterministic; non-learned)
- sig64(t) is a fixed, deterministic, NON-LEARNED 64-bit locality-sensitive binary sketch of a fixed-size feature record:
    F_bytes(t) = metadata(O_t) || p(t).to_bytes()
  where metadata(O_t) is the closed schema in A4.3′.1 and p(t).to_bytes() is fixed-size.

A4.3′.2.3 Forbidden sig64 implementations (hard ban)
- FORBIDDEN: avalanche/cryptographic/fast hashes as sig64 when Hamming distance is used for ranking, including:
  SipHash, xxHash, Blake2, SHA-family, MurmurHash, CityHash, or any hash with avalanche property.
Reason: avalanche hashes destroy neighborhood structure; Hamming distance becomes non-informative.

A4.3′.2.4 Allowed reference sketch (SimHash-style; deterministic)
- Maintain a fixed compile-time matrix R ∈ {−1,+1}^{64×K}, K = |F_bytes|.
- Let v[k] = int(F_bytes[k]) − 128 (centered integer vector).
- For i=0..63: sig64[i] = 1{ Σ_k R[i,k] * v[k] ≥ 0 }.
- R is chosen once at compile time (seeded RNG), versioned, and never learned/updated online.

================================================================================
A4.3′.3 HAMMING-COMPATIBLE STAGE-1 BUCKETING (normative; prevents “rehash sig”)
================================================================================

A4.3′.3.1 Requirement
- Stage-1 bucket probing MUST preserve Hamming locality of sig64.

A4.3′.3.2 Forbidden bucket ids
- FORBIDDEN: bucket_id = Hash(sig64) % N_BUCKETS using avalanche hashes.

A4.3′.3.3 Allowed reference (banded LSH)
- Partition sig64 into T equal-width bands (e.g., T=4 bands × 16 bits).
- Bucket key is (block_id, band_id, band_value).
- Probe all bands in deterministic order, union candidates, cap at C_CAND_MAX.

================================================================================
A4.3′.4 REQUIRED AUDITS FOR SIG64/INDEX HEALTH (must-pass)
================================================================================

Given logged samples of (F_bytes(t), sig64(t), block_id) and any regime label L(t) used for evaluation:

(1) Locality preservation audit:
    kNN_overlap( distance(F_bytes), Hamming(sig64) ) ≥ θ_knn

(2) Index recall audit:
    Recall@K( Stage1+Stage2 vs full-scan Hamming TopK ) ≥ θ_recall

(3) Alias/collision audit:
    bucket multi-label fraction ≤ θ_alias

If any audit fails: change feature schema and/or sketch parameters and/or banding,
NOT memory size and NOT caps.
================================================================================

A4.1 Node/unit structure
[CHEAT-RISK] “Experts” here means footprint-local predictive operators. This is NOT Mixture-of-Experts.
There is NO gating network. Units are combined by PRECISION-WEIGHTED FUSION, not learned routing.
“Expert/unit” is NOT a MoE shard and NOT trained/selected via embedding-similarity gating.

Each unit j contains:
- mask m_j (footprint: which dims it predicts)
- predictor params (W_j, b_j)
- uncertainty Σ_j (diagonal only)
- reliability π_j
- compute cost L_j
- DAG links (optional)

A4.ID Stable UnitId requirement (normative; prevents index corruption)
- UnitId is stable for the unit’s lifetime.
- Library storage MUST support deletion without moving other units:
  either (a) free-list with tombstones, or (b) an indirection table UnitId -> slot.
- Buckets store UnitId (stable), never raw vector indices that can change.
  [CHEAT-RISK] Don’t store direct array indices in buckets if you compact arrays.

A4.2 Bounded working set A_t
- |A_t| ≤ N_work_max
- Σ_{j∈A_t} L_j ≤ L_work_max
- Anchors force-included.

--------------------------------------------------------------------------------
A4.3′ Signature-addressed retrieval (BLOCK-SCOPED; SCAN-PROOF)
--------------------------------------------------------------------------------
Definitions:
- Stored object = executed unit u
- Each unit stores:
  unit_block[u]: which block footprint lies in
  unit_sig64[u]: 64-bit address, IMMUTABLE at creation
  err_ema[u,h_bin]: outcome-vetted error average, h_bin ∈ {NEG, ZERO, POS}
  val_count[u,h_bin]: validation count
  [CHEAT-RISK] unit_sig64 is a FIXED HASH, not a learned embedding. Retrieval uses HAMMING DISTANCE.

Signature computation (strict scope):
- sig64(t) computed from:
  (i) committed observation METADATA from O_t (counts/small histograms only; no per-dim values)
  (ii) ephemeral peripheral gist p(t) available before purge.
- FORBIDDEN: persist p(t), O_t per-dim values, or dense feature vectors in durable state.

Stage 1 (gate; scan-proof):
- Bucket[block b, table j, bucket_id] → ≤ BUCKET_CAP unit_ids
- Probe in fixed deterministic order
- Union + bounded dedup until |Cand| = C_CAND_MAX, then DROP extras deterministically

Stage 2 (exact scoring; candidates only):
- score(u) = −popcount(sig64(t) XOR unit_sig64[u]) − α_err * err_ema[u, h_bin]
- Return Top-K, K ≤ K_MAX
  [CHEAT-RISK] Do NOT replace with learned similarity, attention, or embedding lookup.

Top-K selection MUST be streaming (bounded):
- Maintain a bounded heap of size K_MAX while scoring candidates.
- Forbidden: sorting unbounded lists. Sorting is permitted only because |Cand| ≤ C_CAND_MAX is a hard cap.

Index maintenance boundedness (normative; prevents hidden scans):
- Each unit stores its exact bucket memberships at creation: sig_index = [(table_id, bucket_id)...] (fixed size L_TABLES).
- Removal uses sig_index directly (no searches).
- Rebalancing is incremental: per REST tick touch ≤ BUCKET_MAINT_MAX buckets.
  [CHEAT-RISK] Don’t “rebuild the whole index” in REST; that’s a scan.

Overflow/eviction:
- Bucket full: replace lowest-priority using outcome-vetted stats ONLY (never usage-count)
- N_MAX reached: evict before create, or refuse creation

A4.4 Anti-aliasing at write time
- No two incumbents indistinguishable under footprint/cue basis.

--------------------------------------------------------------------------------
A4.5 Deferred Outcome Validation (SOLE SOURCE OF RETRIEVAL-STAT UPDATES; MUST-FIXED)
--------------------------------------------------------------------------------
Pending record structure:
- (unit_id, t_due, h_bin, frame_id, footprint_id, pred_ref)

When unit u contributes at time t for horizon h ≥ 0:
- write pending record into fixed-capacity ring
- store predicted values into PendingPredStore and set pred_ref to its slot id

PendingPredStore [cross-tick; non-serialized; bounded]:
- Bounded ring of size PREDVAL_MAX
- Stores only what is needed for validation:
  (t_pred, unit_id, frame_id, footprint_dims_sorted[≤FOOT_MAX], pred_values_aligned[≤FOOT_MAX])
- Entries persist until validated at t_due, then freed.

[CHEAT-RISK] “Transient” means non-serialized, NOT “purged each tick.” Predictions must persist until t_due.
[CHEAT-RISK] Do not recompute predictions at t_due with current weights; validate the stored prediction from time t.

A4.5.6 Footprint + snapshot representation (normative)
- footprint_dims are sorted canonical dim ids (bounded length ≤ FOOT_MAX).
- pred_values aligned 1:1 with footprint_dims.
- frame_id/footprint_id must be sufficient to interpret canonical dims (must be stable).

A4.5.7 Validation intersection algorithm (bounded, deterministic)
- At t_due, build commit_dims = sorted canonical dim ids in O_{t_due}.
- Compute E via two-pointer intersection of footprint_dims and commit_dims.
- Error computed only on E using aligned pred_values and y_commit values.
- Forbidden: per-dim global scans; forbidden: re-encoding past snapshots.

At t_due:
- If |E| = 0: no update (prediction unvalidatable)
- Else:
  - compute error on E vs observed y_commit(t_due)[E]
  - update err_ema[u,h_bin] (includes negative evidence)
  - increment val_count[u,h_bin]

Overflow:
- Pending overflow: evict oldest pending record and free its pred_ref storage; log.
- PredStore overflow: if PREDVAL_MAX would be exceeded, evict oldest pending record deterministically; log.

Forbidden updates:
- retrieval frequency, activation frequency, fusion weight, internal similarity

Serialization:
- Pending ring and PendingPredStore are NOT serialized.
- On restart, pending validations are lost (acceptable transient loss).
  [CHEAT-RISK] Don’t serialize PendingPredStore “for continuity.” That becomes episodic sensory memory.

===============================================================================
A5 — SALIENCE & TEMPERATURE (ATTENTION)
===============================================================================

A5.1–A5.5 Salience, temperature τ_eff, activation propensity, active-set selection under caps, log L_eff.

===============================================================================
A6 — FIXED BUDGET, EMERGENT HORIZON
===============================================================================

A6.1 B_rt fixed per tick; allocate internally; enforce degradation.
A6.2 Emergent horizon h(t) arises from remaining slack; increased b_enc reduces h(t).
A6.3 No hidden compute.

===============================================================================
A7 — COMPLETION & FUSION
===============================================================================

A7.1 Belief state x(t); precision-weighted fusion; no hallucinated certainty.
[CHEAT-RISK] Fusion is PRECISION-WEIGHTED AVERAGE of unit predictions. This is NOT attention. NOT learned routing. NOT MoE gating.

A7.2 Confidence explicit and used downstream.

--------------------------------------------------------------------------------
A7.3 Unified Temporal Completion Operator + Reference Contract (MUST; prevents “handwave Complete()”)
--------------------------------------------------------------------------------
A7.3.1 Signature
- Complete(prior, cue, h, context) → estimate(t+h)

A7.3.2 Semantics by horizon
- h = 0 (perception): cue = y_commit(t)|_{O_t}; clamp into prior on observed dims
- h > 0 (prediction/planning): cue may be empty (passive) or encode action hypothesis a_hyp (planning)
- h < 0 (recall): cue = {}; inference-only; NO durable episodic store

A7.3.3 Same model class across h
- Same weights across h; h is conditioning input, not a mode switch.

A7.3.4 Interface contract (normative)
- Inputs:
  prior: mean vector mu_prior and diag precision/variance PrecPriorDiag (or VarPriorDiag)
  cue: committed observations for h=0, or action hypothesis / empty for h>0, or empty for h<0
  h: integer horizon
  context: active units A_t with (mask, params, diag uncertainty, reliability)
- Output:
  mu_est and PrecDiag_est, plus optional predicted block-gist summaries r_b(t) (fixed-size quantized)

A7.3.5 Reference fusion + completion (allowed baseline; bounded; deterministic)
- Each unit u ∈ A_t produces on its footprint:
  mu_u = f_u(mu_prior, cue, h)     (e.g., linear: W_u * mu_prior + b_u [+ action term])
  prec_u = π_u * inv_var_u         (diag precision; π_u scalar reliability)
- Fuse per-dim using precision-weighted averaging:
  Prec[d] = PrecPrior[d] + Σ_{u covers d} prec_u[d]
  mu[d]   = (PrecPrior[d]*mu_prior[d] + Σ prec_u[d]*mu_u[d]) / Prec[d]
- Then clamp committed cue dims for h=0:
  for d ∈ O_t: mu[d] := y_commit[d]; Prec[d] := clamp_precision_large

[CHEAT-RISK] Do not replace this with an end-to-end attention model or MoE gating; any upgrades must preserve:
- explicit masks,
- explicit bounded fusion,
- explicit clamp semantics,
- and outcome-vetted updates only.

A7.3.6 Predicted gist emission (needed by A16.5.2 reference U)
- After fusion, emit per-block predicted gist r_b(t) = Quantize8(gist(mu on block b)) (fixed-size).
- Persist only r_b(t) as bounded per-block scalars if needed; do NOT persist dense fields.

===============================================================================
A8 — LATENCY & COMMITMENT
===============================================================================

A8.1 Latency floor d = ceil(T_proc/Δt).
A8.2 Commit gate: not in REST, h(t)≥d, confidence(d)≥θ_act, safety conditions satisfied.
A8.3 If not committed: execute π_safe (viability-preserving).

===============================================================================
A9 — PREDICTION OBJECTIVE
===============================================================================

A9.1 Multi-step predictive loss (at least depth d; optionally to h(t)).
A9.2 Arousal-modulated horizon weighting (lagged).
A9.3 Threat-gated exploration.

===============================================================================
A10 — EDIT CONTROL
===============================================================================

A10.1 Freeze under external threat (lagged); hard safety lock.
A10.2 Permit parameter updates only with headroom and slack.
A10.3 Responsibility-gated updates: only if overlaps O_t and local error below θ_learn.

===============================================================================
A11 — SEMANTIC INTEGRITY
===============================================================================

A11.1 Track drift_P(t); margin m_S reflects headroom.
A11.2 Edits must not degrade semantic integrity.

===============================================================================
A12 — EDIT ACCEPTANCE (REST; MUST BE BOUNDED AND ANTI-REPLAY)
===============================================================================

A12.1 Typed edits only: CREATE/SPAWN, SPLIT, MERGE, REPLACE/PRUNE.
A12.2 Acceptance evidence-based and audited on same window.
A12.3 Merge fix: per-domain non-worsening.
A12.4 Residual/debt-driven invention (not arbitrary growth).
  [CHEAT-RISK] No “create unit because it helps once.” Must be residual persistence + discriminability gate.

A12.0.1 Edit proposal queue (bounded)
- Maintain bounded FIFO EditQ ≤ EDITQ_MAX containing proposals with:
  type, involved unit_ids, involved block_ids, creation_time, evidence_refs.

A12.0.2 Allowed evidence sources (anti-replay; outcome-vetted)
- Edits may use ONLY:
  (i) outcome-vetted stats: err_ema, val_count, ResidEMA, ChangeRateEMA, P^nov_state, debt/age
  (ii) compressed sufficient statistics accumulated online (A12.0.3)
- Forbidden: training on stored TraceCache entries; forbidden: scanning raw past frames.

A12.0.3 Compressed sufficient statistics (optional but makes CREATE/SPLIT implementable)
- Per block b maintain bounded accumulators updated only when b ∈ O_t:
  Sxx[b], Sxy[b], Syy[b] for a fixed small feature basis φ(x_prior, y_commit) (K_feat dims).
- These are not raw sensory; they are fixed-size aggregates.

A12.0.4 Bounded edit evaluation per REST tick
- Each REST tick may evaluate at most EVAL_MAX proposals.
- Each evaluation must be O(1) in library size (no scans), using only proposal-local units/blocks.

A12.0.5 Typed evidence criteria (reference)
- PRUNE(u): val_count[u,*] ≥ N_prune AND err_ema[u,*] ≥ ε_prune.
- MERGE(u,v): overlap + merged candidate does not worsen validated error proxy beyond margin.
- SPLIT(u): consistent bimodality in u’s footprint-basis sufficient stats and split improves validated proxy.
- CREATE/SPAWN(b): debt(b) high AND P^nov_state(b) high for ≥ T_persist ticks AND discriminability gate passes.

===============================================================================
A13 — UNIFIED STEP ORDERING (CONSTITUTIONAL)
===============================================================================

A13.1 Canonical per-tick ordering (world tick t_w = t)

1) INGRESS
- Receive y_full(t) (ephemeral) + domain observables
- Record τ_start (diagnostics only)

2) MODE
- Compute rest(t) from lagged predicates (A14)

3) SENSORY PREPASS (bounded; no micro-steps)
- Compute peripheral gist p(t) from y_full(t) (ephemeral)
- Compute per-block gist p_b(t) and last-committed summaries q_b(t-1) if available (bounded)
- Compute adviser fields U(b,t), C^rec(b,t where defined); update P^nov_state(b) and U_prev_state(b); compute V(b,t)
- Adviser fields do not redefine residual

3.5) CONTEMPLATIVE GATE (bounded)
- If rest(t)=1: force g_contemplate(t)=0
- Else: compute g_contemplate_perm(t); then g_contemplate(t) per A0.5(d); determine focus_mode

4) ATTENTION COMMIT (authoritative O_t)
- If g_contemplate=0: O_t = greedy_cov(V_t, debt_t, budget_t) (A16.4)
- If g_contemplate=1 & FREE: O_t = sentinel/anchor blocks (min intake)
- If g_contemplate=1 & BOUND: O_t = problem_dims(t) from problem_bound(t)

5) ENCODE/COMMIT
- Form committed cue y_commit(t)|_{O_t}
- Discard remainder of y_full(t) from further dependence (still ephemeral until purge)
  [CHEAT-RISK] No later “peeking” at uncommitted dims within the tick

5.25) DEFERRED VALIDATION RESOLUTION (A4.5)
- For pending with t_due==t: compute overlap with O_t, update err_ema/val_count
- MUST occur before retrieval scoring so err_ema is current

5.5) RETRIEVAL (A4.3′)
- Compute sig64(t) from metadata(O_t) + p(t)
- Run Stage-1 + Stage-2 under caps; NO scans

5.75) WORKING-SET SELECTION (A5)
- Merge retrieved units into candidates; select A_t under caps

6) PRIOR
- Compute x_hat(t|t−1) from A_t and x(t−1)

7) COMPLETE + OPTIONAL MICRO-STEPS
Perception finalization (AUTHORITATIVE):
- x_commit(t) = Complete(prior=x_hat, cue=y_commit, h=0, context=A_t)

If g_contemplate=1, within remaining slack, run bounded micro-steps k that update ONLY staged artifacts:
- Thread_i_stage / pending_artifacts_stage (A0.6 staging requirement)
- (optionally) TraceCache reads (A16.8) for planning context
Write-permission invariant (hard):
- Micro-steps MUST NOT modify x_commit(t)
- MUST NOT modify residual/debt/age
- MUST NOT modify A4.5 stats except via step 5.25
- MUST NOT mutate durable library/memory or Thread_i in-place (use staging)
  [CHEAT-RISK] Don’t “refine perception” inside contemplation because it’s easier; perception is finalized at h=0.

8) RESIDUAL (constitutional definition)
- e_obs(t) = x_commit(t)[O_t] − x_hat(t|t−1)[O_t] on committed dims only
- Update residual stats, debt/age ONCE (t_w-indexed)

9) OPERATING UPDATES
A13.1(9).1 Operating parameter updates (reference rule; bounded; no stat authority)
- Permitted updates:
  (i) unit predictor params (W_u, b_u), diag variance estimates Σ_u,
  (ii) per-block streaming sufficient statistics (compressed; A12.0.3),
  (iii) staged thread artifacts.
- Forbidden updates:
  err_ema, val_count (except via A4.5), bucket priority, retention priority from usage.

A13.1(9).2 Reference learning update (simple, implementable)
- For each unit u that contributed to x_commit at h=0:
  update using e_obs restricted to footprint(u) under A10 gates and within remaining budget.
- All updates must be incremental O(#active units * footprint_size).

- Record new A4.5 pending validations and store predicted values in PendingPredStore (bounded)

10) REST UPDATES
- If rest(t)=1: consolidation/edit evaluation only (per A14/A12)

11) ACTION
- If committed (A8): execute planned a(t)
- Else: π_safe

12) PURGE & COMMIT BARRIER
- Purge: y_full(t), p(t), and ephemeral adviser fields U/C^rec/V (per-tick values)
- Commit only permitted persistent states at tick boundary:
  P^nov_state, U_prev_state, debts/ages/stats (from canonical steps), Thread_i_stage→Thread_i (bounded),
  library edits only via REST/permits, and bounded TraceCache insert/evict
- Update Δτ_ema (diagnostics)
  [CHEAT-RISK] Don’t serialize PendingPredStore or TraceCache “temporarily.” That becomes episodic memory.

A13.2 Residual definition is prediction error on committed dims ONLY.
- Any “residual = Δx” shortcut is non-compliant.

===============================================================================
A14 — MACROSTATES (REST vs OPERATING)
===============================================================================

A14.1 REST is lag-disciplined.
- rest(t) = rest_permitted(t−1) * demand(t−1) * (1 − interrupt(t−1)).

A14.2 Roles
- OPERATING: act + learn (if permitted).
- REST: consolidate/evaluate edits/restructure; no same-step action-planning dependence on new updates.
- REST still executes perception pipeline for safety/threat detection; threat may interrupt REST.

A14.3 External threat can interrupt REST and freeze learning (A10.1).

A14.4 Contemplation vs REST (explicit)
- Contemplation ⊂ OPERATING; not a macrostate; does not enable structural edits.
- Mutual exclusivity per tick: if rest(t)=1 then g_contemplate(t)=0.

A14.5 Consolidation semantics (bounded; no scans; anti-replay)
- REST-time processing may:
  - evaluate edit proposals from EditQ (≤ EVAL_MAX per REST tick),
  - apply MERGE/PRUNE/SPLIT per A12 criteria,
  - perform bounded index maintenance per A4.3′ (≤ BUCKET_MAINT_MAX buckets),
  - update long-term reliability summaries derived from outcome-vetted val_count/err_ema.
- Budget charged to b_cons(t).
- FORBIDDEN: replay raw sensory, train from TraceCache, rebuild whole index, create units without residual/debt persistence triggers.

===============================================================================
A15 — MARGIN DYNAMICS
===============================================================================

A15.1 Margins evolve with work and repair; REST restores E/D/S per domain dynamics.
A15.2 Dynamics must be explicit and auditable.

===============================================================================
A16 — ATTENTION GEOMETRY & COVERAGE-DISCIPLINED FOVEATION
===============================================================================

A16.0 Full ingress, selective commit.
- Agent receives y_full(t), commits only O_t.

A16.1 Block partition.
- D partitioned into disjoint blocks; non-anchor footprints within one block.

A16.2 Per-block age and residual (on committed dims).
- age(b,t): ticks since last committed (resets to 0 when b ∈ O_t; else +1)
- residual(b,t): prediction error stats when committed (A13.2)

A16.2.1 Per-block residual stats (persistent; bounded)
- Maintain ResidEMA[b] updated ONLY when b ∈ O_t:
  ResidEMA[b] ← λ*ResidEMA[b] + (1−λ)*mean_abs(e_obs on b at t)

A16.3 Coverage debt.
- Debt increases with age/residual, decreases when committed and well-explained.

A16.3.1 Coverage debt definition (persistent; bounded)
- debt(b,t) := clamp( w_age*age(b,t) + w_res*ResidEMA[b], 0, debt_max )

A16.4 greedy_cov selection (refined; deterministic; bounded)
A16.4.1 Inputs/outputs
- Input: adviser score V(b,t) (ephemeral), debt(b,t) (persistent), and remaining encode budget B_enc_rem.
- Output: O_t as a set of blocks.

A16.4.2 Reference greedy_cov (deterministic; bounded)
- For each block b define:
  score(b) = (λV*V(b,t) + λD*debt(b,t)) / cost_enc(b)
- Iteratively select the not-yet-selected block with maximal score(b)
  while total_cost ≤ B_enc_rem and |O_t| ≤ F_MAX.
- Tie-breaker: smallest block_id wins.
- Anchors: forced include before greedy pass; anchors count toward F_MAX and budget.

A16.4.3 Boundedness rule
- greedy_cov may only inspect per-block scalars and fixed-size metadata.
- Forbidden: inspecting all units; forbidden: per-dim scans beyond selected blocks.

- Contemplative override: if g_contemplate=1 then bypass greedy_cov per A13.1 step 4.

A16.5 Peripheral adviser (internal prepass; bounded interfaces + reference defs)
A16.5.1 computePGist(y_full(t)) (ephemeral; bounded; deterministic baseline)
- Output: p(t) = {p_b(t)} over blocks b, where each p_b is a fixed-size quantized summary.
- p_b(t) MUST be non-invertible by design: fixed feature count K_gist << |block|.
- Reference definition (allowed baseline):
  p_b(t) = Quantize8([ mean(y_full[b]), mean(|Δy|[b]), var(y_full[b]), sat_count(y_full[b]) ])
  where Δy is difference vs last committed y_commit on that block if available, else 0.
- Boundedness: O(|y_full|) once per tick; no persistence; not serialized.
  [AI-DIVERGENCE RISK] Don’t “upgrade” p(t) into a learned high-capacity encoder whose outputs become reconstructible memory.

A16.5.2 computeU(b,t) (ephemeral per-block uncertainty proxy; bounded)
- Output: U(b,t) is a scalar computed ONLY from:
  (i) current ephemeral p_b(t),
  (ii) last-tick committed cue summary q_b(t−1) (persistent scalar(s)), and/or
  (iii) last-tick predicted summary r_b(t−1) (persistent scalar(s)).
- Reference definition (allowed baseline):
  U(b,t) = clamp01( L1(p_b(t) − r_b(t−1)) )
  where r_b(t−1) is a fixed-size predicted gist produced last tick (see A7.3.6).
- Persistence: U(b,t) is purged at step 12.
  [CHEAT-RISK] Do not compute U(b,t) by scanning full stored history or by reading uncommitted dims after step 5.

A16.5.3 recognized(b,t) predicate (makes C^rec well-defined; bounded)
- recognized(b,t)=1 iff ∃ unit u with unit_block[u]=b AND val_count[u,ZERO] ≥ N_rec AND err_ema[u,ZERO] ≤ ε_rec.
- Otherwise recognized(b,t)=0.

A16.5.4 computeCRec(b,t) (recognized change-likelihood; undefined if not recognized)
- If recognized(b,t)=0: C^rec(b,t) := UNDEFINED.
- If recognized(b,t)=1: C^rec(b,t) := ChangeRateEMA[b] (persistent per-block scalar).
- Update rule for ChangeRateEMA[b] occurs ONLY when block b is committed:
  change_event(b,t) = 1{ mean_abs(e_obs on b at t) ≥ θ_change }
  ChangeRateEMA[b] ← ρ*ChangeRateEMA[b] + (1−ρ)*change_event(b,t)
- Boundedness: O(#committed blocks) updates; no scans.

A16.5.5 Adviser authority constraint
- Adviser outputs bias greedy_cov and sig64 addressing; do not update x outside O_t; do not redefine residual.
  [CHEAT-RISK] Don’t “peek” at non-committed dims later in the tick using cached y_full.

A16.6′ Cold storage retrieval coupling (supersedes prior A16.6).
- Block selection defines retrieval scope: only units whose footprints intersect selected blocks are eligible.
- Signature determines rank within scope (A4.3′ Stage 2); block scores do not define unit rank.
- Clamping by O_t remains authoritative.

--------------------------------------------------------------------------------
A16.7 REST RAW BUFFER SEMANTICS (NO POST-REST PIXELS)
--------------------------------------------------------------------------------
A16.7.1 Permitted only for short-horizon audit during REST
- Rolling raw buffer allowed ONLY in REST for audit/edit evaluation.
- Buffer content must be foveated (committed dims + geometry), not full-field.

A16.7.2 Purge requirement
- Raw buffers purged before leaving REST.
- No pixels (raw or reconstructible full-field) may appear in post-REST durable state or serialization.
  [CHEAT-RISK] Don’t keep a “debug video” buffer around in OPERATING.

A16.7.3 Explicit encoder boundary
- x(t) is encoded abstraction; completion/fusion operate on x(t), not pixels, unless explicitly declared prototype.

--------------------------------------------------------------------------------
A16.8 FOVEATED TRACE CACHE (OPERATING-readable; bounded; non-authoritative)
--------------------------------------------------------------------------------
A16.8.1 Definition
- Maintain volatile ring TraceCache with ≤ TRACE_M entries.
- Each entry stores only committed foveated cues:
  e = (t_w, frame_id, geom(O_t), y_commit(t)|_{O_t})
- No y_full(t), no p(t), no dense intermediates, no reconstructible full-field payload.

A16.8.2 Global sensory-mass cap (prevents “accumulate full scene” loophole)
- Maintain incremental per-block refcount[b] for blocks present in TraceCache.
- Enforce |{b : refcount[b]>0}| ≤ TRACE_B_MAX.
- Enforce incrementally on insert/evict in O(F_MAX) time; no scans.
- If insert would violate TRACE_B_MAX: deterministically drop caching for that tick or drop cached blocks
  from the cache entry payload (NOT from O_t).

A16.8.3 Non-authority
- TraceCache has ZERO authority to update:
  x_commit(t), residual/debt/age, A4.5 stats, library params/structure, retrieval stats, or training replay.
- It may be read only to inform staged planning/thread artifacts.
  [CHEAT-RISK] Do not use TraceCache as a replay dataset for learning updates.

A16.8.4 Serialization
- TraceCache is NOT serialized to durable checkpoints by default.
  [CHEAT-RISK] Don’t serialize TraceCache “temporarily” to debug; that becomes episodic memory.

A16.8.5 Read interface (bounded; deterministic)
- query_trace(t_query, block_filter) → entry or None
- Returns closest entry with t_w ≤ t_query matching block_filter
- Read cost charged to micro-step budget
- Returned cue is NON-AUTHORITATIVE

===============================================================================
A17 — DERIVED DIAGNOSTICS (NO CONTROL AUTHORITY)
===============================================================================

A17.1 Diagnostics are logged observables only unless explicitly referenced by an axiom.
A17.2 Audit requirements
- Log enough to verify lag discipline, budget accounting, purge compliance, O_t decisions, edit acceptance tests,
  and environment non-decisionality.

A17.3 Structural non-authority rule (normative; prevents “diagnostics become control”)
- The tick transition produces two disjoint outputs: DecisionOutputs and DiagnosticsOutputs.
- DecisionOutputs MUST be computed without access to DiagnosticsOutputs.
- Any value used to affect behavior MUST be in DecisionInputs and logged as such.
  [CHEAT-RISK] Don’t leak diagnostics (timers, profilers) into policy except through explicitly declared DecisionInputs.

IO boundary rule (normative)
- Runtime provides τ_now and/or measured capacity B_max(t) as explicit inputs.
- The pure tick transition consumes these inputs; it does not perform IO internally.

===============================================================================
SUGGESTED DEFAULTS (non-normative)
===============================================================================
θ_threat_low = 0.2
θ_urgent_high = 0.8
θ_slack_min = 0.3
θ_persist = 0.3
α = 1.0 (C^rec weight in V)
β = 0.5 (P^nov weight in V)
γ = 0.8 (P^nov EMA decay)
α_err = 0.5 (err_ema penalty in scoring)
k_max = 8 (max micro-steps)
T_MAX = (small, e.g., 2–8)
PREDVAL_MAX = (bounded; sufficient for worst-case pending footprint slices without storing full states)
FOOT_MAX = (bounded; per-unit footprint slice cap)
EDITQ_MAX = (bounded)
EVAL_MAX = (bounded)
BUCKET_MAINT_MAX = (bounded)
N_rec, ε_rec, θ_change, ρ, λ, w_age, w_res, debt_max, λV, λD, cost_enc(b) = (explicit implementation constants)

===============================================================================
IMPLEMENTATION CHECKLIST (AI must verify before claiming compliance)
===============================================================================
[ ] Retrieval uses Hamming distance on sig64, NOT learned embeddings
[ ] Fusion uses precision-weighting, NOT attention or MoE gating
[ ] Units are footprint-local predictors, NOT routed experts
[ ] No "for unit in all_units" on any tick path
[ ] err_ema/val_count updated ONLY via A4.5 deferred validation
[ ] Predictions stored in PendingPredStore until t_due (cross-tick, non-serialized)
[ ] PendingPredStore stores footprint_dims+values (bounded FOOT_MAX), intersection is two-pointer (no scans)
[ ] Micro-steps write to staged buffers only; commit at step 12
[ ] Threads mutate only via Thread_i_stage; deterministic scheduler
[ ] U_prev_state(b) is single scalar per block, not history; P^nov_state persists; U/V/C^rec ephemeral
[ ] TraceCache never used for training; never serialized
[ ] No serialization of ephemeral fields, pending ring, PendingPredStore, or TraceCache
[ ] greedy_cov uses only per-block scalars; deterministic tie-breakers
[ ] Environment/harness makes ZERO attention/routing decisions
[ ] No “upgrade” to MoE/attention/replay/embedding search for convenience

===============================================================================
END OF NUPCA5 AXIOMS v5.02
===============================================================================




# NUPCA5 v5.02 Amendments: Closing Remaining Escape Hatches

These four amendments can be integrated as v5.03 additions. Each addresses a specific underspecification that could enable non-compliant implementations.

---

## Amendment 1: Metadata Schema for sig64

### Problem
A4.3′ says sig64 is computed from "committed observation METADATA from O_t (counts/small histograms only)" but doesn't define the schema. An implementation could smuggle dense information into "metadata."

### Proposed Axiom Text

```
================================================================================
A4.3′.1 sig64 METADATA SCHEMA (normative; prevents dense smuggling)
================================================================================

A4.3′.1.1 Metadata definition
- metadata(O_t) is a fixed-size record M with the following fields ONLY:

  M.block_mask    : BitVec[B_MAX]     -- 1 if block b ∈ O_t, else 0
  M.total_dims    : UInt16            -- |O_t| clamped to 2^16-1
  M.anchor_count  : UInt8             -- number of anchor blocks in O_t
  M.block_counts  : UInt8[F_MAX]      -- per-selected-block dim count (padded/sorted by block_id)

- Total size: ceil(B_MAX/8) + 2 + 1 + F_MAX bytes (fixed, independent of |O_t| content)

A4.3′.1.2 sig64 computation (reference; deterministic)
- sig64(t) = Sketch64( M.to_bytes() ++ p(t).to_bytes() )
- Hash64 is any fixed cryptographic or fast hash (xxHash64, SipHash, etc.) chosen at compile time.
- p(t).to_bytes() is the concatenation of all p_b(t) in block_id order, zero-padded for absent blocks.

A4.3′.1.3 Forbidden metadata content
- Per-dim values from y_commit or y_full
- Floating-point statistics beyond counts
- Variable-length encodings
- Any field not in the schema above

[CHEAT-RISK] Don't add "just one more summary stat" to metadata. The schema is closed.
```

### Implementation Notes
- B_MAX is the maximum number of blocks in the domain (compile-time constant)
- F_MAX already defined in A0.BUDGET
- Total sig64 input is fixed-size, making hash cost constant
- Block_mask enables coarse regime discrimination without per-dim leakage

---

## Amendment 2: Δy-in-Gist Resolution

### Problem
A16.5.1 reference gist uses `mean(|Δy|[b])` where Δy is "difference vs last committed y_commit on that block." This requires storing full y_commit per block—potentially unbounded dense storage.

### Proposed Axiom Text

```
================================================================================
A16.5.1′ GIST COMPUTATION (revised; no dense persistence)
================================================================================

A16.5.1′.1 Per-block committed summary q_b(t) (persistent; fixed-size)
- When block b ∈ O_t, compute and store:
  q_b(t) = Quantize8([ mean(y_commit[b]), var(y_commit[b]) ])
- When block b ∉ O_t: q_b(t) := q_b(t-1) (unchanged)
- Size: 2 bytes per block (fixed)
- This is the ONLY per-block committed-cue persistence.

A16.5.1′.2 Gist computation (revised reference)
- p_b(t) = Quantize8([ 
    mean(y_full[b]),                           -- current level
    |mean(y_full[b]) - q_b(t-1).mean|,         -- change vs last commit (uses summary, not dense)
    var(y_full[b]),                            -- current variance
    sat_count(y_full[b])                       -- saturation count
  ])
- Size: 4 bytes per block (fixed)

A16.5.1′.3 Forbidden dense storage
- y_commit[b] values MUST NOT be stored beyond q_b summary.
- Δy MUST be computed from summaries, not stored per-dim differences.
- No per-dim history arrays.

[CHEAT-RISK] Don't store "just the last frame's y_commit" as a dense array. Use q_b only.
```

### Implementation Notes
- q_b(t) replaces any need to store raw y_commit values
- Change detection uses summary-level comparison, not per-dim
- Total persistent gist storage: 2 bytes × B_MAX (negligible)
- Ephemeral p_b(t) is 4 bytes × B_MAX per tick (purged)

---

## Amendment 3: Scan-Free Recognized Predicate and Block Stats

### Problem
A16.5.3 defines `recognized(b,t) = 1 iff ∃ unit u with unit_block[u]=b AND ...`. Naively, this requires scanning all units in block b. Same issue for any "exists unit satisfying X" predicate.

### Proposed Axiom Text

```
================================================================================
A16.5.6 BLOCK-LEVEL UNIT STATISTICS (scan-free maintenance; normative)
================================================================================

A16.5.6.1 Per-block unit index (bounded; incrementally maintained)
- Maintain per-block summary stats updated ONLY during:
  (i) A4.5 deferred validation (when err_ema/val_count change)
  (ii) A12 edit commits (when units created/deleted/moved)
- No scans; O(1) per update.

A16.5.6.2 BlockStats structure (per block b)
  BlockStats[b] = {
    recognized_count : UInt16    -- count of units where val_count[u,ZERO] ≥ N_rec AND err_ema[u,ZERO] ≤ ε_rec
    best_err_ema     : Float16   -- min err_ema[u,ZERO] among units in block (for diagnostics)
    total_units      : UInt16    -- count of units with unit_block[u] = b
  }

A16.5.6.3 Incremental update rules

On A4.5 validation update for unit u in block b:
  was_recognized := (old_val_count[u,ZERO] ≥ N_rec AND old_err_ema[u,ZERO] ≤ ε_rec)
  is_recognized  := (new_val_count[u,ZERO] ≥ N_rec AND new_err_ema[u,ZERO] ≤ ε_rec)
  if was_recognized AND NOT is_recognized:
    BlockStats[b].recognized_count -= 1
  if NOT was_recognized AND is_recognized:
    BlockStats[b].recognized_count += 1
  BlockStats[b].best_err_ema = min(BlockStats[b].best_err_ema, new_err_ema[u,ZERO])

On unit creation in block b:
  BlockStats[b].total_units += 1
  -- recognized_count unchanged (new units have val_count=0, not recognized)

On unit deletion from block b:
  if unit was recognized:
    BlockStats[b].recognized_count -= 1
  BlockStats[b].total_units -= 1
  -- best_err_ema may become stale; acceptable (conservative) or rebuild lazily in REST

A16.5.6.4 Query (O(1))
  recognized(b,t) := BlockStats[b].recognized_count > 0

A16.5.6.5 ChangeRateEMA persistence
- ChangeRateEMA[b] is always maintained (updated when b ∈ O_t per A16.5.4).
- The recognized(b,t) predicate only gates whether C^rec(b,t) is DEFINED, not whether ChangeRateEMA is stored.

[CHEAT-RISK] Don't scan units to answer recognized(b,t). Use BlockStats.
[CHEAT-RISK] Don't rebuild best_err_ema by scanning; accept staleness or bound REST maintenance.
```

### Implementation Notes
- BlockStats adds ~6 bytes per block (negligible vs. B_MAX)
- All updates are O(1) during existing operations
- best_err_ema may become stale on deletion; can be lazily corrected in REST by scanning only affected block's units (bounded by units-per-block, not N_MAX)
- Generalizes to any "∃ unit in block satisfying X" predicate

---

## Amendment 4: Explicit Update Rule for Unit Parameters

### Problem
A13.1(9).2 says "update using e_obs restricted to footprint(u)" but doesn't specify the rule. This leaves room for smuggling in BPTT, replay, or unbounded computation.

### Proposed Axiom Text

```
================================================================================
A9.4 UNIT PARAMETER UPDATE RULE (normative; bounded; no replay)
================================================================================

A9.4.1 Scope
- Applies to each unit u ∈ A_t that contributed to x_commit(t) at h=0.
- Update occurs in step A13.1(9) under A10 gates.

A9.4.2 Inputs (available at update time)
- e_u(t) = e_obs(t)[footprint(u)]                    -- error on unit's footprint (from step 8)
- x_prior_u = x_hat(t|t-1)[footprint(u)]             -- prior on footprint
- π_u, Σ_u                                           -- current reliability and variance
- Learning rate η (global or per-unit; bounded)
- Arousal modulator s_ar(t-1) (lagged; A0.2)

A9.4.3 Reference update rule (online gradient descent; allowed baseline)

For linear unit with predictor μ_u = W_u · x_input + b_u:

  -- Gradient of squared error on footprint
  g_W = outer(e_u, x_input[relevant_dims])           -- outer product
  g_b = e_u
  
  -- Arousal-modulated learning rate
  η_eff = η * (1 + κ_ar * s_ar(t-1))                 -- higher arousal → faster learning
  
  -- Clipped gradient (prevents instability)
  g_W_clip = clip(g_W, -G_MAX, G_MAX)
  g_b_clip = clip(g_b, -G_MAX, G_MAX)
  
  -- Update (gradient descent)
  W_u := W_u - η_eff * g_W_clip
  b_u := b_u - η_eff * g_b_clip

A9.4.4 Variance/uncertainty update (optional; bounded)
  -- Online variance estimate (Welford-style)
  Σ_u[d] := (1 - α_Σ) * Σ_u[d] + α_Σ * e_u[d]^2
  -- Clamp to prevent collapse or explosion
  Σ_u[d] := clamp(Σ_u[d], Σ_MIN, Σ_MAX)

A9.4.5 Reliability update (optional; bounded)
  -- Reliability tracks inverse average error magnitude
  err_mag = mean(|e_u|)
  π_u := (1 - α_π) * π_u + α_π * (1 / (1 + err_mag))
  π_u := clamp(π_u, π_MIN, π_MAX)

A9.4.6 Forbidden update patterns
- Backpropagation through time across tick boundaries
- Updates using stored sensory (y_full, TraceCache, raw buffers)
- Updates using predictions from units not in A_t
- Updates using error from dims outside footprint(u)
- Batch updates accumulated over multiple ticks
- Any update with cost > O(|footprint(u)|² + |x_input|)

A9.4.7 Multi-step loss extension (optional; bounded)
- If h(t) > 0 and pending predictions exist from prior ticks:
  - A4.5 validation provides err_ema updates (stat authority)
  - Parameter updates MAY weight current e_u by horizon discount:
    e_weighted = e_u * γ^0 + ... (but only from CURRENT tick error, not replay)
- Forbidden: storing gradients across ticks; training from stored predictions

A9.4.8 Budget accounting
- Update cost charged as: C_update = |A_t| * max_footprint_size * C_flop
- Must fit within B_rem after perception/retrieval

[CHEAT-RISK] Don't accumulate gradients across ticks "for stability." Each tick is self-contained.
[CHEAT-RISK] Don't use TraceCache to compute "better" gradients. That's replay.
[CHEAT-RISK] Don't add momentum/Adam state that grows with N_MAX. Per-unit state is bounded.
```

### Implementation Notes

```haskell
-- Reference implementation (simplified)
updateUnit :: Unit -> ErrorVec -> PriorVec -> Float -> Unit
updateUnit u e_u x_prior s_ar = u
  { params_W = Vec.zipWith (-) (params_W u) (Vec.map (* η_eff) g_W_clip)
  , params_b = Vec.zipWith (-) (params_b u) (Vec.map (* η_eff) g_b_clip)
  , variance = Vec.zipWith updateVar (variance u) e_u
  , reliability = clamp π_MIN π_MAX new_π
  }
  where
    η_eff = η_base * (1 + κ_ar * s_ar)
    g_W = outerProduct e_u x_prior  -- or relevant slice
    g_b = e_u
    g_W_clip = Vec.map (clamp (-g_MAX) g_MAX) g_W
    g_b_clip = Vec.map (clamp (-g_MAX) g_MAX) g_b
    updateVar old_σ e = clamp σ_MIN σ_MAX $ (1 - α_Σ) * old_σ + α_Σ * e^2
    new_π = (1 - α_π) * reliability u + α_π / (1 + mean (Vec.map abs e_u))
```

### Constants (suggested defaults)
```
η = 0.01           -- base learning rate
κ_ar = 0.5         -- arousal modulation strength
G_MAX = 10.0       -- gradient clipping threshold
α_Σ = 0.1          -- variance EMA rate
α_π = 0.05         -- reliability EMA rate
Σ_MIN = 1e-6       -- minimum variance (prevent division by zero)
Σ_MAX = 1e3        -- maximum variance (prevent overflow)
π_MIN = 0.01       -- minimum reliability
π_MAX = 1.0        -- maximum reliability
```

---

## Summary of Amendments

| Amendment | Closes Escape Hatch | Key Constraint |
|-----------|--------------------|-----------------| 
| A4.3′.1 (metadata schema) | Dense info in sig64 | Fixed-size bitfield + counts only |
| A16.5.1′ (gist revision) | Dense Δy storage | Use 2-byte q_b summary, not raw y_commit |
| A16.5.6 (block stats) | Scans for ∃ unit predicates | Incremental recognized_count |
| A9.4 (update rule) | Unbounded/replay learning | Online gradient descent, no cross-tick state |

---

## Integration Checklist

```
[ ] A4.3′.1 added; sig64 uses only M.block_mask, M.total_dims, M.anchor_count, M.block_counts
[ ] A16.5.1′ added; q_b is 2 bytes/block; no dense y_commit storage
[ ] A16.5.6 added; BlockStats maintained incrementally; recognized(b,t) is O(1)
[ ] A9.4 added; update rule is explicit gradient descent with clipping
[ ] No "for unit in all_units" anywhere including recognized checks
[ ] No dense arrays persisted beyond fixed-size summaries
[ ] Learning uses only current-tick error on footprint dims
```

These four amendments close the remaining specification gaps. An implementation satisfying v5.02 + these amendments has no remaining degrees of freedom that could enable architectural violations.

===============================================================================
A18 — LEARNED STATISTICAL ENCODER (CORRELATION-BASED ADDRESSING)
===============================================================================

A18.0 PHILOSOPHICAL FOUNDATION (READ FIRST)
--------------------------------------------------------------------------------
The agent learns CORRELATION, not causation. Causation is not modeled, not 
inferred, not relevant. The system discovers statistical regularities:
"What patterns tend to co-occur with what outcomes?"

This is compression via learned co-occurrence statistics. If you observe X, 
what else typically appears? If you see pattern P, what prediction error 
typically follows? The encoder learns to produce codes that group 
statistically-similar situations together.

Analogy: Given "The cat sat on the ___", you fill in "mat" not because you 
model cat physics, but because "cat sat on the mat" has high co-occurrence 
in your statistics. The encoder works the same way.

[PHILOSOPHY-RISK] Do not import causal modeling, intervention logic, 
counterfactual reasoning, or structural causal models. This is pure 
statistical correlation learning.

A18.1 ENCODER STRUCTURE
--------------------------------------------------------------------------------
(a) The encoder E is a learned function:
    code(t) = E(y_full(t), context(t)) → ℤ^C  (fixed-size code, e.g., C=64 bits)

(b) "Learned" means:
    - E has trainable parameters θ_E
    - θ_E updates based on outcome-vetted prediction error (A4.5 authority)
    - E discovers which compressions of input correlate with prediction outcomes

(c) The encoder sees:
    - y_full(t): full sensory field (ephemeral, not stored)
    - context(t): prior belief μ(t-1), peripheral gist, committed history summaries
    - NOT future observations, NOT stored raw sensory from past ticks

A18.2 CORRELATION LEARNING OBJECTIVE
--------------------------------------------------------------------------------
(a) The encoder learns codes such that:
    - Similar codes → similar prediction error statistics
    - Dissimilar codes → dissimilar prediction error statistics

(b) Formally, E is trained to minimize:
    L_E = 𝔼[ divergence( err_distribution(code) || err_distribution(code') ) ]
    where code ≈ code' should imply similar error distributions.

(c) Equivalently: codes should cluster situations where the same units 
    (predictors) perform well, and separate situations where different 
    units perform well.

(d) This is NOT:
    - Supervised classification of regime labels (labels are unknown)
    - Causal inference about hidden states
    - Reconstruction loss on inputs (this is not an autoencoder)
    
    This IS:
    - Learning which input patterns correlate with which prediction outcomes
    - Compression that preserves outcome-relevant distinctions

A18.3 UPDATE AUTHORITY
--------------------------------------------------------------------------------
(a) Encoder parameters θ_E update ONLY via outcome-vetted signal:
    - When A4.5 deferred validation fires, the prediction error on validated 
      dims provides gradient signal to E
    - "code(t_pred) should have retrieved units that predicted well for t_due"

(b) Update occurs at validation time, not observation time:
    - At t_due: compute error, backprop to E's parameters that produced code(t_pred)
    - This requires storing code(t_pred) in pending record (bounded, not raw sensory)

(c) Forbidden update sources:
    - Reconstruction loss on y_full
    - Contrastive loss on raw observations
    - Any signal not derived from prediction error

[CHEAT-RISK] Do not add auxiliary losses "to help the encoder." The only 
signal is prediction outcome. If the encoder can't learn from that, the 
architecture needs rethinking, not loss hacking.

A18.4 RETRIEVAL INTEGRATION
--------------------------------------------------------------------------------
(a) Retrieval uses learned codes:
    - Unit u stores creation_code (learned code at time of creation)
    - Query: code(t) from current observation
    - Distance: Hamming, L1, or other fixed metric on code space

(b) The metric is fixed; the codes are learned:
    - No attention mechanism
    - No learned similarity function
    - The encoder learns to produce codes where fixed-metric neighbors are 
      statistically similar situations

(c) This replaces A4.3′ sig64 for retrieval ranking:
    - Stage 1 bucket addressing may still use hash(code) for efficiency
    - Stage 2 scoring uses distance(code(t), unit.creation_code)

A18.5 WHAT THIS IS NOT
--------------------------------------------------------------------------------
(a) NOT an autoencoder:
    - No reconstruction objective
    - Compression is judged by prediction utility, not reconstruction fidelity

(b) NOT a classifier:
    - No regime labels, no supervised signal
    - Clustering emerges from correlation with prediction error

(c) NOT attention/embedding similarity search:
    - Codes are produced by a feedforward encoder, not retrieved via attention
    - Distance metric is fixed (Hamming/L1), not learned

(d) NOT causal modeling:
    - No DAGs, no interventions, no counterfactuals
    - Pure: "what co-occurs with what"

A18.6 IMPLEMENTATION REFERENCE (bounded, incremental, no replay)
--------------------------------------------------------------------------------
(a) Encoder architecture (allowed baseline):
    - Small MLP or conv net: y_full → hidden → code_logits → discretize
    - Parameters bounded by θ_E_MAX
    - Forward pass bounded by B_encode ≤ fixed cost

(b) Discretization:
    - code = sign(code_logits) or quantize(code_logits) 
    - Straight-through estimator for gradient flow if needed

(c) Update (online, bounded):
    - At validation time, compute gradient of prediction error w.r.t. code(t_pred)
    - Backprop through stored code_logits(t_pred) to θ_E
    - Update θ_E with clipped gradient, learning rate η_E
    - No replay buffer; each validation event updates once and discards

(d) Storage in pending record:
    - Store code(t_pred) and code_logits(t_pred) (bounded, fixed size)
    - NOT y_full(t_pred), NOT raw observations

(e) Incremental cost:
    - Encode: O(|y_full| × hidden_size) per tick
    - Update: O(|pending_validated| × encoder_size) per tick
    - Both bounded by existing B_rt budget

[CHEAT-RISK] Do not cache intermediate activations for all pending predictions.
Store only code and code_logits (fixed size per prediction).

A18.7 INTERACTION WITH EXISTING AXIOMS
--------------------------------------------------------------------------------
(a) A0.BUDGET: Encoder forward/backward costs charged to B_rt.

(b) A4.3′: sig64 is replaced by learned code for retrieval scoring.
    Bucket addressing may still use hash(code) for O(1) lookup.

(c) A4.5: Deferred validation now also provides encoder update signal.
    Pending record stores code(t_pred) alongside pred_values.

(d) A13.1 step 5.5: Retrieval computes code(t) via encoder, then 
    Stage 1 + Stage 2 as before but using code distance.

(e) A16.5: Peripheral gist computation remains separate from encoder.
    Encoder may consume gist as part of context(t).

A18.8 DISCRIMINABILITY GATE (REVISED)
--------------------------------------------------------------------------------
The offline discriminability audit (from A4.3′) now tests:

(a) Encoder discriminability:
    - Train encoder on held-out regime-labeled data (labels for AUDIT ONLY)
    - Measure: 1-NN accuracy on code space for regime classification
    - Threshold: best_acc(regime | code) ≥ 0.90

(b) If discriminability fails:
    - Increase encoder capacity (within θ_E_MAX)
    - Increase code size (within C_MAX)
    - Do NOT bypass by removing the encoder

[CHEAT-RISK] The audit uses regime labels that the online system never sees.
This is offline validation, not supervised training signal.

===============================================================================
END OF A18 ADDENDUM
===============================================================================