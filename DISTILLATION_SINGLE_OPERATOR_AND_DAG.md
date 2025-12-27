# Distillation: Single Completion Operator + DAG/Constellation Semantics

## 1) Canonical per-step invariant (equation-first)

**State estimate invariants per step t** (names match code variables):

1) **Prior** (prediction):
- prior_t = x_hat(t | t-1)
- sigma_prior_diag = diag(Sigma_global(t | t-1))

2) **Observation** (sparse cue):
- cue_t = {dim -> value} from env_obs.x_partial
- O_req = requested dims derived from selected fovea blocks
- O_t = observed dims after filtering cue_t to O_req

3) **Posterior** (completion/clamp):
- x_post = Complete(prior_t, cue_t, O_t)

4) **Log/metrics**:
- e_obs = obs_vals - prior_t[O_t]
- innovation = x_post - prior_t (mean abs = mean(|innovation|))

5) **Update**:
- buffer.x_last <- x_post
- buffer.x_prior <- prior_t
- buffer.observed_dims <- O_t

All perception/recall/prediction calls use the same **Complete()** operator; the only difference is whether the cue is empty and where the prior comes from.

---

## 2) The single operator (Complete / completion)

### Definition
**Complete(cue, mode, prior, sigma_prior_diag, transport_shift) -> (x_completed, Sigma_prior, prior)**

- **Inputs**
  - prior_t: global prior vector x_hat(t | t-1).
    - Obtained from the previous step’s working set fusion (A7.3), or explicitly supplied.
  - sigma_prior_diag: diagonal of Sigma_global(t | t-1).
  - cue_t: sparse dict {dim -> value} (canonical). Dense-with-NaNs is accepted but immediately coerced to sparse.
  - mode in {perception, recall, prediction}
  - transport_shift: optional (dx, dy); if provided, applies a frame shift to prior and sigma prior before clamping.

- **Observation set**
  - O_t = observed_dims(cue_t) after **filtering to O_req** (A16.5); only dims in O_req are treated as observed.

- **Operator**
  - x_completed = prior_t
  - For k in O_t: x_completed[k] = cue_t[k]
  - Unobserved dims remain **prior_t** (A16.5 stale persistence).

- **Outputs**
  - x_completed_t: the posterior estimate x(t)
  - Sigma_prior_t: the A7 global covariance for the prior at time t (not clamped)
  - prior_t: the prior x_hat(t | t-1)

### Prior vs Posterior (where clamping occurs)
- **Prior** is produced by fusion or supplied explicitly; it is *not* altered by the cue except for the returned x_completed value.
- **Posterior** is computed only by **clamping** the observed cue into the prior (no other modification).

### Recall vs Perception vs Prediction (code paths)
- **Perception:** `complete(cue_t, mode="perception", ...)` called in the main step pipeline after filtering cue to O_req. This produces x_post and keeps the A7 prior intact. 
- **Prediction:** `complete(None, mode="prediction", predicted_prior_t=yhat_tp1, predicted_sigma_diag=...)` is used to pass through the prior with **no overwrite**.
- **Recall:** supported in `complete()` as `mode="recall"`; there is **no separate recall path** in the main step loop, but the operator is identical (cue can be empty or sparse).

### Observation set and peripheral channel guardrails
- `env_obs.x_partial` is the **only** observation channel that affects x_post. It is filtered to O_req (A16.5). 
- `env_obs.x_full` is used **only for diagnostics or transport debugging** when `allow_full_state` is set; it does **not** clamp or override x_post.
- `env_obs.periph_full` is used **only** for routing scores / peripheral metrics and does **not** directly update x_post.

---

## 3) DAG/Constellation semantics (nodes, masks, blocks, selection)

### What a “node/expert/constellation” is
A node is a **masked linear-Gaussian operator** over the abstraction vector x(t):
- **Parameters:** W, b, Sigma (per-dim variance), mask, optional input_mask
- **Prediction:** mu_j = W x + b, masked by input_mask; only mask dims are active
- **Precision:** diagonal precision is derived from Sigma and masked
- **Structural fields:** footprint (block ID), anchors, and DAG parents/children

### Block partition constraint (A16-style)
- The D-dimensional state is partitioned into **B disjoint blocks**.
- Blocks are contiguous index ranges with remainder distributed across early blocks.
- Optional peripheral blocks are **appended at the end**; peripheral dims are the final dims of x (D - periph_size .. D-1).
- `mask` footprints are expected to lie entirely within a block for footprint inference.

### DAG / constellation composition
- **DAG structure** exists via node.parents and node.children.
- The DAG is **not used for prediction ordering**; it is used for salience (out-degree) and structural edits.
- Only REST-time structural edits modify DAG edges (parents/children are detached on deletion).

### Working set (candidate pool / selection)
- **Salience:** computed per node using reliability, relevance to observed dims, and context signals.
- **Candidate pool U_t:**
  - A_{t-1} (previous active nodes)
  - Retrieval candidates from incumbents keyed to current fovea blocks
  - Anchors (always included if they fit budget)
- **Selection:** GreedySelect by score (a_j * pi_j) / L_j with budgets on count and load.
- **Anchors:** force-included first; if anchors exceed budget, they are truncated by cost.

### Constellation / block footprint
- Each node has `mask` and optional `input_mask` to define its footprint.
- Footprints are associated with blocks; retrieval candidates are drawn by block (incumbents per block).
- Node masks define which dimensions are covered and thus how fusion and clamping behave.

---

## 4) One step through the main loop (pseudocode)

```
# Inputs: state, env_obs, cfg
D = cfg.D
x_prev = state.buffer.x_last

# (1) Select fovea blocks using t-1 tracking
blocks_t = select_fovea(state.fovea, cfg)
O_req = make_observation_set(blocks_t, cfg)

# (2) Filter observation to requested dims
cue_raw = env_obs.x_partial  # sparse dict
cue_t = filter(cue_raw, dims in O_req and [0,D))
O_t = keys(cue_t)
obs_idx, obs_vals = sorted(O_t), cue_t[obs_idx]

# (3) Optional transport shift (diagnostic / motion hypothesis)
shift = compute transport shift from coarse cues and history

# (4) Completion operator (perception)
# prior_t comes from cached fusion or recomputed fusion
x_post, Sigma_prior, prior_t = complete(cue_t, mode="perception", transport_shift=shift)

# (5) Compute metrics
error_vec[O_t] = obs_vals - prior_t[O_t]   # e_obs
innovation = x_post - prior_t              # clamp delta
innovation_mean_abs = mean(abs(innovation))

# (6) Update observation buffer
state.buffer.x_prior = prior_t
state.buffer.x_last = x_post
state.buffer.observed_dims = O_t

# (7) Salience + working set
salience = compute_salience(state, observed_dims=O_t)
A_t = select_working_set(state, salience)
state.active_set = A_t.active

# (8) Update fovea tracking from observed error
update_fovea_tracking(state.fovea, state.buffer, abs_error=abs(error_vec))

# (9) Prediction for t+1 (same operator, no cue)
yhat_tp1, Sigma_tp1 = fuse_predictions(state.library, A_t, state.buffer, O_t)
# apply Complete() in prediction mode (no overwrite)
yhat_tp1, Sigma_tp1_pred, _ = complete(None, mode="prediction", predicted_prior_t=yhat_tp1,
                                       predicted_sigma_diag=diag(Sigma_tp1))

# (10) Cache learn context and continue with control, rollout, REST, etc.
state.learn_cache = {x_t, yhat_tp1, sigma_tp1_diag, A_t, ...}
```

---

## 5) Terminology mapping (repo name -> conceptual name)

- **completion / complete()** -> cue integration / clamp operator
- **buffer.x_last** -> current posterior state estimate x(t)
- **buffer.x_prior** -> prior estimate x_hat(t | t-1)
- **x_partial / cue_t** -> observation samples (sparse)
- **O_req** -> requested observation set from fovea blocks
- **O_t** -> actual observed dims after filtering
- **WorkingSet (A_t)** -> active constellation set
- **ExpertNode** -> node/constellation operator (masked linear dynamics)
- **mask / input_mask** -> footprint selector for node influence
- **fovea blocks (F_t)** -> observation budget / attention blocks
- **incumbents** -> per-block node indices (retrieval candidates)
- **transport** -> optional frame shift for alignment before clamping
- **periph_full** -> peripheral coarse channel (diagnostic/routing only)

---

## 6) Appendix: minimum source functions/classes used

- nupca3/memory/completion.py
  - complete, apply_cue, _prior_from_cache_or_fusion
- nupca3/step_pipeline_parts/part3.part
  - step_pipeline (perception callsite), O_req/O_t filtering, metrics
- nupca3/step_pipeline_parts/part4.part
  - prediction callsite (complete in prediction mode)
- nupca3/memory/fusion.py
  - fuse_predictions
- nupca3/memory/expert.py
  - predict, precision_vector, sgd_update
- nupca3/memory/working_set.py
  - select_working_set, get_retrieval_candidates
- nupca3/types.py
  - ExpertNode, ExpertLibrary, ObservationBuffer, WorkingSet
- nupca3/geometry/fovea.py
  - build_blocks_from_cfg, make_observation_set, update_fovea_tracking
- nupca3/edits/rest_processor.py
  - DAG hygiene for parents/children
