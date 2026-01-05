# agent.md — NUPCA5 Project Guardrails (for coding agents)

This file is the non-negotiable operating contract for any AI agent modifying this repo. If you violate it, your changes are assumed wrong even if they “work.”

## 0) Prime directive
Do not “simplify” the architecture into something else. Implement **exactly what the axioms require**, using minimal, verifiable changes. If something is structurally broken, stop and report it immediately.

## 1) Ground truth
1) The **NUPCA5 v5.05 or latest axiom list (A0–A18+)** is ground truth.  
2) `test5.py` is the canonical runnable harness unless the user explicitly designates another.  
3) Existing behavior that contradicts axioms is a bug, not a “design choice.”

If you don’t have the axiom list in your context, you must locate it in-repo (or request it) before editing anything.

## 2) Hard “NO” shortcuts (automatic rejection)
Do NOT implement or “approximate” NUPCA5 as any of the following:

- Mixture of Experts (MoE) gating / router networks
- Transformer attention / softmax attention as a substitute for signature-bucket retrieval
- Learned embedding similarity search as the retrieval mechanism
- Replay-buffer training from stored raw sensory (training must be from outcome-validated prediction error only)
- “Just use an LSTM/Transformer” for temporal memory
- “Just add decay/EMA everywhere” as a substitute for the specified validation/debt/novelty mechanics
- Silent changes to semantics (“occupied vs null”, “unobserved vs zero”) to make graphs look nicer

If you think a shortcut is “equivalent,” prove equivalence in terms of the axioms. Otherwise it’s forbidden.

## 3) Structural-failure rule (stop early)
If you detect any of the following, you must stop and report the structural problem with exact file/function locations and why it violates axioms:

- Circular dependencies or updates that violate required ordering (e.g., validations must occur before scoring/retrieval if specified)
- Identifiability/fatal ambiguity: a variable can mean two different things in different code paths
- A subsystem that cannot possibly produce the required signals (e.g., “due validation” exists but nothing ever schedules it)
- Any “always commit” path that forces persistence of stale predictions contrary to purge/validation rules

Do NOT continue “patching around” a broken foundation.

## 4) Workflow: the only acceptable way to change code

### Step A — Reproduce (mandatory)
- Run the canonical test: `python test5.py` (or repo’s documented command).
- Capture the exact traceback + seed/config if present.
- Identify the *first* incorrect state transition (not the final crash).

### Step B — Locate the axiom touchpoints (mandatory)
Before editing, map the failing behavior to specific axiom(s) and the code that should implement them:
- Find the functions that correspond to: observe/perceive, retrieval, predict, commit, validation, novelty/hazard, budgeting, thread mutation.
- Make a short list: “Axiom → file:function → current behavior → required behavior.”

### Step C — Make minimal diffs (mandatory)
- Prefer small targeted edits over rewrites.
- Don’t rename half the repo “for clarity.”
- Don’t introduce new abstractions unless they directly enforce an axiom boundary (e.g., “planning-only write permissions”).

### Step D — Add invariants (mandatory)
Every critical axiom boundary should have at least one invariant check or assertion (cheap, deterministic):
- Ordering invariants (e.g., due-validations processed before scoring)
- Permission invariants (planning-only threads cannot mutate committed state)
- Budget invariants (`B_use(t) <= B_max(t)` enforced deterministically)
- Persistence invariants (no cross-tick full-field sensory; only allowed trace cache)

### Step E — Verify with focused tests (mandatory)
- Rerun `test5.py`.
- If you add a micro-test, it must be small, deterministic, and directly tied to an axiom invariant.
- Do not add “tests that encode the bug” as the new expected behavior.

## 5) Specific NUPCA5 v5.02 non-negotiables (common failure points)
You must not “handwave” these. Implement them concretely.

- **Time-index separation**: world tick vs internal micro-steps vs wallclock. No conflation.
- **Compute budgeting**: enforce `B_use(t) <= B_max(t)` with deterministic degradation if planned spend exceeds capacity. Budget caps must be applied exactly as specified.
- **Contemplation is planning-only**: contemplation must not mutate committed state or residual/debt stats; only planning artifacts allowed.
- **Deferred validation ordering**: if axioms specify validations processed before retrieval scoring, enforce that ordering mechanically.
- **Thread mutation staging**: thread changes must be staged and applied at the correct boundary (no mid-tick implicit mutation).
- **Cross-tick prediction snapshot store**: if required, store snapshots exactly at the boundary specified (not “close enough”).
- **Novelty vs recognized-change split**: do not collapse these into one scalar. If axioms separate them, preserve the split.
- **No raw sensory persistence loophole**: if a foveated trace cache exists, it must be bounded and non-authoritative; prevent reconstructing the full field by accumulation.

If any of these are unclear in code, treat that as a defect and resolve it by aligning to the axioms, not by inventing a new interpretation.

## 6) Semantics: “unobserved” is not “zero”
Do not clamp occluded/unobserved cells to zero unless explicitly required. Preserve “unknown/unobserved” as a distinct state and ensure scoring/reward does not accidentally reward null→null matches unless specified.

## 7) Logging and instrumentation rules
- Logs must be actionable: show state transitions, gating decisions, and invariant violations.
- No megadumps of entire tensors/grids every tick unless explicitly requested.
- If you add debug output, gate it behind a flag and keep defaults quiet.

## 8) Dependency and reuse policy
- Prefer using existing, standard components only when they do **not** alter semantics and do **not** smuggle in forbidden architectures.
- Do not add heavyweight frameworks “because it’s easier.”
- If you propose an external dependency, justify it in one sentence: what axiom it helps enforce, and why it doesn’t change semantics.

## 9) Communication requirements (how you report changes)
When you propose or implement changes, you must provide:
- The exact files and functions to edit
- The precise behavioral change in terms of state transitions and invariants
- Why it satisfies specific axiom(s)
- What could break, and the test that would catch it

Forbidden reporting:
- “Should work now”
- “Cleaner”
- “More robust”
- “Equivalent”
…unless you back it with explicit invariants or tests.

## 10) Completion criteria
A change set is “done” only when:
- `test5.py` passes
- No forbidden shortcuts were introduced
- The modified code path enforces the relevant axiom invariants
- Any previous “legacy placeholder/shim” behavior that violates axioms is removed, not hidden

If you can’t achieve this in the current change set, stop and report the blocker precisely.
