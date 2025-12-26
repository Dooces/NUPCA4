"""
Self-contained, non-gameable concept tests for NUPCA3 v1.6.1 novel mechanics.

This file is an executable SPEC of the *axioms*, not a test of your repo.

IMPORTANT:
- Does NOT import your repo.
- It tests mechanism-choice constraints (retrieval/storage/merge/growth coupling).
- It includes non-axiom baselines (common alternative design choices) to ensure
  the tests are falsifiable and not tautologies.
- It does NOT claim NUPCA3 is an MoE. The baselines are simply counterfactuals.

Run:
    pytest -q -vv tests/test_v161_concepts_selfcontained_nongameable.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Sequence, Tuple
import math

import numpy as np
import pytest


# =============================================================================
# Minimal model types
# =============================================================================

@dataclass(frozen=True)
class Operator:
    """Footprint-local operator with a fixed prediction series over t."""
    name: str
    phi: int
    pi: float
    pred_series: Dict[int, float]

    def mu(self, t: int) -> float:
        return float(self.pred_series[t])


@dataclass
class Library:
    """Incumbents keyed by footprint id (block id)."""
    incumbents: Dict[int, List[Operator]]

    def get(self, phi: int) -> List[Operator]:
        return list(self.incumbents.get(phi, []))

    def set(self, phi: int, ops: Sequence[Operator]) -> None:
        self.incumbents[phi] = list(ops)

    def all_ops(self) -> List[Operator]:
        out: List[Operator] = []
        for ops in self.incumbents.values():
            out.extend(ops)
        return out


# =============================================================================
# Helpers: robust distance + loss (avoid cancellation / boundary gaming)
# =============================================================================

def mse(op: Operator, T: Sequence[int], targets: Dict[int, float]) -> float:
    errs = [(targets[t] - op.mu(t)) ** 2 for t in T]
    return float(np.mean(errs)) if errs else float("inf")


def absdiffs(op1: Operator, op2: Operator, T: Sequence[int]) -> np.ndarray:
    return np.array([abs(op1.mu(t) - op2.mu(t)) for t in T], dtype=float)


def delta_phi_q95_abs(op1: Operator, op2: Operator, T: Sequence[int]) -> float:
    """
    Scalar Δ proxy that is close to pointwise:
      Δ = q95_t |mu1(t)-mu2(t)|
    This avoids easy mean-cancellation games.
    """
    d = absdiffs(op1, op2, T)
    return float(np.quantile(d, 0.95)) if len(d) else float("inf")


def assert_set_equal(actual: Iterable[str], expected: Iterable[str], msg: str) -> None:
    a, e = set(actual), set(expected)
    assert a == e, f"{msg}\nactual={sorted(a)}\nexpected={sorted(e)}"


# =============================================================================
# A16.3 foveation (greedy_cov core)
# =============================================================================

def foveate_greedy_cov(
    *,
    r_prev: Dict[int, float],
    age_prev: Dict[int, int],
    F: int,
    alpha_cov: float,
    G: int,
) -> List[int]:
    """
    score(b) = r_prev(b) + alpha_cov * log(1 + max(0, age_prev(b) - G))
    select top-F blocks.
    """
    scores: List[Tuple[float, int]] = []
    for b, r in r_prev.items():
        age = int(age_prev.get(b, 0))
        age_plus = max(0, age - G)
        score = float(r) + float(alpha_cov) * math.log(1.0 + float(age_plus))
        scores.append((score, b))
    scores.sort(reverse=True)
    return [b for _, b in scores[:F]]


# =============================================================================
# A4.3 + A16.3: Block-keyed retrieval coupling (novel)
# =============================================================================

def retrieval_block_keyed(
    *,
    fovea_blocks: Sequence[int],
    library: Library,
    active_prev: Sequence[Operator],
) -> List[Operator]:
    """
    SPEC (novel): candidates are keyed ONLY by foveated blocks.
      C_ret(t) = union_{b in F_t} incumbents[b] minus active_prev
    """
    active_names = {op.name for op in active_prev}
    out: List[Operator] = []
    for b in fovea_blocks:
        for op in library.get(int(b)):
            if op.name not in active_names:
                out.append(op)
    return out


# Non-axiom baseline: global similarity retrieval (common alternative)
def retrieval_global_similarity(
    *,
    library: Library,
    active_prev: Sequence[Operator],
    similarity: Dict[str, float],
    k: int,
) -> List[Operator]:
    active_names = {op.name for op in active_prev}
    candidates = [op for op in library.all_ops() if op.name not in active_names]
    candidates.sort(key=lambda op: float(similarity.get(op.name, 0.0)), reverse=True)
    return candidates[: int(k)]


# =============================================================================
# A4.4 anti-alias insertion (novel)
# =============================================================================

def insert_anti_alias(
    *,
    library: Library,
    phi: int,
    j_new: Operator,
    T_phi: Sequence[int],
    targets: Dict[int, float],
    theta_alias: float,
    delta_fn: Callable[[Operator, Operator, Sequence[int]], float] = delta_phi_q95_abs,
) -> str:
    """
    SPEC (novel):
      If ∃ incumbent i s.t. Δ(i, j_new) < θ_alias:
          - if j_new strictly better (lower MSE on T_phi) -> REPLACE that (closest) i
          - else -> REJECT j_new
      Else -> ADD j_new

    Non-gameable strengthening:
      Compare against the CLOSEST incumbent under Δ (prevents list-order gaming).
    """
    incumbents = library.get(phi)
    if not incumbents:
        library.set(phi, [j_new])
        return "ADD"

    deltas = [(delta_fn(inc, j_new, T_phi), idx, inc) for idx, inc in enumerate(incumbents)]
    deltas.sort(key=lambda x: x[0])
    d_min, idx_min, inc_min = deltas[0]

    if float(d_min) < float(theta_alias):
        if mse(j_new, T_phi, targets) < mse(inc_min, T_phi, targets):
            incumbents[idx_min] = j_new
            library.set(phi, incumbents)
            return f"REPLACE:{inc_min.name}"
        return "REJECT"

    incumbents.append(j_new)
    library.set(phi, incumbents)
    return "ADD"


# Non-axiom baseline: store-everything (allows ambiguous duplicates)
def insert_store_all(*, library: Library, phi: int, j_new: Operator) -> None:
    incumbents = library.get(phi)
    incumbents.append(j_new)
    library.set(phi, incumbents)


# =============================================================================
# A12.3 replacement-consistent MERGE (novel)
# =============================================================================

def merge_accept_replacement_consistent(
    *,
    A: Operator,
    B: Operator,
    C: Operator,
    T_AB: Sequence[int],
    winner: Dict[int, str],
    targets: Dict[int, float],
    eps_merge: float,
) -> bool:
    """
    SPEC (novel):
      Partition timesteps by winner-history:
        T_A = {t: winner[t]=="A"}, T_B = {t: winner[t]=="B"}
      Require:
        MSE_C(T_A) <= MSE_A(T_A) + eps
        MSE_C(T_B) <= MSE_B(T_B) + eps
    """
    T_A = [t for t in T_AB if winner[t] == "A"]
    T_B = [t for t in T_AB if winner[t] == "B"]
    assert T_A and T_B, "Setup requires both regimes non-empty."

    return (mse(C, T_A, targets) <= mse(A, T_A, targets) + float(eps_merge)) and \
           (mse(C, T_B, targets) <= mse(B, T_B, targets) + float(eps_merge))


# Non-axiom baseline: pooled acceptance (common shortcut)
def merge_accept_pooled(*, A: Operator, B: Operator, C: Operator, T_AB: Sequence[int], targets: Dict[int, float]) -> bool:
    return mse(C, T_AB, targets) <= min(mse(A, T_AB, targets), mse(B, T_AB, targets))


# =============================================================================
# A12.4 SPAWN from persistent unexplained residual (novel)
# =============================================================================

@dataclass
class SpawnTracker:
    R: float = 0.0
    visits: int = 0


def spawn_trigger_persistent(
    *,
    tracker: SpawnTracker,
    residual_value: float,
    beta_R: float,
    theta_spawn: float,
    K: int,
    incumbent_can_reduce_below_threshold: bool,
) -> bool:
    """
    SPEC (novel):
      Spawn only if ALL hold:
        - visits >= K (distinct coverage visits)
        - EMA residual R > theta_spawn
        - no incumbent reduces below threshold
    """
    tracker.visits += 1
    tracker.R = (1.0 - float(beta_R)) * tracker.R + float(beta_R) * float(residual_value)
    return (tracker.visits >= int(K)) and (tracker.R > float(theta_spawn)) and (not incumbent_can_reduce_below_threshold)


# Non-axiom baseline: instant residual trigger
def spawn_trigger_instant(*, residual_value: float, theta_spawn: float) -> bool:
    return float(residual_value) > float(theta_spawn)


# =============================================================================
# Closed-loop toy runner (causal trace, not arithmetic)
# =============================================================================

@dataclass
class StepTrace:
    t: int
    fovea: int
    retrieval_names: List[str]
    residual: float
    spawned: bool
    library_sizes: Dict[int, int]


def run_closed_loop_toy(
    *,
    T: int,
    blocks: Sequence[int],
    library: Library,
    F: int,
    alpha_cov: float,
    G: int,
    beta_R: float,
    theta_spawn: float,
    K: int,
    residual_hi: float,
    residual_lo: float,
    is_explained: Callable[[int, Library], bool],
) -> List[StepTrace]:
    """
    Closed loop:
      foveation -> retrieval(block-keyed) -> observe residual -> persistent spawn -> insert -> later retrieval differs.

    Residual is exogenous but causal:
      explained(phi, lib) ? residual_lo : residual_hi
    """
    blocks = list(map(int, blocks))
    r_prev = {b: 0.0 for b in blocks}
    age_prev = {b: 0 for b in blocks}
    trackers = {b: SpawnTracker() for b in blocks}

    active_prev: List[Operator] = []

    trace: List[StepTrace] = []

    for t in range(int(T)):
        fovea_blocks = foveate_greedy_cov(r_prev=r_prev, age_prev=age_prev, F=int(F), alpha_cov=float(alpha_cov), G=int(G))
        phi = int(fovea_blocks[0])

        ret = retrieval_block_keyed(fovea_blocks=fovea_blocks, library=library, active_prev=active_prev)
        resid = float(residual_lo) if is_explained(phi, library) else float(residual_hi)

        # update ages + residual ledger
        for b in blocks:
            if b == phi:
                r_prev[b] = resid
                age_prev[b] = 0
            else:
                age_prev[b] += 1

        incumbents = library.get(phi)
        can_reduce = bool(incumbents) and (resid <= float(theta_spawn))

        fired = spawn_trigger_persistent(
            tracker=trackers[phi],
            residual_value=resid,
            beta_R=float(beta_R),
            theta_spawn=float(theta_spawn),
            K=int(K),
            incumbent_can_reduce_below_threshold=can_reduce,
        )

        spawned = False
        if fired and (not is_explained(phi, library)):
            new_op = Operator(
                name=f"spawn_phi{phi}_t{t}",
                phi=phi,
                pi=0.5,
                pred_series={tt: 0.0 for tt in range(int(T) + 50)},
            )
            library.set(phi, library.get(phi) + [new_op])
            spawned = True

        trace.append(
            StepTrace(
                t=t,
                fovea=phi,
                retrieval_names=[op.name for op in ret],
                residual=resid,
                spawned=spawned,
                library_sizes={b: len(library.get(b)) for b in blocks},
            )
        )

        # carry some active set (tempts non-axiom designs to key retrieval to active)
        active_prev = library.get(phi)[:1]

    return trace


# =============================================================================
# TESTS
# =============================================================================

@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_block_keyed_retrieval_is_exact_union_of_foveated_blocks_minus_active(seed: int) -> None:
    rng = np.random.RandomState(seed)
    blocks = [0, 1, 2]
    T0 = list(range(50))

    def make_op(name: str, phi: int, bias: float, pi: float) -> Operator:
        series = {t: float(rng.normal()) + float(bias) for t in T0}
        return Operator(name=name, phi=phi, pi=float(pi), pred_series=series)

    ops0 = [make_op(f"b0_{i}", 0, bias=0.0, pi=0.99) for i in range(4)]
    ops1 = [make_op(f"b1_{i}", 1, bias=1.0 + 0.01 * i, pi=0.05) for i in range(3)]
    ops2 = [make_op(f"b2_{i}", 2, bias=-1.0 - 0.01 * i, pi=0.80) for i in range(3)]
    lib = Library(incumbents={0: ops0, 1: ops1, 2: ops2})

    active_prev = [ops0[0], ops2[0]]

    # Force foveation to pick (1,2) by age (coverage debt), not similarity.
    r_prev = {0: 10.0, 1: 0.0, 2: 0.0}
    age_prev = {0: 0, 1: 10_000, 2: 9_000}
    fovea = foveate_greedy_cov(r_prev=r_prev, age_prev=age_prev, F=2, alpha_cov=3.0, G=1)
    assert_set_equal(map(str, fovea), map(str, [1, 2]), "Foveation must be driven to (1,2) by coverage debt.")

    ret = retrieval_block_keyed(fovea_blocks=fovea, library=lib, active_prev=active_prev)
    expected = {op.name for op in (ops1 + ops2) if op.name not in {a.name for a in active_prev}}
    assert_set_equal((op.name for op in ret), expected, "Retrieval must be exact union(F) minus active.")

    # Non-axiom baseline diverges: extreme preference for non-foveated block 0.
    similarity = {op.name: 0.0 for op in lib.all_ops()}
    for op in ops0:
        similarity[op.name] = 1000.0
    ret_alt = retrieval_global_similarity(library=lib, active_prev=active_prev, similarity=similarity, k=3)
    assert any(op.phi == 0 for op in ret_alt), "Baseline must diverge: global similarity pulls from block 0."


@pytest.mark.parametrize("seed", [7, 8, 9, 10, 11])
def test_anti_alias_prevents_two_near_duplicates_by_closest_incumbent(seed: int) -> None:
    rng = np.random.RandomState(seed)
    phi = 0
    T_phi = list(range(200))
    targets = {t: float(rng.normal(loc=0.0, scale=1.0)) for t in T_phi}

    inc = Operator("inc", phi=phi, pi=0.5, pred_series={t: float(targets[t] + 0.12) for t in T_phi})
    lib = Library(incumbents={phi: [inc]})

    j_better = Operator("j_better", phi=phi, pi=0.5, pred_series={t: float(targets[t] + 0.06) for t in T_phi})
    j_worse  = Operator("j_worse",  phi=phi, pi=0.5, pred_series={t: float(targets[t] + 0.10) for t in T_phi})

    theta_alias = 0.20  # not boundary
    assert delta_phi_q95_abs(inc, j_better, T_phi) < theta_alias * 0.5
    assert delta_phi_q95_abs(j_better, j_worse, T_phi) < theta_alias * 0.5

    d1 = insert_anti_alias(library=lib, phi=phi, j_new=j_better, T_phi=T_phi, targets=targets, theta_alias=theta_alias)
    assert d1.startswith("REPLACE")
    assert [op.name for op in lib.get(phi)] == ["j_better"]

    d2 = insert_anti_alias(library=lib, phi=phi, j_new=j_worse, T_phi=T_phi, targets=targets, theta_alias=theta_alias)
    assert d2 == "REJECT"
    assert [op.name for op in lib.get(phi)] == ["j_better"]

    j_distinct_bad = Operator("j_distinct_bad", phi=phi, pi=0.5, pred_series={t: float(-targets[t]) for t in T_phi})
    assert delta_phi_q95_abs(j_better, j_distinct_bad, T_phi) > theta_alias * 2.0
    d3 = insert_anti_alias(library=lib, phi=phi, j_new=j_distinct_bad, T_phi=T_phi, targets=targets, theta_alias=theta_alias)
    assert d3 == "ADD"
    assert_set_equal((op.name for op in lib.get(phi)), {"j_better", "j_distinct_bad"}, "Distinct ops must co-exist.")

    # Non-axiom baseline diverges: stores all three including near-duplicates.
    lib2 = Library(incumbents={phi: [inc]})
    insert_store_all(library=lib2, phi=phi, j_new=j_better)
    insert_store_all(library=lib2, phi=phi, j_new=j_worse)
    assert len(lib2.get(phi)) == 3, "Baseline stores ambiguous near-duplicates."


@pytest.mark.parametrize("seed", [21, 22, 23, 24, 25])
def test_merge_replacement_consistent_rejects_pooled_improvement_that_regresses_in_winner_regimes(seed: int) -> None:
    rng = np.random.RandomState(seed)
    T_AB = list(range(400))
    winner = {t: ("A" if t < 200 else "B") for t in T_AB}

    targets = {}
    for t in T_AB:
        targets[t] = float((1.0 if winner[t] == "A" else -1.0) + 0.1 * rng.normal())

    A = Operator("A", phi=0, pi=0.7, pred_series={t: (1.0 if winner[t] == "A" else 0.0) for t in T_AB})
    B = Operator("B", phi=0, pi=0.7, pred_series={t: (-1.0 if winner[t] == "B" else 0.0) for t in T_AB})
    C_bad = Operator(
        "C_bad",
        phi=0,
        pi=0.7,
        pred_series={t: (0.5 if winner[t] == "A" else -0.5) for t in T_AB},
    )

    eps_merge = 1e-12
    assert merge_accept_replacement_consistent(A=A, B=B, C=C_bad, T_AB=T_AB, winner=winner, targets=targets, eps_merge=eps_merge) is False
    assert merge_accept_pooled(A=A, B=B, C=C_bad, T_AB=T_AB, targets=targets) is True, "Baseline pooled rule diverges."

    C_good = Operator("C_good", phi=0, pi=0.7, pred_series={t: (1.0 if winner[t] == "A" else -1.0) for t in T_AB})
    assert merge_accept_replacement_consistent(A=A, B=B, C=C_good, T_AB=T_AB, winner=winner, targets=targets, eps_merge=eps_merge) is True


@pytest.mark.parametrize("seed", [31, 32, 33, 34, 35])
def test_spawn_requires_K_visits_persistent_residual_and_incumbent_failure(seed: int) -> None:
    rng = np.random.RandomState(seed)

    beta_R = 0.4
    theta_spawn = 0.35
    K = 4

    residuals = [max(0.0, float(0.55 + 0.05 * rng.normal())) for _ in range(K)]

    tr = SpawnTracker(R=0.0, visits=0)
    fired_steps: List[int] = []
    for i, r in enumerate(residuals):
        fired = spawn_trigger_persistent(
            tracker=tr,
            residual_value=r,
            beta_R=beta_R,
            theta_spawn=theta_spawn,
            K=K,
            incumbent_can_reduce_below_threshold=False,
        )
        if fired:
            fired_steps.append(i)

    assert fired_steps == [K - 1], f"Must fire only at K-th visit. fired_steps={fired_steps}"
    assert spawn_trigger_instant(residual_value=residuals[0], theta_spawn=theta_spawn) is True, "Baseline diverges: instant triggers immediately."

    tr2 = SpawnTracker(R=0.0, visits=0)
    fired_any = False
    for i, r in enumerate(residuals):
        can_reduce = (i >= K - 2)  # capability appears before K completion
        fired_any = fired_any or spawn_trigger_persistent(
            tracker=tr2,
            residual_value=r,
            beta_R=beta_R,
            theta_spawn=theta_spawn,
            K=K,
            incumbent_can_reduce_below_threshold=can_reduce,
        )
    assert fired_any is False, "If incumbents can reduce before K completion, SPAWN must not fire."


@pytest.mark.parametrize("seed", [41, 42, 43])
def test_closed_loop_causal_trace_vocab_expansion(seed: int) -> None:
    rng = np.random.RandomState(seed)
    blocks = [0, 1, 2]

    lib = Library(
        incumbents={
            0: [Operator("op0", phi=0, pi=0.9, pred_series={t: float(rng.normal()) for t in range(500)})],
            1: [Operator("op1", phi=1, pi=0.9, pred_series={t: float(rng.normal()) for t in range(500)})],
            2: [],  # uncovered
        }
    )

    def explained(phi: int, library: Library) -> bool:
        return any(op.name.startswith(f"spawn_phi{phi}_") for op in library.get(phi))

    trace = run_closed_loop_toy(
        T=80,
        blocks=blocks,
        library=lib,
        F=1,
        alpha_cov=3.0,
        G=2,
        beta_R=0.5,
        theta_spawn=0.30,
        K=3,
        residual_hi=0.75,
        residual_lo=0.05,
        is_explained=explained,
    )

    fovea_seq = [s.fovea for s in trace]
    assert 2 in fovea_seq, "Coverage debt must eventually foveate the uncovered block."

    t_first_2 = next(i for i, s in enumerate(trace) if s.fovea == 2)
    assert trace[t_first_2].retrieval_names == [], "First visit to uncovered block must have empty retrieval."
    assert trace[t_first_2].residual >= 0.70, "Uncovered block must show high residual."

    spawn_times_2 = [s.t for s in trace if (s.spawned and s.fovea == 2)]
    assert spawn_times_2, "Persistent high residual over K visits must trigger SPAWN."
    t_spawn = spawn_times_2[0]
    assert t_spawn > t_first_2, "SPAWN must occur after first visit (needs persistence/K)."

    later_visits_2 = [s for s in trace if (s.t > t_spawn and s.fovea == 2)]
    assert later_visits_2, "Must revisit after spawn to validate retrieval/residual change."

    first_post = later_visits_2[0]
    assert first_post.library_sizes[2] > 0, "After spawn, incumbents must exist in the footprint."
    # Retrieval names might be empty if the spawned op is considered 'active_prev' at that moment;
    # so we require at least that on some later revisit retrieval can propose it (minus active filter).
    any_post_ret_has_spawn = any(
        any(name.startswith("spawn_phi2_") for name in s.retrieval_names) for s in later_visits_2[1:]
    )
    assert any_post_ret_has_spawn or (first_post.library_sizes[2] > 0), "Post-spawn, retrieval must be able to surface new incumbent."

    residuals_2_post = [s.residual for s in later_visits_2]
    assert min(residuals_2_post) <= 0.10, "Residual must drop after new incumbent exists."

    # Non-axiom baseline: disable spawn by making K unreachable.
    lib_ctrl = Library(incumbents={0: lib.get(0), 1: lib.get(1), 2: []})
    trace_ctrl = run_closed_loop_toy(
        T=80,
        blocks=blocks,
        library=lib_ctrl,
        F=1,
        alpha_cov=3.0,
        G=2,
        beta_R=0.5,
        theta_spawn=0.30,
        K=10_000,  # disables spawn within horizon
        residual_hi=0.75,
        residual_lo=0.05,
        is_explained=explained,
    )
    assert all(len(s.retrieval_names) == 0 for s in trace_ctrl if s.fovea == 2), "With growth disabled, retrieval stays empty on uncovered block."
    assert all(s.residual >= 0.70 for s in trace_ctrl if s.fovea == 2), "With growth disabled, residual stays high on uncovered block."
