"""
Self-contained concept tests for NUPCA3 v1.6.1 novel mechanics.

IMPORTANT:
- This file does NOT import or depend on your repo code.
- It includes a tiny reference implementation of the semantics under test.
- It also includes "standard MoE" control behaviors that must differ.

Run:
    pytest -q tests/test_v161_concepts_selfcontained.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import math

import numpy as np


# =============================================================================
# Minimal reference model (just enough to test semantics)
# =============================================================================

@dataclass(frozen=True)
class Operator:
    """
    A footprint-local operator.

    pred_series[t] is the predicted value for the footprint at timestep t
    (we treat that as mu(t+1|t) for testing; indexing details are irrelevant here).
    """
    name: str
    phi: int
    pi: float
    pred_series: Dict[int, float]

    def mu(self, t: int) -> float:
        return float(self.pred_series[t])


@dataclass
class Library:
    """Incumbents keyed by footprint (block id)."""
    incumbents: Dict[int, List[Operator]]

    def get(self, phi: int) -> List[Operator]:
        return list(self.incumbents.get(phi, []))

    def set(self, phi: int, ops: List[Operator]) -> None:
        self.incumbents[phi] = list(ops)

    def all_ops(self) -> List[Operator]:
        out: List[Operator] = []
        for ops in self.incumbents.values():
            out.extend(ops)
        return out


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
    fovea_blocks: List[int],
    library: Library,
    active_prev: List[Operator],
) -> List[Operator]:
    """
    Novel requirement:
      Candidate pool is keyed ONLY by the foveated blocks.
      It is NOT keyed by active set similarity or state similarity.

    Here we model: union_{b in F_t} incumbents[b] \ active_prev
    """
    active_names = {op.name for op in active_prev}
    out: List[Operator] = []
    for b in fovea_blocks:
        for op in library.get(b):
            if op.name not in active_names:
                out.append(op)
    return out


# Control: MoE-like retrieval by similarity (not fovea keyed)
def retrieval_moe_similarity(
    *,
    library: Library,
    active_prev: List[Operator],
    similarity: Dict[str, float],
    k: int = 2,
) -> List[Operator]:
    active_names = {op.name for op in active_prev}
    candidates = [op for op in library.all_ops() if op.name not in active_names]
    candidates.sort(key=lambda op: float(similarity.get(op.name, 0.0)), reverse=True)
    return candidates[:k]


# =============================================================================
# A4.4 anti-alias insertion (novel)
# =============================================================================

def mse(op: Operator, T: List[int], targets: Dict[int, float]) -> float:
    errs = [(targets[t] - op.mu(t)) ** 2 for t in T]
    return float(np.mean(errs)) if errs else float("inf")


def delta_phi(op1: Operator, op2: Operator, T: List[int]) -> float:
    diffs = [abs(op1.mu(t) - op2.mu(t)) for t in T]
    return float(np.mean(diffs)) if diffs else float("inf")


def insert_anti_alias(
    *,
    library: Library,
    phi: int,
    j_new: Operator,
    T_phi: List[int],
    targets: Dict[int, float],
    theta_alias: float,
) -> str:
    """
    Novel requirement:
      If Δ_φ(i, j_new) < θ_alias for some incumbent i:
          - if j_new strictly better -> REPLACE i
          - else -> REJECT j_new
      Else -> ADD j_new

    Returns: "REPLACE:<inc_name>" | "REJECT" | "ADD"
    """
    incumbents = library.get(phi)

    for idx, inc in enumerate(incumbents):
        if delta_phi(inc, j_new, T_phi) < theta_alias:
            if mse(j_new, T_phi, targets) < mse(inc, T_phi, targets):
                incumbents[idx] = j_new
                library.set(phi, incumbents)
                return f"REPLACE:{inc.name}"
            return "REJECT"

    incumbents.append(j_new)
    library.set(phi, incumbents)
    return "ADD"


# Control: store everything (allows ambiguous duplicates)
def insert_moe_allow_ambiguity(*, library: Library, phi: int, j_new: Operator) -> None:
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
    T_AB: List[int],
    aA: Dict[int, float],
    aB: Dict[int, float],
    targets: Dict[int, float],
    eps_merge: float,
) -> bool:
    """
    Novel requirement:
      Partition timesteps by dominance:
        T_A = {t: aA(t) >= aB(t)}, T_B = rest
      Require:
        MSE_C(T_A) <= MSE_A(T_A) + eps
        MSE_C(T_B) <= MSE_B(T_B) + eps
    """
    T_A = [t for t in T_AB if float(aA[t]) >= float(aB[t])]
    T_B = [t for t in T_AB if t not in set(T_A)]
    if not T_A or not T_B:
        raise AssertionError("Test setup requires both regimes non-empty.")

    return (mse(C, T_A, targets) <= mse(A, T_A, targets) + eps_merge) and \
           (mse(C, T_B, targets) <= mse(B, T_B, targets) + eps_merge)


# Control: pooled MSE acceptance (common non-axiom shortcut)
def merge_accept_pooled(*, A: Operator, B: Operator, C: Operator, T_AB: List[int], targets: Dict[int, float]) -> bool:
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
    Novel requirement:
      Spawn only if:
        - visits >= K (distinct coverage visits)
        - EMA residual R > theta_spawn
        - no incumbent reduces below threshold
    """
    tracker.visits += 1
    tracker.R = (1.0 - beta_R) * tracker.R + beta_R * float(residual_value)

    if tracker.visits >= K and tracker.R > theta_spawn and (not incumbent_can_reduce_below_threshold):
        return True
    return False


# Control: naive spawn on instant residual
def spawn_trigger_instant(*, residual_value: float, theta_spawn: float) -> bool:
    return float(residual_value) > theta_spawn


# =============================================================================
# Closed-loop toy runner for loop-level semantic (novel)
# =============================================================================

def run_closed_loop_toy(
    *,
    T: int,
    blocks: List[int],
    library: Library,
    F: int,
    alpha_cov: float,
    G: int,
    beta_R: float,
    theta_spawn: float,
    K: int,
    residual_if_uncovered: float,
    residual_if_covered: float,
) -> Dict[str, object]:
    """
    Full-loop toy:
      foveation -> retrieval -> (residual observation) -> persistent spawn -> insert -> retrieval changes.
    Residual is exogenous based solely on "has incumbents" to avoid testing fusion/residual math.
    """
    r_prev = {b: 0.0 for b in blocks}
    age_prev = {b: 0 for b in blocks}
    trackers = {b: SpawnTracker() for b in blocks}

    trace_fovea: List[int] = []
    trace_ret_sizes: List[int] = []
    trace_resid: List[Tuple[int, float]] = []
    trace_spawns: List[int] = []

    active_prev: List[Operator] = []

    for t in range(T):
        fovea = foveate_greedy_cov(r_prev=r_prev, age_prev=age_prev, F=F, alpha_cov=alpha_cov, G=G)
        b = fovea[0]
        trace_fovea.append(b)

        ret = retrieval_block_keyed(fovea_blocks=fovea, library=library, active_prev=active_prev)
        trace_ret_sizes.append(len(ret))

        incumbents = library.get(b)
        resid = residual_if_covered if incumbents else residual_if_uncovered
        trace_resid.append((b, resid))

        # update ages + residuals
        for bb in blocks:
            if bb == b:
                r_prev[bb] = resid
                age_prev[bb] = 0
            else:
                age_prev[bb] += 1

        tracker = trackers[b]
        incumbent_can_reduce = bool(incumbents) and (resid <= theta_spawn)

        if spawn_trigger_persistent(
            tracker=tracker,
            residual_value=resid,
            beta_R=beta_R,
            theta_spawn=theta_spawn,
            K=K,
            incumbent_can_reduce_below_threshold=incumbent_can_reduce,
        ) and (not incumbents):
            # "spawn" inserts one new incumbent
            new_op = Operator(
                name=f"spawn_phi{b}_t{t}",
                phi=b,
                pi=0.5,
                pred_series={tt: 0.0 for tt in range(T + 10)},
            )
            library.set(b, library.get(b) + [new_op])
            trace_spawns.append(b)

    return {
        "trace_fovea": trace_fovea,
        "trace_ret_sizes": trace_ret_sizes,
        "trace_resid": trace_resid,
        "trace_spawns": trace_spawns,
        "library": library,
    }


# =============================================================================
# TESTS
# =============================================================================

def test_block_keyed_retrieval_coupling_keys_only_to_fovea_blocks():
    """
    Novel claim:
      Retrieval pool is exactly incumbents from foveated block(s), not from active set or similarity.

    Control:
      MoE-style similarity retrieval will pull from a different block if similarity says so.
    """
    lib = Library(
        incumbents={
            0: [
                Operator("op0_hi", phi=0, pi=0.99, pred_series={0: 0.0}),
                Operator("op0_mid", phi=0, pi=0.80, pred_series={0: 0.0}),
            ],
            1: [
                Operator("op1_low", phi=1, pi=0.05, pred_series={0: 0.0}),
                Operator("op1_low2", phi=1, pi=0.01, pred_series={0: 0.0}),
            ],
        }
    )
    active_prev = [lib.get(0)[0]]  # active includes best op on block 0

    # Force foveation to choose block 1 via coverage debt
    r_prev = {0: 1.0, 1: 0.0}
    age_prev = {0: 0, 1: 10_000}
    fovea = foveate_greedy_cov(r_prev=r_prev, age_prev=age_prev, F=1, alpha_cov=1.0, G=1)
    assert fovea == [1]

    ret = retrieval_block_keyed(fovea_blocks=fovea, library=lib, active_prev=active_prev)
    assert {op.phi for op in ret} == {1}
    assert {op.name for op in ret} == {"op1_low", "op1_low2"}

    similarity = {"op0_mid": 0.99, "op1_low": 0.10, "op1_low2": 0.05}
    ret_moe = retrieval_moe_similarity(library=lib, active_prev=active_prev, similarity=similarity, k=2)
    assert any(op.phi == 0 for op in ret_moe), "Control must differ: MoE retrieval not keyed to fovea."


def test_anti_aliasing_disallows_two_indistinguishable_ops_but_allows_distinguishable_even_if_worse():
    """
    Novel claim:
      You cannot store two ambiguous operators in same footprint.
      Better replaces; otherwise reject. Distinguishable can be stored even if worse.

    Control:
      MoE-like library happily stores all ambiguous operators.
    """
    phi = 0
    T_phi = [0, 1, 2, 3]
    targets = {t: 0.0 for t in T_phi}
    theta_alias = 0.04

    inc = Operator("inc", phi=phi, pi=0.5, pred_series={t: 0.02 for t in T_phi})
    lib = Library(incumbents={phi: [inc]})

    j_better = Operator("j_better", phi=phi, pi=0.5, pred_series={t: 0.01 for t in T_phi})
    d1 = insert_anti_alias(library=lib, phi=phi, j_new=j_better, T_phi=T_phi, targets=targets, theta_alias=theta_alias)
    assert d1.startswith("REPLACE")
    assert [op.name for op in lib.get(phi)] == ["j_better"]

    j_worse = Operator("j_worse", phi=phi, pi=0.5, pred_series={t: 0.03 for t in T_phi})
    d2 = insert_anti_alias(library=lib, phi=phi, j_new=j_worse, T_phi=T_phi, targets=targets, theta_alias=theta_alias)
    assert d2 == "REJECT"
    assert [op.name for op in lib.get(phi)] == ["j_better"]

    j_distinct_bad = Operator("j_distinct_bad", phi=phi, pi=0.5, pred_series={t: 10.0 for t in T_phi})
    d3 = insert_anti_alias(library=lib, phi=phi, j_new=j_distinct_bad, T_phi=T_phi, targets=targets, theta_alias=theta_alias)
    assert d3 == "ADD"
    assert set(op.name for op in lib.get(phi)) == {"j_better", "j_distinct_bad"}

    lib2 = Library(incumbents={phi: [inc]})
    insert_moe_allow_ambiguity(library=lib2, phi=phi, j_new=j_better)
    insert_moe_allow_ambiguity(library=lib2, phi=phi, j_new=j_worse)
    assert len(lib2.get(phi)) == 3, "Control differs: ambiguity not eliminated at storage time."


def test_merge_replacement_consistent_rejects_pooled_improvement_that_regresses_in_winner_regimes():
    """
    Novel claim:
      MERGE must be replacement-consistent per regime partitions, not pooled-error.

    Control:
      Pooled-error acceptance would accept the pooled-improving candidate.
    """
    T_AB = list(range(10))
    targets = {t: 0.0 for t in T_AB}
    aA = {t: (1.0 if t < 5 else 0.0) for t in T_AB}
    aB = {t: (0.0 if t < 5 else 1.0) for t in T_AB}

    A = Operator("A", phi=0, pi=0.5, pred_series={t: (0.0 if t < 5 else 2.0) for t in T_AB})
    B = Operator("B", phi=0, pi=0.5, pred_series={t: (2.0 if t < 5 else 0.0) for t in T_AB})
    C_bad = Operator("C_bad", phi=0, pi=0.5, pred_series={t: 1.0 for t in T_AB})  # pooled better, regime worse

    eps_merge = 1e-9
    assert merge_accept_replacement_consistent(
        A=A, B=B, C=C_bad, T_AB=T_AB, aA=aA, aB=aB, targets=targets, eps_merge=eps_merge
    ) is False
    assert merge_accept_pooled(A=A, B=B, C=C_bad, T_AB=T_AB, targets=targets) is True, \
        "Control differs: pooled rule accepts pooled-improving candidate."

    C_good = Operator("C_good", phi=0, pi=0.5, pred_series={t: 0.0 for t in T_AB})
    assert merge_accept_replacement_consistent(
        A=A, B=B, C=C_good, T_AB=T_AB, aA=aA, aB=aB, targets=targets, eps_merge=eps_merge
    ) is True


def test_spawn_requires_persistence_K_visits_and_incumbent_failure_to_reduce():
    """
    Novel claim:
      SPAWN triggers only after K distinct coverage visits with persistent high residual AND
      no incumbent can reduce it below threshold.

    Control:
      Instant residual rule triggers immediately (not axiom-faithful).
    """
    beta_R = 0.5
    theta_spawn = 0.25
    K = 3

    tr = SpawnTracker(R=0.0, visits=0)
    residuals = [0.40, 0.40, 0.40]
    fired = False
    for r in residuals:
        fired = spawn_trigger_persistent(
            tracker=tr,
            residual_value=r,
            beta_R=beta_R,
            theta_spawn=theta_spawn,
            K=K,
            incumbent_can_reduce_below_threshold=False,
        )
    assert fired is True

    tr2 = SpawnTracker(R=0.0, visits=0)
    fired_first = spawn_trigger_persistent(
        tracker=tr2,
        residual_value=0.40,
        beta_R=beta_R,
        theta_spawn=theta_spawn,
        K=K,
        incumbent_can_reduce_below_threshold=False,
    )
    assert fired_first is False, "Must not spawn on first visit under persistence rule."
    assert spawn_trigger_instant(residual_value=0.40, theta_spawn=theta_spawn) is True, \
        "Control differs: instant rule triggers immediately."

    tr3 = SpawnTracker(R=0.0, visits=0)
    residuals2 = [0.40, 0.05, 0.05, 0.05]
    fired_any = False
    for r in residuals2:
        fired_any = spawn_trigger_persistent(
            tracker=tr3,
            residual_value=r,
            beta_R=beta_R,
            theta_spawn=theta_spawn,
            K=K,
            incumbent_can_reduce_below_threshold=True,
        ) or fired_any
    assert fired_any is False, "If incumbents can reduce below threshold, SPAWN must not trigger."


def test_closed_loop_constellation_dynamics_end_to_end_vocab_expansion():
    """
    Novel claim:
      foveation debt -> uncov block visited -> retrieval empty -> residual high -> SPAWN -> insert incumbent ->
      subsequent retrieval returns incumbent -> residual drops.
    """
    blocks = [0, 1]
    lib = Library(
        incumbents={
            0: [Operator("op0", phi=0, pi=0.8, pred_series={t: 0.0 for t in range(200)})],
            1: [],
        }
    )

    out = run_closed_loop_toy(
        T=40,
        blocks=blocks,
        library=lib,
        F=1,
        alpha_cov=2.0,
        G=2,
        beta_R=0.5,
        theta_spawn=0.25,
        K=2,
        residual_if_uncovered=0.60,
        residual_if_covered=0.05,
    )

    trace_fovea = out["trace_fovea"]
    trace_resid = out["trace_resid"]
    trace_spawns = out["trace_spawns"]
    final_lib: Library = out["library"]

    assert 1 in trace_fovea, "Coverage debt must eventually foveate the uncovered block."

    saw_high_on_1 = any((b == 1 and r >= 0.50) for (b, r) in trace_resid)
    assert saw_high_on_1, "Uncovered footprint must show high residual on visit."

    assert 1 in trace_spawns, "Persistent high residual on uncovered footprint must trigger SPAWN."
    assert len(final_lib.get(1)) > 0, "SPAWN must insert an incumbent."

    saw_low_on_1 = any((b == 1 and r <= 0.10) for (b, r) in trace_resid)
    assert saw_low_on_1, "After insert, later visits must show low residual."

    # Control statement (conceptual): without SPAWN, incumbents(1) would remain empty forever.
    assert len(final_lib.get(1)) > 0
