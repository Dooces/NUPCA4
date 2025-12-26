"""nupca3/control/budget.py

Compute accounting and emergent horizon.

Axiom coverage:
  - A2.4: compute slack x_C(t)
  - A6.1–A6.3: effective-load decomposition, rollout cost, horizon h(t)

Key requirements
----------------
- REST horizon is forced to 0.
- b_cons is explicitly accounted for when provided (A2.4, A6.2).
- Horizon is capped by cfg.h_max to prevent accidental O(h) blowups.

Notes
-----
The repository uses a lightweight model of compute accounting. v1.5b defines
an effective-load decomposition and a compute slack variable x_C(t). This
module keeps the interface compact but makes the dynamics explicit.


[AXIOM_CLARIFICATION_ADDENDUM — Representation & Naming]

- Terminology: identifiers like "Expert" in this codebase refer to NUPCA3 **abstraction/resonance nodes** (a "constellation"), not conventional Mixture-of-Experts "experts" or router-based MoE.

- Representation boundary (clarified intent of v1.5b): the completion/fusion operator (A7) is defined over an **encoded, multi-resolution abstraction vector** \(x(t)\). Raw pixels may exist only in a transient observation buffer for the current step; **raw pixel values must never be inserted into long-term storage** (library/cold storage) and must not persist across REST boundaries.

- Decomposition intuition: each node is an operator that *factors out* a predictable/resonant component on its footprint, leaving residual structure for other nodes (or for REST-time proposal) to capture. This is the intended "FFT-like" interpretation of masks/constellations.
"""

from __future__ import annotations

from ..config import AgentConfig
from ..types import ExpertLibrary, WorkingSet, BudgetBreakdown


def effective_load(lib: ExpertLibrary, A_t: WorkingSet, cfg: AgentConfig) -> float:
    """Compute L_eff(t) = Σ_{j∈A_t} a_j(t)·L_j (A5.5 / A6.2).

    Inputs:
      - lib: expert library
      - A_t: WorkingSet (contains per-node activation weights)
      - cfg: configuration

    Outputs:
      - L_eff(t) float

    Implementation detail:
      Prefer A_t.effective_load if present (computed during selection). Falls
      back to a direct summation over A_t.active.
    """
    L_eff = getattr(A_t, "effective_load", None)
    if L_eff is not None:
        return float(L_eff)

    total = 0.0
    for node_id in getattr(A_t, "active", []) or []:
        node = lib.nodes[int(node_id)]
        a_j = float(getattr(A_t, "weights", {}).get(int(node_id), 0.0))
        # Node cost field is normalized by library; fall back to L.
        L_j = float(getattr(node, "cost", getattr(node, "L", 1.0)))
        total += a_j * L_j
    return float(total)


def compute_budget_and_horizon(
    rest: bool,
    cfg: AgentConfig,
    L_eff: float,
    L_eff_roll: float | None = None,
    L_eff_anc: float | None = None,
    b_cons: float = 0.0,
) -> BudgetBreakdown:
    """Compute b_enc, b_roll, b_cons, horizon h, and slack x_C.

    Axiom forms (v1.5b):
      b_enc(t)  = b_enc,0 + b_anc,0 · (ε + L_eff,anc(t))           (A6.2)
      b_roll(t) = b_roll,0 · (ε + L_eff,roll(t))                  (A6.2)
      h(t)      = floor((B_rt - b_enc(t)) / b_roll(t))            (A6.3)
      x_C(t)    = B_rt - ( b_enc(t) + (1-rest(t))·h(t)·b_roll(t)
                           + rest(t)·b_cons(t) )                  (A2.4)

    Inputs:
      - rest: whether we are in REST at time t
      - cfg: AgentConfig (B_rt, b_enc_base, b_roll_base, eps_budget, h_max)
      - L_eff: effective maintenance load (A5.5)
      - L_eff_roll: effective rollout load L_eff,roll (A6.2); defaults to L_eff
      - L_eff_anc: effective anchor maintenance load L_eff,anc (A6.2); defaults to 0
      - b_cons: consolidation cost channel (A2.4); typically 0 in WAKE

    Outputs:
      - BudgetBreakdown with fields b_enc, b_roll, b_cons, h, x_C
    """

    eps = float(getattr(cfg, "eps_budget", 1e-6))
    h_max = int(getattr(cfg, "h_max", 32))

    if L_eff_roll is None:
        L_eff_roll = float(L_eff)
    if L_eff_anc is None:
        L_eff_anc = 0.0

    # Effective-load decomposition (A6.2)
    b_enc_0 = float(getattr(cfg, "b_enc_base", 3.2))
    b_anc_0 = float(getattr(cfg, "b_anc_base", 0.0))
    b_roll_0 = float(getattr(cfg, "b_roll_base", 0.85))

    b_enc = b_enc_0 + b_anc_0 * (eps + float(L_eff_anc))
    b_roll = b_roll_0 * (eps + float(L_eff_roll))

    # Consolidation cost b_cons is paid only in REST (A2.4).
    b_cons = float(b_cons)

    if rest:
        h = 0
    else:
        budget = float(getattr(cfg, "B_rt", 0.0)) - b_enc
        if budget <= 0.0:
            h = 0
        else:
            h = int(budget / max(b_roll, 1e-12))
            if h < 0:
                h = 0
            if h > h_max:
                h = h_max

    x_C = float(getattr(cfg, "B_rt", 0.0)) - (b_enc + ((0 if rest else h) * b_roll) + (b_cons if rest else 0.0))

    return BudgetBreakdown(
        b_enc=float(b_enc),
        b_roll=float(b_roll),
        b_cons=float(b_cons),
        h=int(h),
        x_C=float(x_C),
    )
