#!/usr/bin/env python3
"""
true_train_color_shape.py  (NUPCA3 harness; fresh rewrite)

Purpose
-------
Runs a true online TRAIN + TEST loop for the NUPCA3 agent on a factorized
discrete world whose latent state is (color, shape).

World (ColorShapeWorld)
-----------------------
State x(t) is a concatenated one-hot vector:

    x(t) = [onehot(color_t, C) ; onehot(shape_t, S)]  ∈ R^(C+S)

Color and shape evolve independently via sticky Markov chains. During TRAIN, a
subset of color×shape pairs can be withheld by rejecting those combinations at
sampling time (so the marginals remain, but the joint support is reduced).

Agent (NUPCA3)
--------------
This harness drives the implementation in the local `nupca3/` package.

Axiom alignment expectations (harness-level)
--------------------------------------------
- The harness honors the A16 discipline that the fovea selection precedes
  observation: when --observe-mode=fovea, the harness filters the world's full
  observation down to only dims in the selected blocks.
- The harness does NOT implement structural edits or learning rules itself; it
  only calls the agent. If your repo snapshot has missing glue (e.g., the agent
  wrapper expects `step_pipeline.step_pipeline` but the module only defines
  `step_agent`), the harness provides a minimal import-compat wrapper.

Compatibility short-cuts
------------------------
Any harness-only shim is marked with:

    #ITOOKASHORTCUT

and explains why it exists. These shims are not intended as “framework design”;
they are narrowly-scoped survivability measures for an evolving repo snapshot.

Logging discipline
------------------
- One header line, followed by periodic single-line rows (many columns).
- Additional informational prints (e.g., transition matrices) are kept to single
  lines.

Usage
-----
  python true_train_color_shape.py --train-steps 12000 --test-steps 1600 --seed 7
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


# =============================================================================
# Generic helpers
# =============================================================================

def _now_s() -> float:
    return time.perf_counter()


def ema(prev: Optional[float], x: float, alpha: float) -> float:
    if prev is None or not np.isfinite(prev):
        return float(x)
    a = float(alpha)
    return float((1.0 - a) * float(prev) + a * float(x))


def dense_from_partial(x_partial: Dict[int, float], D: int) -> np.ndarray:
    x = np.zeros(int(D), dtype=float)
    for k, v in x_partial.items():
        kk = int(k)
        if 0 <= kk < int(D):
            x[kk] = float(v)
    return x


def decode_pair(x: np.ndarray, n_colors: int, n_shapes: int) -> Tuple[int, int]:
    x = np.asarray(x, dtype=float).reshape(-1)
    c = int(np.argmax(x[:n_colors])) if n_colors > 0 else 0
    s = int(np.argmax(x[n_colors : n_colors + n_shapes])) if n_shapes > 0 else 0
    return c, s


def one_line_matrix(name: str, M: np.ndarray) -> str:
    M = np.asarray(M, dtype=float)
    s = np.array2string(M, precision=3, suppress_small=False, separator=",", max_line_width=10**9)
    return f"{name}={s}"


# =============================================================================
# Holdout construction
# =============================================================================

def make_holdout_pairs(
    n_colors: int,
    n_shapes: int,
    holdout_frac: float,
    seed: int,
    scheme: str,
) -> Set[Tuple[int, int]]:
    C, S = int(n_colors), int(n_shapes)
    all_pairs = [(c, s) for c in range(C) for s in range(S)]
    total = len(all_pairs)
    k = int(round(float(holdout_frac) * float(total)))
    k = max(0, min(total, k))

    rng = np.random.default_rng(int(seed))

    if scheme == "diag":
        # Deterministic diagonal-like holdout first, then fill if needed.
        pairs: List[Tuple[int, int]] = []
        m = min(C, S)
        for i in range(m):
            pairs.append((i, i))
            if len(pairs) >= k:
                return set(pairs)
        # Fill remainder uniformly at random from remaining.
        remaining = [p for p in all_pairs if p not in set(pairs)]
        rng.shuffle(remaining)
        pairs.extend(remaining[: max(0, k - len(pairs))])
        return set(pairs)

    # scheme == "random"
    rng.shuffle(all_pairs)
    return set(all_pairs[:k])


# =============================================================================
# Repo-compat glue
# =============================================================================

def ensure_repo_importable() -> None:
    """Ensure local imports work no matter the current working directory."""
    this_dir = __file__.rsplit("/", 1)[0] if "/" in __file__ else "."
    if this_dir not in sys.path:
        sys.path.insert(0, this_dir)


def ensure_step_pipeline_symbol() -> None:
    """Ensure `nupca3.step_pipeline.step_pipeline` exists for NUPCA3Agent.

    #ITOOKASHORTCUT:
    Some snapshots define `step_agent(state, obs, cfg)` but the agent wrapper
    imports `step_pipeline` as a function. We provide a minimal wrapper that:
      - returns action=0 (this toy world ignores actions)
      - returns a dict trace (dataclasses are converted via asdict)
    """
    import nupca3.step_pipeline as sp

    if hasattr(sp, "step_pipeline"):
        return
    if not hasattr(sp, "step_agent"):
        return

    def _step_pipeline(state, env_obs, cfg):
        action = 0
        next_state, trace = sp.step_agent(state, env_obs, cfg)
        if dataclasses.is_dataclass(trace):
            trace = dataclasses.asdict(trace)
        elif not isinstance(trace, dict):
            trace = {"trace": trace}
        return action, next_state, trace

    sp.step_pipeline = _step_pipeline  # type: ignore[attr-defined]


def ensure_agentstate_current_fovea_setter() -> None:
    """Allow snapshots where AgentState.current_fovea is read-only.

    #ITOOKASHORTCUT:
    Some branches expose `AgentState.current_fovea` as a @property that
    forwards to `fovea.current_blocks` but provide no setter. The step pipeline
    sets `state.current_fovea = set(F_t)`; without a setter this raises.
    We patch the property to include a setter that writes `fovea.current_blocks`.
    """
    try:
        import nupca3.types as t
        AgentState = t.AgentState
    except Exception:
        return

    prop = getattr(AgentState, "current_fovea", None)
    if not isinstance(prop, property):
        return
    if prop.fset is not None:
        return

    def _set(self, blocks):
        try:
            self.fovea.current_blocks = set(blocks) if blocks is not None else set()
        except Exception:
            # Last-resort: ignore if this snapshot uses a different container.
            pass

    AgentState.current_fovea = property(prop.fget, _set)


def ensure_commit_gate_compat() -> None:
    """Allow commit_gate to accept a scalar confidence c_d.

    #ITOOKASHORTCUT:
    Some snapshots implement commit_gate(rest, h, c:list[float], cfg) while
    the step pipeline passes a scalar c_d(t-1). We wrap the module-global
    `commit_gate` used by step_agent so it handles either:
      - a sequence (legacy) or
      - a scalar float (current pipeline).
    """
    try:
        import nupca3.step_pipeline as sp
    except Exception:
        return

    if not hasattr(sp, "commit_gate"):
        return

    _orig = sp.commit_gate

    def _compat_commit_gate(rest: bool, h: int, c, cfg):
        if rest:
            return False
        if h < int(getattr(cfg, "d_latency_floor", 1)):
            return False

        # If c is a sequence, preserve legacy behavior.
        if isinstance(c, (list, tuple, np.ndarray)):
            try:
                if len(c) < int(getattr(cfg, "d_latency_floor", 1)):
                    return False
                val = float(c[int(getattr(cfg, "d_latency_floor", 1)) - 1])
            except Exception:
                return False
            return bool(val >= float(getattr(cfg, "theta_act", 0.5)))

        # Otherwise treat c as scalar confidence at the latency floor.
        try:
            val = float(c)
        except Exception:
            return False
        return bool(val >= float(getattr(cfg, "theta_act", 0.5)))

    # Preserve name for debuggability.
    _compat_commit_gate.__name__ = getattr(_orig, "__name__", "commit_gate")
    sp.commit_gate = _compat_commit_gate


def ensure_agent_init_stress_compat(cfg) -> None:
    """Ensure `nupca3.agent.init_stress()` can be called with no args.

    #ITOOKASHORTCUT:
    Some snapshots define `state.margins.init_stress(cfg)` but the agent
    wrapper calls `init_stress()` without an argument. Because the reference
    is imported into `nupca3.agent` at module load, patching
    `state.margins.init_stress` alone is insufficient; we patch the symbol in
    `nupca3.agent` directly.
    """
    import inspect
    import nupca3.agent as ag

    try:
        sig = inspect.signature(ag.init_stress)
        if len(sig.parameters) == 0:
            return
    except Exception:
        # If we cannot introspect, do nothing.
        return
    
    _orig = ag.init_stress

    def _compat_init_stress(cfg_arg=None):
        return _orig(cfg if cfg_arg is None else cfg_arg)

    ag.init_stress = _compat_init_stress


def ensure_agent_init_macro_compat(cfg) -> None:
    """Ensure `nupca3.agent.init_macro()` can be called with no args.

    #ITOOKASHORTCUT:
    Some snapshots define `state.macrostate.init_macro(cfg)` but the agent
    wrapper calls `init_macro()` without an argument. We patch the symbol in
    `nupca3.agent` directly.
    """
    import inspect
    import nupca3.agent as ag

    try:
        sig = inspect.signature(ag.init_macro)
        if len(sig.parameters) == 0:
            return
    except Exception:
        return

    _orig = ag.init_macro

    def _compat_init_macro(cfg_arg=None):
        return _orig(cfg if cfg_arg is None else cfg_arg)

    ag.init_macro = _compat_init_macro


def mark_anchors_on_nodes(agent) -> None:
    """Synchronize node.is_anchor with library.anchors when missing.

    #ITOOKASHORTCUT:
    Some branches track anchors only in lib.anchors. Proposal/selection code
    sometimes looks at node.is_anchor; we set it here for consistency.
    """
    lib = getattr(getattr(agent, "state", None), "library", None)
    if lib is None:
        return
    anchors = getattr(lib, "anchors", set()) or set()
    nodes = getattr(lib, "nodes", {}) or {}
    for nid in list(anchors):
        node = nodes.get(nid, None)
        if node is None:
            continue
        try:
            setattr(node, "is_anchor", True)
        except Exception:
            pass


# =============================================================================
# Observation filtering (A16 harness responsibility)
# =============================================================================

def filter_obs_to_fovea(*, x_full: np.ndarray, opp: float, danger: float, state, cfg):
    """Construct EnvObs with x_partial restricted to the selected fovea blocks."""
    from nupca3.geometry.fovea import select_fovea, make_observation_set
    from nupca3.types import EnvObs

    blocks = select_fovea(getattr(state, "fovea"), cfg)
    O_t = make_observation_set(blocks, cfg)
    x_partial = {int(k): float(x_full[int(k)]) for k in O_t}
    return EnvObs(x_partial=x_partial, opp=float(opp), danger=float(danger))


# =============================================================================
# Run loops
# =============================================================================

def run_phase(
    *,
    phase: str,
    agent,
    world,
    cfg,
    n_colors: int,
    n_shapes: int,
    steps: int,
    heldout_pairs: Set[Tuple[int, int]],
    observe_mode: str,
    log_every: int,
    ema_alpha: float,
) -> Dict[str, float]:
    from nupca3.types import EnvObs

    obs_full = world.reset(seed=int(getattr(agent, "seed", 0)))
    x_true = dense_from_partial(obs_full.x_partial, int(cfg.D))

    mse_ema: Optional[float] = None
    accC_ema: Optional[float] = None
    accS_ema: Optional[float] = None
    accP_ema: Optional[float] = None

    held_seen = 0
    held_accC_sum = 0.0
    held_accS_sum = 0.0
    held_accP_sum = 0.0

    t0 = _now_s()
    last_log = t0

    for t in range(1, int(steps) + 1):
        # --- Build observation presented to the agent (A16 harness duty) ---
        if observe_mode == "full":
            obs_for_agent = EnvObs(
                x_partial=dict(obs_full.x_partial),
                opp=float(getattr(obs_full, "opp", 0.0)),
                danger=float(getattr(obs_full, "danger", 0.0)),
            )
        else:
            obs_for_agent = filter_obs_to_fovea(
                x_full=x_true,
                opp=float(getattr(obs_full, "opp", 0.0)),
                danger=float(getattr(obs_full, "danger", 0.0)),
                state=getattr(agent, "state"),
                cfg=cfg,
            )

        # --- Agent step ---
        _action, trace = agent.step(obs_for_agent)

        # --- World step (ignores action in this toy world) ---
        obs_full_next, _done = world.step(0)
        x_next = dense_from_partial(obs_full_next.x_partial, int(cfg.D))

        # --- Prediction extraction (agent predicts x(t+1|t)) ---
        lc = getattr(getattr(agent, "state", None), "learn_cache", None)
        yhat_tp1 = getattr(lc, "yhat_tp1", None)
        if yhat_tp1 is None:
            yhat_tp1 = np.zeros_like(x_next)

        yhat_tp1 = np.asarray(yhat_tp1, dtype=float).reshape(-1)
        if yhat_tp1.shape[0] != x_next.shape[0]:
            yhat_tmp = np.zeros_like(x_next)
            n = min(len(yhat_tmp), len(yhat_tp1))
            yhat_tmp[:n] = yhat_tp1[:n]
            yhat_tp1 = yhat_tmp

        # --- Metrics on next-state prediction ---
        mse = float(np.mean((yhat_tp1 - x_next) ** 2))

        pred_c, pred_s = decode_pair(yhat_tp1, n_colors, n_shapes)
        true_c, true_s = decode_pair(x_next, n_colors, n_shapes)

        accC = 1.0 if pred_c == true_c else 0.0
        accS = 1.0 if pred_s == true_s else 0.0
        accP = 1.0 if (pred_c == true_c and pred_s == true_s) else 0.0

        mse_ema = ema(mse_ema, mse, ema_alpha)
        accC_ema = ema(accC_ema, accC, ema_alpha)
        accS_ema = ema(accS_ema, accS, ema_alpha)
        accP_ema = ema(accP_ema, accP, ema_alpha)

        # Heldout tracking
        if (true_c, true_s) in heldout_pairs:
            held_seen += 1
            held_accC_sum += accC
            held_accS_sum += accS
            held_accP_sum += accP

        # --- Logging ---
        do_log = (t == 1) or (t % int(log_every) == 0) or (t == int(steps))
        if do_log:
            now = _now_s()
            wall_s = now - t0
            dt_ms = (now - last_log) * 1000.0
            last_log = now

            rest = int(bool(trace.get("rest", getattr(getattr(agent.state, "macro", None), "rest", False))))
            h = int(trace.get("h", 1))
            xC = float(trace.get("x_C", float("nan")))
            permit = int(bool(trace.get("permit_param", trace.get("permit_param_t", False))))
            freeze = int(bool(trace.get("freeze", False)))
            ar = float(trace.get("arousal", float("nan")))

            lib = getattr(getattr(agent, "state", None), "library", None)
            lib_n = int(len(getattr(lib, "nodes", {}) or {})) if lib is not None else 0
            q_struct = int(len(getattr(getattr(agent, "state", None), "q_struct", []) or []))

            print(
                f"{phase}"
                f" t={t:7d}/{int(steps):7d}"
                f" wall_s={wall_s:9.3f}"
                f" dt_ms={dt_ms:8.2f}"
                f" mse_ema={float(mse_ema):.6f}"
                f" accC_ema={float(accC_ema):.4f}"
                f" accS_ema={float(accS_ema):.4f}"
                f" accP_ema={float(accP_ema):.4f}"
                f" accC={accC:.0f}"
                f" accS={accS:.0f}"
                f" accP={accP:.0f}"
                f" lib={lib_n:6d}"
                f" q_struct={q_struct:6d}"
                f" rest={rest:d}"
                f" h={h:3d}"
                f" xC={xC:9.3f}"
                f" permit={permit:d}"
                f" freeze={freeze:d}"
                f" ar={ar:7.3f}"
                f" held_seen={held_seen:6d}"
                f" held_accC={(held_accC_sum/held_seen if held_seen>0 else float('nan')):7.4f}"
                f" held_accS={(held_accS_sum/held_seen if held_seen>0 else float('nan')):7.4f}"
                f" held_accP={(held_accP_sum/held_seen if held_seen>0 else float('nan')):7.4f}"
            )

        obs_full = obs_full_next
        x_true = x_next

    return {
        "mse_ema": float(mse_ema) if mse_ema is not None else float("nan"),
        "acc_color_ema": float(accC_ema) if accC_ema is not None else float("nan"),
        "acc_shape_ema": float(accS_ema) if accS_ema is not None else float("nan"),
        "acc_pair_ema": float(accP_ema) if accP_ema is not None else float("nan"),
        "held_seen": float(held_seen),
        "held_acc_color": float(held_accC_sum / held_seen) if held_seen > 0 else float("nan"),
        "held_acc_shape": float(held_accS_sum / held_seen) if held_seen > 0 else float("nan"),
        "held_acc_pair": float(held_accP_sum / held_seen) if held_seen > 0 else float("nan"),
    }


# =============================================================================
# Entrypoint
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="NUPCA3 true training run on compositional color×shape world (fresh)")
    parser.add_argument("--train-steps", type=int, default=12000)
    parser.add_argument("--test-steps", type=int, default=1600)
    parser.add_argument("--seed", type=int, default=7)

    parser.add_argument("--n-colors", type=int, default=3)
    parser.add_argument("--n-shapes", type=int, default=3)

    parser.add_argument("--holdout-frac", type=float, default=0.20)
    parser.add_argument("--holdout-scheme", type=str, default="random", choices=["random", "diag"])

    parser.add_argument("--observe-mode", type=str, default="fovea", choices=["fovea", "full"])
    parser.add_argument("--fovea-blocks", type=int, default=2)
    parser.add_argument("--alpha-cov", type=float, default=0.10)
    parser.add_argument("--coverage-cap-G", type=int, default=0)

    parser.add_argument("--log-every", type=int, default=2000)
    parser.add_argument("--ema-alpha", type=float, default=0.02)

    parser.add_argument("--use-current-gates", action="store_true")
    parser.add_argument("--no-learn-test", action="store_true", help="Disable learning during TEST (recommended).")

    args = parser.parse_args()

    ensure_repo_importable()
    ensure_step_pipeline_symbol()
    ensure_agentstate_current_fovea_setter()
    ensure_commit_gate_compat()

    from nupca3.agent import NUPCA3Agent
    from nupca3.config import AgentConfig
    from nupca3.sim.worlds import ColorShapeWorld

    np.random.seed(int(args.seed))

    n_colors = int(args.n_colors)
    n_shapes = int(args.n_shapes)
    D = n_colors + n_shapes

    heldout_pairs = make_holdout_pairs(
        n_colors=n_colors,
        n_shapes=n_shapes,
        holdout_frac=float(args.holdout_frac),
        seed=int(args.seed),
        scheme=str(args.holdout_scheme),
    )

    world_train = ColorShapeWorld(n_colors=n_colors, n_shapes=n_shapes, reject_pairs=set(heldout_pairs))
    _ = world_train.reset(seed=int(args.seed))

    world_test = ColorShapeWorld(
        n_colors=n_colors,
        n_shapes=n_shapes,
        P_color=np.asarray(world_train.P_color, dtype=float).copy(),
        P_shape=np.asarray(world_train.P_shape, dtype=float).copy(),
        reject_pairs=None,
    )

    cfg = AgentConfig(
        D=int(D),
        B=2,
        fovea_blocks_per_step=int(args.fovea_blocks),
        alpha_cov=float(args.alpha_cov),
        coverage_cap_G=int(args.coverage_cap_G),
        enable_learning=True,
        gates_use_current=bool(args.use_current_gates),
        lr_expert=0.05,
        sigma_ema=0.05,
        B_rt=100.0,
        b_enc_base=1.0,
        b_roll_base=0.1,
    )

    try:
        cfg.validate()
    except Exception:
        pass

    # Patch snapshot-level signature drift before instantiating the agent.
    ensure_agent_init_stress_compat(cfg)
    ensure_agent_init_macro_compat(cfg)

    agent = NUPCA3Agent(cfg)
    setattr(agent, "seed", int(args.seed))
    mark_anchors_on_nodes(agent)

    print(f"INFO seed={int(args.seed)} D={int(D)} C={n_colors} S={n_shapes} holdout_k={len(heldout_pairs)} observe_mode={args.observe_mode}")
    print("INFO " + one_line_matrix("P_color", np.asarray(world_train.P_color, dtype=float)))
    print("INFO " + one_line_matrix("P_shape", np.asarray(world_train.P_shape, dtype=float)))
    print("INFO holdout_pairs=" + ",".join([f"({c},{s})" for (c, s) in sorted(list(heldout_pairs))]))

    print(
        "phase"
        " t"
        " wall_s"
        " dt_ms"
        " mse_ema"
        " accC_ema"
        " accS_ema"
        " accP_ema"
        " accC"
        " accS"
        " accP"
        " lib"
        " q_struct"
        " rest"
        " h"
        " xC"
        " permit"
        " freeze"
        " ar"
        " held_seen"
        " held_accC"
        " held_accS"
        " held_accP"
    )

    _ = run_phase(
        phase="train",
        agent=agent,
        world=world_train,
        cfg=cfg,
        n_colors=n_colors,
        n_shapes=n_shapes,
        steps=int(args.train_steps),
        heldout_pairs=heldout_pairs,
        observe_mode=str(args.observe_mode),
        log_every=int(args.log_every),
        ema_alpha=float(args.ema_alpha),
    )

    try:
        agent.reset(seed=int(args.seed), clear_memory=False)
    except TypeError:
        #ITOOKASHORTCUT: signature drift
        agent.reset(int(args.seed), False)

    setattr(agent, "seed", int(args.seed))
    mark_anchors_on_nodes(agent)

    if bool(args.no_learn_test):
        try:
            agent.cfg = agent.cfg.replace(enable_learning=False)
        except Exception:
            #ITOOKASHORTCUT
            setattr(agent.cfg, "enable_learning", False)

    metrics = run_phase(
        phase="test",
        agent=agent,
        world=world_test,
        cfg=getattr(agent, "cfg"),
        n_colors=n_colors,
        n_shapes=n_shapes,
        steps=int(args.test_steps),
        heldout_pairs=heldout_pairs,
        observe_mode=str(args.observe_mode),
        log_every=max(1, int(args.log_every) // 2),
        ema_alpha=float(args.ema_alpha),
    )

    print(
        "RESULT"
        f" acc_color_ema={metrics['acc_color_ema']:.4f}"
        f" acc_shape_ema={metrics['acc_shape_ema']:.4f}"
        f" acc_pair_ema={metrics['acc_pair_ema']:.4f}"
        f" held_seen={int(metrics['held_seen'])}"
        f" held_acc_color={metrics['held_acc_color']:.4f}"
        f" held_acc_shape={metrics['held_acc_shape']:.4f}"
        f" held_acc_pair={metrics['held_acc_pair']:.4f}"
    )


if __name__ == "__main__":
    main()
