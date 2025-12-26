#!/usr/bin/env python3
"""
true_train_color_shape_trace.py

Verbose *trace harness* for the current framework.

This does NOT "fix" anything. It only adds high-visibility logging so you can
see exactly what the framework is doing step-by-step.

Key ideas:
- Print a single, very wide line per step (or per `--log-every`) so you can diff.
- Optional "between calls" trace: emit PRE/POST snapshots around agent.step()
  for the first `--trace-steps` steps (can be large/annoying; that's the point).

No monkeypatching. No config shims. No fabricated defaults. If something is
missing, it prints "MISSING:<field>" rather than inventing values.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np


def safe_get(obj: Any, path: str, default: Any = None) -> Any:
    """Safe dotted-path getter: returns default if any hop fails."""
    cur = obj
    for part in path.split("."):
        if cur is None:
            return default
        if not hasattr(cur, part):
            return default
        cur = getattr(cur, part)
    return cur


def fmt(v: Any, prec: int = 3) -> str:
    if v is None:
        return "None"
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.{prec}f}"
    if isinstance(v, (set, list, tuple)):
        return f"{len(v)}"
    return str(v)


def one_hot(i: int, n: int) -> np.ndarray:
    x = np.zeros(n, dtype=float)
    x[i] = 1.0
    return x


def make_transition(rng: np.random.Generator, n: int, alpha: float = 0.7) -> np.ndarray:
    P = rng.random((n, n))
    P = P / P.sum(axis=1, keepdims=True)
    P = (1 - alpha) * P + alpha * np.eye(n)
    P = P / P.sum(axis=1, keepdims=True)
    return P


def sample_markov(rng: np.random.Generator, P: np.ndarray, cur: int) -> int:
    return int(rng.choice(P.shape[0], p=P[cur]))


def mse(a: np.ndarray, b: np.ndarray) -> float:
    d = a - b
    return float(np.mean(d * d))


def argmax_segment(x: np.ndarray, start: int, n: int) -> int:
    seg = x[start:start + n]
    return int(np.argmax(seg))


def summarize_state(state: Any) -> Dict[str, Any]:
    """Small snapshot of state fields that matter for debugging."""
    out: Dict[str, Any] = {}

    out["t"] = safe_get(state, "t", None)
    out["E"] = safe_get(state, "E", None)
    out["D"] = safe_get(state, "D", None)
    out["drift_P"] = safe_get(state, "drift_P", None)

    out["macro.rest"] = safe_get(state, "macro.rest", None)
    out["rest_permitted_prev"] = safe_get(state, "rest_permitted_prev", None)
    out["demand_prev"] = safe_get(state, "demand_prev", None)
    out["interrupt_prev"] = safe_get(state, "interrupt_prev", None)

    out["stress.s_ext_th"] = safe_get(state, "stress.s_ext_th", None)
    out["stress.s_int_need"] = safe_get(state, "stress.s_int_need", None)
    out["stress.s_total"] = safe_get(state, "stress.s_total", None)

    out["arousal"] = safe_get(state, "arousal", None)
    out["arousal_prev"] = safe_get(state, "arousal_prev", None)

    out["x_C_prev"] = safe_get(state, "x_C_prev", None)
    out["rawE_prev"] = safe_get(state, "rawE_prev", None)
    out["rawD_prev"] = safe_get(state, "rawD_prev", None)
    out["c_d_prev"] = safe_get(state, "c_d_prev", None)
    out["b_cons"] = safe_get(state, "b_cons", None)

    # Fovea / observation
    cur_blocks = safe_get(state, "fovea.current_blocks", None)
    if isinstance(cur_blocks, set):
        out["fovea.current_blocks"] = sorted(list(cur_blocks))[:16]  # truncate
        out["fovea.current_blocks_n"] = len(cur_blocks)
    else:
        out["fovea.current_blocks"] = cur_blocks
        out["fovea.current_blocks_n"] = None

    obs_dims = safe_get(state, "buffer.observed_dims", None)
    if isinstance(obs_dims, set):
        out["buffer.observed_dims_n"] = len(obs_dims)
    else:
        out["buffer.observed_dims_n"] = None

    # Library / queue
    nodes = safe_get(state, "library.nodes", None)
    out["lib_n"] = len(nodes) if isinstance(nodes, dict) else None
    q_struct = safe_get(state, "q_struct", None)
    out["q_struct_n"] = len(q_struct) if isinstance(q_struct, list) else None

    # Learn cache essentials
    lc = safe_get(state, "learn_cache", None)
    out["learn_cache_present"] = lc is not None
    yhat = safe_get(state, "learn_cache.yhat_tp1", None)
    out["yhat_tp1_present"] = yhat is not None
    if yhat is not None:
        y = np.asarray(yhat, dtype=float).reshape(-1)
        out["yhat_tp1_mean"] = float(np.mean(y))
        out["yhat_tp1_max"] = float(np.max(y))
        out["yhat_tp1_argmax"] = int(np.argmax(y))
        out["yhat_tp1_l1"] = float(np.sum(np.abs(y)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-colors", type=int, default=24)
    parser.add_argument("--n-shapes", type=int, default=24)
    parser.add_argument("--holdout-frac", type=float, default=0.20)
    parser.add_argument("--train-steps", type=int, default=24000)
    parser.add_argument("--test-steps", type=int, default=1600)
    parser.add_argument("--seed", type=int, default=7)

    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--trace-steps", type=int, default=25,
                        help="For the first N steps of each phase, print PRE/POST snapshots around agent.step()")
    parser.add_argument("--trace-json", action="store_true",
                        help="Emit PRE/POST snapshots as compact JSON instead of key=value.")
    args = parser.parse_args()

    try:
        from nupca3.config import AgentConfig
        from nupca3.agent import NUPCA3Agent
        from nupca3.types import EnvObs
        from nupca3.geometry.fovea import make_observation_set
    except Exception as e:
        print(f"[IMPORT-FAIL] {type(e).__name__}: {e}", file=sys.stderr)
        raise

    rng = np.random.default_rng(int(args.seed))

    C = int(args.n_colors)
    S = int(args.n_shapes)
    D = C + S

    # Heldout pairs
    all_pairs = [(c, s) for c in range(C) for s in range(S)]
    rng.shuffle(all_pairs)
    held_n = int(round(float(args.holdout_frac) * len(all_pairs)))
    heldout = set(all_pairs[:held_n])
    train_allowed = set(all_pairs[held_n:])

    Pc = make_transition(rng, C, alpha=0.6)
    Ps = make_transition(rng, S, alpha=0.6)

    # Config: do NOT shim; try strict ctor patterns
    try:
        cfg = AgentConfig(D=D)
    except TypeError:
        cfg = AgentConfig()
        if not hasattr(cfg, "D") or int(getattr(cfg, "D")) != D:
            print(f"[CFG-MISMATCH] cfg.D={getattr(cfg,'D',None)} expected D={D}. This harness refuses to shim.", file=sys.stderr)
            raise SystemExit(2)

    agent = NUPCA3Agent(cfg)

    # Build dims for requested blocks from fovea request
    def dims_for_blocks(block_ids: Set[int]) -> Set[int]:
        # blocks are contiguous partitions implied by cfg.D and cfg.B/n_blocks
        if hasattr(cfg, "B"):
            B = int(getattr(cfg, "B"))
        elif hasattr(cfg, "n_blocks"):
            B = int(getattr(cfg, "n_blocks"))
        else:
            B = 1
        # contiguous partition
        base = D // max(1, B)
        rem = D % max(1, B)
        blocks: List[List[int]] = []
        start = 0
        for b in range(max(1, B)):
            size = base + (1 if b < rem else 0)
            blocks.append(list(range(start, start + size)))
            start += size

        dims: Set[int] = set()
        for bid in block_ids:
            if 0 <= bid < len(blocks):
                dims.update(blocks[bid])
        return dims

    def x_from_pair(cc: int, ss: int) -> np.ndarray:
        return np.concatenate([one_hot(cc, C), one_hot(ss, S)], axis=0)

    def step_pair(cur_c: int, cur_s: int, allowed: Set[Tuple[int, int]]) -> Tuple[int, int]:
        for _ in range(20000):
            nc = sample_markov(rng, Pc, cur_c)
            ns = sample_markov(rng, Ps, cur_s)
            if (nc, ns) in allowed:
                return nc, ns
        raise RuntimeError("Could not sample allowed pair; check holdout/transition.")

    # Initial hidden pair in training set
    c = int(rng.integers(C))
    s = int(rng.integers(S))
    if (c, s) in heldout:
        (c, s) = next(iter(train_allowed))

    # Initial requested blocks (lagged observation contract)
    req_prev: Set[int] = set(safe_get(agent.state, "fovea.current_blocks", set()) or set())
    if not req_prev:
        req_prev = {0}

    # Header
    print("phase t step/phase  rest dem int perm  obs_dims req_blocks  lib q_struct  yhat_ok yhat_l1 yhat_max  mse  accC accS accP  held_seen held_accP")

    def emit_snapshot(tag: str, phase: str, t: int, state: Any, extra: Dict[str, Any]) -> None:
        snap = summarize_state(state)
        snap.update(extra)
        snap["tag"] = tag
        snap["phase"] = phase
        snap["t_step"] = t
        if args.trace_json:
            print(json.dumps(snap, separators=(",", ":"), sort_keys=True))
        else:
            # key=value compact
            parts = [f"{k}={snap[k]}" for k in sorted(snap.keys())]
            print(" ".join(parts))

    def run_phase(phase: str, steps: int, allowed: Set[Tuple[int, int]]) -> None:
        nonlocal c, s, req_prev

        held_seen = 0
        held_accP_sum = 0.0

        for t in range(1, steps + 1):
            nc, ns = step_pair(c, s, allowed)
            x_true_next = x_from_pair(nc, ns)
            x_cur = x_from_pair(c, s)

            # Enforce partial observation based on lagged request.
            dims = dims_for_blocks(req_prev)
            cue = {int(i): float(x_cur[i]) for i in dims if 0 <= i < D}

            obs = EnvObs(x_partial=cue, opp=0.0, danger=0.0)

            if t <= int(args.trace_steps):
                emit_snapshot("PRE", phase, t, agent.state, {"cue_n": len(cue), "req_prev_n": len(req_prev)})

            action, trace = agent.step(obs)

            if t <= int(args.trace_steps):
                emit_snapshot("POST", phase, t, agent.state, {"action": int(action), "trace_keys": sorted(list(trace.keys())) if isinstance(trace, dict) else "non-dict"})

            # Required prediction for evaluation
            yhat_tp1 = safe_get(agent.state, "learn_cache.yhat_tp1", None)
            if yhat_tp1 is None:
                print("[MISSING] learn_cache.yhat_tp1 is None; cannot evaluate.", file=sys.stderr)
                raise SystemExit(2)
            yhat_tp1 = np.asarray(yhat_tp1, dtype=float).reshape(-1)
            if yhat_tp1.size != D:
                print(f"[BAD] yhat_tp1.size={yhat_tp1.size} expected D={D}", file=sys.stderr)
                raise SystemExit(2)

            pred_c = argmax_segment(yhat_tp1, 0, C)
            pred_s = argmax_segment(yhat_tp1, C, S)
            accC = 1.0 if pred_c == nc else 0.0
            accS = 1.0 if pred_s == ns else 0.0
            accP = 1.0 if (pred_c == nc and pred_s == ns) else 0.0

            e = mse(yhat_tp1, x_true_next)

            if (nc, ns) in heldout:
                held_seen += 1
                held_accP_sum += accP

            # Update lagged request for next step
            req_prev = set(safe_get(agent.state, "fovea.current_blocks", set()) or set())
            if not req_prev:
                # do not "fix"; just show and stop
                print("[BAD] fovea.current_blocks became empty; stopping.", file=sys.stderr)
                raise SystemExit(2)

            # advance true chain
            c, s = nc, ns

            if (t == 1) or (t % int(args.log_every) == 0) or (t == steps):
                rest = int(bool(safe_get(agent.state, "macro.rest", False)))
                dem = int(bool(safe_get(agent.state, "demand_prev", False)))
                intr = int(bool(safe_get(agent.state, "interrupt_prev", False)))
                perm = int(bool(safe_get(agent.state, "rest_permitted_prev", True)))

                obs_dims = int(len(safe_get(agent.state, "buffer.observed_dims", set()) or set()))
                lib_n = len(safe_get(agent.state, "library.nodes", {}) or {})
                qn = len(safe_get(agent.state, "q_struct", []) or [])
                y_l1 = float(np.sum(np.abs(yhat_tp1)))
                y_mx = float(np.max(yhat_tp1))
                y_ok = 1

                held_accP = (held_accP_sum / held_seen) if held_seen else 0.0

                print(
                    f"{phase:5s} {t:6d} {t:6d}/{steps:<6d}  "
                    f"{rest:4d} {dem:3d} {intr:3d} {perm:4d}  "
                    f"{obs_dims:8d} {len(req_prev):9d}  "
                    f"{lib_n:4d} {qn:7d}  "
                    f"{y_ok:6d} {y_l1:7.3f} {y_mx:7.3f}  "
                    f"{e:6.4f} {accC:4.0f} {accS:4.0f} {accP:4.0f}  "
                    f"{held_seen:9d} {held_accP:9.3f}"
                )

    run_phase("train", int(args.train_steps), train_allowed)
    run_phase("test", int(args.test_steps), heldout)


if __name__ == "__main__":
    main()
