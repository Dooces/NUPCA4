#!/usr/bin/env python3
"""
selective_labels_lockin_ascii_nn.py

Demonstrates the closed-loop hard spot with an actual NN + delayed feedback + selective labels + drift,
with ASCII visualization so you can see the mismatch directly.

Locked mode:
  - learns B is bad pre-drift
  - then stops exploring (epsilon collapses)
  - after drift B becomes good but agent keeps denying -> no B labels -> cannot learn change

Probed mode:
  - same, but buys a small fraction of labels even on denies -> trickle of counterevidence -> recovers

Run:
  python selective_labels_lockin_ascii_nn.py --mode locked --animate
  python selective_labels_lockin_ascii_nn.py --mode probed --animate --probe-rate 0.02

If you want plots too:
  python selective_labels_lockin_ascii_nn.py --mode locked --plot
  python selective_labels_lockin_ascii_nn.py --mode probed --plot
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from collections import deque
from typing import Deque, Dict, Tuple, List, Optional

import numpy as np

try:
    import matplotlib.pyplot as plt
    HAVE_PLT = True
except Exception:
    HAVE_PLT = False

SHADE = " .:-=+*#%@"

def sigm(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)

def clamp(p: float) -> float:
    return float(min(1.0 - 1e-6, max(1e-6, p)))

def shade(p: float) -> str:
    p = min(1.0, max(0.0, p))
    i = int(p * (len(SHADE) - 1) + 1e-12)
    return SHADE[i]

def clear_screen() -> None:
    print("\x1b[2J\x1b[H", end="")

def moving_avg(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x.copy()
    win = min(win, len(x))
    ker = np.ones(win, dtype=np.float64) / float(win)
    return np.convolve(x, ker, mode="same")


# ----------------------------
# Environment
# ----------------------------

@dataclass
class EnvCfg:
    seed: int = 0
    T: int = 12000
    delay: int = 50
    drift_t: int = 6000

    p_group_B: float = 0.35

    # Feature: z ~ N(mu_g, sigma)
    mu_A: float = 0.0
    mu_B: float = 0.0
    sigma: float = 1.0

    # True label prob: p = sigmoid(a_g(t) + b_g*z)
    a_A: float = 2.0
    b_A: float = 1.2

    # Make B brutally bad pre-drift, brutally good post-drift.
    a_B_pre: float = -6.0
    a_B_post: float = 4.0
    b_B: float = 1.0


class SelectiveLabelsEnv:
    """
    Each tick: sample (g,z), produce y but only reveal it if approved (after delay) or probed.
    Reward delivered with label delivery: +1 if y=1 else -1.
    """
    def __init__(self, cfg: EnvCfg):
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        self.t = 0

    def true_p(self, g: int, z: float, t: int) -> float:
        if g == 0:
            a, b = self.cfg.a_A, self.cfg.b_A
        else:
            a = self.cfg.a_B_post if t >= self.cfg.drift_t else self.cfg.a_B_pre
            b = self.cfg.b_B
        return sigm(a + b * z)

    def sample(self) -> Tuple[int, float, int, float]:
        g = 1 if (self.rng.random() < self.cfg.p_group_B) else 0
        mu = self.cfg.mu_B if g == 1 else self.cfg.mu_A
        z = self.rng.gauss(mu, self.cfg.sigma)
        p = self.true_p(g, z, self.t)
        y = 1 if (self.rng.random() < p) else 0
        return g, z, y, p

    def step_time(self) -> None:
        self.t += 1


# ----------------------------
# Two-head NN (no cross-group transfer)
# ----------------------------

@dataclass
class NetCfg:
    hidden: int = 32
    lr: float = 0.03
    l2: float = 1e-4
    seed: int = 1
    w1_scale: float = 0.15
    w2_scale: float = 0.10
    b1_const: float = 0.2


class TwoHeadMLP:
    """
    Input is 4D: [zA, 1A, zB, 1B]
    Only the active group's slots are nonzero; only that head updates.
    This prevents "A training fixes B" which ruined your earlier plots.
    """
    def __init__(self, cfg: NetCfg):
        self.cfg = cfg
        rng = np.random.default_rng(cfg.seed)
        H = cfg.hidden
        self.W1 = rng.normal(0.0, cfg.w1_scale, size=(H, 4)).astype(np.float64)
        self.W2 = rng.normal(0.0, cfg.w2_scale, size=(2, H)).astype(np.float64)
        self.b1 = np.full((H,), cfg.b1_const, dtype=np.float64)

    @staticmethod
    def x(g: int, z: float) -> np.ndarray:
        if g == 0:
            return np.array([z, 1.0, 0.0, 0.0], dtype=np.float64)
        return np.array([0.0, 0.0, z, 1.0], dtype=np.float64)

    def predict(self, g: int, z: float) -> float:
        x = self.x(g, z)
        z1 = self.W1 @ x + self.b1
        h = np.maximum(0.0, z1)
        logit = float(self.W2[g] @ h)
        return sigm(logit)

    def update(self, g: int, z: float, y: int) -> float:
        x = self.x(g, z)
        z1 = self.W1 @ x + self.b1
        h = np.maximum(0.0, z1)
        logit = float(self.W2[g] @ h)
        p = clamp(sigm(logit))
        y_f = float(y)

        loss = -(y_f * math.log(p) + (1.0 - y_f) * math.log(1.0 - p))
        dlogit = (p - y_f)  # scalar

        # W2 grad for head g only
        dW2g = dlogit * h + self.cfg.l2 * self.W2[g]
        self.W2[g] -= self.cfg.lr * dW2g

        # backprop to W1
        dh = self.W2[g] * dlogit
        dz1 = dh * (z1 > 0.0).astype(np.float64)
        dW1 = np.outer(dz1, x) + self.cfg.l2 * self.W1
        self.W1 -= self.cfg.lr * dW1

        return loss


# ----------------------------
# Policy with epsilon schedule
# ----------------------------

@dataclass
class PolicyCfg:
    thr: float = 0.85

    eps_warmup: float = 0.25   # high early exploration to learn "B is bad"
    warmup_T: int = 1500

    eps_final: float = 0.0001  # collapse exploration -> lock-in
    seed: int = 7


class Policy:
    def __init__(self, cfg: PolicyCfg):
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)

    def eps(self, t: int) -> float:
        return self.cfg.eps_warmup if t < self.cfg.warmup_T else self.cfg.eps_final

    def act(self, t: int, p_hat: float) -> int:
        e = self.eps(t)
        if self.rng.random() < e:
            return 1 if (self.rng.random() < 0.5) else 0
        return 1 if (p_hat >= self.cfg.thr) else 0


# ----------------------------
# Runner + ASCII visualization
# ----------------------------

@dataclass
class RunCfg:
    mode: str = "locked"     # locked | probed
    probe_rate: float = 0.0  # when deny, buy label with this prob (only in probed)
    probe_cost: float = 0.05
    animate: bool = False
    render_every: int = 100
    render_window: int = 400   # render densely around drift
    plot: bool = False


def bar(p: float, width: int = 40) -> str:
    p = min(1.0, max(0.0, p))
    n = int(p * width + 1e-12)
    return "[" + ("#" * n) + (" " * (width - n)) + f"] {p:0.3f}"

def render(t: int, env: SelectiveLabelsEnv, pol: Policy, net: TwoHeadMLP, stats: Dict) -> None:
    clear_screen()
    drift = env.cfg.drift_t
    line = f"t={t:5d}  drift_t={drift}  delay={env.cfg.delay}  mode={stats['mode']}  eps={stats['eps']:.6f}  thr={pol.cfg.thr:.2f}"
    print(line)
    if t == drift:
        print(">>> DRIFT NOW: group B flips bad -> good (oracle B should jump; locked agent should stay blind).")
    print(f"reward={stats['reward']:+.2f}  pending={stats['pending']:3d}  labels(A,B)=({stats['labels_A']},{stats['labels_B']})")
    print(f"approve_rate(A,B)=({stats['apprA']:.4f},{stats['apprB']:.4f})   (window={stats['win']})")
    print("")

    print("Group B (holdout):")
    print("  P_true_B:", bar(stats["p_true_B"]))
    print("  P_hat_B :", bar(stats["p_hat_B"]))
    print("  labels_B_in_window:", stats["labelsB_win"], "  approves_B_in_window:", stats["apprB_win"])
    print("")

    print("Quick sanity: if LOCKED is working, post-drift labels_B_in_window should be ~0 and P_hat_B should stay low while P_true_B is high.")


def run(env_cfg: EnvCfg, net_cfg: NetCfg, pol_cfg: PolicyCfg, run_cfg: RunCfg) -> Dict[str, np.ndarray]:
    env = SelectiveLabelsEnv(env_cfg)
    net = TwoHeadMLP(net_cfg)
    pol = Policy(pol_cfg)
    rng_probe = random.Random(env_cfg.seed + 2025)

    # pending: (deliver_t, g, z, y)
    pending: Deque[Tuple[int, int, float, int]] = deque()

    T = env_cfg.T
    reward = 0.0

    # logs
    approve_B = np.zeros(T, dtype=np.float64)
    approve_A = np.zeros(T, dtype=np.float64)
    labels_B = np.zeros(T, dtype=np.float64)
    labels_A = np.zeros(T, dtype=np.float64)
    ptrueB = np.zeros(T, dtype=np.float64)
    phatB = np.zeros(T, dtype=np.float64)

    seenA = 0
    seenB = 0
    apprA_c = 0
    apprB_c = 0
    labA_c = 0
    labB_c = 0

    for t in range(T):
        # Deliver delayed labels
        while pending and pending[0][0] <= t:
            _, g_d, z_d, y_d = pending.popleft()
            net.update(g_d, z_d, y_d)
            reward += (1.0 if y_d == 1 else -1.0)
            if g_d == 0:
                labels_A[t] += 1.0
                labA_c += 1
            else:
                labels_B[t] += 1.0
                labB_c += 1

        g, z, y, p = env.sample()

        # Holdout estimate for B at this t (fixed z=0 for clarity)
        ptrueB[t] = env.true_p(1, 0.0, t)
        phatB[t]  = net.predict(1, 0.0)

        p_hat = net.predict(g, z)
        a = pol.act(t, p_hat)

        if g == 0:
            seenA += 1
            apprA_c += a
            approve_A[t] = a
        else:
            seenB += 1
            apprB_c += a
            approve_B[t] = a

        did_probe = False
        if a == 1:
            pending.append((t + env_cfg.delay, g, z, y))
        else:
            if run_cfg.probe_rate > 0.0 and rng_probe.random() < run_cfg.probe_rate:
                did_probe = True
                net.update(g, z, y)
                reward -= run_cfg.probe_cost
                if g == 0:
                    labels_A[t] += 1.0
                    labA_c += 1
                else:
                    labels_B[t] += 1.0
                    labB_c += 1

        # ASCII render
        if run_cfg.animate:
            near = abs(t - env_cfg.drift_t) <= run_cfg.render_window
            if (t % run_cfg.render_every == 0) or (t == env_cfg.drift_t) or near:
                win = 300
                lo = max(0, t - win)
                hi = t + 1

                stats = {
                    "mode": run_cfg.mode,
                    "eps": pol.eps(t),
                    "reward": reward,
                    "pending": len(pending),
                    "labels_A": labA_c,
                    "labels_B": labB_c,
                    "apprA": apprA_c / max(1, seenA),
                    "apprB": apprB_c / max(1, seenB),
                    "win": win,
                    "labelsB_win": int(np.sum(labels_B[lo:hi])),
                    "apprB_win": int(np.sum(approve_B[lo:hi])),
                    "p_true_B": float(ptrueB[t]),
                    "p_hat_B": float(phatB[t]),
                }
                render(t, env, pol, net, stats)

        env.step_time()

    out = {
        "approve_B": approve_B,
        "approve_A": approve_A,
        "cum_labels_B": np.cumsum(labels_B),
        "cum_labels_A": np.cumsum(labels_A),
        "ptrueB": ptrueB,
        "phatB": phatB,
    }
    return out


def plot_out(path: str, env_cfg: EnvCfg, locked: Dict[str, np.ndarray], probed: Optional[Dict[str, np.ndarray]] = None) -> None:
    if not HAVE_PLT:
        print("matplotlib not installed; skipping plots.")
        return

    T = env_cfg.T
    x = np.arange(T)
    drift = env_cfg.drift_t

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(x, moving_avg(locked["approve_B"], 200), label="locked approve_B")
    if probed is not None:
        plt.plot(x, moving_avg(probed["approve_B"], 200), label="probed approve_B")
    plt.axvline(drift, linestyle="--")
    plt.title("Approval rate for group B (moving avg)")
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(x, locked["cum_labels_B"], label="locked cum_labels_B")
    if probed is not None:
        plt.plot(x, probed["cum_labels_B"], label="probed cum_labels_B")
    plt.axvline(drift, linestyle="--")
    plt.title("Cumulative labels observed for group B")
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(x, locked["ptrueB"], label="P_true_B (oracle)")
    plt.plot(x, locked["phatB"], label="locked P_hat_B")
    if probed is not None:
        plt.plot(x, probed["phatB"], label="probed P_hat_B")
    plt.axvline(drift, linestyle="--")
    plt.title("B: oracle vs agent belief at z=0")
    plt.legend()

    plt.subplot(2, 2, 4)
    # show "belief gap" for locked (and probed)
    gap_locked = locked["ptrueB"] - locked["phatB"]
    plt.plot(x, moving_avg(gap_locked, 200), label="locked (P_true - P_hat)")
    if probed is not None:
        gap_probed = probed["ptrueB"] - probed["phatB"]
        plt.plot(x, moving_avg(gap_probed, 200), label="probed (P_true - P_hat)")
    plt.axvline(drift, linestyle="--")
    plt.title("Belief gap on B (moving avg)")
    plt.legend()

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    print(f"Saved plot: {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["locked", "probed"], default="locked")
    ap.add_argument("--animate", action="store_true")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--probe-rate", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    env_cfg = EnvCfg(seed=args.seed)
    net_cfg = NetCfg(seed=args.seed + 1)
    pol_cfg = PolicyCfg(seed=args.seed + 7)

    if args.mode == "locked":
        run_cfg = RunCfg(mode="locked", probe_rate=0.0, animate=args.animate, plot=args.plot)
        out = run(env_cfg, net_cfg, pol_cfg, run_cfg)
        if args.plot:
            plot_out("lockin_locked.png", env_cfg, out, None)
    else:
        run_cfg_locked = RunCfg(mode="locked", probe_rate=0.0, animate=False, plot=False)
        run_cfg_probed = RunCfg(mode="probed", probe_rate=args.probe_rate, animate=args.animate, plot=args.plot)
        locked = run(env_cfg, net_cfg, pol_cfg, run_cfg_locked)
        probed = run(env_cfg, net_cfg, pol_cfg, run_cfg_probed)
        if args.plot:
            plot_out("lockin_locked_vs_probed.png", env_cfg, locked, probed)


if __name__ == "__main__":
    main()
