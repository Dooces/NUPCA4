"""
nupca3_scale_test.py

Tests whether NUPCA3 scales beyond trivial environments.

Test progression:
1. Dimension scaling: D=8 → D=64 → D=256 → D=1024
2. Nonlinearity: Linear dynamics → Nonlinear dynamics
3. Visual patterns: State vectors → Synthetic images with structure
4. Foveation realism: Full observation → Small patch observation

This will reveal where the system breaks.

Run: python nupca3_scale_test.py
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import time


# =============================================================================
# Node (same as before, but handles larger D)
# =============================================================================

@dataclass
class Node:
    name: str
    block_id: int
    mask: np.ndarray
    W: np.ndarray
    b: np.ndarray
    Sigma: np.ndarray
    pi: float = 0.5
    L: float = 1.0
    is_anchor: bool = False
    
    times_active: int = 0
    times_responsible: int = 0
    cumulative_error: float = 0.0
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.clip(self.W @ x + self.b, -5, 5)
    
    def precision(self) -> np.ndarray:
        prec = np.zeros_like(self.Sigma)
        mask_bool = self.mask.astype(bool)
        prec[mask_bool] = 1.0 / np.maximum(self.Sigma[mask_bool], 1e-6)
        return prec
    
    def avg_error(self) -> float:
        if self.times_active == 0:
            return float('inf')
        return self.cumulative_error / self.times_active


@dataclass
class WitnessBuffer:
    max_size: int = 50
    xs: List[np.ndarray] = field(default_factory=list)
    ys: List[np.ndarray] = field(default_factory=list)
    
    def add(self, x: np.ndarray, y: np.ndarray):
        self.xs.append(x.copy())
        self.ys.append(y.copy())
        if len(self.xs) > self.max_size:
            self.xs.pop(0)
            self.ys.pop(0)
    
    def get_recent(self, n: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        return self.xs[-n:], self.ys[-n:]


# =============================================================================
# Environments: Increasing Difficulty
# =============================================================================

class LinearEnvironment:
    """Original toy environment - baseline."""
    
    def __init__(self, D: int, B: int, n_regimes: int = 3):
        self.D = D
        self.B = B
        self.n_regimes = n_regimes
        self.block_size = D // B
        self.block_dims = [
            np.arange(b * self.block_size, (b + 1) * self.block_size)
            for b in range(B)
        ]
        self.regime_duration = 150
        self.noise_std = 0.02
        
        np.random.seed(42)
        self.dynamics = {}
        for r in range(n_regimes):
            for b in range(B):
                bs = self.block_size
                W = np.eye(bs) * (0.85 + 0.05 * r)
                bias = np.ones(bs) * (0.05 * (r - 1))
                self.dynamics[(r, b)] = (W, bias)
        
        self.t = 0
        self.state = np.zeros(D)
        self.reset()
    
    def reset(self):
        self.t = 0
        self.state = np.random.randn(self.D) * 0.1
    
    def current_regime(self) -> int:
        return (self.t // self.regime_duration) % self.n_regimes
    
    def step(self) -> Tuple[np.ndarray, int]:
        regime = self.current_regime()
        next_state = np.zeros(self.D)
        for b in range(self.B):
            dims = self.block_dims[b]
            W, bias = self.dynamics[(regime, b)]
            next_state[dims] = W @ self.state[dims] + bias
        
        next_state = np.clip(next_state, -3, 3)
        obs = next_state + np.random.randn(self.D) * self.noise_std
        self.state = next_state.copy()
        self.t += 1
        return obs, regime


class NonlinearEnvironment:
    """Nonlinear dynamics - tests A7.1 linear substrate limitation."""
    
    def __init__(self, D: int, B: int, n_regimes: int = 3):
        self.D = D
        self.B = B
        self.n_regimes = n_regimes
        self.block_size = D // B
        self.block_dims = [
            np.arange(b * self.block_size, (b + 1) * self.block_size)
            for b in range(B)
        ]
        self.regime_duration = 150
        self.noise_std = 0.02
        
        self.t = 0
        self.state = np.zeros(D)
        self.reset()
    
    def reset(self):
        self.t = 0
        self.state = np.random.randn(self.D) * 0.1
    
    def current_regime(self) -> int:
        return (self.t // self.regime_duration) % self.n_regimes
    
    def step(self) -> Tuple[np.ndarray, int]:
        regime = self.current_regime()
        next_state = np.zeros(self.D)
        
        for b in range(self.B):
            dims = self.block_dims[b]
            x_block = self.state[dims]
            
            if regime == 0:
                # Tanh saturation
                next_state[dims] = np.tanh(x_block * 1.5) * 0.8
            elif regime == 1:
                # Quadratic
                next_state[dims] = 0.5 * x_block - 0.3 * x_block ** 2
            else:
                # Oscillatory nonlinear
                next_state[dims] = 0.7 * np.sin(x_block * 2) + 0.2 * x_block
        
        next_state = np.clip(next_state, -3, 3)
        obs = next_state + np.random.randn(self.D) * self.noise_std
        self.state = next_state.copy()
        self.t += 1
        return obs, regime


class SyntheticVisualEnvironment:
    """
    Synthetic "image" environment with spatial structure.
    
    State is a flattened 2D grid with moving patterns.
    Tests whether foveation over spatial blocks captures structure.
    """
    
    def __init__(self, grid_size: int = 16, n_regimes: int = 3):
        self.grid_size = grid_size
        self.D = grid_size * grid_size
        self.B = 4  # 2x2 blocks
        self.n_regimes = n_regimes
        self.regime_duration = 200
        self.noise_std = 0.05
        
        self.block_grid = grid_size // 2
        self.block_size = self.block_grid * self.block_grid
        
        # Block dims: top-left, top-right, bottom-left, bottom-right
        self.block_dims = []
        for by in range(2):
            for bx in range(2):
                dims = []
                for y in range(self.block_grid):
                    for x in range(self.block_grid):
                        gy = by * self.block_grid + y
                        gx = bx * self.block_grid + x
                        dims.append(gy * grid_size + gx)
                self.block_dims.append(np.array(dims))
        
        self.t = 0
        self.state = np.zeros(self.D)
        
        # Pattern state (position, velocity)
        self.pattern_pos = np.array([grid_size // 4, grid_size // 4], dtype=float)
        self.pattern_vel = np.array([0.3, 0.2])
        
        self.reset()
    
    def reset(self):
        self.t = 0
        self.pattern_pos = np.array([self.grid_size // 4, self.grid_size // 4], dtype=float)
        self.state = self._render()
    
    def current_regime(self) -> int:
        return (self.t // self.regime_duration) % self.n_regimes
    
    def _render(self) -> np.ndarray:
        """Render current pattern to flat state."""
        img = np.zeros((self.grid_size, self.grid_size))
        regime = self.current_regime()
        
        cx, cy = int(self.pattern_pos[0]) % self.grid_size, int(self.pattern_pos[1]) % self.grid_size
        
        if regime == 0:
            # Gaussian blob
            for y in range(self.grid_size):
                for x in range(self.grid_size):
                    dx = min(abs(x - cx), self.grid_size - abs(x - cx))
                    dy = min(abs(y - cy), self.grid_size - abs(y - cy))
                    img[y, x] = np.exp(-(dx**2 + dy**2) / 8)
        elif regime == 1:
            # Horizontal bar
            for y in range(max(0, cy-1), min(self.grid_size, cy+2)):
                img[y, :] = 0.8
        else:
            # Diagonal edge
            for y in range(self.grid_size):
                for x in range(self.grid_size):
                    if abs(x - y - (cx - cy)) < 2:
                        img[y, x] = 0.8
        
        return img.flatten()
    
    def step(self) -> Tuple[np.ndarray, int]:
        regime = self.current_regime()
        
        # Move pattern
        if regime == 0:
            self.pattern_vel = np.array([0.3, 0.2])
        elif regime == 1:
            self.pattern_vel = np.array([0.0, 0.4])
        else:
            self.pattern_vel = np.array([0.25, 0.25])
        
        self.pattern_pos += self.pattern_vel
        self.pattern_pos = self.pattern_pos % self.grid_size
        
        next_state = self._render()
        obs = next_state + np.random.randn(self.D) * self.noise_std
        
        self.state = next_state.copy()
        self.t += 1
        return obs, regime


# =============================================================================
# NUPCA3 Agent (handles variable D)
# =============================================================================

class NUPCA3Agent:
    def __init__(self, D: int, B: int, block_dims: List[np.ndarray], F: int = 1):
        self.D = D
        self.B = B
        self.F = F
        self.block_dims = block_dims
        self.block_size = len(block_dims[0])
        
        # Parameters (may need scaling with D)
        self.alpha_cov = 0.35
        self.G = 50  # Reduced for faster testing
        self.beta_r = 0.20
        self.theta_learn = 0.15  # Slightly relaxed
        self.theta_spawn = 0.10  # Lower for larger D
        self.beta_R = 0.10
        self.K_spawn = 3
        self.theta_alias = 0.05
        self.learning_rate = 0.02  # Reduced for stability
        self.max_grad = 0.3
        self.pi_decay = 0.95
        self.pi_boost = 1.01
        self.max_nodes_per_block = 32
        
        # State
        self.library: Dict[int, List[Node]] = defaultdict(list)
        self.witness_buffers: Dict[int, WitnessBuffer] = {
            b: WitnessBuffer() for b in range(B)
        }
        self.block_age = np.zeros(B, dtype=int)
        self.block_residual = np.ones(B) * 0.5
        self.persistent_residual = np.zeros(B)
        self.coverage_visits = np.zeros(B, dtype=int)
        self.high_residual_streak = np.zeros(B, dtype=int)
        self.x_prev = np.zeros(D)
        self.t = 0
        
        # Stats
        self.spawns = 0
        self.rejected_aliases = 0
        self.replacements = 0
        self.update_time = 0.0
        self.fusion_time = 0.0
        
        self._init_anchors()
    
    def _init_anchors(self):
        for b in range(self.B):
            block_dims = self.block_dims[b]
            mask = np.zeros(self.D, dtype=int)
            mask[block_dims] = 1
            
            # Block-local identity
            W = np.zeros((self.D, self.D))
            for k in block_dims:
                W[k, k] = 0.9
            
            anchor = Node(
                name=f"anchor_b{b}",
                block_id=b,
                mask=mask,
                W=W,
                b=np.zeros(self.D),
                Sigma=np.ones(self.D) * 0.5,
                pi=0.5,
                is_anchor=True
            )
            self.library[b].append(anchor)
    
    def select_fovea(self) -> List[int]:
        age_plus = np.maximum(0, self.block_age - self.G)
        score = self.block_residual + self.alpha_cov * np.log1p(age_plus.astype(float))
        return list(np.argsort(-score)[:self.F])
    def fusion_predict(self, x: np.ndarray, active_set: List[Node]) -> Tuple[np.ndarray, np.ndarray]:
        t0 = time.time()

        numer = np.zeros(self.D)
        denom = np.zeros(self.D)

        for node in active_set:
            mu = node.predict(x)
            prec = node.pi * node.precision()  # A10.3: punish affects inference
            numer += prec * mu
            denom += prec

        x_hat = np.zeros(self.D)
        Sigma_global = np.full(self.D, np.inf)

        covered = denom > 0
        x_hat[covered] = numer[covered] / denom[covered]
        x_hat[~covered] = x[~covered]
        Sigma_global[covered] = 1.0 / denom[covered]

        self.fusion_time += time.time() - t0
        return x_hat, Sigma_global


    def get_active_set(self) -> List[Node]:
        active = []
        for nodes in self.library.values():
            active.extend(nodes)
        return active
    
    def update_node(
        self,
        node: Node,
        x_prev: np.ndarray,
        x_completed: np.ndarray,
        x_predicted: np.ndarray,   # kept for signature compatibility; unused by A10.3 here
        observed_dims: np.ndarray
    ) -> str:
        """
        A10.3 Responsibility-Gated Learning (axiom-accurate, node-specific):

        - Compute node-specific error on observed overlap of its footprint using the node's own prediction:
                mu_j = node.predict(x_prev)
                err_j = mean(|x_completed[overlap] - mu_j[overlap]|)

        - Responsible iff err_j <= theta_learn.

        - If responsible: update (W, b) on footprint; increase pi.
        - If not responsible: update pi only (punish), do NOT update (W, b).

        Notes:
        - x_predicted is intentionally ignored; using fused prediction here destroys node-specific pi semantics.
        - Uses observed_dims ∩ node.mask as the measurement support, matching your footprint-local axiom.
        """
        t0 = time.time()

        # Node footprint indices
        node_dims = np.where(node.mask)[0]
        obs_overlap = np.intersect1d(node_dims, observed_dims)

        if len(obs_overlap) == 0:
            return "SKIPPED"

        # Track activity timestamps (safe dynamic attrs; optional but useful for REST-only pruning later)
        node.last_active_t = self.t
        if not hasattr(node, "last_responsible_t"):
            node.last_responsible_t = -10**9

        # Node's own one-step prediction (A7.1 form, node-local)
        mu_j = node.predict(x_prev)

        # Node-specific error on observed overlap (A10.3)
        err = float(np.mean(np.abs(x_completed[obs_overlap] - mu_j[obs_overlap])))

        node.times_active += 1
        node.cumulative_error += err

        responsible = err <= self.theta_learn

        if responsible:
            node.times_responsible += 1
            node.last_responsible_t = self.t

            # Footprint-local parameter update toward x_completed
            error = x_completed - mu_j

            footprint = node.mask.astype(bool)
            error[~footprint] = 0.0

            grad = np.clip(error, -self.max_grad, self.max_grad)

            # Bias update
            node.b = np.clip(node.b + self.learning_rate * grad, -2, 2)

            # Row-wise W update restricted to footprint rows (your large-D optimization)
            for i in node_dims:
                node.W[i, :] = np.clip(
                    node.W[i, :] + self.learning_rate * grad[i] * x_prev,
                    -2, 2
                )

            # Reliability reward
            node.pi = min(1.0, node.pi * self.pi_boost)

            self.update_time += time.time() - t0
            return "UPDATED"
        else:
            # Punish-only: preserve learned dynamics, degrade influence
            node.pi = max(0.01, node.pi * self.pi_decay)

            self.update_time += time.time() - t0
            return "PUNISHED"

    def compute_delta(self, node_p: Node, node_q: Node, block_id: int) -> float:
        xs, _ = self.witness_buffers[block_id].get_recent(10)
        if len(xs) < 3:
            return float('inf')
        
        block_dims = self.block_dims[block_id]
        deltas = []
        for x in xs:
            mu_p = node_p.predict(x)
            mu_q = node_q.predict(x)
            delta = np.mean(np.abs(mu_p[block_dims] - mu_q[block_dims]))
            deltas.append(delta)
        
        return np.mean(deltas)
    def update_margins_and_mode(self) -> None:
        # Lazy init to avoid touching __init__
        if not hasattr(self, "rest"):
            self.rest = False
            self.P_rest = 0.0
            self.T_since_rest = 0
            self.T_in_rest = 0

            # Default schedule (tune later; this is just to make REST exist)
            self.theta_demand_enter = 0.90
            self.theta_demand_exit = 0.75
            self.T_max_wake = 600
            self.T_max_rest = 120

        if not self.rest:
            # OPERATING: accumulate rest pressure
            self.T_since_rest += 1
            self.P_rest += 0.005

            if (self.P_rest > self.theta_demand_enter) or (self.T_since_rest > self.T_max_wake):
                self.rest = True
                self.T_in_rest = 0
        else:
            # REST: discharge rest pressure
            self.T_in_rest += 1
            self.P_rest *= 0.93

            if (self.P_rest < self.theta_demand_exit) or (self.T_in_rest > self.T_max_rest):
                self.rest = False
                self.T_since_rest = 0

    def insert_with_anti_aliasing(
        self,
        block_id: int,
        candidate: Node,
        x_prev: np.ndarray,
        x_completed: np.ndarray
    ) -> str:
        """
        A4.4 Anti-aliasing (axiom-faithful):
        - At insertion time, compare candidate to incumbents using Δ computed under the system's own cue basis
            (witness-buffer x cues).
        - If aliased (Δ < θ_alias): REPLACE only if candidate is strictly better under witness-evaluated prediction error;
            otherwise REJECT.
        - If distinguishable: ADD if under hard cap else REJECTED_CAP.
        Notes:
        - Replacement/betterment is evaluated on witness buffer pairs, excluding the most recent witness
            to avoid tautology when the candidate was fit from the current (x_prev, x_completed).
        - Anchors are treated as normal incumbents for aliasing/replacement decisions (no exemption).
        - No eviction/garbage-collection is performed here (cap is respected).
        """
        DEBUG = bool(getattr(self, "debug", False))
        incumbents = self.library[block_id]
        block_dims = self.block_dims[block_id]

        # Hard cap: do not perform eviction here (keeps this function purely insertion-discipline).
        if len(incumbents) >= self.max_nodes_per_block:
            if DEBUG:
                print(
                    f"[AA-REJECTED_CAP] t={self.t} block={block_id} "
                    f"cap={self.max_nodes_per_block} current={len(incumbents)} candidate={candidate.name}"
                )
            return "REJECTED_CAP"

        # Candidate summary for debug
        def cand_summary() -> str:
            b_block = candidate.b[block_dims]
            return (
                f"name={candidate.name} pi={candidate.pi:.3f} "
                f"Sigma={float(candidate.Sigma[block_dims][0]):.3f} "
                f"b_absmean={float(np.mean(np.abs(b_block))):.3f}"
            )

        # Pull witness buffer evidence
        wb = self.witness_buffers[block_id]
        xs = wb.xs
        ys = wb.ys

        # Exclude the most recent witness to avoid evaluating on the same sample the candidate was fit to.
        # (In your step(), witness_buffers[b].add(self.x_prev, x_completed) happens before maybe_spawn().)
        xs_eval = xs[:-1] if len(xs) >= 2 else []
        ys_eval = ys[:-1] if len(ys) >= 2 else []

        MIN_EVIDENCE = 5  # require enough cue diversity to justify replacement of an aliased incumbent

        def witness_mse(node: Node) -> float:
            """Mean squared error on footprint over witness evaluation set."""
            if len(xs_eval) < MIN_EVIDENCE:
                return float("inf")
            errs = []
            for xw, yw in zip(xs_eval, ys_eval):
                pred = node.predict(xw)[block_dims]
                tgt = yw[block_dims]
                errs.append(np.mean((tgt - pred) ** 2))
            return float(np.mean(errs)) if errs else float("inf")

        min_delta = float("inf")
        closest_name = None

        # Anti-aliasing against ALL incumbents, including anchors.
        for i, inc in enumerate(incumbents):
            delta = self.compute_delta(inc, candidate, block_id)
            if delta < min_delta:
                min_delta = delta
                closest_name = inc.name

            if delta < self.theta_alias:
                # Aliased under cue basis.
                inc_err = witness_mse(inc)
                cand_err = witness_mse(candidate)

                # If insufficient witness evidence, do NOT replace on tautological current-sample fit.
                if not np.isfinite(inc_err) or not np.isfinite(cand_err):
                    if DEBUG:
                        print(
                            f"[AA-REJECT] t={self.t} block={block_id} aliased_with={inc.name} "
                            f"Δ={delta:.5f} (<θ={self.theta_alias:.5f}) "
                            f"reason=insufficient_witness (need≥{MIN_EVIDENCE}, have={len(xs_eval)}) "
                            f"candidate=({cand_summary()})"
                        )
                    self.rejected_aliases += 1
                    return "REJECTED"

                # Strictly better under witness-evaluated error
                if cand_err < 0.9 * (inc_err + 1e-12):
                    if DEBUG:
                        ratio = cand_err / (inc_err + 1e-12)
                        print(
                            f"[AA-REPLACE] t={self.t} block={block_id} aliased_with={inc.name} "
                            f"Δ={delta:.5f} (<θ={self.theta_alias:.5f}) "
                            f"inc_wMSE={inc_err:.6f} cand_wMSE={cand_err:.6f} (cand/inc={ratio:.3f}) "
                            f"witness_n={len(xs_eval)} candidate=({cand_summary()})"
                        )
                    incumbents[i] = candidate
                    self.replacements += 1
                    return "REPLACED"
                else:
                    if DEBUG:
                        ratio = cand_err / (inc_err + 1e-12)
                        print(
                            f"[AA-REJECT] t={self.t} block={block_id} aliased_with={inc.name} "
                            f"Δ={delta:.5f} (<θ={self.theta_alias:.5f}) "
                            f"inc_wMSE={inc_err:.6f} cand_wMSE={cand_err:.6f} (cand/inc={ratio:.3f}) "
                            f"witness_n={len(xs_eval)} candidate=({cand_summary()})"
                        )
                    self.rejected_aliases += 1
                    return "REJECTED"

        # Distinguishable under cue basis: add.
        incumbents.append(candidate)

        if DEBUG:
            md = min_delta if np.isfinite(min_delta) else float("inf")
            print(
                f"[AA-ADD] t={self.t} block={block_id} distinguishable "
                f"(minΔ={md:.5f} vs θ={self.theta_alias:.5f}, closest={closest_name}, "
                f"witness_eval_n={len(xs_eval)}) candidate=({cand_summary()})"
            )

        return "ADDED"


    def maybe_spawn(self, block_id: int, x_prev: np.ndarray, x_completed: np.ndarray) -> Optional[str]:
        """
        A12.4 SPAWN (axiom-faithful):
        Spawn iff:
            - coverage_visits >= K_spawn
            - persistent_residual > theta_spawn
            - high_residual_streak >= 2  (your harness gate)
            - AND no incumbent explains under witness cues.

        Critical axiom alignment:
        If an incumbent explains, that residual is NOT "unexplained", so we extinguish the
        *unexplained* streak and damp the persistent residual accumulator for the block.
        """
        DEBUG = bool(getattr(self, "debug", False))

        # ---- gates ----
        if self.coverage_visits[block_id] < self.K_spawn:
            return None
        if self.persistent_residual[block_id] <= self.theta_spawn:
            return None
        if self.high_residual_streak[block_id] < 2:
            return None

        block_dims = self.block_dims[block_id]
        incumbents = self.library[block_id]

        # ---- incumbent-explain check (witness cue basis; exclude newest to avoid tautology) ----
        wb = self.witness_buffers[block_id]
        xs = wb.xs
        ys = wb.ys
        xs_eval = xs[:-1] if len(xs) >= 2 else []
        ys_eval = ys[:-1] if len(ys) >= 2 else []

        MIN_EVIDENCE = 5

        def wMAE(node: Node) -> float:
            errs = []
            for xw, yw in zip(xs_eval, ys_eval):
                pred = node.predict(xw)[block_dims]
                tgt = yw[block_dims]
                errs.append(np.mean(np.abs(tgt - pred)))
            return float(np.mean(errs)) if errs else float("inf")

        best_node = None
        best_mae = float("inf")

        if incumbents and len(xs_eval) >= MIN_EVIDENCE:
            for n in incumbents:
                mae = wMAE(n)
                if mae < best_mae:
                    best_mae = mae
                    best_node = n

            # If an incumbent explains under the cue basis, don't spawn.
            if best_mae <= self.theta_spawn:
                # AXIOM-CRITICAL: this residual is explained => extinguish "unexplained persistence"
                self.high_residual_streak[block_id] = 0
                self.persistent_residual[block_id] *= 0.5

                if DEBUG:
                    print(
                        f"[SPAWN-SKIP] t={self.t} block={block_id} reason=incumbent_explains "
                        f"best={best_node.name if best_node else None} best_wMAE={best_mae:.4f} "
                        f"(<=θ_spawn={self.theta_spawn:.4f}) "
                        f"persist→{self.persistent_residual[block_id]:.4f} streak→{int(self.high_residual_streak[block_id])} "
                        f"witness_eval_n={len(xs_eval)}"
                    )
                return None

        if DEBUG:
            print(
                f"[SPAWN-TRY] t={self.t} block={block_id} "
                f"coverage={int(self.coverage_visits[block_id])} "
                f"persist={self.persistent_residual[block_id]:.4f} (θ={self.theta_spawn:.4f}) "
                f"streak={int(self.high_residual_streak[block_id])} witness_eval_n={len(xs_eval)}"
            )

        # ---- candidate construction (footprint-local) ----
        mask = np.zeros(self.D, dtype=int)
        mask[block_dims] = 1

        W = np.zeros((self.D, self.D))
        for k in block_dims:
            W[k, k] = 0.9

        b = np.zeros(self.D)
        pred = W @ x_prev
        b_block = np.clip(x_completed[block_dims] - pred[block_dims], -1, 1)
        b[block_dims] = b_block

        # ---- cold-start (untrusted until responsible) ----
        PI_SPAWN = 0.05
        SIGMA_SPAWN = 1.0
        Sigma = np.ones(self.D) * SIGMA_SPAWN

        candidate = Node(
            name=f"node_b{block_id}_t{self.t}",
            block_id=block_id,
            mask=mask,
            W=W,
            b=b,
            Sigma=Sigma,
            pi=PI_SPAWN
        )

        result = self.insert_with_anti_aliasing(block_id, candidate, x_prev, x_completed)

        if result in ("ADDED", "REPLACED"):
            self.spawns += 1
            self.persistent_residual[block_id] *= 0.5
            self.high_residual_streak[block_id] = 0

            if DEBUG:
                print(
                    f"[SPAWN-{result}] t={self.t} block={block_id} name={candidate.name} "
                    f"pi0={candidate.pi:.3f} Sigma0={float(candidate.Sigma[block_dims][0]):.3f} "
                    f"Wdiag_mean={float(np.mean([candidate.W[k, k] for k in block_dims])):.3f} "
                    f"b_mean={float(np.mean(b_block)):.3f} b_absmean={float(np.mean(np.abs(b_block))):.3f}"
                )
        else:
            if DEBUG:
                print(f"[SPAWN-{result}] t={self.t} block={block_id} name={candidate.name}")

        return result

    def step(self, y_obs: np.ndarray) -> Dict:
        self.t += 1

        # REST/OPERATING macrostate (axiom: consolidation phase exists)
        self.update_margins_and_mode()
        in_rest = bool(self.rest)

        fovea_blocks = self.select_fovea()
        observed_dims = np.concatenate([self.block_dims[b] for b in fovea_blocks])

        active_set = self.get_active_set()
        x_hat, _ = self.fusion_predict(self.x_prev, active_set)

        x_completed = x_hat.copy()
        x_completed[observed_dims] = y_obs[observed_dims]

        block_errors = {}

        for b in fovea_blocks:
            block_dims = self.block_dims[b]
            obs_in_block = np.intersect1d(block_dims, observed_dims)

            block_residual = float(np.mean(np.abs(y_obs[obs_in_block] - x_hat[obs_in_block]))) if len(obs_in_block) else 0.0
            block_errors[b] = block_residual

            # Tracking (A16.* style)
            self.block_age[b] = 0
            self.coverage_visits[b] += 1

            self.block_residual[b] = (1 - self.beta_r) * self.block_residual[b] + self.beta_r * block_residual
            self.persistent_residual[b] = (1 - self.beta_R) * self.persistent_residual[b] + self.beta_R * block_residual

            if block_residual > self.theta_spawn:
                self.high_residual_streak[b] += 1
            else:
                self.high_residual_streak[b] = max(0, self.high_residual_streak[b] - 1)

            # Witness buffer for anti-aliasing cues (A4.4 discipline)
            self.witness_buffers[b].add(self.x_prev.copy(), x_completed.copy())

            if not in_rest:
                # OPERATING: learning updates only (A10.3)
                for node in self.library[b]:
                    self.update_node(node, self.x_prev, x_completed, x_hat, observed_dims)
            else:
                # REST: structural edits only (SPAWN/PRUNE/MERGE would live here)
                # SPAWN in REST (axiom-faithful)
                self.maybe_spawn(b, self.x_prev, x_completed)

                # REST-only PRUNE: remove dead wood by reliability + exposure
                PI_PRUNE = 0.05
                MIN_ACTIVE = 200
                T_UNRESP = 1500  # allow long grace to avoid premature deletions

                kept = []
                for node in self.library[b]:
                    if getattr(node, "is_anchor", False):
                        kept.append(node)
                        continue

                    last_resp = getattr(node, "last_responsible_t", -10**9)
                    stale = (self.t - last_resp) > T_UNRESP
                    enough = node.times_active >= MIN_ACTIVE

                    if enough and stale and (node.pi < PI_PRUNE):
                        continue

                    kept.append(node)

                self.library[b] = kept

        for b in range(self.B):
            if b not in fovea_blocks:
                self.block_age[b] += 1

        self.x_prev = x_completed.copy()

        return {
            "fovea": fovea_blocks,
            "block_errors": block_errors,
            "rest": in_rest,
            "mean_residual": np.mean(list(block_errors.values())) if block_errors else 0.0
        }

# =============================================================================
# Test Runner
# =============================================================================

def run_test(env, agent, n_steps: int, name: str) -> Dict:
    """Run a single test configuration."""
    print(f"\n  Running {name}...")
    
    env.reset()
    agent.x_prev = env.state.copy() if hasattr(env, 'state') else np.zeros(env.D)
    
    error_history = []
    t0 = time.time()
    
    for step in range(n_steps):
        y_obs, regime = env.step()
        diag = agent.step(y_obs)
        error_history.append(diag["mean_residual"])
        
        if (step + 1) % 500 == 0:
            recent_err = np.mean(error_history[-100:])
            print(f"    Step {step+1}: error={recent_err:.4f}, nodes={sum(len(n) for n in agent.library.values())}")
    
    elapsed = time.time() - t0
    
    early_error = np.mean(error_history[:min(300, len(error_history)//3)])
    final_error = np.mean(error_history[-min(300, len(error_history)//3):])
    
    total_nodes = sum(len(nodes) for nodes in agent.library.values())
    
    return {
        "name": name,
        "D": env.D,
        "B": env.B,
        "early_error": early_error,
        "final_error": final_error,
        "improvement": (early_error - final_error) / (early_error + 1e-9),
        "total_nodes": total_nodes,
        "spawns": agent.spawns,
        "elapsed": elapsed,
        "steps_per_sec": n_steps / elapsed,
        "fusion_time": agent.fusion_time,
        "update_time": agent.update_time,
    }


def main():
    print("=" * 70)
    print("NUPCA3 v1.6.1 — SCALING & PERCEPTION TEST")
    print("=" * 70)
    
    results = []


    # =========================================================================
    # Test 1: Dimension Scaling (Linear Dynamics)
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 1: DIMENSION SCALING (Linear Dynamics)")
    print("=" * 70)
    
    for D in [8, 64, 256]:
        B = 4
        block_size = D // B
        block_dims = [np.arange(b * block_size, (b + 1) * block_size) for b in range(B)]
        
        env = LinearEnvironment(D, B, n_regimes=3)
        agent = NUPCA3Agent(D, B, block_dims, F=1)
        agent.debug = True
        n_steps = 10000 if D <= 64 else 2000
        result = run_test(env, agent, n_steps, f"Linear D={D}")
        results.append(result)
    
    # =========================================================================
    # Test 2: Nonlinear Dynamics
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 2: NONLINEAR DYNAMICS")
    print("=" * 70)
    
    for D in [8, 64]:
        B = 4
        block_size = D // B
        block_dims = [np.arange(b * block_size, (b + 1) * block_size) for b in range(B)]
        
        env = NonlinearEnvironment(D, B, n_regimes=3)
        agent = NUPCA3Agent(D, B, block_dims, F=1)
        agent.debug = True
        result = run_test(env, agent, 2500, f"Nonlinear D={D}")
        results.append(result)
    
    # =========================================================================
    # Test 3: Synthetic Visual Patterns
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 3: SYNTHETIC VISUAL PATTERNS")
    print("=" * 70)
    
    for grid_size in [8, 16]:
        env = SyntheticVisualEnvironment(grid_size=grid_size, n_regimes=3)
        agent = NUPCA3Agent(env.D, env.B, env.block_dims, F=1)
        agent.debug = True
        result = run_test(env, agent, 5000, f"Visual {grid_size}x{grid_size}")
        results.append(result)
    
    # =========================================================================
    # Results Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    print(f"\n{'Test':<25} {'D':>6} {'Improve':>10} {'Nodes':>6} {'Steps/s':>10} {'Status':<10}")
    print("-" * 70)
    
    for r in results:
        status = "✓" if r["improvement"] > 0.1 else ("~" if r["improvement"] > 0 else "✗")
        print(f"{r['name']:<25} {r['D']:>6} {r['improvement']*100:>9.1f}% {r['total_nodes']:>6} {r['steps_per_sec']:>10.1f} {status:<10}")
    
    # =========================================================================
    # Analysis
    # =========================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    
    # Check dimension scaling
    linear_results = [r for r in results if "Linear" in r["name"]]
    if len(linear_results) >= 2:
        small = linear_results[0]
        large = linear_results[-1]
        speedup_loss = small["steps_per_sec"] / large["steps_per_sec"]
        print(f"\n1. Dimension scaling:")
        print(f"   D={small['D']}: {small['steps_per_sec']:.1f} steps/s, {small['improvement']*100:.1f}% improvement")
        print(f"   D={large['D']}: {large['steps_per_sec']:.1f} steps/s, {large['improvement']*100:.1f}% improvement")
        print(f"   Speed ratio: {speedup_loss:.1f}x slower")
        
        if large["improvement"] > 0.05:
            print(f"   → Scales to D={large['D']} with learning preserved")
        else:
            print(f"   → FAILS to scale: learning degraded at D={large['D']}")
    
    # Check nonlinearity
    nonlinear_results = [r for r in results if "Nonlinear" in r["name"]]
    if nonlinear_results:
        print(f"\n2. Nonlinear dynamics:")
        for r in nonlinear_results:
            if r["improvement"] > 0.05:
                print(f"   D={r['D']}: {r['improvement']*100:.1f}% improvement → Linear substrate handles some nonlinearity")
            else:
                print(f"   D={r['D']}: {r['improvement']*100:.1f}% improvement → Linear substrate FAILS on nonlinear dynamics")
    
    # Check visual
    visual_results = [r for r in results if "Visual" in r["name"]]
    if visual_results:
        print(f"\n3. Visual patterns:")
        for r in visual_results:
            if r["improvement"] > 0.1:
                print(f"   {r['name']}: {r['improvement']*100:.1f}% improvement → Captures spatial structure")
            else:
                print(f"   {r['name']}: {r['improvement']*100:.1f}% improvement → FAILS on spatial patterns")
    
    # Overall verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    
    passed = sum(1 for r in results if r["improvement"] > 0.05)
    total = len(results)
    
    if passed == total:
        print(f"\n✓ All {total} tests show learning. System scales within tested range.")
    elif passed > total // 2:
        print(f"\n~ {passed}/{total} tests show learning. Partial scaling.")
    else:
        print(f"\n✗ Only {passed}/{total} tests show learning. System does NOT scale.")
    
    # Identify breaking points
    failures = [r for r in results if r["improvement"] <= 0.05]
    if failures:
        print("\nBreaking points:")
        for r in failures:
            print(f"  - {r['name']}: {r['improvement']*100:.1f}% (failed)")
    
    return 0 if passed >= total // 2 else 1


if __name__ == "__main__":
    np.random.seed(42)
    exit(main())