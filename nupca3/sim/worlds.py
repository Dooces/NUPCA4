"""nupca3/sim/worlds.py

Toy worlds to exercise the agent. Non-authoritative harness.

These worlds are placeholders; they are not part of the axioms.

The key is that they emit partial observations and accept fovea selections if you expand the API.


[AXIOM_CLARIFICATION_ADDENDUM — Representation & Naming]

- Terminology: identifiers like "Expert" in this codebase refer to NUPCA3 **abstraction/resonance nodes** (a "constellation"), not conventional Mixture-of-Experts "experts" or router-based MoE.

- Representation boundary (clarified intent of v1.5b): the completion/fusion operator (A7) is defined over an **encoded, multi-resolution abstraction vector** \(x(t)\). Raw pixels may exist only in a transient observation buffer for the current step; **raw pixel values must never be inserted into long-term storage** (library/cold storage) and must not persist across REST boundaries.

- Decomposition intuition: each node is an operator that *factors out* a predictable/resonant component on its footprint, leaving residual structure for other nodes (or for REST-time proposal) to capture. This is the intended "FFT-like" interpretation of masks/constellations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple, Dict

import time

import numpy as np

from ..types import EnvObs


@dataclass
class ToyWorld:
    D: int
    x: np.ndarray
    _t_w: int = field(init=False, default=0)

    def reset(self, seed: int = 0) -> EnvObs:
        rng = np.random.default_rng(seed)
        self.x = rng.normal(size=self.D)
        self._t_w = 1
        return EnvObs(
            x_partial={0: float(self.x[0])},
            opp=0.0,
            danger=0.0,
            t_w=self._t_w,
            wall_ms=int(time.perf_counter() * 1000),
        )

    def step(self, action: int) -> Tuple[EnvObs, bool]:
        # Simple linear drift + noise
        self.x = 0.99 * self.x + 0.01 * np.random.normal(size=self.D)
        done = False
        self._t_w = int(getattr(self, "_t_w", 0)) + 1
        wall_ms = int(time.perf_counter() * 1000)
        # Partial observe dim 0 only
        return (
            EnvObs(
                x_partial={0: float(self.x[0])},
                opp=0.0,
                danger=0.0,
                t_w=self._t_w,
                wall_ms=wall_ms,
            ),
            done,
        )


@dataclass
class ColorShapeWorld:
    """Discrete color×shape world used to test compositional generalization.

    State is a concatenated one-hot vector:
      x = [onehot(color, C) ; onehot(shape, S)]

    Dynamics are factorized (color and shape evolve independently) unless you use
    the `reject_pairs` option during training to withhold combinations.

    This world emits full observations by default (all dims in x_partial).
    """

    n_colors: int = 3
    n_shapes: int = 3
    P_color: np.ndarray = None  # shape (C,C)
    P_shape: np.ndarray = None  # shape (S,S)
    reject_pairs: set[tuple[int, int]] | None = None
    rng: np.random.Generator | None = None
    color: int = 0
    shape: int = 0
    _t_w: int = field(init=False, default=0)

    @property
    def D(self) -> int:
        return self.n_colors + self.n_shapes

    def _onehot(self) -> np.ndarray:
        x = np.zeros(self.D, dtype=float)
        x[self.color] = 1.0
        x[self.n_colors + self.shape] = 1.0
        return x

    def reset(self, seed: int = 0) -> EnvObs:
        self.rng = np.random.default_rng(seed)
        if self.P_color is None:
            self.P_color = _random_sticky_markov(self.n_colors, self.rng)
        if self.P_shape is None:
            self.P_shape = _random_sticky_markov(self.n_shapes, self.rng)
        # sample an initial pair that is allowed under reject_pairs
        while True:
            self.color = int(self.rng.integers(self.n_colors))
            self.shape = int(self.rng.integers(self.n_shapes))
            if not self.reject_pairs or (self.color, self.shape) not in self.reject_pairs:
                break
        x = self._onehot()
        self._t_w = 1
        return EnvObs(
            x_partial={i: float(x[i]) for i in range(self.D)},
            opp=0.0,
            danger=0.0,
            t_w=self._t_w,
            wall_ms=int(time.perf_counter() * 1000),
        )

    def step(self, action: int = 0) -> tuple[EnvObs, bool]:
        # independent transitions
        self.color = int(_sample_cat(self.P_color[self.color], self.rng))
        self.shape = int(_sample_cat(self.P_shape[self.shape], self.rng))
        # optionally reject withheld combinations (training-time only)
        if self.reject_pairs:
            tries = 0
            while (self.color, self.shape) in self.reject_pairs and tries < 50:
                # resample shape only to preserve marginal color dynamics
                self.shape = int(_sample_cat(self.P_shape[self.shape], self.rng))
                tries += 1
        x = self._onehot()
        done = False
        self._t_w = int(getattr(self, "_t_w", 0)) + 1
        wall_ms = int(time.perf_counter() * 1000)
        return (
            EnvObs(
                x_partial={i: float(x[i]) for i in range(self.D)},
                opp=0.0,
                danger=0.0,
                t_w=self._t_w,
                wall_ms=wall_ms,
            ),
            done,
        )


def _sample_cat(p: np.ndarray, rng: np.random.Generator) -> int:
    r = float(rng.random())
    c = 0.0
    for i, pi in enumerate(p):
        c += float(pi)
        if r <= c:
            return i
    return int(len(p) - 1)


def _random_sticky_markov(n: int, rng: np.random.Generator, stickiness: float = 0.7) -> np.ndarray:
    """Create a simple Markov chain with diagonal preference."""
    P = np.zeros((n, n), dtype=float)
    for i in range(n):
        base = rng.random(n)
        base = base / base.sum()
        base = (1.0 - stickiness) * base
        base[i] += stickiness
        base = base / base.sum()
        P[i] = base
    return P
