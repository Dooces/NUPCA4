"""nupca3/sim/runner.py

Non-authoritative runner for quick smoke tests.
"""

from __future__ import annotations

from typing import List

from ..agent import NUPCA3Agent
from ..types import StepTrace
from .worlds import ToyWorld


def run_smoke(agent: NUPCA3Agent, world: ToyWorld, n_steps: int = 100, seed: int = 0) -> List[dict]:
    obs = world.reset(seed=seed)
    traces: List[dict] = []
    for _ in range(n_steps):
        action, trace = agent.step(obs)
        traces.append(trace)
        obs, done = world.step(action)
        if done:
            break
    return traces
