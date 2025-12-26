"""nupca3/geometry/buffer.py

Persistent observation buffer for partial observability.

Axiom coverage: A16.5 support (observation geometry) + supports A13 completion.


[AXIOM_CLARIFICATION_ADDENDUM — Buffer Semantics vs. "No Long-Term Pixel Storage"]

- A16.5 allows stale persistence only as an **input-side buffer** under partial observability. Under the clarified intent, this buffer is permitted for \(x(t)\) (abstraction state) and/or for raw observations *within the current step*, but **must be cleared at REST boundaries** and must never be used as a mechanism for long-term pixel storage.

- Current code may persist `x_last` across REST because REST/OPERATING boundaries are not yet wired into this helper. Treat that as a temporary scaffold; the authoritative rule is the axiom clarification above.
"""

from __future__ import annotations

import numpy as np

from ..config import AgentConfig
from ..types import ObservationBuffer, EnvObs


def init_observation_buffer(cfg: AgentConfig) -> ObservationBuffer:
    return ObservationBuffer(x_last=np.zeros(cfg.D, dtype=float))


def buffer_update(buf: ObservationBuffer, obs: EnvObs, cfg: AgentConfig) -> None:
    """Update stored last-seen values with partial observations.

    Inputs:
      - buf: ObservationBuffer (mutated in place)
      - obs: EnvObs (provides obs.x_partial)
      - cfg: AgentConfig (for bounds)

    Side effects:
      - updates buf.x_last on observed dims; leaves unobserved dims unchanged
      - sets buf.observed_dims to the set of in-range observed indices (A16.5)
    """
    observed = set()
    for k, v in obs.x_partial.items():
        k = int(k)
        if 0 <= k < int(cfg.D):
            buf.x_last[k] = float(v)
            observed.add(k)
    buf.observed_dims = observed
