"""v5 kernel entrypoint (no legacy shims)."""
from __future__ import annotations

import time
from typing import Any, Dict, Tuple

from ..config import AgentConfig
from ..types import Action, AgentState, EnvObs
from ._v5_pipeline import step_pipeline


def _extract_env_tick(env_obs: EnvObs) -> int:
    """Return the declared environment tick; error if missing or non-positive."""
    t_val = getattr(env_obs, "t_w")
    t_int = int(t_val)
    if t_int <= 0:
        raise AssertionError(f"env_obs.t_w must be positive, got {t_int}")
    return t_int


def step_v5_kernel(state: AgentState, env_obs: EnvObs, cfg: AgentConfig) -> Tuple[Action, AgentState, Dict[str, Any]]:
    """v5 kernel entrypoint that executes the v5 pipeline directly."""
    env_tick = _extract_env_tick(env_obs)
    expected_tick = int(state.t_w) + 1
    if env_tick != expected_tick:
        raise AssertionError(
            f"Non-monotonic t_w: env_obs.t_w={env_tick} vs expected={expected_tick}"
        )

    action, next_state, trace = step_pipeline(state, env_obs, cfg)
    if int(next_state.t_w) != env_tick:
        raise AssertionError(
            f"Post-step t_w mismatch: state.t_w={next_state.t_w} expected={env_tick}"
        )

    wall_ms = getattr(env_obs, "wall_ms", None)
    next_state.wall_ms = int(wall_ms) if wall_ms is not None else int(time.perf_counter() * 1000)

    metadata: Dict[str, Any] = dict(trace or {})
    metadata["kernel"] = "v5"
    return action, next_state, metadata
