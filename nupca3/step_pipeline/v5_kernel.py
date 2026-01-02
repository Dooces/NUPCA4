"""Minimal v5 kernel entrypoint that wraps the legacy pipeline (Step 1)."""
from __future__ import annotations

import time
from typing import Any, Dict, Tuple

from ..config import AgentConfig
from ..types import Action, AgentState, EnvObs
from ._v5_pipeline import step_pipeline as _legacy_step_pipeline


def _extract_env_tick(env_obs: EnvObs) -> int | None:
    t_val = getattr(env_obs, "t_w", 0)
    try:
        t_int = int(t_val)
    except Exception:
        return None
    if t_int <= 0:
        return None
    return t_int


def step_v5_kernel(state: AgentState, env_obs: EnvObs, cfg: AgentConfig) -> Tuple[Action, AgentState, Dict[str, Any]]:
    """
    v5 kernel entrypoint that will eventually replace ``core.step_pipeline``.

    For now, the function delegates to the existing pipeline logic while
    annotating the trace with ``kernel=v5`` so we can detect the new path.
    """
    env_tick = _extract_env_tick(env_obs)
    expected_tick = int(getattr(state, "t_w", 0)) + 1
    if env_tick is not None and env_tick != expected_tick:
        raise AssertionError(
            f"Non-monotonic t_w: env_obs.t_w={env_tick} vs expected={expected_tick}"
        )

    action, next_state, trace = _legacy_step_pipeline(state, env_obs, cfg)
    if env_tick is not None and int(getattr(next_state, "t_w", 0)) != env_tick:
        raise AssertionError(
            f"Post-step t_w mismatch: state.t_w={next_state.t_w} expected={env_tick}"
        )

    wall_ms = getattr(env_obs, "wall_ms", None)
    next_state.wall_ms = int(wall_ms) if wall_ms is not None else int(time.perf_counter() * 1000)

    metadata: Dict[str, Any] = dict(trace or {})
    metadata["kernel"] = "v5"
    return action, next_state, metadata
