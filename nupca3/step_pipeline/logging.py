"""
Utilities for tracing and fovea event logging inside the step pipeline.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

DEBUG = False
_LOG_START_TIME: float | None = None
_LAST_LOG_TIME: float | None = None


def _dbg(msg: str, *, state: object | None = None) -> None:
    """Print a debug line when DEBUG is enabled. Does not alter control flow."""
    if not DEBUG:
        return
    bracket = msg.find("]")
    tail = msg[bracket + 1 :] if bracket >= 0 else msg
    if "=" not in tail:
        return
    global _LAST_LOG_TIME, _LOG_START_TIME
    try:
        t = getattr(state, "t_w", None) if state is not None else None
    except Exception:
        t = None
    now = time.monotonic()
    if _LOG_START_TIME is None:
        _LOG_START_TIME = now
    if _LAST_LOG_TIME is None:
        delta = 0.0
    else:
        delta = now - _LAST_LOG_TIME
    _LAST_LOG_TIME = now
    elapsed = now - _LOG_START_TIME
    prefix = (
        f"[step_pipeline t_w={int(t):6d} dt={delta:7.3f}s elapsed={elapsed:7.3f}s] "
        if t is not None
        else f"[step_pipeline dt={delta:7.3f}s elapsed={elapsed:7.3f}s] "
    )
    print(prefix + str(msg))


FOVEA_LOGGER = logging.getLogger("nupca3_grid_harness")
FOVEA_LOGGER_START = time.perf_counter()


def _log_fovea_event(event: str, details: dict[str, Any]) -> None:
    """Emit a JSON payload when the harness logger is configured."""
    if not FOVEA_LOGGER.handlers:
        return
    payload = {"event": event}
    payload["timestamp"] = float(time.perf_counter() - FOVEA_LOGGER_START)
    payload.update(details)
    try:
        FOVEA_LOGGER.info(json.dumps(payload, sort_keys=True))
    except Exception:
        FOVEA_LOGGER.info(f"{event} {details}")
