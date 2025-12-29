"""
Feature and learning helpers used by the step pipeline.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ..config import AgentConfig
from ..state.margins import compute_margins
from ..types import AgentState, EnvObs, Margins


def _feature_probe_vectors(
    *,
    state: AgentState,
    obs: EnvObs,
    abs_error: np.ndarray,
    observed_dims: set[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    A3.3 stability metrics plumbing: feed low-dimensional probe/features into rolling windows.
    """
    probe_vec = np.asarray([float(getattr(obs, "opp", 0.0)), float(getattr(obs, "danger", 0.0))], dtype=float)

    fovea = getattr(state, "fovea", None)
    if fovea is None:
        feature_vec = np.zeros(4, dtype=float)
        return probe_vec, feature_vec

    br = np.asarray(getattr(fovea, "block_residual", np.zeros(0)), dtype=float)
    ba = np.asarray(getattr(fovea, "block_age", np.zeros(0)), dtype=float)

    if observed_dims:
        od = sorted(int(k) for k in observed_dims)
        mean_abs_err = float(np.mean(abs_error[od]))
    else:
        mean_abs_err = 0.0

    feature_vec = np.asarray(
        [
            float(np.mean(br)) if br.size else 0.0,
            float(np.std(br)) if br.size else 0.0,
            float(np.mean(ba)) if ba.size else 0.0,
            float(mean_abs_err),
        ],
        dtype=float,
    )
    return probe_vec, feature_vec


def _build_training_mask(*, obs_mask: np.ndarray, x_obs: np.ndarray, cfg: AgentConfig) -> np.ndarray:
    """Objective shaping: focus updates on active, observed signal."""
    mask = obs_mask.astype(float)
    if not np.any(mask):
        return mask
    if bool(getattr(cfg, "train_active_only", False)):
        thresh = float(getattr(cfg, "train_active_threshold", 0.0))
        active = np.abs(x_obs) > thresh
        mask = mask * active.astype(float)
    if bool(getattr(cfg, "train_weight_by_value", False)):
        power = float(getattr(cfg, "train_value_power", 1.0))
        power = max(power, 0.0)
        weights = np.abs(x_obs) ** power
        mask = mask * weights
    return mask


def _derive_margins(
    *,
    E: float,
    D: float,
    drift_P: float,
    opp: float,
    x_C: float,
    cfg: AgentConfig,
) -> tuple[Margins, float, float, float]:
    """
    Build v(t) = (m_E, m_D, m_L, m_C, m_S) using A2.1–A2.4.

    m_L uses the external opportunity signal opp(t), not a proxy.
    """
    margins, rawE, rawD, rawS = compute_margins(
        E=E,
        D=D,
        drift_P=drift_P,
        opp=opp,
        x_C=x_C,
        cfg=cfg,
    )
    return margins, rawE, rawD, rawS


class LearningProcessor:
    def __init__(self):
        self.low_streak: int = 0
        self.high_streak: int = 0

    def update_streaks(self, mean_delta: float) -> None:
        ADD_DELTA_THRESHOLD = 0.08
        HIGH_DELTA_THRESHOLD = 0.15
        STREAK_STEPS = 20  # Align with original test harness constant

        if mean_delta < ADD_DELTA_THRESHOLD:
            self.low_streak += 1
            self.high_streak = 0
        elif mean_delta > HIGH_DELTA_THRESHOLD:
            self.high_streak += 1
            self.low_streak = 0
        else:
            self.low_streak = 0
            self.high_streak = 0
