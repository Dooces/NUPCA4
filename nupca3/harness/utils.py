"""Utility helpers for the harness."""

from __future__ import annotations

from typing import Dict, List

import numpy as np


def build_partial_obs(full_x: np.ndarray, obs_dims: List[int]) -> Dict[int, float]:
    return {int(k): float(full_x[int(k)]) for k in obs_dims}
