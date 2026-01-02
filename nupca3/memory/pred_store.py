"""Prediction store / deferred validation helpers for v5."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Iterable, List, Sequence, Tuple

import numpy as np


@dataclass
class PredSnapshot:
    """Captured prediction metadata for deferred validation."""

    t_emit: int
    t_due: int
    sig64_emit: int | None
    selected_blocks: Tuple[int, ...]
    dims_idx: np.ndarray
    pred_vals: np.ndarray
    attribution: Tuple[int, ...]


class PredStore:
    """Deterministic ring buffer for pending prediction snapshots."""

    def __init__(self, capacity: int = 64) -> None:
        self.capacity = max(1, int(capacity))
        self._buffer: deque[PredSnapshot] = deque(maxlen=self.capacity)

    def append(self, snapshot: PredSnapshot) -> None:
        if snapshot is None:
            return
        self._buffer.append(snapshot)

    def pop_due(self, tick: int) -> List[PredSnapshot]:
        if tick is None:
            return []
        due: List[PredSnapshot] = []
        keep: List[PredSnapshot] = []
        for entry in self._buffer:
            if entry.t_due == tick:
                due.append(entry)
            else:
                keep.append(entry)
        if len(keep) != len(self._buffer):
            self._buffer = deque(keep, maxlen=self.capacity)
        return due

    def clear(self) -> None:
        self._buffer.clear()


def _prepare_obs_map(obs_idx: np.ndarray, obs_vals: np.ndarray) -> dict[int, float]:
    return {
        int(dim): float(val)
        for dim, val in zip(obs_idx.tolist(), obs_vals.tolist())
        if np.isfinite(val)
    }


def process_due_validations(
    state: Any,
    cfg: Any,
    tick: int,
    obs_idx: np.ndarray,
    obs_vals: np.ndarray,
) -> int:
    if not bool(getattr(cfg, "sig_enable_validation", True)):
        return 0
    store = getattr(state, "pred_store", None)
    if store is None:
        return 0
    obs_map = _prepare_obs_map(obs_idx, obs_vals)
    if not obs_map:
        return 0

    due = store.pop_due(tick)
    if not due:
        return 0

    lib = getattr(state, "library", None)
    nodes = getattr(lib, "nodes", {}) if lib is not None else {}
    sig_index = getattr(lib, "sig_index", None) if lib is not None else None
    beta_err = float(getattr(cfg, "sig_err_ema_beta", 0.0))
    beta_err = 0.0 if beta_err < 0.0 else (1.0 if beta_err > 1.0 else beta_err)
    h_bin = 2
    processed = 0

    for snap in due:
        if snap.dims_idx.size == 0:
            continue
        common_indices: List[int] = [
            idx for idx, dim in enumerate(snap.dims_idx.tolist()) if int(dim) in obs_map
        ]
        if not common_indices:
            continue
        preds = np.asarray(snap.pred_vals, dtype=float).reshape(-1)[common_indices]
        actual = np.array([obs_map[int(snap.dims_idx[i])] for i in common_indices], dtype=float)
        diff = preds - actual
        finite = np.isfinite(diff)
        if not np.any(finite):
            continue
        err = float(np.mean(np.abs(diff[finite])))
        if not snap.attribution:
            continue
        for nid in snap.attribution:
            node = nodes.get(int(nid))
            if node is None:
                continue
            try:
                ema = np.asarray(getattr(node, "err_ema", np.zeros(3, dtype=np.float32)), dtype=np.float32).reshape(-1)
                if ema.size != 3:
                    ema = np.zeros(3, dtype=np.float32)
                ema[h_bin] = (1.0 - beta_err) * float(ema[h_bin]) + beta_err * err
                node.err_ema = ema
                cnt = np.asarray(
                    getattr(node, "val_count", np.zeros(3, dtype=np.int32)), dtype=np.int32
                ).reshape(-1)
                if cnt.size != 3:
                    cnt = np.zeros(3, dtype=np.int32)
                cnt[h_bin] = int(cnt[h_bin]) + 1
                node.val_count = cnt
                if sig_index is not None and hasattr(sig_index, "update_error"):
                    sig_index.update_error(int(nid), int(h_bin), float(ema[h_bin]))
            except Exception:
                continue
        processed += 1

    return processed


def emit_pred_snapshot(
    state: Any,
    cfg: Any,
    *,
    tick: int,
    obs_idx: np.ndarray,
    prior: np.ndarray,
    blocks: Sequence[int],
    active_set: Iterable[int],
    sig64: int | None,
) -> None:
    if not bool(getattr(cfg, "sig_enable_validation", True)):
        return
    store = getattr(state, "pred_store", None)
    if store is None:
        return
    if not getattr(cfg, "pred_store_capacity", 0):
        return
    if obs_idx is None or obs_idx.size == 0:
        return
    dims = np.asarray(obs_idx, dtype=int).reshape(-1)
    if dims.size == 0:
        return
    prior_arr = np.asarray(prior, dtype=float).reshape(-1)
    if prior_arr.size < dims.max(initial=-1) + 1:
        prior_arr = np.resize(prior_arr, (max(int(dims.max(initial=-1) + 1), prior_arr.size),))
    preds = prior_arr[dims]
    snapshot = PredSnapshot(
        t_emit=int(tick),
        t_due=int(tick) + 1,
        sig64_emit=int(sig64) if sig64 is not None else None,
        selected_blocks=tuple(int(b) for b in blocks),
        dims_idx=dims.copy(),
        pred_vals=preds.copy(),
        attribution=tuple(sorted(int(nid) for nid in active_set if nid is not None)),
    )
    store.append(snapshot)
