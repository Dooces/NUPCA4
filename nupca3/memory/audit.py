"""Audit helpers for sig64 metadata and packed sig_index health."""

from __future__ import annotations

from typing import Any, List

import numpy as np


def _effective_B_cap(cfg: Any) -> int:
    B = int(getattr(cfg, "B", 0))
    B_max = int(getattr(cfg, "B_max", 0))
    return B_max if B_max > 0 else B


def _effective_F_cap(cfg: Any) -> int:
    fovea = int(getattr(cfg, "fovea_blocks_per_step", 0))
    F_max = int(getattr(cfg, "F_max", 0))
    return F_max if F_max > 0 else fovea


def audit_sig64_index_health(state: Any, cfg: Any) -> List[str]:
    """Return a list of issues detected in sig64 metadata/index state."""
    issues: List[str] = []

    B_cap = _effective_B_cap(cfg)
    F_cap = _effective_F_cap(cfg)

    mask = np.asarray(
        getattr(state, "sig_prev_hist", np.zeros(0, dtype=np.uint16)), dtype=np.uint16
    ).reshape(-1)
    counts = np.asarray(
        getattr(state, "sig_prev_counts", np.zeros(0, dtype=np.int16)), dtype=np.int16
    ).reshape(-1)

    expected_mask_bytes = max(1, (max(0, B_cap) + 7) // 8) if B_cap > 0 else 0
    if expected_mask_bytes and mask.size != expected_mask_bytes:
        issues.append(f"sig_prev_hist expected {expected_mask_bytes} bytes, found {mask.size}")
    if F_cap > 0 and counts.size != F_cap:
        issues.append(f"sig_prev_counts expected {F_cap}, found {counts.size}")

    lib = getattr(state, "library", None)
    sig_index = getattr(lib, "sig_index", None) if lib is not None else None
    if sig_index is None:
        issues.append("library.sig_index missing")
        return issues

    expected_tables = int(getattr(cfg, "sig_tables", getattr(sig_index, "tables", 0)))
    if int(getattr(sig_index, "tables", -1)) != expected_tables:
        issues.append(f"sig_index.tables expected {expected_tables}, found {getattr(sig_index, 'tables', None)}")

    expected_bucket_bits = int(getattr(cfg, "sig_bucket_bits", getattr(sig_index, "bucket_bits", 0)))
    if int(getattr(sig_index, "bucket_bits", -1)) != expected_bucket_bits:
        issues.append(
            f"sig_index.bucket_bits expected {expected_bucket_bits}, found {getattr(sig_index, 'bucket_bits', None)}"
        )
    expected_bucket_cap = int(getattr(cfg, "sig_bucket_cap", getattr(sig_index, "bucket_cap", 0)))
    if int(getattr(sig_index, "bucket_cap", -1)) != expected_bucket_cap:
        issues.append(
            f"sig_index.bucket_cap expected {expected_bucket_cap}, found {getattr(sig_index, 'bucket_cap', None)}"
        )

    n_buckets = int(getattr(sig_index, "n_buckets", 0))
    buckets = getattr(sig_index, "buckets", None)
    if buckets is not None and n_buckets > 0 and expected_bucket_cap > 0:
        for tbl in buckets:
            for row in tbl.values():
                arr = np.asarray(row)
                if arr.shape != (n_buckets, expected_bucket_cap):
                    issues.append(
                        f"sig_index bucket shape mismatch: expected {(n_buckets, expected_bucket_cap)}, found {arr.shape}"
                    )
                    break
            if issues:
                break

    err_cache = getattr(sig_index, "_err_cache", None)
    if err_cache is not None:
        arr = np.asarray(err_cache)
        err_bins = int(getattr(sig_index, "err_bins", arr.shape[1] if arr.ndim >= 2 else 0))
        if arr.ndim != 2 or arr.shape[1] != err_bins:
            issues.append(f"sig_index.err_cache shape mismatch: expected (*, {err_bins}), found {arr.shape}")

    return issues
