"""nupca3/memory/expert.py

Operations for masked local linear experts.

Axiom coverage:
  - A4.1: masked local experts
  - A7.1: local prediction and precision

Note: ExpertNode dataclass is defined in `nupca3/types.py` for shared typing.


[AXIOM_CLARIFICATION_ADDENDUM — "Expert" = Abstraction/Resonance Node]

- This module uses the term "expert" purely as a code label. Conceptually, these are **abstractions/constellations**: sparse-footprint operators that detect/complete/resonate with a component of the abstraction state and thereby reduce residual prediction error.

- The mask is a *footprint selector* in abstraction space, not a pixel mask. Any future ARC/vision integration must introduce an encoder producing multi-resolution abstractions and ensure the library stores only abstraction-domain parameters.
"""

from __future__ import annotations

import numpy as np

from ..types import ExpertNode


def predict(node: ExpertNode, x: np.ndarray) -> np.ndarray:
    """Local 1-step prediction mu_j(t+1|t) = W_j x + b_j, masked to node.mask."""
    def _compact_indices():
        out_idx = getattr(node, "out_idx", None)
        in_idx = getattr(node, "in_idx", None)
        if out_idx is None:
            out_idx = np.where(node.mask > 0.5)[0]
        if in_idx is None:
            in_mask_local = getattr(node, "input_mask", None)
            if in_mask_local is None:
                in_mask_local = node.mask
            in_idx = np.where(np.asarray(in_mask_local) > 0.5)[0]
        return np.asarray(out_idx, dtype=int), np.asarray(in_idx, dtype=int)

    binding = getattr(node, "binding_map", None)
    if binding is not None:
        fwd = np.asarray(getattr(binding, "forward", []), dtype=int)
        D = int(fwd.shape[0])
        x_canon = np.zeros(D, dtype=float)
        for i in range(D):
            j = int(fwd[i])
            if j >= 0 and j < x.shape[0]:
                x_canon[i] = x[j]
        in_mask = getattr(node, "input_mask", None)
        if in_mask is None:
            in_mask = node.mask
        x_masked = x_canon * in_mask
        if node.W.ndim == 2 and node.W.shape == (D, D):
            y_canon = node.W @ x_masked + node.b
        else:
            out_idx, in_idx = _compact_indices()
            y_canon = np.zeros(D, dtype=float)
            if out_idx.size and in_idx.size:
                y_canon[out_idx] = node.W @ x_masked[in_idx] + node.b[out_idx]
        y_world = np.zeros_like(y_canon)
        for i in range(D):
            j = int(fwd[i])
            if j >= 0 and j < y_world.shape[0]:
                y_world[j] = y_canon[i]
        return y_world
    in_mask = getattr(node, "input_mask", None)
    if in_mask is None:
        in_mask = node.mask
    x_masked = x * in_mask
    D = x.shape[0]
    if node.W.ndim == 2 and node.W.shape == (D, D):
        return node.W @ x_masked + node.b
    out_idx, in_idx = _compact_indices()
    y = np.zeros(D, dtype=float)
    if out_idx.size and in_idx.size:
        y[out_idx] = node.W @ x_masked[in_idx] + node.b[out_idx]
    return y


def precision_vector(node: ExpertNode) -> np.ndarray:
    """Return masked diagonal precision vector for the expert.

    Skeleton: uses diagonal of Sigma^{-1} where defined; zeros elsewhere.
    """
    # Avoid full inversion; treat Sigma as diagonal placeholder.
    diag = np.diag(node.Sigma) if node.Sigma.ndim == 2 else node.Sigma
    binding = getattr(node, "binding_map", None)
    if binding is not None:
        fwd = np.asarray(getattr(binding, "forward", []), dtype=int)
        prec = np.zeros_like(diag, dtype=float)
        mask = node.mask.astype(bool)
        for i in np.where(mask)[0]:
            j = int(fwd[i])
            if j >= 0 and j < prec.shape[0]:
                prec[j] = 1.0 / np.maximum(diag[i], 1e-8)
        return prec
    prec = np.zeros_like(diag, dtype=float)
    prec[node.mask.astype(bool)] = 1.0 / np.maximum(diag[node.mask.astype(bool)], 1e-8)
    return prec


def sgd_update(
    node: ExpertNode,
    x_t: np.ndarray,
    y_target: np.ndarray,
    out_mask: np.ndarray,
    lr: float,
    sigma_ema: float,
) -> None:
    """Online parameter update for a masked linear expert.

    Implements a simple SGD step on squared prediction error:
        minimize 0.5 * || (y_target - (W x + b)) ⊙ out_mask ||^2

    Constraints:
      - Inputs are masked to node.mask (A4.1).
      - Outputs updated only where out_mask == 1.

    Side effects:
      - Updates node.W, node.b in-place.
      - Updates node.Sigma diagonal as an EMA of squared residuals on updated outputs.

    Axiom coverage: A10 (parameter updates gated elsewhere), A4.1, A7.1.
    """
    if lr <= 0.0:
        return
    binding = getattr(node, "binding_map", None)
    if binding is not None:
        fwd = np.asarray(getattr(binding, "forward", []), dtype=int)
        D = int(fwd.shape[0])
        x_canon = np.zeros(D, dtype=float)
        y_canon = np.zeros(D, dtype=float)
        out_canon = np.zeros(D, dtype=float)
        for i in range(D):
            j = int(fwd[i])
            if j >= 0 and j < x_t.shape[0]:
                x_canon[i] = x_t[j]
                y_canon[i] = y_target[j]
                out_canon[i] = out_mask[j]
        x_t = x_canon
        y_target = y_canon
        out_mask = out_canon
    D = x_t.shape[0]
    in_mask = getattr(node, "input_mask", None)
    if in_mask is None:
        in_mask = node.mask
    in_mask = np.asarray(in_mask, dtype=bool)
    out_mask = np.asarray(out_mask, dtype=float)
    out_mask_b = out_mask > 0.0
    if not np.any(out_mask_b):
        return

    x_in = x_t.copy()
    x_in[~in_mask] = 0.0

    if node.W.ndim == 2 and node.W.shape == (D, D):
        # Current prediction
        y_hat = node.W @ x_in + node.b
        err = (y_target - y_hat)
        err = err * out_mask
        err[~out_mask_b] = 0.0

        # SGD update: W[k,:] += lr * err[k] * x_in[:]
        # Only update output rows where out_mask is true.
        rows = np.where(out_mask_b)[0]
        for k in rows:
            node.W[k, in_mask] += lr * err[k] * x_in[in_mask]
            node.b[k] += lr * err[k]
    else:
        out_idx = getattr(node, "out_idx", None)
        in_idx = getattr(node, "in_idx", None)
        if out_idx is None:
            out_idx = np.where(node.mask > 0.5)[0]
        if in_idx is None:
            in_idx = np.where(in_mask)[0]
        out_idx = np.asarray(out_idx, dtype=int)
        in_idx = np.asarray(in_idx, dtype=int)
        if out_idx.size == 0 or in_idx.size == 0:
            return
        x_in_compact = x_in[in_idx]
        y_hat_out = node.W @ x_in_compact + node.b[out_idx]
        err_out = (y_target[out_idx] - y_hat_out)
        out_mask_local = out_mask[out_idx]
        out_mask_b_local = out_mask_local > 0.0
        if not np.any(out_mask_b_local):
            return
        err_out = err_out * out_mask_local
        rows = np.where(out_mask_b_local)[0]
        for r in rows:
            node.W[r, :] += lr * err_out[r] * x_in_compact
            node.b[out_idx[r]] += lr * err_out[r]

    # Update diagonal covariance estimate for updated outputs.
    if node.Sigma.ndim == 2:
        diag = np.diag(node.Sigma).copy()
    else:
        diag = node.Sigma.copy()
    alpha = float(np.clip(sigma_ema, 0.0, 1.0))
    if node.W.ndim == 2 and node.W.shape == (D, D):
        diag[rows] = (1.0 - alpha) * diag[rows] + alpha * (err[rows] ** 2)
    else:
        diag[out_idx[rows]] = (1.0 - alpha) * diag[out_idx[rows]] + alpha * (err_out[rows] ** 2)
    diag = np.maximum(diag, 1e-8)
    if node.Sigma.ndim == 2:
        node.Sigma = np.diag(diag)
    else:
        node.Sigma = diag
