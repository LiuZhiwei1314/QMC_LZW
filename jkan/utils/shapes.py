"""Shape helpers for JKAN."""

from __future__ import annotations

import jax.numpy as jnp


Array = jnp.ndarray


def ensure_2d(x: Array) -> Array:
    """Normalizes vector input to shape (batch, features)."""

    if x.ndim == 1:
        return x[None, :]
    if x.ndim != 2:
        raise ValueError(f"Expected a 1D or 2D array, got shape {x.shape}.")
    return x
