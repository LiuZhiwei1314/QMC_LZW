"""Linear algebra helpers for JKAN."""

from __future__ import annotations

import jax.numpy as jnp


Array = jnp.ndarray


def solve_full_lstsq(a: Array, b: Array, ridge: float = 1e-8) -> Array:
    """Batched least-squares solver with small ridge stabilization."""

    a_t = jnp.swapaxes(a, -1, -2)
    eye = jnp.eye(a.shape[-1], dtype=a.dtype)
    gram = a_t @ a + ridge * eye
    rhs = a_t @ b
    return jnp.linalg.solve(gram, rhs)
