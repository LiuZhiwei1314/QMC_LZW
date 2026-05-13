"""Envelope functions for QMC wavefunctions."""

from __future__ import annotations

from typing import Sequence

import jax.numpy as jnp

from networks import Array


def init_isotropic_envelope(natom: int, output_dims: Sequence[int]) -> list[dict[str, Array]]:
    """Initialize FermiNet-style isotropic exponential envelope parameters."""

    return [
        {
            "pi": jnp.ones((natom, int(output_dim))),
            "sigma": jnp.ones((natom, int(output_dim))),
        }
        for output_dim in output_dims
    ]


def apply_isotropic_envelope(*, r_ae: Array, pi: Array, sigma: Array) -> Array:
    """Evaluate sum_a pi_a exp(-sigma_a r_ae) for one spin channel."""

    return jnp.sum(jnp.exp(-r_ae * sigma) * pi, axis=1)
