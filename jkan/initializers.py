"""Initialization helpers for JKAN."""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp


Array = jnp.ndarray
InitFn = Callable[[jax.Array, tuple[int, ...]], Array]


def _normal(stddev: float) -> InitFn:
    def init(key: jax.Array, shape: tuple[int, ...]) -> Array:
        return jax.random.normal(key, shape=shape) * stddev

    return init


def _glorot_uniform() -> InitFn:
    def init(key: jax.Array, shape: tuple[int, ...]) -> Array:
        if len(shape) < 2:
            fan_in = fan_out = shape[0]
        else:
            fan_in, fan_out = shape[-1], shape[-2]
        limit = jnp.sqrt(6.0 / float(fan_in + fan_out))
        return jax.random.uniform(key, shape=shape, minval=-limit, maxval=limit)

    return init


def _zeros() -> InitFn:
    def init(key: jax.Array, shape: tuple[int, ...]) -> Array:
        del key
        return jnp.zeros(shape)

    return init


def get_initializer(name: str, stddev: float = 0.1) -> InitFn:
    """Returns a lightweight initializer by name."""

    name = name.lower()
    if name == "normal":
        return _normal(stddev)
    if name == "glorot_uniform":
        return _glorot_uniform()
    if name == "zeros":
        return _zeros()
    raise ValueError(f"Unknown initializer: {name}")


def init_base_params(
    *,
    key: jax.Array,
    n_in: int,
    n_out: int,
    n_basis: int,
    residual: bool = True,
    external_weights: bool = True,
    add_bias: bool = True,
    basis_init: str = "normal",
    basis_stddev: float = 0.1,
    residual_init: str = "glorot_uniform",
) -> Tuple[Array, Optional[Array], Optional[Array], Optional[Array]]:
    """Initializes the main trainable arrays for a spline KAN layer."""

    keys = jax.random.split(key, 4)
    basis_fn = get_initializer(basis_init, stddev=basis_stddev)
    residual_fn = get_initializer(residual_init)

    c_basis = basis_fn(keys[0], (n_in * n_out, n_basis))
    c_res = residual_fn(keys[1], (n_out, n_in)) if residual else None
    c_spl = jnp.ones((n_out, n_in)) if external_weights else None
    bias = jnp.zeros((n_out,)) if add_bias else None
    return c_basis, c_res, c_spl, bias
