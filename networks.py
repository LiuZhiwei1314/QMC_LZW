"""QMC-specific wavefunction components for trainer-side assembly."""

from __future__ import annotations

import functools
from typing import Any, Iterable, MutableMapping, Optional, Sequence, Union

import chex
import jax.numpy as jnp


Array = jnp.ndarray
ParamTree = Union[Array, Iterable["ParamTree"], MutableMapping[Any, "ParamTree"]]


@chex.dataclass
class KANetsData:
    positions: Any
    spins: Any
    atoms: Any
    charges: Any


def construct_input_features(
    pos: Array,
    atoms: Array,
    ndim: int = 3,
) -> tuple[Array, Array, Array, Array]:
    """Construct electron-atom/electron-electron vectors and distances."""

    if atoms.shape[1] != ndim:
        raise ValueError(f"Expected atom coordinates with ndim={ndim}, got {atoms.shape}.")
    ae = jnp.reshape(pos, [-1, 1, ndim]) - atoms[None, ...]
    ee = jnp.reshape(pos, [1, -1, ndim]) - jnp.reshape(pos, [-1, 1, ndim])
    r_ae = jnp.linalg.norm(ae, axis=2, keepdims=True)
    n = ee.shape[0]
    r_ee = jnp.linalg.norm(ee + jnp.eye(n)[..., None], axis=-1) * (1.0 - jnp.eye(n))
    return ae, ee, r_ae, r_ee[..., None]


def active_spin_channels(nspins: Sequence[int]) -> list[int]:
    return [int(spin) for spin in nspins if int(spin) > 0]


def slogdet(x: Array) -> tuple[Array, Array]:
    """Compute determinant phase/sign and log magnitude."""

    if x.shape[-1] == 1:
        value = x[..., 0, 0]
        if value.dtype in (jnp.complex64, jnp.complex128):
            sign = value / jnp.abs(value)
        else:
            sign = jnp.sign(value)
        logdet = jnp.log(jnp.abs(value))
    else:
        sign, logdet = jnp.linalg.slogdet(x)
    return sign, logdet


def logdet_matmul(xs: Sequence[Array], w: Optional[Array] = None) -> tuple[Array, Array]:
    """Combine determinant blocks in log-domain."""

    det1d = functools.reduce(
        lambda a, b: a * b,
        [x.reshape(-1) for x in xs if x.shape[-1] == 1],
        1,
    )
    phase_in, logdet = functools.reduce(
        lambda a, b: (a[0] * b[0], a[1] + b[1]),
        [slogdet(x) for x in xs if x.shape[-1] > 1],
        (1, 0),
    )

    maxlogdet = jnp.max(logdet)
    det = phase_in * det1d * jnp.exp(logdet - maxlogdet)
    if w is None:
        result = jnp.sum(det)
    else:
        result = jnp.matmul(det, w)[0]

    if result.dtype in (jnp.complex64, jnp.complex128):
        phase_out = result / jnp.abs(result)
    else:
        phase_out = jnp.sign(result)
    log_out = jnp.log(jnp.abs(result)) + maxlogdet
    return phase_out, log_out
