import functools
from typing import Optional, Sequence, Tuple

import jax.numpy as jnp


def slogdet(x):
    """Compute sign/phase and log absolute determinant, with a 1x1 fast path."""
    if x.shape[-1] == 1:
        if x.dtype == jnp.complex64 or x.dtype == jnp.complex128:
            sign = x[..., 0, 0] / jnp.abs(x[..., 0, 0])
        else:
            sign = jnp.sign(x[..., 0, 0])
        logdet = jnp.log(jnp.abs(x[..., 0, 0]))
    else:
        sign, logdet = jnp.linalg.slogdet(x)
    return sign, logdet


def logdet_matmul(
    xs: Sequence[jnp.ndarray],
    w: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Combine one or more determinants in the log domain."""
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

    if result.dtype == jnp.complex64 or result.dtype == jnp.complex128:
        phase_out = result / jnp.abs(result)
    else:
        phase_out = jnp.sign(result)
    log_out = jnp.log(jnp.abs(result)) + maxlogdet
    return phase_out, log_out

