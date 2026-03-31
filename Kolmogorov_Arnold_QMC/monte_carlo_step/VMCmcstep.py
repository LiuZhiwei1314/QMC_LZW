"""This module tells us how to move the walkers i.e. the calculation of T and A . We dont use the algorithm in Ferminet."""

import chex
import jax
from jax import lax
from jax import numpy as jnp
import Kolmogorov_Arnold_QMC.kan_wavefunction_case_one.kan_networks_case_one as networks
from GaussianNet import constants
from GaussianNet.tools.utils import utils


def _mcmc_impl(
    f,
    ndim: int,
    nelectrons: int,
    steps: int,
    data: networks.KANetsData,
    params: networks.ParamTree,
    key: chex.PRNGKey,
    tstep,
):
    logabs_f = utils.select_output(f, 1)
    sqrt_tstep = jnp.sqrt(tstep)
    inv_two_tstep = 0.5 / tstep

    pos = data.positions
    expected_last_dim = nelectrons * ndim
    squeeze_output = pos.ndim == 1
    if pos.ndim not in (1, 2):
        raise ValueError(
            f'positions must have rank 1 or 2 (got shape {pos.shape}). '
            'Expected (ncoord,) or (batch, ncoord).'
        )
    x0 = pos[None, ...] if squeeze_output else pos
    if x0.shape[-1] != expected_last_dim:
        raise ValueError(
            f'positions last dim mismatch: got {x0.shape[-1]}, '
            f'expected nelectrons*ndim={expected_last_dim}.'
        )
    batch_size = x0.shape[0]

    def _logprob(x):
        lp = 2.0 * logabs_f(params, x, data.spins, data.atoms, data.charges)
        lp = jnp.asarray(lp)
        if lp.ndim == 0:
            return lp[None]
        return jnp.reshape(lp, (batch_size,))

    def _drift(x):
        # Sum over walkers so grad wrt x returns per-walker gradients.
        def total_logabs(xx):
            out = logabs_f(params, xx, data.spins, data.atoms, data.charges)
            return jnp.sum(jnp.asarray(out))
        return jax.grad(total_logabs)(x)

    def one_step(_, carry):
        x1, key, lp_1, num_accepts = carry
        key, noise_key = jax.random.split(key)
        drift1 = _drift(x1)

        x2 = x1 + tstep * drift1 + sqrt_tstep * jax.random.normal(
            noise_key, shape=x1.shape
        )
        lp_2 = _logprob(x2)
        drift2 = _drift(x2)

        delta_forward = x2 - x1 - tstep * drift1
        delta_backward = x1 - x2 - tstep * drift2
        sum_axes = tuple(range(1, delta_forward.ndim))
        lq_1 = -inv_two_tstep * jnp.sum(delta_forward**2, axis=sum_axes)
        lq_2 = -inv_two_tstep * jnp.sum(delta_backward**2, axis=sum_axes)
        ratio = lp_2 + lq_2 - lp_1 - lq_1

        key, accept_key = jax.random.split(key)
        rnd = jnp.log(jax.random.uniform(accept_key, shape=ratio.shape))
        accept = ratio > rnd
        x_new = jnp.where(accept[..., None], x2, x1)
        lp_new = jnp.where(accept, lp_2, lp_1)
        num_accepts += accept.astype(jnp.int32)
        return x_new, key, lp_new, num_accepts

    init = (x0, key, _logprob(x0), jnp.zeros(batch_size, dtype=jnp.int32))
    x_new, key, _, num_accepts = lax.fori_loop(0, steps, one_step, init)
    pmove = jnp.mean(num_accepts / steps)
    # Single-device: identity. Multi-device (inside pmap): cross-device mean.
    pmove = constants.pmean(pmove)
    x_new = x_new[0] if squeeze_output else x_new
    new_data = networks.KANetsData(
        positions=x_new,
        spins=data.spins,
        atoms=data.atoms,
        charges=data.charges,
    )
    return new_data, key, pmove


def make_mcmc_step(
    f,
    ndim: int,
    nelectrons: int,
    steps: int = 1,
):
    """Builds a training-loop compatible MCMC step: (params, data, key, width)->data."""
    if steps <= 0:
        raise ValueError('steps must be positive.')

    def mcmc_step(
        params: networks.ParamTree,
        data: networks.KANetsData,
        key: chex.PRNGKey,
        width,
    ):
        new_data, _, pmove = _mcmc_impl(
            f=f,
            ndim=ndim,
            nelectrons=nelectrons,
            steps=steps,
            data=data,
            params=params,
            key=key,
            tstep=jnp.asarray(width),
        )
        return new_data, pmove

    return mcmc_step

def walkers_update(f,
                   tstep: float,
                   ndim: int,
                   nelectrons: int,
                   steps: int = 1,):
    """All-electron MALA update in FermiNet-like loop structure.

    The external interface is unchanged: returns mcstep(data, params, key).
    """
    if tstep <= 0:
        raise ValueError('tstep must be positive.')
    if steps <= 0:
        raise ValueError('steps must be positive.')

    def mcstep(data: networks.KANetsData, params: networks.ParamTree, key: chex.PRNGKey,):
        new_data, key, _ = _mcmc_impl(
            f=f,
            ndim=ndim,
            nelectrons=nelectrons,
            steps=steps,
            data=data,
            params=params,
            key=key,
            tstep=jnp.asarray(tstep),
        )
        return new_data, key
    return mcstep
