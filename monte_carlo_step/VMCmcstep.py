"""This module tells us how to move the walkers i.e. the calculation of T and A . We dont use the algorithm in Ferminet."""

import chex
import jax
from jax import lax
from jax import numpy as jnp
import kan_wavefunction_case_one.kan_networks_case_one as networks
import constants
from tools.utils import utils


def _canonical_atoms(atoms: jnp.ndarray, ndim: int) -> jnp.ndarray:
    atoms = jnp.asarray(atoms)
    if atoms.ndim == 3:
        # Allow atoms shaped as (1, natoms, ndim).
        if atoms.shape[0] != 1:
            raise ValueError(
                f'Expected atoms with leading dimension 1 for shared geometry; got {atoms.shape}.'
            )
        atoms = atoms[0]
    if atoms.ndim != 2 or atoms.shape[-1] != ndim:
        raise ValueError(
            f'Expected atoms shape (natoms, {ndim}) or (1, natoms, {ndim}); got {atoms.shape}.'
        )
    return atoms


def _harmonic_mean_scale(
    x: jnp.ndarray,
    atoms: jnp.ndarray,
    nelectrons: int,
    ndim: int,
    eps: float = 1e-6,
) -> jnp.ndarray:
    """Per-coordinate scale from harmonic mean electron-nuclear distance."""
    x_e = jnp.reshape(x, (x.shape[0], nelectrons, ndim))
    ae = x_e[:, :, None, :] - atoms[None, None, :, :]
    r_ae = jnp.linalg.norm(ae, axis=-1)
    r_ae = jnp.maximum(r_ae, eps)
    h = 1.0 / jnp.mean(1.0 / r_ae, axis=-1)  # (batch, nelectrons)
    h = jnp.repeat(h[..., None], repeats=ndim, axis=-1)
    return jnp.reshape(h, x.shape)


def _log_prob_diag_gaussian(
    x: jnp.ndarray,
    mean: jnp.ndarray,
    sigma: jnp.ndarray,
) -> jnp.ndarray:
    sigma = jnp.maximum(sigma, 1e-12)
    z = (x - mean) / sigma
    return -0.5 * jnp.sum(z * z, axis=-1) - jnp.sum(jnp.log(sigma), axis=-1)


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
    atoms = _canonical_atoms(data.atoms, ndim)

    def _logabs_and_logprob(x):
        logabs = logabs_f(params, x, data.spins, data.atoms, data.charges)
        logabs = jnp.asarray(logabs)
        if logabs.ndim == 0:
            logabs = logabs[None]
        logabs = jnp.reshape(logabs, (batch_size,))
        # Keep exact old behavior:
        # - proposal drift from grad(log|psi|)
        # - acceptance ratio from logprob = 2 * log|psi|
        logprob = 2.0 * logabs
        return logabs, logprob

    def _logabs_sum_with_logprob(x):
        logabs, logprob = _logabs_and_logprob(x)
        return jnp.sum(logabs), logprob

    # Single pass for both values and gradients.
    _logprob_and_drift_fn = jax.value_and_grad(_logabs_sum_with_logprob, has_aux=True)

    def _logprob_and_drift(x):
        (_, lp), drift = _logprob_and_drift_fn(x)
        return lp, drift

    def _proposal_terms(x, drift):
        h = _harmonic_mean_scale(x, atoms, nelectrons, ndim)
        sigma = sqrt_tstep * h
        # Keep MALA scaling consistent with sigma^2:
        # sigma^2 = tstep * h^2, drift coeff = sigma^2 / 2 * grad(log pi)
        # and grad(log pi) = 2 * grad(log|psi|), so coeff is tstep * h^2.
        mean = x + tstep * (h * h) * drift
        return mean, sigma

    def one_step(_, carry):
        x1, key, lp_1, drift1, num_accepts = carry
        key, noise_key = jax.random.split(key)

        mean_12, sigma_12 = _proposal_terms(x1, drift1)
        x2 = mean_12 + sigma_12 * jax.random.normal(noise_key, shape=x1.shape)
        lp_2, drift2 = _logprob_and_drift(x2)

        mean_21, sigma_21 = _proposal_terms(x2, drift2)
        lq_1 = _log_prob_diag_gaussian(x2, mean_12, sigma_12)
        lq_2 = _log_prob_diag_gaussian(x1, mean_21, sigma_21)
        ratio = lp_2 + lq_2 - lp_1 - lq_1

        key, accept_key = jax.random.split(key)
        rnd = jnp.log(jax.random.uniform(accept_key, shape=ratio.shape))
        accept = ratio > rnd
        x_new = jnp.where(accept[..., None], x2, x1)
        lp_new = jnp.where(accept, lp_2, lp_1)
        drift_new = jnp.where(accept[..., None], drift2, drift1)
        num_accepts += accept.astype(jnp.int32)
        return x_new, key, lp_new, drift_new, num_accepts

    lp_0, drift_0 = _logprob_and_drift(x0)
    init = (x0, key, lp_0, drift_0, jnp.zeros(batch_size, dtype=jnp.int32))
    x_new, key, _, _, num_accepts = lax.fori_loop(0, steps, one_step, init)
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
    """Builds a training-loop compatible MCMC step: (params, data, key, width)->(new_data, pmove)."""
    if steps <= 0:
        raise ValueError('steps must be positive.')

    def mcmc_step(
        params: networks.ParamTree,
        data: networks.KANetsData,
        key: chex.PRNGKey,
        width,
    ):
        width = jnp.maximum(jnp.asarray(width), jnp.asarray(1e-12))
        new_data, _, pmove = _mcmc_impl(
            f=f,
            ndim=ndim,
            nelectrons=nelectrons,
            steps=steps,
            data=data,
            params=params,
            key=key,
            tstep=width,
        )
        return new_data, pmove

    return jax.jit(mcmc_step)
