"""Scalar pretraining for the direct MKAN wavefunction."""

from typing import Callable, Optional, Tuple

import chex
import jax
import jax.numpy as jnp
import optax
from tqdm.auto import trange

from kan_wavefunction_case_one import kan_networks_case_one as networks
from monte_carlo_step import VMCmcstep


def make_pretrain_step(
    *,
    batch_network,
    batch_log_network,
    optimizer_update: optax.TransformUpdateFn,
    electrons: Tuple[int, int],
    scf_fraction: float = 1.0,
    phase_weight: float = 1.0e-2,
    mcmc_steps: int = 1,
    mcmc_width: float = 0.02,
):
  """Creates one MKAN scalar pretraining step.

  The old pretraining path matches orbital matrices.  A direct MKAN wavefunction
  only exposes a scalar log-amplitude, so here we match centered SCF
  log|determinant| values instead.  The centering removes the arbitrary
  normalization constant in the wavefunction.
  """
  if scf_fraction > 1 or scf_fraction < 0:
    raise ValueError('scf_fraction must be in between 0 and 1, inclusive.')

  def scf_logabs(scf_approx, pos):
    return scf_approx.eval_slater(pos, electrons)[1]

  if scf_fraction < 1e-6:
    def mcmc_logabs(full_params, pos, spins, atoms, charges):
      return batch_network(full_params['mkan'], pos, spins, atoms, charges)
  elif scf_fraction > 0.999999:
    def mcmc_logabs(full_params, pos, spins, atoms, charges):
      del spins, atoms, charges
      return scf_logabs(full_params['scf'], pos)
  else:
    def mcmc_logabs(full_params, pos, spins, atoms, charges):
      log_mkan = batch_network(full_params['mkan'], pos, spins, atoms, charges)
      log_scf = scf_logabs(full_params['scf'], pos)
      return (1 - scf_fraction) * log_mkan + scf_fraction * log_scf

  def mcmc_signed_network(full_params, pos, spins, atoms, charges):
    logmag = mcmc_logabs(full_params, pos, spins, atoms, charges)
    phase = jnp.ones_like(logmag)
    return phase, logmag

  mcmc_step = VMCmcstep.make_mcmc_step(
      f=mcmc_signed_network,
      ndim=3,
      nelectrons=sum(electrons),
      steps=mcmc_steps,
  )

  def loss_fn(params, data: networks.KANetsData, scf_approx):
    target_logabs = scf_logabs(scf_approx, data.positions)
    pred_logabs = batch_network(
        params, data.positions, data.spins, data.atoms, data.charges)

    target_logabs = target_logabs - jnp.mean(target_logabs)
    pred_logabs = pred_logabs - jnp.mean(pred_logabs)
    logabs_loss = jnp.mean((pred_logabs - target_logabs) ** 2)

    if batch_log_network is None or phase_weight <= 0.0:
      return logabs_loss

    pred_log = batch_log_network(
        params, data.positions, data.spins, data.atoms, data.charges)
    phase_loss = jnp.mean(jnp.imag(pred_log) ** 2)
    return logabs_loss + phase_weight * phase_loss

  def pretrain_step(data, params, state, key, scf_approx):
    val_and_grad = jax.value_and_grad(loss_fn, argnums=0)
    loss_val, grads = val_and_grad(params, data, scf_approx)
    updates, state = optimizer_update(grads, state, params)
    params = optax.apply_updates(params, updates)
    if scf_fraction < 1e-6:
      full_params = {'mkan': params}
    elif scf_fraction > 0.999999:
      full_params = {'scf': scf_approx}
    else:
      full_params = {'mkan': params, 'scf': scf_approx}
    data, _ = mcmc_step(full_params, data, key, width=mcmc_width)
    return data, params, state, loss_val

  return pretrain_step


def pretrain_scalar_wavefunction(
    *,
    params,
    positions: jnp.ndarray,
    spins: jnp.ndarray,
    atoms: jnp.ndarray,
    charges: jnp.ndarray,
    batch_network,
    batch_log_network,
    sharded_key: chex.PRNGKey,
    electrons: Tuple[int, int],
    scf_approx,
    iterations: int = 1000,
    logger: Optional[Callable[[int, float], None]] = None,
    checkpoint_callback: Optional[
        Callable[[int, float, networks.ParamTree, optax.OptState,
                  networks.KANetsData, chex.PRNGKey], None]
    ] = None,
    scf_fraction: float = 1.0,
    phase_weight: float = 1.0e-2,
    mcmc_steps: int = 1,
    mcmc_width: float = 0.02,
    start_iteration: int = 0,
    opt_state: Optional[optax.OptState] = None,
    data: Optional[networks.KANetsData] = None,
):
  """Pretrains MKAN log-amplitudes to match an HF/DFT Slater reference."""
  optimizer = optax.adam(3.e-4)
  opt_state_pt = optimizer.init(params) if opt_state is None else opt_state

  pretrain_step = make_pretrain_step(
      batch_network=batch_network,
      batch_log_network=batch_log_network,
      optimizer_update=optimizer.update,
      electrons=electrons,
      scf_fraction=scf_fraction,
      phase_weight=phase_weight,
      mcmc_steps=mcmc_steps,
      mcmc_width=mcmc_width,
  )

  if data is None:
    data = networks.KANetsData(
        positions=positions, spins=spins, atoms=atoms, charges=charges
    )

  iterator = trange(
      start_iteration, iterations, desc='Pretrain-MKAN', dynamic_ncols=True
  )

  for t in iterator:
    sharded_key, subkeys = jax.random.split(sharded_key)
    data, params, opt_state_pt, loss = pretrain_step(
        data, params, opt_state_pt, subkeys, scf_approx)
    step = t + 1
    loss_value = float(jnp.real(loss))
    iterator.set_postfix(iter=step, loss=f'{loss_value:.6f}')

    if logger:
      logger(step, loss_value)
    if checkpoint_callback:
      checkpoint_callback(
          step, loss_value, params, opt_state_pt, data, sharded_key
      )

  return params, data, opt_state_pt, sharded_key
