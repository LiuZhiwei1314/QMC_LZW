"""we need pretrain to reduce the risk of optimization.10.11.2025."""
"""we need also modify this pretrain part including parallel strategy because the networks has been changed to ours."""
from typing import Callable, Mapping, Sequence, Tuple, Union, Optional

from absl import logging
import chex
#import constants
from monte_carlo_step import VMCmcstep
from kan_wavefunction_case_one import kan_networks_case_one as networks
from tools.utils import scf
from tools.utils import system
import jax
from jax import numpy as jnp
#import kfac_jax
import numpy as np
import optax
import pyscf
from tqdm.auto import trange


def get_hf(molecule: Optional[Sequence[system.Atom]] = None,
           nspins: Optional[Tuple[int, int]] = None,
           basis: Optional[str] = 'sto-3g',
           ecp: Optional[Mapping[str, str]] = None,
           core_electrons: Optional[Mapping[str, int]] = None,
           pyscf_mol: Optional[pyscf.gto.Mole] = None,
           restricted: Optional[bool] = False,
           states: int = 0,
           excitation_type: str = 'ordered') -> scf.Scf:
  """Returns an Scf object with the Hartree-Fock solution to the system.

  Args:
    molecule: the molecule in internal format.
    nspins: tuple with number of spin up and spin down electrons.
    basis: basis set to use in Hartree-Fock calculation.
    ecp: dictionary of the ECP to use for different atoms.
    core_electrons: dictionary of the number of core electrons excluded by the
      pseudopotential/effective core potential.
    pyscf_mol: pyscf Mole object defining the molecule. If supplied,
      molecule, nspins and basis are ignored.
    restricted: If true, perform a restricted Hartree-Fock calculation,
      otherwise perform an unrestricted Hartree-Fock calculation.
    states: Number of excited states.  If nonzero, compute all single and double
      excitations of the Hartree-Fock solution and return coefficients for the
      lowest ones.
    excitation_type: The way to construct different states for excited state
      pretraining. One of 'ordered' or 'random'. 'Ordered' tends to work better,
      but 'random' is necessary for some systems, especially double excitaitons.
  """
  if pyscf_mol:
    scf_approx = scf.Scf(pyscf_mol=pyscf_mol,
                         restricted=restricted)
  else:
    scf_approx = scf.Scf(molecule,
                         nelectrons=nspins,
                         basis=basis,
                         ecp=ecp,
                         core_electrons=core_electrons,
                         restricted=restricted)
  scf_approx.run(excitations=max(states - 1, 0),
                 excitation_type=excitation_type)
  return scf_approx


def eval_orbitals(scf_approx: scf.Scf, pos: Union[np.ndarray, jnp.ndarray],
                  nspins: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
  """Evaluates SCF orbitals from PySCF at a set of positions.

  Args:
    scf_approx: an scf.Scf object that contains the result of a PySCF
      calculation.
    pos: an array of electron positions to evaluate the orbitals at, of shape
      (..., nelec*3), where the leading dimensions are arbitrary, nelec is the
      number of electrons and the spin up electrons are ordered before the spin
      down electrons.
    nspins: tuple with number of spin up and spin down electrons.

  Returns:
    tuple with matrices of orbitals for spin up and spin down electrons, with
    the same leading dimensions as in pos.
  """
  if not isinstance(pos, np.ndarray):  # works even with JAX array
    try:
      pos = pos.copy()
    except AttributeError as exc:
      raise ValueError('Input must be either NumPy or JAX array.') from exc
  leading_dims = pos.shape[:-1]
  # split into separate electrons
  pos = np.reshape(pos, [-1, 3])  # (batch*nelec, 3)
  mos = scf_approx.eval_mos(pos)  # (batch*nelec, nbasis), (batch*nelec, nbasis)
  # Reshape into (batch, nelec, nbasis) for each spin channel.
  mos = [np.reshape(mo, leading_dims + (sum(nspins), -1)) for mo in mos]
  # Return (using Aufbau principle) the matrices for the occupied alpha and
  # beta orbitals. Number of alpha electrons given by nspins[0].
  alpha_spin = mos[0][..., :nspins[0], :nspins[0]]
  beta_spin = mos[1][..., nspins[0]:, :nspins[1]]
  return alpha_spin, beta_spin


def make_pretrain_step(
    batch_orbitals,
    batch_network,
    optimizer_update: optax.TransformUpdateFn,
    electrons: Tuple[int, int],
    batch_size: int = 0,
    full_det: bool = False,
    scf_fraction: float = 0.0,
    states: int = 0,
  mcmc_steps: int = 1,
  mcmc_width: float = 0.02,
):
  """Creates function for performing one step of Hartre-Fock pretraining.

  Args:
    batch_orbitals: callable with signature f(params, data), which given network
      parameters and a batch of electron positions, returns the orbitals in the
      network evaluated at those positions.
    batch_network: callable with signature f(params, data), which given network
      parameters and a batch of electron positions, returns the log of the
      magnitude of the (wavefunction) network  evaluated at those positions.
    optimizer_update: callable for transforming the gradients into an update (ie
      conforms to the optax API).
    electrons: number of spin-up and spin-down electrons.
    batch_size: number of walkers per device, used to make MCMC step.
    full_det: If true, evaluate all electrons in a single determinant.
      Otherwise, evaluate products of alpha- and beta-spin determinants.
    scf_fraction: What fraction of the wavefunction sampled from is the SCF
      wavefunction and what fraction is the neural network wavefunction?
    states: Number of excited states, if not 0.

  Returns:
    Callable for performing a single pretraining optimisation step.
  """

  # Create a function which gives either the SCF ansatz, the neural network
  # ansatz, or a weighted mixture of the two.
  if scf_fraction > 1 or scf_fraction < 0:
    raise ValueError('scf_fraction must be in between 0 and 1, inclusive.')

  if states:
    def scf_network(fn, x):
      x = x.reshape(x.shape[:-1] + (states, -1))
      slater_fn = jax.vmap(fn, in_axes=(-2, None), out_axes=-2)
      slogdets = slater_fn(x, electrons)
      # logsumexp trick
      maxlogdet = jnp.max(slogdets[1])
      dets = slogdets[0] * jnp.exp(slogdets[1] - maxlogdet)
      result = jnp.linalg.slogdet(dets)
      return result[1] + maxlogdet * slogdets[1].shape[-1]
  else:
    scf_network = lambda fn, x: fn(x, electrons)[1]

  if scf_fraction < 1e-6:
    def mcmc_network(full_params, pos, spins, atoms, charges):
      return batch_network(full_params['ferminet'], pos, spins, atoms, charges)
  elif scf_fraction > 0.999999:
    def mcmc_network(full_params, pos, spins, atoms, charges):
      del spins, atoms, charges
      return scf_network(full_params['scf'].eval_slater, pos)
  else:
    def mcmc_network(full_params, pos, spins, atoms, charges):
      log_ferminet = batch_network(full_params['ferminet'], pos, spins, atoms,
                                   charges)
      log_scf = scf_network(full_params['scf'].eval_slater, pos)
      return (1 - scf_fraction) * log_ferminet + scf_fraction * log_scf

  def mcmc_signed_network(full_params, pos, spins, atoms, charges):
      logmag = mcmc_network(full_params, pos, spins, atoms, charges)
      phase = jnp.zeros_like(logmag)
      return phase, logmag

  mcmc_step = VMCmcstep.make_mcmc_step(
      f=mcmc_signed_network,
      ndim=3,
      nelectrons=sum(electrons),
      steps=mcmc_steps,
  )

  def loss_fn(
      params: networks.ParamTree,
      data: networks.KANetsData,
      target,
  ):
    pos = data.positions
    spins = data.spins
    if states:
      # Make vmap-ed versions of eval_orbitals and batch_orbitals over the
      # states dimension.
      # (batch, states, nelec*ndim)
      pos = jnp.reshape(pos, pos.shape[:-1] + (states, -1))
      # (batch, states, nelec)
      spins = jnp.reshape(spins, spins.shape[:-1] + (states, -1))

      scf_orbitals = jax.vmap(
          scf_approx.eval_orbitals, in_axes=(-2, None), out_axes=-4
      )

      def net_orbitals(params, pos, spins, atoms, charges):
        vmapped_orbitals = jax.vmap(
            batch_orbitals, in_axes=(None, -2, -2, None, None), out_axes=-4
        )
        # Dimensions of result are
        # [(batch, states, ndet*states, nelec, nelec)]
        result = vmapped_orbitals(params, pos, spins, atoms, charges)
        result = [
            jnp.reshape(r, r.shape[:-3] + (states, -1) + r.shape[-2:])
            for r in result
        ]
        result = [jnp.transpose(r, (0, 3, 1, 2, 4, 5)) for r in result]
        # We draw distinct samples for each excited state (electron
        # configuration), and then evaluate each state within each sample.
        # Output dimensions are:
        # (batch, det, electron configuration,
        # excited state, electron, orbital)
        return result

    else:
      net_orbitals = batch_orbitals

    target = jax.lax.stop_gradient(target)
    orbitals = net_orbitals(params, pos, spins, data.atoms, data.charges)
    def cnorm(x, y):
      diff = x - y
      return jnp.real(diff) ** 2 + jnp.imag(diff) ** 2
    if full_det:
      dims = target[0].shape[:-2]  # (batch) or (batch, states).
      na = target[0].shape[-2]
      nb = target[1].shape[-2]
      target = jnp.concatenate(
          (
              jnp.concatenate(
                  (target[0], jnp.zeros(dims + (na, nb))), axis=-1),
              jnp.concatenate(
                  (jnp.zeros(dims + (nb, na)), target[1]), axis=-1),
          ),
          axis=-2,
      )
      result = jnp.mean(cnorm(target[:, None, ...], orbitals[0])).real
    else:
      result = jnp.array([
          jnp.mean(cnorm(t[:, None, ...], o)).real
          for t, o in zip(target, orbitals)
      ]).sum()
    return result

  def pretrain_step(data, params, state, key, scf_approx):
    """One iteration of pretraining to match HF."""
    target = scf_approx.eval_orbitals(data.positions, electrons)
    val_and_grad = jax.value_and_grad(loss_fn, argnums=0)
    loss_val, search_direction = val_and_grad(params, data, target)
    search_direction = search_direction
    updates, state = optimizer_update(search_direction, state, params)
    params = optax.apply_updates(params, updates)
    if scf_fraction < 1e-6:
      full_params = {'ferminet': params}
    elif scf_fraction > 0.999999:
      full_params = {'scf': scf_approx}
    else:
      full_params = {'ferminet': params, 'scf': scf_approx}
    mcmc_out = mcmc_step(full_params, data, key, width=mcmc_width)
    data = mcmc_out[0] if isinstance(mcmc_out, tuple) else mcmc_out
    return data, params, state, loss_val,

  return pretrain_step


def pretrain_hartree_fock(
    *,
    params: networks.ParamTree,
    positions: jnp.ndarray,
    spins: jnp.ndarray,
    atoms: jnp.ndarray,
    charges: jnp.ndarray,
    batch_network,
    batch_orbitals,
    sharded_key: chex.PRNGKey,
    electrons: Tuple[int, int],
    scf_approx: scf.Scf,
    iterations: int = 1000,
    batch_size: int = 0,
    logger: Optional[Callable[[int, float], None]] = None,
    checkpoint_callback: Optional[Callable[[int, float, networks.ParamTree, optax.OptState, networks.KANetsData, chex.PRNGKey], None]] = None,
    scf_fraction: float = 0.0,
    states: int = 0,
    mcmc_steps: int = 1,
    mcmc_width: float = 0.02,
    start_iteration: int = 0,
    opt_state: Optional[optax.OptState] = None,
    data: Optional[networks.KANetsData] = None,
):
  """Performs training to match initialization as closely as possible to HF."""
  optimizer = optax.adam(3.e-4)
  opt_state_pt = optimizer.init(params) if opt_state is None else opt_state

  pretrain_step = make_pretrain_step(
      batch_orbitals,
      batch_network,
      optimizer.update,
      electrons=electrons,
      batch_size=batch_size,
      full_det=True,
      scf_fraction=scf_fraction,
      states=states,
      mcmc_steps=mcmc_steps,
      mcmc_width=mcmc_width,
  )

  if data is None:
    data = networks.KANetsData(
        positions=positions, spins=spins, atoms=atoms, charges=charges
    )

  iterator = trange(start_iteration, iterations, desc='Pretrain', dynamic_ncols=True)

  for t in iterator:
    sharded_key, subkeys = jax.random.split(sharded_key)
    data, params, opt_state_pt, loss, = pretrain_step(
        data, params, opt_state_pt, subkeys, scf_approx)
    step = t + 1
    loss_value = float(jnp.real(loss))
    iterator.set_postfix(iter=step, loss=f'{loss_value:.6f}')

    if logger:
      logger(step, loss_value)
    if checkpoint_callback:
      checkpoint_callback(step, loss_value, params, opt_state_pt, data, sharded_key)

  return params, data, opt_state_pt, sharded_key
