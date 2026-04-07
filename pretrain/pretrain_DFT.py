"""KS-DFT orbital pretraining for the KAN wavefunction."""

from typing import Callable, Mapping, Optional, Sequence, Tuple

from absl import logging
import chex
from kan_wavefunction_case_one import kan_networks_case_one as networks
import jax
from jax import numpy as jnp
import optax
import pyscf
import pyscf.dft
from pretrain.pretrain_HF import make_pretrain_step
from tools.utils import scf
from tools.utils import system
from tqdm.auto import trange


class Dft(scf.Scf):
  """Helper class for running Kohn-Sham DFT and exposing HF-like interfaces."""

  def __init__(self,
               molecule: Optional[Sequence[system.Atom]] = None,
               nelectrons: Optional[Tuple[int, int]] = None,
               basis: Optional[str] = 'sto-3g',
               ecp: Optional[Mapping[str, str]] = None,
               core_electrons: Optional[Mapping[str, int]] = None,
               pyscf_mol: Optional[pyscf.gto.Mole] = None,
               restricted: bool = False,
               xc: str = 'pbe,pbe',
               grid_level: Optional[int] = None):
    super().__init__(
        molecule=molecule,
        nelectrons=nelectrons,
        basis=basis,
        ecp=ecp,
        core_electrons=core_electrons,
        pyscf_mol=pyscf_mol,
        restricted=restricted,
    )
    self.mean_field = (pyscf.dft.RKS(self._mol) if restricted
                       else pyscf.dft.UKS(self._mol))
    self.mean_field.xc = xc
    if grid_level is not None:
      self.mean_field.grids.level = grid_level
    self.xc = xc
    self.grid_level = grid_level

  def run(self,
          dm0=None,
          excitations: int = 0,
          excitation_type: str = 'ordered'):
    del excitation_type
    if excitations:
      raise ValueError('DFT pretraining currently supports only ground states.')
    try:
      self.mean_field.kernel(dm0=dm0)
    except TypeError:
      logging.info('Mean-field solver does not support specifying an initial '
                   'density matrix.')
      self.mean_field.kernel()
    return self.mean_field


def dft_flatten(dft: Dft):
  children = ()
  aux_data = (dft.mo_coeff,
              dft._mol_jax._spec,
              dft._mol,
              dft.restricted,
              dft.xc,
              dft.grid_level)
  return children, aux_data


def dft_unflatten(aux_data, children) -> Dft:
  assert not children
  mo_coeff, spec, mol, restricted, xc, grid_level = aux_data
  dft = Dft(
      pyscf_mol=mol.copy(),
      restricted=restricted,
      xc=xc,
      grid_level=grid_level,
  )
  dft.mo_coeff = mo_coeff
  dft._mol_jax._spec = spec
  return dft


jax.tree_util.register_pytree_node(Dft, dft_flatten, dft_unflatten)


def get_dft(molecule: Optional[Sequence[system.Atom]] = None,
            nspins: Optional[Tuple[int, int]] = None,
            basis: Optional[str] = 'sto-3g',
            ecp: Optional[Mapping[str, str]] = None,
            core_electrons: Optional[Mapping[str, int]] = None,
            pyscf_mol: Optional[pyscf.gto.Mole] = None,
            restricted: Optional[bool] = False,
            xc: str = 'pbe,pbe',
            grid_level: Optional[int] = None,
            states: int = 0) -> Dft:
  """Returns a KS-DFT reference object for orbital pretraining."""
  if states:
    raise ValueError('DFT pretraining currently supports only ground states.')
  dft_approx = Dft(
      molecule=molecule,
      nelectrons=nspins,
      basis=basis,
      ecp=ecp,
      core_electrons=core_electrons,
      pyscf_mol=pyscf_mol,
      restricted=restricted,
      xc=xc,
      grid_level=grid_level,
  )
  dft_approx.run()
  return dft_approx


def pretrain_ks_dft(
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
    dft_approx: Dft,
    iterations: int = 1000,
    batch_size: int = 0,
    logger: Optional[Callable[[int, float], None]] = None,
    checkpoint_callback: Optional[Callable[[int, float, networks.ParamTree, optax.OptState, networks.KANetsData, chex.PRNGKey], None]] = None,
    scf_fraction: float = 0.0,
    start_iteration: int = 0,
    opt_state: Optional[optax.OptState] = None,
    data: Optional[networks.KANetsData] = None,
):
  """Performs pretraining to match the KAN orbitals to KS-DFT orbitals."""
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
      states=0,
  )

  if data is None:
    data = networks.KANetsData(
        positions=positions, spins=spins, atoms=atoms, charges=charges
    )

  iterator = trange(
      start_iteration, iterations, desc='Pretrain-DFT', dynamic_ncols=True
  )

  for t in iterator:
    sharded_key, subkeys = jax.random.split(sharded_key)
    data, params, opt_state_pt, loss = pretrain_step(
        data, params, opt_state_pt, subkeys, dft_approx)
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
