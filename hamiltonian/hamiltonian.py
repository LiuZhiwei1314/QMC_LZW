from typing import Any, Callable, Optional, Sequence, Tuple, Union, Protocol, cast
import chex
from kan_wavefunction_case_one import kan_networks_case_one as networks
from tools.utils import utils
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
import folx




Array = Union[jnp.ndarray, np.ndarray]

WaveFunctionOutput = Callable[
    [networks.ParamTree, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    jnp.ndarray,
]


class LocalEnergy(Protocol):

  def __call__(
      self,
      params: networks.ParamTree,
      key: chex.PRNGKey,
      data: networks.KANetsData,
  ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
    """Returns the local energy of a Hamiltonian at a configuration.

    Args:
      params: network parameters.
      key: JAX PRNG state.
      data: MCMC configuration to evaluate.
    """
    ...


class MakeLocalEnergy(Protocol):

  def __call__(
      self,
      f,
      charges: jnp.ndarray,
      nspins: Sequence[int],
      use_scan: bool = False,
      complex_output: bool = False,
      **kwargs: Any
  ) -> LocalEnergy:
    """Builds the LocalEnergy function.

    Args:
      f: Callable which evaluates the sign and log of the magnitude of the
        wavefunction.
      charges: nuclear charges.
      nspins: Number of particles of each spin.
      use_scan: Whether to use a `lax.scan` for computing the laplacian.
      complex_output: If true, the output of f is complex-valued.
      **kwargs: additional kwargs to use for creating the specific Hamiltonian.
    """
    ...


KineticEnergy = Callable[
    [networks.ParamTree, networks.KANetsData], jnp.ndarray
]


def local_kinetic_energy(
    f: Callable[..., Any],
    use_scan: bool = False,
    complex_output: bool = False,
    laplacian_method: str = 'default',
) -> KineticEnergy:
  r"""Creates a function to for the local kinetic energy, -1/2 \nabla^2 ln|f|.

  Args:
    f: Callable which evaluates the wavefunction as a
      (sign or phase, log magnitude) tuple.
    use_scan: Whether to use a `lax.scan` for computing the laplacian.
    complex_output: If true, the output of f is complex-valued.
    laplacian_method: Laplacian calculation method. One of:
      'default': take jvp(grad), looping over inputs
      'folx': use Microsoft's implementation of forward laplacian

  Returns:
    Callable which evaluates the local kinetic energy,
    -1/2f \nabla^2 f = -1/2 (\nabla^2 log|f| + (\nabla log|f|)^2).
  """

  phase_f = cast(WaveFunctionOutput, utils.select_output(f, 0))
  logabs_f = cast(WaveFunctionOutput, utils.select_output(f, 1))

  if laplacian_method == 'default':

    def _lapl_over_f(
      params: networks.ParamTree,
      data: networks.KANetsData,
    ) -> jnp.ndarray:
      n = data.positions.shape[0]
      #jax.debug.print("n:{}", n)
      eye = jnp.eye(n)
      grad_f = jax.grad(logabs_f, argnums=1)
      def grad_f_closure(x: jnp.ndarray) -> jnp.ndarray:
        return grad_f(params, x, data.spins, data.atoms, data.charges)
      primal, dgrad_f = jax.linearize(grad_f_closure, data.positions)

      if complex_output:
        # Modified by 2026.3.16: Apply Method 3 (chain rule on U=e^{i\theta}) to avoid branch cut jumps.
        # [NEW CODE START: 2026.3.16 - Chain rule for continuous U]
        def u_real_closure(x: jnp.ndarray) -> jnp.ndarray:
            return jnp.real(phase_f(params, x, data.spins, data.atoms, data.charges))
        def u_imag_closure(x: jnp.ndarray) -> jnp.ndarray:
            return jnp.imag(phase_f(params, x, data.spins, data.atoms, data.charges))

        grad_u_real = jax.grad(u_real_closure, argnums=0)
        grad_u_imag = jax.grad(u_imag_closure, argnums=0)

        primal_u_real, dgrad_u_real = jax.linearize(grad_u_real, data.positions)
        primal_u_imag, dgrad_u_imag = jax.linearize(grad_u_imag, data.positions)

        u_val = u_real_closure(data.positions) + 1.j * u_imag_closure(data.positions)
        grad_u = primal_u_real + 1.j * primal_u_imag

        grad_phase = -1.j * grad_u / u_val

        hessian_u_diagonal = lambda i: dgrad_u_real(eye[i])[i] + 1.j * dgrad_u_imag(eye[i])[i]
        def compute_hessian_phase_diag(i: int) -> jnp.ndarray:
            lapl_u_i = hessian_u_diagonal(i)
            return -1.j * (lapl_u_i / u_val - (grad_u[i] / u_val)**2)

        hessian_diagonal = (
            lambda i: dgrad_f(eye[i])[i] + 1.j * compute_hessian_phase_diag(i))
        # [NEW CODE END: 2026.3.16]
      else:
        hessian_diagonal = lambda i: dgrad_f(eye[i])[i]

      if use_scan:
        _, diagonal = lax.scan(
            lambda i, _: (i + 1, hessian_diagonal(i)), 0, None, length=n)
        result = -0.5 * jnp.sum(diagonal)
      else:
        initial_value = jnp.asarray(0.0j if complex_output else 0.0)
        result = -0.5 * lax.fori_loop(
            0, n, lambda i, val: val + hessian_diagonal(i), initial_value)
      result -= 0.5 * jnp.sum(primal ** 2)

      if complex_output:
        # Modified by 2026.3.16: Expand kinetic energy strictly using stable grad_phase        
        # [NEW CODE START: 2026.3.16 - Energy expansion]
        result += 0.5 * jnp.sum(grad_phase ** 2)
        result -= 1.j * jnp.sum(primal * grad_phase)
        # [NEW CODE END: 2026.3.16]
      return result

    return _lapl_over_f

  elif laplacian_method == 'folx':
    if folx is None:
      raise ModuleNotFoundError(
          "folx is required when laplacian_method='folx'."
      )

    def _lapl_over_f(
        params: networks.ParamTree,
        data: networks.KANetsData,
    ) -> jnp.ndarray:
      f_closure = lambda x: f(params, x, data.spins, data.atoms, data.charges)
      f_wrapped = folx.forward_laplacian(f_closure, sparsity_threshold=0)
      output = f_wrapped(data.positions)
      
      A_lapl = output[1].laplacian
      A_grad = output[1].jacobian.dense_array
      
      result = -0.5 * (A_lapl + jnp.sum(A_grad ** 2))
      
      if complex_output:
        U_val = output[0].x
        U_lapl = output[0].laplacian
        U_grad = output[0].jacobian.dense_array
        
        result -= 0.5 * (U_lapl / U_val)
        result -= jnp.sum((U_grad / U_val) * A_grad)
        
      return result

    return _lapl_over_f
  
  else:
    raise NotImplementedError(f"Laplacian method '{laplacian_method}' is not implemented.")


def potential_electron_electron(r_ee: Array) -> jnp.ndarray:
  """Returns the electron-electron potential.

  Args:
    r_ee: Shape (neletrons, nelectrons, :). r_ee[i,j,0] gives the distance
      between electrons i and j. Other elements in the final axes are not
      required.
  """
  r_ee = r_ee[jnp.triu_indices_from(r_ee[..., 0], 1)]
  return (1.0 / r_ee).sum()


def potential_electron_nuclear(charges: Array, r_ae: Array) -> jnp.ndarray:
  """Returns the electron-nuclearpotential.

  Args:
    charges: Shape (natoms). Nuclear charges of the atoms.
    r_ae: Shape (nelectrons, natoms). r_ae[i, j] gives the distance between
      electron i and atom j.
  """
  return -jnp.sum(charges[None, :] / r_ae[..., 0])


def potential_nuclear_nuclear(charges: Array, atoms: Array) -> jnp.ndarray:
  """Returns the nuclear-nuclear potential.

  Args:
    charges: Shape (natoms). Nuclear charges of the atoms.
    atoms: Shape (natoms, ndim). Positions of the atoms.
  """
  r_aa = jnp.linalg.norm(atoms[None, ...] - atoms[:, None], axis=-1)
  return jnp.sum(
      jnp.triu((charges[None, ...] * charges[..., None]) / r_aa, k=1))


def potential_energy(r_ae: Array, r_ee: Array, atoms: Array,
                     charges: Array) -> jnp.ndarray:
  """Returns the potential energy for this electron configuration.

  Args:
    r_ae: Shape (nelectrons, natoms). r_ae[i, j] gives the distance between
      electron i and atom j.
    r_ee: Shape (neletrons, nelectrons, :). r_ee[i,j,0] gives the distance
      between electrons i and j. Other elements in the final axes are not
      required.
    atoms: Shape (natoms, ndim). Positions of the atoms.
    charges: Shape (natoms). Nuclear charges of the atoms.
  """
  return (potential_electron_electron(r_ee) +
          potential_electron_nuclear(charges, r_ae) +
          potential_nuclear_nuclear(charges, atoms))


def local_energy(
    f,
    charges: jnp.ndarray,
    nspins: Sequence[int],
    use_scan: bool = False,
    complex_output: bool = False,
    laplacian_method: str = 'default',) -> LocalEnergy:
  """Creates the function to evaluate the local energy."""
  del nspins

  ke = local_kinetic_energy(f,
                              use_scan=use_scan,
                              complex_output=complex_output,
                              laplacian_method=laplacian_method)
  

  def _e_l(params: networks.ParamTree,
           key: chex.PRNGKey,
           data: networks.KANetsData,
   ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
    """Returns the total energy.

    Args:
      params: network parameters.
      key: RNG state.
      data: MCMC configuration.
    """

    ae, _, r_ae, r_ee = networks.construct_input_features(
        data.positions, data.atoms
    )
    potential = (potential_energy(r_ae, r_ee, data.atoms, data.charges))
    """something is wrong in the kinetic energy calculation.31.10.2025."""
    #jax.debug.print("data:{}", data)
    kinetic = ke(params, data)
    #total_energy = potential
    total_energy = potential + kinetic
    energy_mat = None  # Not necessary for ground state
    return total_energy, energy_mat

  return _e_l
