from typing import Any, Iterable, MutableMapping, Union, Tuple

import chex
import jax.numpy as jnp


ParamTree = Union[jnp.ndarray, Iterable['ParamTree'], MutableMapping[Any, 'ParamTree']]
Param = MutableMapping[str, jnp.ndarray]


@chex.dataclass
class KANetsData:
    positions: Any
    spins: Any
    atoms: Any
    charges: Any


def construct_input_features(
        pos: jnp.ndarray,
        atoms: jnp.ndarray,
        ndim: int = 3) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Constructs inputs to Fermi Net from raw electron and atomic positions.
    For KANets, we need do the normalization for the input layer to be [-1, 1].
    currently the optimization is not stable. We consider to reconstruct it as FermiNet did before?
    They used mean value to stabilize the networks."""
    #jax.debug.print("atoms:{}", atoms)
    assert atoms.shape[1] == ndim
    ae = jnp.reshape(pos, [-1, 1, ndim]) - atoms[None, ...]
    ee = jnp.reshape(pos, [1, -1, ndim]) - jnp.reshape(pos, [-1, 1, ndim])
    r_ae = jnp.linalg.norm(ae, axis=2, keepdims=True)
    n = ee.shape[0]
    r_ee = (jnp.linalg.norm(ee + jnp.eye(n)[..., None], axis=-1) * (1.0 - jnp.eye(n)))
    return ae, ee, r_ae, r_ee[..., None]

