"""Jastrow factors for QMC wavefunctions."""

from typing import Callable, Mapping, Sequence

import jax.numpy as jnp

from networks import Array


def spin_pair_indices(nspins: Sequence[int]) -> tuple[Array, Array]:
    """Return same-spin and opposite-spin electron-pair indices."""

    n_alpha, n_beta = (int(nspins[0]), int(nspins[1]))
    alpha = list(range(n_alpha))
    beta = list(range(n_alpha, n_alpha + n_beta))

    same_spin = []
    for group in (alpha, beta):
        for i, electron_i in enumerate(group):
            for electron_j in group[i + 1 :]:
                same_spin.append((electron_i, electron_j))

    opposite_spin = [(electron_i, electron_j) for electron_i in alpha for electron_j in beta]

    same_spin = same_spin or [(0, 0)]
    opposite_spin = opposite_spin or [(0, 0)]
    return jnp.asarray(same_spin, dtype=jnp.int32), jnp.asarray(opposite_spin, dtype=jnp.int32)


def spin_pair_indices_or_empty(nspins: Sequence[int]) -> tuple[Array, Array]:
    """Return same-spin and opposite-spin pair indices without sentinel pairs."""

    n_alpha, n_beta = (int(nspins[0]), int(nspins[1]))
    alpha = list(range(n_alpha))
    beta = list(range(n_alpha, n_alpha + n_beta))

    same_spin = []
    for group in (alpha, beta):
        for i, electron_i in enumerate(group):
            for electron_j in group[i + 1 :]:
                same_spin.append((electron_i, electron_j))

    opposite_spin = [(electron_i, electron_j) for electron_i in alpha for electron_j in beta]
    return (
        jnp.asarray(same_spin, dtype=jnp.int32).reshape((-1, 2)),
        jnp.asarray(opposite_spin, dtype=jnp.int32).reshape((-1, 2)),
    )


def init_pade_ee_jastrow() -> dict[str, Array]:
    """Initialize electron-electron Pade Jastrow variational parameters."""

    return {
        "ee_par": jnp.ones((1,)),
        "ee_anti": jnp.ones((1,)),
    }


def init_ferminet_ee_jastrow() -> dict[str, Array]:
    """Initialize FermiNet simple electron-electron Jastrow parameters."""

    return {
        "ee_par": jnp.ones((1,)),
        "ee_anti": jnp.ones((1,)),
    }


def _pair_distances(r_ee: Array, pair_indices: Array) -> Array:
    return r_ee[pair_indices[:, 0], pair_indices[:, 1], 0]


def _pade_cusp(r: Array, cusp: float, alpha: Array) -> Array:
    return cusp * r / (1.0 + alpha * r)


def _ferminet_cusp(r: Array, cusp: float, alpha: Array) -> Array:
    return -(cusp * alpha**2) / (alpha + r)


def _sum_pair_cusps(
    r_ee: Array,
    params: Mapping[str, Array],
    same_spin_pairs: Array,
    opposite_spin_pairs: Array,
    cusp_fn: Callable[[Array, float, Array], Array],
) -> Array:
    total = jnp.asarray(0.0)
    if same_spin_pairs.shape[0] > 0:
        r_parallel = _pair_distances(r_ee, same_spin_pairs)
        total = total + jnp.sum(cusp_fn(r_parallel, 0.25, params["ee_par"]))
    if opposite_spin_pairs.shape[0] > 0:
        r_antiparallel = _pair_distances(r_ee, opposite_spin_pairs)
        total = total + jnp.sum(cusp_fn(r_antiparallel, 0.5, params["ee_anti"]))
    return total


def apply_pade_ee_jastrow(
    r_ee: Array,
    params: Mapping[str, Array],
    same_spin_pairs: Array,
    opposite_spin_pairs: Array,
) -> Array:
    """Evaluate log J_ee for Pade electron-electron cusp factors."""

    return _sum_pair_cusps(
        r_ee,
        params,
        same_spin_pairs,
        opposite_spin_pairs,
        _pade_cusp,
    )


def apply_ferminet_ee_jastrow(
    r_ee: Array,
    params: Mapping[str, Array],
    same_spin_pairs: Array,
    opposite_spin_pairs: Array,
) -> Array:
    """Evaluate the FermiNet simple electron-electron Jastrow factor."""

    return _sum_pair_cusps(
        r_ee,
        params,
        same_spin_pairs,
        opposite_spin_pairs,
        _ferminet_cusp,
    )
