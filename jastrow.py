"""Jastrow factors for QMC wavefunctions."""

from __future__ import annotations

from typing import Mapping, Sequence

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


def init_pade_ee_jastrow() -> dict[str, Array]:
    """Initialize electron-electron Pade Jastrow variational parameters."""

    return {
        "ee_par": jnp.ones((1,)),
        "ee_anti": jnp.ones((1,)),
    }


def _pair_distances(r_ee: Array, pair_indices: Array) -> Array:
    return r_ee[pair_indices[:, 0], pair_indices[:, 1], 0]


def _pade_cusp(r: Array, cusp: float, alpha: Array) -> Array:
    return cusp * r / (1.0 + alpha * r)


def apply_pade_ee_jastrow(
    r_ee: Array,
    params: Mapping[str, Array],
    same_spin_pairs: Array,
    opposite_spin_pairs: Array,
) -> Array:
    """Evaluate log J_ee for Pade electron-electron cusp factors."""

    r_parallel = _pair_distances(r_ee, same_spin_pairs)
    r_antiparallel = _pair_distances(r_ee, opposite_spin_pairs)
    j_parallel = jnp.sum(_pade_cusp(r_parallel, 0.25, params["ee_par"]))
    j_antiparallel = jnp.sum(_pade_cusp(r_antiparallel, 0.5, params["ee_anti"]))
    return j_parallel + j_antiparallel
