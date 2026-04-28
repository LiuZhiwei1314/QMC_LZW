"""Minimal spline-based KAN layer for JKAN."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import jax
import jax.numpy as jnp

from ..grids import SplineGrid
from ..initializers import init_base_params
from ..utils import solve_full_lstsq


Array = jnp.ndarray


def silu_residual(x: Array) -> Array:
    return jax.nn.silu(x)


@dataclass
class SplineKANLayer:
    """Research-first spline KAN layer."""

    n_in: int
    n_out: int
    k: int = 3
    G: int = 5
    grid_range: tuple[float, float] = (-1.0, 1.0)
    grid_eps: float = 0.05
    residual: Optional[Callable[[Array], Array]] = silu_residual
    external_weights: bool = True
    add_bias: bool = True
    seed: int = 42

    def __post_init__(self) -> None:
        self.grid = SplineGrid(
            n_nodes=self.n_in * self.n_out,
            k=self.k,
            G=self.G,
            grid_range=self.grid_range,
            grid_eps=self.grid_eps,
        )
        c_basis, c_res, c_spl, bias = init_base_params(
            key=jax.random.key(self.seed),
            n_in=self.n_in,
            n_out=self.n_out,
            n_basis=self.G + self.k,
            residual=self.residual is not None,
            external_weights=self.external_weights,
            add_bias=self.add_bias,
        )
        self.c_basis = c_basis
        self.c_res = c_res
        self.c_spl = c_spl
        self.bias = bias

    def basis(self, x: Array) -> Array:
        batch = x.shape[0]
        x_ext = jnp.einsum("bi,o->boi", x, jnp.ones(self.n_out)).reshape(
            batch, self.n_in * self.n_out
        )
        x_ext = jnp.transpose(x_ext, (1, 0))
        grid = jnp.expand_dims(self.grid.item, axis=2)
        x_exp = jnp.expand_dims(x_ext, axis=1)
        basis = ((x_exp >= grid[:, :-1]) & (x_exp < grid[:, 1:])).astype(x.dtype)

        for order in range(1, self.k + 1):
            left = (x_exp - grid[:, : -(order + 1)]) / (
                grid[:, order:-1] - grid[:, : -(order + 1)]
            )
            right = (grid[:, order + 1 :] - x_exp) / (
                grid[:, order + 1 :] - grid[:, 1:-order]
            )
            basis = left * basis[:, :-1] + right * basis[:, 1:]

        return jnp.nan_to_num(basis)

    def spline_values(self, x: Array) -> Array:
        basis = self.basis(x)
        weights = self.c_basis
        if self.c_spl is not None:
            weights = weights * self.c_spl.reshape(self.n_in * self.n_out, 1)
        values = jnp.einsum("ij,ijk->ik", weights, basis)
        return values.T.reshape(x.shape[0], self.n_in, self.n_out)

    def __call__(self, x: Array) -> Array:
        spline = self.spline_values(x)
        output = spline

        if self.residual is not None and self.c_res is not None:
            base = self.residual(x)
            residual = base[:, :, None] * self.c_res.T[None, :, :]
            output = output + residual

        output = jnp.sum(output, axis=1)
        if self.bias is not None:
            output = output + self.bias[None, :]
        return output

    def update_grid(self, x: Array, G_new: int) -> Array:
        old_basis = self.basis(x)
        old_values = jnp.einsum("ij,ijk->ik", self.c_basis, old_basis)
        self.grid.update(x, G_new)
        self.G = G_new
        new_basis = self.basis(x)
        design = jnp.transpose(new_basis, (0, 2, 1))
        targets = jnp.expand_dims(old_values, axis=-1)
        self.c_basis = jnp.squeeze(solve_full_lstsq(design, targets), axis=-1)
        return self.grid.item
