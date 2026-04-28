"""Spline grid utilities for JKAN."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp


Array = jnp.ndarray


@dataclass
class SplineGrid:
    """Stores and updates a spline knot grid."""

    n_nodes: int
    k: int
    G: int
    grid_range: tuple[float, float] = (-1.0, 1.0)
    grid_eps: float = 0.05

    def __post_init__(self) -> None:
        self.item = self.initialize()

    def initialize(self) -> Array:
        h = (self.grid_range[1] - self.grid_range[0]) / self.G
        grid = jnp.arange(-self.k, self.G + self.k + 1, dtype=jnp.float32) * h
        grid = grid + self.grid_range[0]
        return jnp.expand_dims(grid, axis=0).repeat(self.n_nodes, axis=0)

    def update(self, x: Array, G_new: int) -> Array:
        batch = x.shape[0]
        x_sorted = jnp.sort(x, axis=0)
        ids = jnp.concatenate(
            (jnp.floor(batch / G_new * jnp.arange(G_new)).astype(int), jnp.array([-1]))
        )
        grid_adaptive = x_sorted[ids]

        margin = 0.01
        uniform_step = (x_sorted[-1] - x_sorted[0] + 2.0 * margin) / G_new
        grid_uniform = (
            jnp.arange(G_new + 1, dtype=jnp.float32)[:, None] * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1.0 - self.grid_eps) * grid_adaptive
        h = (grid[-1] - grid[0]) / G_new
        left = h * jnp.arange(self.k, 0, -1)[:, None]
        right = h * jnp.arange(1, self.k + 1)[:, None]
        grid = jnp.concatenate([grid[:1] - left, grid, grid[-1:] + right], axis=0)

        self.item = grid.T
        self.G = G_new
        return self.item
