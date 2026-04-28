"""Minimal sequential KAN model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import jax.numpy as jnp

from ..layers import SplineKANLayer
from ..utils import ensure_2d


Array = jnp.ndarray


@dataclass
class KAN:
    """Small research-first sequential KAN wrapper."""

    layer_dims: Sequence[int]
    k: int = 3
    G: int = 5
    grid_range: tuple[float, float] = (-1.0, 1.0)
    grid_eps: float = 0.05
    seed: int = 42
    layers: list[SplineKANLayer] = field(init=False)

    def __post_init__(self) -> None:
        if len(self.layer_dims) < 2:
            raise ValueError("layer_dims must include at least input and output dimensions.")
        self.layers = [
            SplineKANLayer(
                n_in=self.layer_dims[i],
                n_out=self.layer_dims[i + 1],
                k=self.k,
                G=self.G,
                grid_range=self.grid_range,
                grid_eps=self.grid_eps,
                seed=self.seed + i,
            )
            for i in range(len(self.layer_dims) - 1)
        ]

    def __call__(self, x: Array) -> Array:
        x = ensure_2d(x)
        for layer in self.layers:
            x = layer(x)
        return x

    def update_grids(self, x: Array, G_new: int) -> None:
        x = ensure_2d(x)
        for layer in self.layers:
            layer.update_grid(x, G_new)
            x = layer(x)
