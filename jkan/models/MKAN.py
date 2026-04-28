from jax import numpy as jnp

from flax import nnx

from ..layers import get_layer

from typing import List, Sequence, Union


class MultKAN(nnx.Module):
    """
    KAN model with multiplication nodes inspired by pykan's MultKAN.

    Width format:
    - Plain int: number of additive nodes, e.g. [2, 8, 1]
    - Pair [n_sum, n_mult]: additive and multiplicative nodes for a layer,
      e.g. [2, [5, 3], 1]
    """

    def __init__(
        self,
        width: Sequence[Union[int, Sequence[int]]],
        layer_type: str = "base",
        required_parameters: Union[None, dict] = None,
        mult_arity: Union[int, Sequence[Sequence[int]]] = 2,
        affine_trainable: bool = False,
        seed: int = 42,
    ):
        del affine_trainable

        LayerClass = get_layer(layer_type.lower())

        if required_parameters is None:
            raise ValueError(
                "required_parameters must be provided as a dictionary for the selected layer_type."
            )

        self.width = [
            [int(item), 0] if isinstance(item, int) else [int(item[0]), int(item[1])]
            for item in width
        ]
        self.depth = len(self.width) - 1

        if isinstance(mult_arity, int):
            self.mult_homo = True
        else:
            self.mult_homo = False
        self.mult_arity = mult_arity

        self.layers = nnx.List(
            [
                LayerClass(
                    n_in=self.width_in[i],
                    n_out=self.width_out[i + 1],
                    **required_parameters,
                    seed=seed + i,
                )
                for i in range(self.depth)
            ]
        )

        self.node_bias = nnx.List(
            [nnx.Param(jnp.zeros((self.width_in[i + 1],))) for i in range(self.depth)]
        )
        self.node_scale = nnx.List(
            [nnx.Param(jnp.ones((self.width_in[i + 1],))) for i in range(self.depth)]
        )
        self.subnode_bias = nnx.List(
            [nnx.Param(jnp.zeros((self.width_out[i + 1],))) for i in range(self.depth)]
        )
        self.subnode_scale = nnx.List(
            [nnx.Param(jnp.ones((self.width_out[i + 1],))) for i in range(self.depth)]
        )

    def _arity_list_for_width(self, width_idx: int) -> List[int]:
        dim_mult = self.width[width_idx][1]
        if dim_mult == 0:
            return []
        if self.mult_homo:
            return [int(self.mult_arity)] * dim_mult
        if width_idx >= len(self.mult_arity):
            raise ValueError(
                f"Missing multiplication arities for width index {width_idx}."
            )
        arities = [int(v) for v in self.mult_arity[width_idx]]
        if len(arities) != dim_mult:
            raise ValueError(
                f"Expected {dim_mult} multiplication arities at width index {width_idx}, got {len(arities)}."
            )
        return arities

    @property
    def width_in(self) -> List[int]:
        return [layer[0] + layer[1] for layer in self.width]

    @property
    def width_out(self) -> List[int]:
        width_out = []
        for idx, layer in enumerate(self.width):
            n_sum, _ = layer
            width_out.append(n_sum + sum(self._arity_list_for_width(idx)))
        return width_out

    def _apply_multiplication(self, x, layer_idx: int):
        width_idx = layer_idx + 1
        dim_sum = self.width[width_idx][0]
        arities = self._arity_list_for_width(width_idx)

        if not arities:
            return x[:, :dim_sum]

        x_sum = x[:, :dim_sum]
        offset = dim_sum
        mult_terms = []
        for arity in arities:
            mult_terms.append(jnp.prod(x[:, offset : offset + arity], axis=1, keepdims=True))
            offset += arity
        x_mult = jnp.concatenate(mult_terms, axis=1)
        return jnp.concatenate([x_sum, x_mult], axis=1)

    def update_grids(self, x, G_new):
        for idx, layer in enumerate(self.layers):
            layer.update_grid(x, G_new)
            x = layer(x)
            x = self.subnode_scale[idx][...][None, :] * x + self.subnode_bias[idx][...][None, :]
            x = self._apply_multiplication(x, idx)
            x = self.node_scale[idx][...][None, :] * x + self.node_bias[idx][...][None, :]

    def __call__(self, x):
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            x = self.subnode_scale[idx][...][None, :] * x + self.subnode_bias[idx][...][None, :]
            x = self._apply_multiplication(x, idx)
            x = self.node_scale[idx][...][None, :] * x + self.node_bias[idx][...][None, :]
        return x
