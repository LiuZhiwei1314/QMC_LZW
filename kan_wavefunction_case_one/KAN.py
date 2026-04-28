from typing import Any, Mapping, Optional, Sequence

from flax import nnx

from jkan.layers import RBFLayer, SineLayer
from jkan.models import KAN as JKAN


def _default_required_parameters(
    layer_type: str,
    required_parameters: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Build minimal defaults so a KAN can be created from layer_dims only."""
    layer_type = layer_type.lower()

    if layer_type in ('base', 'spline'):
        params = {
            'k': 3,
            'G': 5,
            'grid_range': (-1.0, 1.0),
            'grid_e': 0.05,
            'residual': nnx.silu,
            'external_weights': True,
            'add_bias': True,
        }
    elif layer_type in ('chebyshev', 'legendre'):
        params = {
            'D': 6,
            'flavor': 'default',
            'residual': None,
            'external_weights': False,
            'add_bias': True,
        }
    elif layer_type == 'fourier':
        params = {
            'D': 8,
            'smooth_init': True,
            'add_bias': True,
        }
    elif layer_type == 'rbf':
        params = {
            'D': 8,
            'kernel': {'type': 'gaussian'},
            'add_bias': True,
        }
    elif layer_type == 'sine':
        params = {
            'D': 8,
            'add_bias': True,
        }
    else:
        raise ValueError(f'Unsupported layer_type: {layer_type}')

    if required_parameters:
        params.update(dict(required_parameters))
    return params


class KAN(nnx.Module):
    """Thin wrapper over jkan.models.KAN with practical default parameters."""

    def __init__(
        self,
        layer_dims: Sequence[int],
        layer_type: str = 'base',
        required_parameters: Optional[Mapping[str, Any]] = None,
        seed: int = 42,
    ):
        self.layer_dims = tuple(int(v) for v in layer_dims)
        self.layer_type = layer_type.lower()
        self.required_parameters = _default_required_parameters(
            self.layer_type,
            required_parameters=required_parameters,
        )
        self.net = JKAN(
            layer_dims=list(self.layer_dims),
            layer_type=self.layer_type,
            required_parameters=self.required_parameters,
            seed=seed,
        )

    def __call__(self, x):
        return self.net(x)

    def update_grids(self, x, g_new: int) -> None:
        if hasattr(self.net, 'update_grids'):
            self.net.update_grids(x=x, G_new=g_new)


def build_kan(
    layer_dims: Sequence[int],
    layer_type: str = 'base',
    seed: int = 42,
    required_parameters: Optional[Mapping[str, Any]] = None,
) -> KAN:
    """Minimal constructor used by training/inference call sites."""
    return KAN(
        layer_dims=layer_dims,
        layer_type=layer_type,
        required_parameters=required_parameters,
        seed=seed,
    )


class CustomKAN(nnx.Module):
    """Two-stage RBF -> GELU -> Sine stack."""

    def __init__(
        self,
        rbf_layers: Sequence[int],
        sine_layers: Sequence[int],
        add_bias: bool = True,
        seed: int = 42,
    ):
        self.r_layers = nnx.List([
            RBFLayer(
                n_in=rbf_layers[i],
                n_out=rbf_layers[i + 1],
                D=5,
                kernel={'type': 'gaussian'},
                add_bias=add_bias,
                seed=seed + i,
            )
            for i in range(len(rbf_layers) - 1)
        ])

        self.s_layers = nnx.List([
            SineLayer(
                n_in=sine_layers[i],
                n_out=sine_layers[i + 1],
                D=8,
                add_bias=add_bias,
                seed=seed + 100 + i,
            )
            for i in range(len(sine_layers) - 1)
        ])

    def __call__(self, x):
        for layer in self.r_layers:
            x = layer(x)

        x = nnx.gelu(x, approximate=True)

        for layer in self.s_layers:
            x = layer(x)

        return x