import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
from flax import nnx

import hamiltonian
import envelope
import jastrow
import networks
from jkan.models import MultKAN
from tools.utils import system


def _load_config(config_path: Path) -> ml_collections.ConfigDict:
    raw_cfg = json.loads(config_path.read_text())
    molecule = [
        system.Atom(
            atom['symbol'],
            atom['coords'],
            charge=atom['charge'],
            atomic_number=atom['atomic_number'],
            units=atom.get('units', 'bohr'),
        )
        for atom in raw_cfg['system']['molecule']
    ]
    raw_cfg['system']['molecule'] = molecule
    return ml_collections.ConfigDict(raw_cfg)


def _load_checkpoint(checkpoint_path: Path) -> dict[str, Any]:
    with checkpoint_path.open('rb') as handle:
        return pickle.load(handle)


def _load_positions(path: Path | None, checkpoint_data) -> jnp.ndarray:
    if path is None:
        return checkpoint_data.positions
    if path.suffix == '.npy':
        return jnp.array(np.load(path))
    payload = json.loads(path.read_text())
    return jnp.array(payload)


def _first_int(values, default: int) -> int:
    if values is None:
        return default
    arr = np.asarray(values).reshape(-1)
    return int(arr[0]) if arr.size else default


def _first_grid_range(values, default=(-1.0, 1.0)) -> tuple[float, float]:
    if values is None:
        return tuple(default)
    arr = np.asarray(values)
    if arr.ndim == 1 and arr.size >= 2:
        return (float(arr[0]), float(arr[1]))
    if arr.ndim >= 2 and arr.shape[-1] >= 2:
        flat = arr.reshape(-1, arr.shape[-1])
        return (float(flat[0, 0]), float(flat[0, 1]))
    return tuple(default)


def _array_partitions(sizes):
    return list(np.cumsum(tuple(int(size) for size in sizes)))[:-1]


def _build_network(cfg: ml_collections.ConfigDict):
    molecule = cfg.system.molecule
    electrons = tuple(cfg.system.electrons)
    nelectrons = sum(electrons)
    natoms = len(molecule)
    nfeatures = int(cfg.nfeatures)

    atoms = jnp.array([atom.coords for atom in molecule])
    charges = jnp.array([atom.charge for atom in molecule])
    spins_list = [1] * electrons[0] + [-1] * electrons[1]
    spins = jnp.array([spins_list])

    mkan_cfg = cfg.get('mkan', {})
    layer_type = str(mkan_cfg.get('layer_type', 'spline')).lower()
    mkan_input_dim = int(nfeatures if mkan_cfg.get('input_dim', None) is None else mkan_cfg.input_dim)
    output_default = (2 * nelectrons) if bool(cfg.complex_output) else nelectrons
    mkan_output_dim = int(output_default if mkan_cfg.get('output_dim', None) is None else mkan_cfg.output_dim)

    if mkan_cfg.get('width', None) is None:
        hidden_dims = [int(v) for v in np.asarray(cfg.layer_dims).reshape(-1)[1:-1]]
        width = [mkan_input_dim, *hidden_dims, mkan_output_dim]
    else:
        width = list(mkan_cfg.width)
        width[0] = mkan_input_dim
        width[-1] = mkan_output_dim

    required_parameters = mkan_cfg.get('required_parameters', None)
    if required_parameters is None:
        if layer_type in ('chebyshev', 'legendre'):
            required_parameters = {
                'D': _first_int(cfg.k, 3),
                'flavor': 'exact' if layer_type == 'chebyshev' else None,
                'external_weights': bool(cfg.external_weights),
                'add_bias': bool(cfg.add_bias),
            }
        elif layer_type in ('base', 'spline'):
            required_parameters = {
                'k': _first_int(cfg.k, 3),
                'G': _first_int(cfg.g, 5),
                'grid_range': _first_grid_range(cfg.grid_range),
                'external_weights': bool(cfg.external_weights),
                'add_bias': bool(cfg.add_bias),
            }
        elif layer_type == 'rbf':
            required_parameters = {
                'D': _first_int(cfg.k, 5),
                'grid_range': _first_grid_range(cfg.grid_range, default=(-2.0, 2.0)),
                'external_weights': bool(cfg.external_weights),
                'add_bias': bool(cfg.add_bias),
            }
        elif layer_type == 'sine':
            required_parameters = {
                'D': _first_int(cfg.k, 5),
                'external_weights': bool(cfg.external_weights),
                'add_bias': bool(cfg.add_bias),
            }
        elif layer_type == 'fourier':
            required_parameters = {
                'D': _first_int(cfg.k, 5),
                'add_bias': bool(cfg.add_bias),
            }
        else:
            raise ValueError(f'Unsupported inference MKAN layer_type: {layer_type}')
    else:
        required_parameters = dict(required_parameters)

    model_template = MultKAN(
        width=width,
        layer_type=layer_type,
        required_parameters=required_parameters,
        mult_arity=mkan_cfg.get('mult_arity', 2),
        seed=int(cfg.seed),
    )
    graphdef, _, static_state = nnx.split(model_template, nnx.Param, ...)
    jastrow_type = str(cfg.get('jastrow', {}).get('type', 'pade')).lower()
    if jastrow_type == 'pade':
        same_spin_pairs, opposite_spin_pairs = jastrow.spin_pair_indices(electrons)
        apply_jastrow = jastrow.apply_pade_ee_jastrow
    elif jastrow_type == 'ferminet':
        same_spin_pairs, opposite_spin_pairs = jastrow.spin_pair_indices_or_empty(electrons)
        apply_jastrow = jastrow.apply_ferminet_ee_jastrow
    else:
        raise ValueError(f'Unsupported jastrow.type={jastrow_type!r}.')
    active_spin_channels = networks.active_spin_channels(electrons)

    def apply_mkan(params, features):
        model_params = params['mkan'] if isinstance(params, dict) and 'mkan' in params else params
        model = nnx.merge(graphdef, model_params, static_state)
        return model(features)

    def orbitals_apply(params, pos, spins_, atoms_, charges_):
        del spins_, charges_
        ae, _, r_ae, _ = networks.construct_input_features(pos, atoms_, ndim=3)
        h_one = jnp.concatenate((r_ae, ae), axis=2).reshape(nelectrons, -1)
        orbital_values = apply_mkan(params, h_one)
        if bool(cfg.complex_output):
            orbital_values = (
                orbital_values[..., 0:2 * nelectrons:2]
                + 1.0j * orbital_values[..., 1:2 * nelectrons:2]
            )
        else:
            orbital_values = orbital_values[..., :nelectrons]

        spin_partitions = _array_partitions(electrons)
        orbital_row_channels = jnp.split(orbital_values, spin_partitions, axis=0)
        orbital_channels = [
            channel[:, start : start + spin]
            for channel, spin, start in zip(
                orbital_row_channels,
                electrons,
                (0, int(electrons[0])),
            )
            if spin > 0
        ]
        if bool(cfg.envelope_simple):
            r_ae_channels = jnp.split(r_ae, spin_partitions, axis=0)
            r_ae_channels = [
                channel for channel, spin in zip(r_ae_channels, electrons) if spin > 0
            ]
            envelope_type = str(cfg.get('envelope_type', 'isotropic')).lower()
            if envelope_type == 'isotropic':
                apply_envelope = envelope.apply_isotropic_envelope
            elif envelope_type == 'chebyshev':
                apply_envelope = envelope.apply_chebyshev_envelope
            else:
                raise ValueError(f'Unsupported envelope_type={envelope_type!r}.')
            orbital_channels = [
                channel * apply_envelope(r_ae=r_ae_channel, **envelope_param)
                for channel, r_ae_channel, envelope_param in zip(
                    orbital_channels, r_ae_channels, params['envelope']
                )
            ]

        shapes = [(spin, -1, spin) for spin in active_spin_channels]
        orbital_channels = [
            jnp.reshape(channel, shape)
            for channel, shape in zip(orbital_channels, shapes)
        ]
        orbital_channels = [jnp.transpose(channel, (1, 0, 2)) for channel in orbital_channels]
        return orbital_channels

    def signed_network(params, pos, spins_, atoms_, charges_):
        determinant = orbitals_apply(params, pos, spins_, atoms_, charges_)
        phase, logmag = networks.logdet_matmul(determinant)
        if bool(cfg.get('jastrow', {}).get('ee', True)):
            _, _, _, r_ee = networks.construct_input_features(pos, atoms_, ndim=3)
            logmag = logmag + apply_jastrow(
                r_ee,
                params['jastrow_ee'],
                same_spin_pairs,
                opposite_spin_pairs,
            )
        return phase, logmag

    return signed_network, orbitals_apply, atoms, charges, spins, electrons


def _jsonable(value: Any) -> Any:
    if isinstance(value, (jnp.ndarray, np.ndarray)):
        return np.asarray(value).tolist()
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, complex):
        return {'real': value.real, 'imag': value.imag}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable(val) for key, val in value.items()}
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description='Run inference from a saved InsightQMC checkpoint.')
    parser.add_argument('--run-dir', default='outputs/default', help='Training output directory containing config.json and checkpoints/.')
    parser.add_argument('--checkpoint', default=None, help='Optional explicit checkpoint path. Defaults to <run-dir>/checkpoints/last.pkl.')
    parser.add_argument('--positions-file', default=None, help='Optional JSON or .npy file with positions for inference.')
    parser.add_argument('--compute-local-energy', action='store_true', help='Also evaluate local energy at the provided positions.')
    parser.add_argument('--output-json', default=None, help='Optional file to write inference results as JSON.')
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser()
    checkpoint_path = Path(args.checkpoint).expanduser() if args.checkpoint else run_dir / 'checkpoints' / 'last.pkl'
    config_path = run_dir / 'config.json'

    cfg = _load_config(config_path)
    checkpoint = _load_checkpoint(checkpoint_path)
    params = checkpoint['params']
    checkpoint_data = checkpoint['data']

    signed_network, orbitals_apply, atoms, charges, spins, electrons = _build_network(cfg)
    positions = _load_positions(Path(args.positions_file).expanduser() if args.positions_file else None, checkpoint_data)

    batch_signed_network = jax.vmap(signed_network, in_axes=(None, 0, None, None, None), out_axes=(0, 0))
    signs, logabs = batch_signed_network(params, positions, spins, atoms, charges)
    orbitals = jax.vmap(orbitals_apply, in_axes=(None, 0, None, None, None), out_axes=0)(params, positions, spins, atoms, charges)

    results: dict[str, Any] = {
        'checkpoint': str(checkpoint_path),
        'stage': checkpoint.get('stage'),
        'step': checkpoint.get('step'),
        'sign': signs,
        'logabs': logabs,
        'orbitals': orbitals,
    }

    if args.compute_local_energy:
        local_energy_fn = hamiltonian.local_energy(
            f=signed_network,
            nspins=electrons,
            charges=charges,
            use_scan=bool(cfg.use_scan),
            complex_output=bool(cfg.complex_output),
            laplacian_method=cfg.laplacian_method,
        )
        energy_keys = jax.random.split(jax.random.PRNGKey(int(cfg.seed)), positions.shape[0])
        local_energy_values, energy_mat = jax.vmap(
            local_energy_fn,
            in_axes=(None, 0, 0, None, None, None),
            out_axes=(0, 0),
        )(params, energy_keys, positions, spins, atoms, charges)
        results['local_energy'] = local_energy_values
        results['local_energy_mat'] = energy_mat

    json_ready = _jsonable(results)
    if args.output_json:
        output_path = Path(args.output_json).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(json_ready, indent=2))
    else:
        print(json.dumps(json_ready, indent=2))


if __name__ == '__main__':
    main()
