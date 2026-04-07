import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import ml_collections
import numpy as np

from hamiltonian import hamiltonian
from kan_wavefunction_case_one.kan_networks_case_one import make_kan_net
from kan_wavefunction_case_one.spin_indices import jastrow_indices_ee
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


def _build_network(cfg: ml_collections.ConfigDict):
    molecule = cfg.system.molecule
    electrons = tuple(cfg.system.electrons)
    nelectrons = sum(electrons)
    natoms = len(molecule)
    nfeatures = int(cfg.nfeatures)

    atoms = jnp.array([atom.coords for atom in molecule])
    charges = jnp.array([atom.charge for atom in molecule])
    spins_list = [1] * electrons[0] + [-1] * electrons[1]
    spins_jastrow = jnp.array(spins_list)
    spins = jnp.array([spins_list])
    g = jnp.array(cfg.g)
    k = jnp.array(cfg.k)
    layer_dims = jnp.array(cfg.layer_dims)
    grid_range_envelope = jnp.array(cfg.envelope.grid_range_envelope)

    parallel_indices, antiparallel_indices, n_parallel, n_antiparallel = jastrow_indices_ee(
        spins=spins_jastrow,
        nelectrons=nelectrons,
    )
    _, signed_network, orbitals_apply = make_kan_net(
        nspins=electrons,
        charges=charges,
        nelectrons=nelectrons,
        nfeatures=nfeatures,
        n_parallel=n_parallel,
        n_antiparallel=n_antiparallel,
        parallel_indices=parallel_indices,
        antiparallel_indices=antiparallel_indices,
        grid_range=cfg.grid_range,
        g=g,
        k=k,
        natoms=natoms,
        ndims=3,
        layer_dims=layer_dims,
        g_envelope=int(cfg.envelope.g_envelope),
        k_envelope=int(cfg.envelope.k_envelope),
        grid_range_envelope=grid_range_envelope,
        chebyshev=bool(cfg.chebyshev),
        spline=bool(cfg.spline),
        add_residual=bool(cfg.add_residual),
        add_bias=bool(cfg.add_bias),
        external_weights=bool(cfg.external_weights),
        envelope_chebyshev=bool(cfg.envelope_chebyshev),
        envelope_spline=bool(cfg.envelope_spline),
        envelope_simple=bool(cfg.envelope_simple),
    )
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
