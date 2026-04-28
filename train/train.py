from typing import Any, Callable, Optional

import flax
from flax import nnx
from flax.training import train_state
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
from tqdm.auto import trange

from hamiltonian import hamiltonian
from initialization import electrons_initialization
from jkan.models import MultKAN
from kan_wavefunction_case_one import normal_network_blocks
from kan_wavefunction_case_one.kan_networks_case_one import KANetsData
from loss_function import loss as qmc_loss_functions
from monte_carlo_step import VMCmcstep
from optimizer.opt import make_opt_update_step, make_training_step
from train.pretrain_runner import PretrainRunner
from train.training_io import RunManager


def _to_scalar(x):
    return float(jnp.asarray(x).reshape(-1)[0])


def _first_int(values, default: int) -> int:
    if values is None:
        return default
    arr = np.asarray(values).reshape(-1)
    if arr.size == 0:
        return default
    return int(arr[0])


def _first_grid_range(values, default=(-1.0, 1.0)) -> tuple[float, float]:
    if values is None:
        return tuple(default)
    arr = np.asarray(values)
    if arr.ndim == 1 and arr.size >= 2:
        return (float(arr[0]), float(arr[1]))
    if arr.ndim >= 2 and arr.shape[-1] >= 2:
        return (float(arr.reshape(-1, arr.shape[-1])[0, 0]), float(arr.reshape(-1, arr.shape[-1])[0, 1]))
    return tuple(default)


def _array_partitions(sizes):
    return list(np.cumsum(tuple(int(size) for size in sizes)))[:-1]


def _construct_input_features(pos: jnp.ndarray, atoms: jnp.ndarray, ndim: int = 3):
    ae = jnp.reshape(pos, [-1, 1, ndim]) - atoms[None, ...]
    ee = jnp.reshape(pos, [1, -1, ndim]) - jnp.reshape(pos, [-1, 1, ndim])
    r_ae = jnp.linalg.norm(ae, axis=2, keepdims=True)
    n = ee.shape[0]
    r_ee = (jnp.linalg.norm(ee + jnp.eye(n)[..., None], axis=-1) * (1.0 - jnp.eye(n)))
    return ae, ee, r_ae, r_ee[..., None]


@flax.struct.dataclass
class RuntimeState:
    data: KANetsData
    key: Any
    mcmc_width: jnp.ndarray
    pmoves: np.ndarray


class VMCTrainer:
    """Flax-style trainer that keeps setup and loop logic modular."""

    def __init__(self, cfg: ml_collections.ConfigDict):
        self.cfg = cfg
        self.run_manager = RunManager(cfg.output)
        self.run_manager.save_config(cfg)
        self._read_config()
        self.pretrain_runner = PretrainRunner(
            run_manager=self.run_manager,
            build_checkpoint_state=self._build_checkpoint_state,
            enabled=self.run_pretrain,
            preiterations=self.preiterations,
            method=self.pretrain_method,
            pyscf_mol=self.pyscf_mol,
            molecule=self.molecule,
            electrons=self.electrons,
            restricted=self.pretrain_restricted,
            basis=self.pretrain_basis,
            core_electrons=self.core_electrons,
            hf_states=self.hf_states,
            hf_excitation_type=self.hf_excitation_type,
            dft_xc=self.dft_xc,
            dft_grid_level=self.dft_grid_level,
            scf_fraction=self.scf_fraction,
            batch_size=self.batch_size,
            pretrain_mcmc_steps=self.pretrain_mcmc_steps,
            pretrain_mcmc_width=self.pretrain_mcmc_width,
            debug=self.debug,
            scalar_pretrain=False,
            phase_weight=self.mkan_pretrain_phase_weight,
        )

    def _read_config(self) -> None:
        cfg = self.cfg
        self.molecule = cfg.system.molecule
        self.electrons = tuple(cfg.system.electrons)
        self.nelectrons = sum(self.electrons)
        self.natoms = len(self.molecule)

        self.batch_size = int(cfg.batch_size)
        self.nfeatures = int(cfg.nfeatures)
        self.atoms = jnp.array([atom.coords for atom in self.molecule])
        self.charges = jnp.array([atom.charge for atom in self.molecule])

        self.spins_list = [1] * self.electrons[0] + [-1] * self.electrons[1]
        self.spins_jastrow = jnp.array(self.spins_list)
        self.spins = jnp.array([self.spins_list])

        self.g = jnp.array(cfg.g)
        self.k = jnp.array(cfg.k)
        self.layer_dims = jnp.array(cfg.layer_dims)
        self.grid_range = cfg.grid_range
        self.grid_range_envelope = jnp.array(cfg.envelope.grid_range_envelope)

        self.seed = int(cfg.seed)
        self.seed_electrons_coords = int(cfg.seed_electrons_coords)
        self.init_width = float(cfg.init_width)
        self.core_electrons = cfg.core_electrons

        self.pretrain_method = str(cfg.get('pretrain_method', 'hf')).lower()
        self.pretrain_basis = cfg.get('pretrain_basis', cfg.get('hf_basis', 'ccpvdz'))
        self.pretrain_restricted = bool(
            cfg.get('pretrain_restricted', cfg.get('hf_restricted', False))
        )
        self.hf_states = int(cfg.get('hf_states', 0))
        self.hf_excitation_type = cfg.get('hf_excitation_type', 'ordered')
        self.dft_xc = cfg.get('dft_xc', 'pbe,pbe')
        self.dft_grid_level = cfg.get('dft_grid_level', 3)
        self.pyscf_mol = cfg.system.get('pyscf_mol')

        self.mcmc_steps = int(cfg.mcmc_steps)
        self.mcmc_width = float(cfg.mcmc_width)
        self.pretrain_mcmc_steps = int(cfg.get('pretrain_mcmc_steps', 1))
        self.pretrain_mcmc_width = float(cfg.get('pretrain_mcmc_width', 0.02))

        self.clip_local_energy = float(cfg.clip_local_energy)
        self.use_scan = bool(cfg.use_scan)
        self.complex_output = bool(cfg.complex_output)
        self.laplacian_method = cfg.laplacian_method
        self.scf_fraction = float(cfg.scf_fraction)
        self.t_init = int(cfg.t_init)
        self.debug = bool(cfg.debug)

        self.learning_rate = float(cfg.learning_rate)
        self.learning_rate_decay = float(cfg.learning_rate_decay)
        self.preiterations = int(cfg.preiterations)
        self.run_pretrain = bool(cfg.run_pretrain)
        self.iterations = int(cfg.iterations)

        self.chebyshev = bool(cfg.chebyshev)
        self.spline = bool(cfg.spline)
        self.add_residual = bool(cfg.add_residual)
        self.add_bias = bool(cfg.add_bias)
        self.external_weights = bool(cfg.external_weights)
        self.envelope_chebyshev = bool(cfg.envelope_chebyshev)
        self.envelope_spline = bool(cfg.envelope_spline)
        self.envelope_simple = bool(cfg.envelope_simple)
        self.g_envelope = int(cfg.envelope.g_envelope)
        self.k_envelope = int(cfg.envelope.k_envelope)

        mkan_cfg = cfg.get('mkan', {})
        self.mkan_layer_type = str(mkan_cfg.get(
            'layer_type',
            'chebyshev' if self.chebyshev else 'spline',
        )).lower()
        self.mkan_mult_arity = mkan_cfg.get('mult_arity', 2)
        self.mkan_width = mkan_cfg.get('width', None)
        self.mkan_required_parameters = mkan_cfg.get('required_parameters', None)
        self.mkan_pretrain_phase_weight = float(mkan_cfg.get('pretrain_phase_weight', 1.0e-2))
        mkan_input_dim = mkan_cfg.get('input_dim', None)
        mkan_output_dim = mkan_cfg.get('output_dim', None)
        self.mkan_input_dim = int(self.nfeatures if mkan_input_dim is None else mkan_input_dim)
        self.mkan_output_dim = int(
            ((2 * self.nelectrons) if self.complex_output else self.nelectrons)
            if mkan_output_dim is None else mkan_output_dim
        )
        min_output_dim = (2 * self.nelectrons) if self.complex_output else self.nelectrons
        if self.mkan_output_dim < min_output_dim:
            raise ValueError(
                f'mkan.output_dim must be at least {min_output_dim} for '
                'orbital MKAN wavefunctions.'
            )
        self.adapt_frequency = int(cfg.get('mcmc_adapt_frequency', 20))
        self.pmove_min = float(cfg.get('mcmc_pmove_min', 0.50))
        self.pmove_max = float(cfg.get('mcmc_pmove_max', 0.60))
        self.width_scale = float(cfg.get('mcmc_width_scale', 1.05))

    def _build_checkpoint_state(
        self,
        *,
        stage: str,
        step: int,
        params,
        data: KANetsData,
        key,
        pretrain_opt_state=None,
        train_opt_state=None,
    ):
        return {
            'stage': stage,
            'step': int(step),
            'params': params,
            'data': data,
            'key': key,
            'pretrain_opt_state': pretrain_opt_state,
            'train_opt_state': train_opt_state,
        }

    def _build_networks(self):
        model_template = self._make_mkan_template()
        graphdef, initial_params, static_state = nnx.split(model_template, nnx.Param, ...)

        def kan_init(key):
            del key
            return initial_params

        def apply_mkan(params, features):
            model = nnx.merge(graphdef, params, static_state)
            if features.shape[-1] != self.mkan_input_dim:
                raise ValueError(
                    f'MKAN input dimension mismatch: got {features.shape[-1]}, '
                    f'expected {self.mkan_input_dim}. Set cfg.mkan.input_dim '
                    'or cfg.mkan.width if you want a different feature size.'
                )
            return model(features)

        def orbitals_apply(params, pos, spins, atoms, charges):
            del spins, charges
            ae, _, r_ae, _ = _construct_input_features(pos, atoms, ndim=3)
            h_one = jnp.concatenate((r_ae, ae), axis=2).reshape(self.nelectrons, -1)
            orbital_values = apply_mkan(params, h_one)
            if self.complex_output:
                orbital_values = (
                    orbital_values[..., 0:2 * self.nelectrons:2]
                    + 1.0j * orbital_values[..., 1:2 * self.nelectrons:2]
                )
            else:
                orbital_values = orbital_values[..., :self.nelectrons]

            spin_partitions = _array_partitions(self.electrons)
            orbital_channels = jnp.split(orbital_values, spin_partitions, axis=0)
            active_spin_channels = [spin for spin in self.electrons if spin > 0]
            orbital_channels = [
                channel for channel, spin in zip(orbital_channels, self.electrons) if spin > 0
            ]
            shapes = [(spin, -1, self.nelectrons) for spin in active_spin_channels]
            orbital_channels = [
                jnp.reshape(channel, shape)
                for channel, shape in zip(orbital_channels, shapes)
            ]
            orbital_channels = [
                jnp.transpose(channel, (1, 0, 2))
                for channel in orbital_channels
            ]
            return [jnp.concatenate(orbital_channels, axis=1)]

        def signed_network(params, pos, spins, atoms, charges):
            determinant = orbitals_apply(params, pos, spins, atoms, charges)
            return normal_network_blocks.logdet_matmul(determinant)

        def logabs_network(params, pos, spins, atoms, charges):
            return signed_network(params, pos, spins, atoms, charges)[1]

        def log_network(params, pos, spins, atoms, charges):
            phase, mag = signed_network(params, pos, spins, atoms, charges)
            if self.complex_output:
                return mag + 1.j * phase
            return mag

        batch_network = jax.vmap(logabs_network, in_axes=(None, 0, None, None, None), out_axes=0)
        batch_log_network = jax.vmap(log_network, in_axes=(None, 0, None, None, None), out_axes=0)
        orbitals_vmap = jax.vmap(orbitals_apply, in_axes=(None, 0, None, None, None), out_axes=0)

        return kan_init, signed_network, logabs_network, log_network, batch_network, batch_log_network, orbitals_vmap

    def _make_mkan_template(self):
        if self.mkan_width is None:
            hidden_dims = [int(v) for v in np.asarray(self.layer_dims).reshape(-1)[1:-1]]
            width = [self.mkan_input_dim, *hidden_dims, self.mkan_output_dim]
        else:
            width = list(self.mkan_width)
            width[0] = self.mkan_input_dim
            width[-1] = self.mkan_output_dim

        required_parameters = self._mkan_required_parameters()
        return MultKAN(
            width=width,
            layer_type=self.mkan_layer_type,
            required_parameters=required_parameters,
            mult_arity=self.mkan_mult_arity,
            seed=self.seed,
        )

    def _mkan_required_parameters(self):
        if self.mkan_required_parameters is not None:
            return dict(self.mkan_required_parameters)

        if self.mkan_layer_type in ('chebyshev', 'legendre'):
            return {
                'D': _first_int(self.k, 3),
                'flavor': 'exact' if self.mkan_layer_type == 'chebyshev' else None,
                'external_weights': self.external_weights,
                'add_bias': self.add_bias,
            }
        if self.mkan_layer_type in ('base', 'spline'):
            return {
                'k': _first_int(self.k, 3),
                'G': _first_int(self.g, 5),
                'grid_range': _first_grid_range(self.grid_range),
                'external_weights': self.external_weights,
                'add_bias': self.add_bias,
            }
        if self.mkan_layer_type == 'rbf':
            return {
                'D': _first_int(self.k, 5),
                'grid_range': _first_grid_range(self.grid_range, default=(-2.0, 2.0)),
                'external_weights': self.external_weights,
                'add_bias': self.add_bias,
            }
        if self.mkan_layer_type == 'sine':
            return {
                'D': _first_int(self.k, 5),
                'external_weights': self.external_weights,
                'add_bias': self.add_bias,
            }
        if self.mkan_layer_type == 'fourier':
            return {
                'D': _first_int(self.k, 5),
                'add_bias': self.add_bias,
            }
        raise ValueError(f'Unsupported MKAN layer_type: {self.mkan_layer_type}')

    def _initialize_params_and_data(self, kan_init):
        resume_state = self.run_manager.load_last_checkpoint()

        key = jax.random.PRNGKey(self.seed)
        key, subkey = jax.random.split(key)
        params = kan_init(subkey)
        sharded_key = key

        pretrain_start_step = 0
        train_start_step = self.t_init
        pretrain_opt_state = None
        train_opt_state = None
        data = None

        if resume_state is not None:
            params = resume_state['params']
            data = resume_state['data']
            sharded_key = resume_state['key']
            stage = resume_state.get('stage')
            if stage == 'pretrain':
                pretrain_start_step = int(resume_state.get('step', 0))
                pretrain_opt_state = resume_state.get('pretrain_opt_state')
            elif stage == 'train':
                train_start_step = int(resume_state.get('step', self.t_init))
                train_opt_state = resume_state.get('train_opt_state')

        if data is None:
            key_electrons_coords = jax.random.PRNGKey(self.seed_electrons_coords)
            key_electrons_coords, subkey_electrons_coords = jax.random.split(key_electrons_coords)
            pos, _ = electrons_initialization.init_electrons(
                subkey_electrons_coords,
                self.molecule,
                self.electrons,
                batch_size=self.batch_size,
                init_width=self.init_width,
                core_electrons=self.core_electrons,
            )
            data = KANetsData(positions=pos, spins=self.spins, atoms=self.atoms, charges=self.charges)

        return params, data, sharded_key, pretrain_start_step, train_start_step, pretrain_opt_state, train_opt_state

    def _build_optimizer(self):
        def learning_rate_schedule(t_: jnp.ndarray) -> jnp.ndarray:
            return self.learning_rate * jnp.power((1.0 / (1.0 + (t_ / 1.0))), self.learning_rate_decay)

        return optax.chain(
            optax.scale_by_adam(b1=0.9, b2=0.999, eps=1e-6),
            optax.scale_by_schedule(learning_rate_schedule),
            optax.scale(-1.0),
        )

    def _build_train_step(self, signed_network: Callable, logabs_network: Callable, log_network: Callable):
        loss_network = log_network if self.complex_output else logabs_network
        local_energy = hamiltonian.local_energy(
            f=signed_network,
            nspins=self.electrons,
            charges=self.charges,
            use_scan=self.use_scan,
            complex_output=self.complex_output,
            laplacian_method=self.laplacian_method,
        )
        evaluate_loss = qmc_loss_functions.make_loss(
            loss_network,
            local_energy,
            clip_local_energy=self.clip_local_energy,
            clip_from_median=True,
            center_at_clipped_energy=True,
            complex_output=self.complex_output,
        )

        optimizer = self._build_optimizer()
        batch_signed_network = jax.vmap(
            signed_network, in_axes=(None, 0, None, None, None), out_axes=(0, 0)
        )
        monte_carlo = VMCmcstep.make_mcmc_step(
            f=batch_signed_network,
            ndim=3,
            nelectrons=self.nelectrons,
            steps=self.mcmc_steps,
        )
        step_fn = make_training_step(
            mcmc_step=monte_carlo,
            optimizer_step=make_opt_update_step(evaluate_loss, optimizer),
            reset_if_nan=True,
        )
        return optimizer, step_fn

    def _build_train_state(self, params, optimizer, train_opt_state):
        state = train_state.TrainState.create(
            apply_fn=lambda *_args, **_kwargs: None,
            params=params,
            tx=optimizer,
        )
        if train_opt_state is not None:
            state = state.replace(opt_state=train_opt_state)
        return state

    def _run_train_loop(self, *, train_start_step: int, runtime: RuntimeState, state: train_state.TrainState, step_fn):
        initial_state = self._build_checkpoint_state(
            stage='train',
            step=train_start_step,
            params=state.params,
            data=runtime.data,
            key=runtime.key,
            train_opt_state=state.opt_state,
        )
        self.run_manager.checkpoints.save_last(initial_state)

        if self.debug:
            jax.debug.print('sharded_key:{}', runtime.key)

        iterator: Any = trange(train_start_step, self.iterations, desc='Training', dynamic_ncols=True)
        for t in iterator:
            key, subkeys = jax.random.split(runtime.key, 2)
            data, params, opt_state, loss, aux_data, pmove = step_fn(
                runtime.data,
                state.params,
                state.opt_state,
                subkeys,
                runtime.mcmc_width,
            )
            state = state.replace(step=state.step + 1, params=params, opt_state=opt_state)

            pmove_mean = jnp.mean(pmove)
            t_since_update = t % self.adapt_frequency
            runtime.pmoves[t_since_update] = _to_scalar(pmove_mean)
            if t > 0 and t_since_update == 0:
                mean_pmove = float(np.mean(runtime.pmoves))
                if mean_pmove > self.pmove_max:
                    runtime = runtime.replace(mcmc_width=runtime.mcmc_width * self.width_scale)
                elif mean_pmove < self.pmove_min:
                    runtime = runtime.replace(mcmc_width=runtime.mcmc_width / self.width_scale)

            window_size = min(t + 1, self.adapt_frequency)
            pmove_window_mean = float(np.mean(runtime.pmoves[:window_size]))
            step_id = t + 1
            loss_value = float(jnp.real(loss))
            variance_value = float(aux_data.variance)
            iterator.set_postfix(iter=step_id, loss=f'{loss_value:.6f}')

            if self.run_manager.should_log(step_id, self.iterations):
                self.run_manager.log_scalars(
                    'train',
                    step_id,
                    {
                        'loss': loss_value,
                        'variance': variance_value,
                        'pmove': _to_scalar(pmove_mean),
                        'pmove_window': pmove_window_mean,
                        'mcmc_width': _to_scalar(runtime.mcmc_width),
                    },
                )

            checkpoint_state = self._build_checkpoint_state(
                stage='train',
                step=step_id,
                params=state.params,
                data=data,
                key=key,
                train_opt_state=state.opt_state,
            )
            if self.run_manager.should_checkpoint(step_id, self.iterations):
                self.run_manager.checkpoints.save_step('train', step_id, checkpoint_state)

            runtime = runtime.replace(data=data, key=key)

    def run(self) -> None:
        try:
            (
                kan_init,
                signed_network,
                logabs_network,
                log_network,
                batch_network,
                batch_log_network,
                orbitals_vmap,
            ) = self._build_networks()
            params, data, sharded_key, pretrain_start_step, train_start_step, pretrain_opt_state, train_opt_state = (
                self._initialize_params_and_data(kan_init)
            )

            params, data, sharded_key, pretrain_opt_state, train_start_step = self.pretrain_runner.run(
                params=params,
                data=data,
                sharded_key=sharded_key,
                pretrain_start_step=pretrain_start_step,
                train_opt_state=train_opt_state,
                pretrain_opt_state=pretrain_opt_state,
                batch_network=batch_network,
                batch_log_network=batch_log_network,
                orbitals_vmap=orbitals_vmap,
                t_init=self.t_init,
            )
            del pretrain_opt_state

            optimizer, step_fn = self._build_train_step(signed_network, logabs_network, log_network)
            state = self._build_train_state(params, optimizer, train_opt_state)

            runtime = RuntimeState(
                data=data,
                key=sharded_key,
                mcmc_width=jnp.asarray(self.mcmc_width),
                pmoves=np.zeros((self.adapt_frequency,), dtype=np.float32),
            )
            self._run_train_loop(train_start_step=train_start_step, runtime=runtime, state=state, step_fn=step_fn)
        finally:
            self.run_manager.close()


def train(cfg: ml_collections.ConfigDict):
    """Main training loop entry."""
    trainer = VMCTrainer(cfg)
    trainer.run()
