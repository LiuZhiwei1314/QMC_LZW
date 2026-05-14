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

import hamiltonian
import electrons_initialization
import envelope
import jastrow
import networks
from jkan.models import MultKAN
import loss as qmc_loss_functions
import vmcmc
from opt import make_opt_update_step, make_training_step
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
    return networks.construct_input_features(pos, atoms, ndim=ndim)


@flax.struct.dataclass
class RuntimeState:
    data: networks.KANetsData
    key: Any
    mcmc_width: jnp.ndarray
    pmoves: np.ndarray


class VMCTrainer:
    """Flax-style trainer that keeps setup and loop logic modular."""

    def __init__(self, cfg: ml_collections.ConfigDict):
        self.cfg = cfg
        self._mkan_graphdef = None
        self._mkan_static_state = None
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
        nfeatures = int(cfg.nfeatures)
        self.atoms = jnp.array([atom.coords for atom in self.molecule])
        self.charges = jnp.array([atom.charge for atom in self.molecule])

        spins = [1] * self.electrons[0] + [-1] * self.electrons[1]
        self.spins = jnp.array([spins])

        self.g = jnp.array(cfg.g)
        self.k = jnp.array(cfg.k)
        self.layer_dims = jnp.array(cfg.layer_dims)
        self.grid_range = cfg.grid_range

        self.seed = int(cfg.seed)
        self.seed_electrons_coords = int(cfg.seed_electrons_coords)
        self.init_width = float(cfg.init_width)
        self.core_electrons = cfg.core_electrons

        self.pretrain_method = str(cfg.get('pretrain_method', 'hf')).lower()
        self.pretrain_basis = cfg.get('pretrain_basis', 'ccpvdz')
        self.pretrain_restricted = bool(cfg.get('pretrain_restricted', False))
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

        self.add_bias = bool(cfg.add_bias)
        self.external_weights = bool(cfg.external_weights)
        self.envelope_simple = bool(cfg.envelope_simple)
        self.envelope_type = str(cfg.get('envelope_type', 'isotropic')).lower()
        self.envelope_degree = int(cfg.get('envelope_degree', 5))
        jastrow_cfg = cfg.get('jastrow', {})
        self.jastrow_ee = bool(jastrow_cfg.get('ee', True))
        self.jastrow_type = str(jastrow_cfg.get('type', 'pade')).lower()

        mkan_cfg = cfg.get('mkan', {})
        self.mkan_layer_type = str(mkan_cfg.get('layer_type', 'spline')).lower()
        self.mkan_mult_arity = mkan_cfg.get('mult_arity', 2)
        self.mkan_width = mkan_cfg.get('width', None)
        self.mkan_required_parameters = mkan_cfg.get('required_parameters', None)
        self.mkan_pretrain_phase_weight = float(mkan_cfg.get('pretrain_phase_weight', 1.0e-2))
        mkan_input_dim = mkan_cfg.get('input_dim', None)
        mkan_output_dim = mkan_cfg.get('output_dim', None)
        self.mkan_input_dim = int(nfeatures if mkan_input_dim is None else mkan_input_dim)
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

        grid_extension_cfg = cfg.get('grid_extension', {})
        self.grid_extension_enabled = bool(grid_extension_cfg.get('enabled', False))
        self.grid_extension_steps = tuple(int(v) for v in grid_extension_cfg.get('steps', ()))
        self.grid_extension_g_values = tuple(int(v) for v in grid_extension_cfg.get('g_values', ()))
        sample_size = grid_extension_cfg.get('sample_size', None)
        self.grid_extension_sample_size = None if sample_size is None else int(sample_size)
        if self.grid_extension_enabled:
            if self.mkan_layer_type not in ('base', 'spline'):
                raise ValueError(
                    'grid_extension currently supports only base/spline MKAN layers. '
                    f'Got layer_type={self.mkan_layer_type!r}.'
                )
            if len(self.grid_extension_steps) != len(self.grid_extension_g_values):
                raise ValueError('grid_extension.steps and grid_extension.g_values must have the same length.')
            if any(step <= 0 for step in self.grid_extension_steps):
                raise ValueError('grid_extension.steps must contain positive training step numbers.')
            if any(g <= 0 for g in self.grid_extension_g_values):
                raise ValueError('grid_extension.g_values must contain positive grid sizes.')

    def _build_checkpoint_state(
        self,
        *,
        stage: str,
        step: int,
        params,
        data: networks.KANetsData,
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
            'mkan_static_state': self._mkan_static_state,
        }

    def _build_networks(self):
        model_template = self._make_mkan_template()
        self._mkan_graphdef, initial_params, self._mkan_static_state = nnx.split(
            model_template, nnx.Param, ...
        )
        active_spin_channels = networks.active_spin_channels(self.electrons)
        envelope_output_dims = active_spin_channels
        envelope_params = None
        if self.envelope_simple:
            if self.envelope_type == 'isotropic':
                envelope_params = envelope.init_isotropic_envelope(
                    self.natoms, envelope_output_dims
                )
            elif self.envelope_type == 'chebyshev':
                envelope_params = envelope.init_chebyshev_envelope(
                    self.natoms,
                    envelope_output_dims,
                    degree=self.envelope_degree,
                )
            else:
                raise ValueError(f'Unsupported envelope_type={self.envelope_type!r}.')
        if self.jastrow_type == 'pade':
            same_spin_pairs, opposite_spin_pairs = jastrow.spin_pair_indices(self.electrons)
            init_jastrow = jastrow.init_pade_ee_jastrow
            apply_jastrow = jastrow.apply_pade_ee_jastrow
        elif self.jastrow_type == 'ferminet':
            same_spin_pairs, opposite_spin_pairs = jastrow.spin_pair_indices_or_empty(self.electrons)
            init_jastrow = jastrow.init_ferminet_ee_jastrow
            apply_jastrow = jastrow.apply_ferminet_ee_jastrow
        else:
            raise ValueError(f'Unsupported jastrow.type={self.jastrow_type!r}.')
        jastrow_params = init_jastrow() if self.jastrow_ee else None

        def kan_init(key):
            del key
            params = {'mkan': initial_params}
            if envelope_params is not None:
                params['envelope'] = envelope_params
            if jastrow_params is not None:
                params['jastrow_ee'] = jastrow_params
            return params

        def apply_mkan(params, features):
            model_params = params['mkan'] if isinstance(params, dict) and 'mkan' in params else params
            model = nnx.merge(self._mkan_graphdef, model_params, self._mkan_static_state)
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
            orbital_row_channels = jnp.split(orbital_values, spin_partitions, axis=0)
            active_spin_channels = [spin for spin in self.electrons if spin > 0]
            orbital_channels = [
                channel[:, start : start + spin]
                for channel, spin, start in zip(
                    orbital_row_channels,
                    self.electrons,
                    (0, int(self.electrons[0])),
                )
                if spin > 0
            ]
            if self.envelope_simple:
                if not (isinstance(params, dict) and 'envelope' in params):
                    raise ValueError('Missing envelope parameters for simple envelope.')
                r_ae_channels = jnp.split(r_ae, spin_partitions, axis=0)
                r_ae_channels = [
                    channel for channel, spin in zip(r_ae_channels, self.electrons) if spin > 0
                ]
                apply_envelope = (
                    envelope.apply_chebyshev_envelope
                    if self.envelope_type == 'chebyshev'
                    else envelope.apply_isotropic_envelope
                )
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
            orbital_channels = [
                jnp.transpose(channel, (1, 0, 2))
                for channel in orbital_channels
            ]
            return orbital_channels

        def signed_network(params, pos, spins, atoms, charges):
            determinant = orbitals_apply(params, pos, spins, atoms, charges)
            phase, logmag = networks.logdet_matmul(determinant)
            if self.jastrow_ee:
                if not (isinstance(params, dict) and 'jastrow_ee' in params):
                    raise ValueError('Missing Jastrow parameters for electron-electron Jastrow.')
                _, _, _, r_ee = _construct_input_features(pos, atoms, ndim=3)
                logmag = logmag + apply_jastrow(
                    r_ee,
                    params['jastrow_ee'],
                    same_spin_pairs,
                    opposite_spin_pairs,
                )
            return phase, logmag

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

        def extend_mkan_grid(params, data, g_new: int):
            if not (isinstance(params, dict) and 'mkan' in params):
                raise ValueError('Grid extension requires trainer params with an mkan entry.')
            samples = self._grid_extension_samples(data)
            model = nnx.merge(self._mkan_graphdef, params['mkan'], self._mkan_static_state)
            model.extend_grids(samples, int(g_new))
            self._mkan_graphdef, mkan_params, self._mkan_static_state = nnx.split(
                model, nnx.Param, ...
            )
            new_params = dict(params)
            new_params['mkan'] = mkan_params
            return new_params

        return (
            kan_init,
            signed_network,
            logabs_network,
            log_network,
            batch_network,
            batch_log_network,
            orbitals_vmap,
            extend_mkan_grid,
        )

    def _grid_extension_samples(self, data: networks.KANetsData) -> jnp.ndarray:
        positions = jnp.reshape(data.positions, (-1, self.nelectrons * 3))

        def single_position_features(pos):
            ae, _, r_ae, _ = _construct_input_features(pos, data.atoms, ndim=3)
            return jnp.concatenate((r_ae, ae), axis=2).reshape(self.nelectrons, -1)

        samples = jax.vmap(single_position_features)(positions)
        samples = jnp.reshape(samples, (-1, self.mkan_input_dim))
        if self.grid_extension_sample_size is not None:
            samples = samples[:self.grid_extension_sample_size]
        return samples

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
            if resume_state.get('mkan_static_state') is not None:
                self._mkan_static_state = resume_state['mkan_static_state']
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
            data = networks.KANetsData(positions=pos, spins=self.spins, atoms=self.atoms, charges=self.charges)

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
        monte_carlo = vmcmc.make_vmcmc_step(
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

    def _run_train_loop(
        self,
        *,
        train_start_step: int,
        runtime: RuntimeState,
        state: train_state.TrainState,
        step_fn,
        signed_network: Callable,
        logabs_network: Callable,
        log_network: Callable,
        extend_mkan_grid: Callable,
    ):
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
        grid_extension_targets = (
            dict(zip(self.grid_extension_steps, self.grid_extension_g_values))
            if self.grid_extension_enabled
            else {}
        )
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

            if step_id in grid_extension_targets:
                g_new = grid_extension_targets[step_id]
                params = extend_mkan_grid(state.params, data, g_new)
                optimizer, step_fn = self._build_train_step(signed_network, logabs_network, log_network)
                step = state.step
                state = self._build_train_state(params, optimizer, train_opt_state=None)
                state = state.replace(step=step)
                iterator.write(f'Extended MKAN grid to G={g_new} at train step {step_id}.')
                self.run_manager.log_scalars('train', step_id, {'grid_G': float(g_new)})

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
                extend_mkan_grid,
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
            self._run_train_loop(
                train_start_step=train_start_step,
                runtime=runtime,
                state=state,
                step_fn=step_fn,
                signed_network=signed_network,
                logabs_network=logabs_network,
                log_network=log_network,
                extend_mkan_grid=extend_mkan_grid,
            )
        finally:
            self.run_manager.close()


def train(cfg: ml_collections.ConfigDict):
    """Main training loop entry."""
    trainer = VMCTrainer(cfg)
    trainer.run()
