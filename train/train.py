import ml_collections
import jax.numpy as jnp
import jax
import optax
from typing import Any, Optional
from optimizer.opt import make_training_step, make_opt_update_step
from kan_wavefunction_case_one.kan_networks_case_one import make_kan_net, KANetsData
import os
import numpy as np
from kan_wavefunction_case_one.spin_indices import jastrow_indices_ee
from monte_carlo_step import VMCmcstep
from hamiltonian import hamiltonian
from loss_function import loss as qmc_loss_functions
from initialization import electrons_initialization
from pretrain import pretrain_DFT, pretrain_HF
from train.training_io import RunManager
from tqdm.auto import trange


def _to_scalar(x):
    return float(jnp.asarray(x).reshape(-1)[0])


def train(cfg: ml_collections.ConfigDict):
    """Main training loop."""
    molecule = cfg.system.molecule
    electrons = tuple(cfg.system.electrons)
    nelectrons = sum(electrons)
    natoms = len(molecule)
    batch_size = int(cfg.batch_size)
    nfeatures = int(cfg.nfeatures)

    atoms = jnp.array([atom.coords for atom in molecule])
    charges = jnp.array([atom.charge for atom in molecule])
    spins_list = [1] * electrons[0] + [-1] * electrons[1]
    spins_jastrow = jnp.array(spins_list)
    spins = jnp.array([spins_list])
    g = jnp.array(cfg.g)
    k = jnp.array(cfg.k)
    layer_dims = jnp.array(cfg.layer_dims)
    grid_range = cfg.grid_range
    grid_range_envelope = jnp.array(cfg.envelope.grid_range_envelope)

    seed_electrons_coords = int(cfg.seed_electrons_coords)
    seed = int(cfg.seed)
    init_width = float(cfg.init_width)
    core_electrons = cfg.core_electrons

    pretrain_method = str(cfg.get('pretrain_method', 'hf')).lower()
    pretrain_basis = cfg.get('pretrain_basis', cfg.get('hf_basis', 'ccpvdz'))
    pretrain_restricted = bool(
        cfg.get('pretrain_restricted', cfg.get('hf_restricted', False))
    )
    hf_basis = cfg.get('hf_basis', pretrain_basis)
    hf_restricted = bool(cfg.get('hf_restricted', pretrain_restricted))
    hf_states = int(cfg.get('hf_states', 0))
    hf_excitation_type = cfg.get('hf_excitation_type', 'ordered')
    dft_xc = cfg.get('dft_xc', 'pbe,pbe')
    dft_grid_level = cfg.get('dft_grid_level', 3)
    pyscf_mol = cfg.system.get('pyscf_mol')

    mcmc_batch_per_device = int(cfg.mcmc_batch_per_device)
    mcmc_steps = int(cfg.mcmc_steps)
    mcmc_blocks = int(cfg.mcmc_blocks)
    mcmc_width = float(cfg.mcmc_width)

    clip_local_energy = float(cfg.clip_local_energy)
    use_scan = bool(cfg.use_scan)
    complex_output = bool(cfg.complex_output)
    laplacian_method = cfg.laplacian_method
    scf_fraction = float(cfg.scf_fraction)
    t_init = int(cfg.t_init)
    debug = bool(cfg.debug)

    learning_rate = float(cfg.learning_rate)
    learning_rate_decay = float(cfg.learning_rate_decay)
    preiterations = int(cfg.preiterations)
    run_pretrain = bool(cfg.run_pretrain)
    iterations = int(cfg.iterations)

    chebyshev = bool(cfg.chebyshev)
    spline = bool(cfg.spline)
    add_residual = bool(cfg.add_residual)
    add_bias = bool(cfg.add_bias)
    external_weights = bool(cfg.external_weights)
    envelope_chebyshev = bool(cfg.envelope_chebyshev)
    envelope_spline = bool(cfg.envelope_spline)
    envelope_simple = bool(cfg.envelope_simple)
    g_envelope = int(cfg.envelope.g_envelope)
    k_envelope = int(cfg.envelope.k_envelope)

    run_manager = RunManager(cfg.output)
    run_manager.save_config(cfg)

    def build_checkpoint_state(
        *,
        stage: str,
        step: int,
        params,
        data: KANetsData,
        key,
        pretrain_opt_state=None,
        train_opt_state=None
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

    try:
        resume_state = run_manager.load_last_checkpoint()

        pretrain_start_step = 0
        train_start_step = t_init
        pretrain_opt_state = None
        train_opt_state = None
        data = None

        # Build electron-pair index sets used by the Jastrow terms.
        parallel_indices, antiparallel_indices, n_parallel, n_antiparallel = jastrow_indices_ee(
            spins=spins_jastrow,
            nelectrons=nelectrons,
        )

        # Construct the neural wavefunction and orbital heads.
        kan_init, kan_apply, orbitals_apply = make_kan_net(
            nspins=electrons,
            charges=charges,
            nelectrons=nelectrons,
            nfeatures=nfeatures,
            n_parallel=n_parallel,
            n_antiparallel=n_antiparallel,
            parallel_indices=parallel_indices,
            antiparallel_indices=antiparallel_indices,
            grid_range=grid_range,
            g=g,
            k=k,
            natoms=natoms,
            ndims=3,
            layer_dims=layer_dims,
            g_envelope=g_envelope,
            k_envelope=k_envelope,
            grid_range_envelope=grid_range_envelope,
            chebyshev=chebyshev,
            spline=spline,
            add_residual=add_residual,
            add_bias=add_bias,
            external_weights=external_weights,
            envelope_chebyshev=envelope_chebyshev,
            envelope_spline=envelope_spline,
            envelope_simple=envelope_simple,
        )
        signed_network = kan_apply
        logabs_network = lambda *args, **kwargs: signed_network(*args, **kwargs)[1]
        batch_network = jax.vmap(logabs_network, in_axes=(None, 0, None, None, None), out_axes=0)
        orbitals_vmap = jax.vmap(orbitals_apply, in_axes=(None, 0, None, None, None), out_axes=0)

        # Convert the signed network output to the complex log-amplitude used by the loss.
        def log_network(*args, **kwargs):
            phase, mag = signed_network(*args, **kwargs)
            return mag + 1.j * phase

        # Initialize network parameters.
        key = jax.random.PRNGKey(seed)
        key, subkey = jax.random.split(key)
        params = kan_init(subkey)
        sharded_key = key

        if resume_state is not None:
            params = resume_state['params']
            data = resume_state['data']
            sharded_key = resume_state['key']
            stage = resume_state.get('stage')
            if stage == 'pretrain':
                pretrain_start_step = int(resume_state.get('step', 0))
                pretrain_opt_state = resume_state.get('pretrain_opt_state')
            elif stage == 'train':
                train_start_step = int(resume_state.get('step', t_init))
                train_opt_state = resume_state.get('train_opt_state')

        if data is None:
            # Sample initial walker positions around the configured molecule.
            key_electrons_coords = jax.random.PRNGKey(seed_electrons_coords)
            key_electrons_coords, subkey_electrons_coords = jax.random.split(key_electrons_coords)
            pos, _ = electrons_initialization.init_electrons(
                subkey_electrons_coords,
                molecule,
                electrons,
                batch_size=batch_size,
                init_width=init_width,
                core_electrons=core_electrons,
            )
            data = KANetsData(positions=pos, spins=spins, atoms=atoms, charges=charges)

        needs_pretrain = run_pretrain and train_opt_state is None and pretrain_start_step < preiterations
        if needs_pretrain:
            def log_pretrain(step: int, loss_value: float) -> None:
                if run_manager.should_log(step, preiterations):
                    run_manager.log_scalars('pretrain', step, {'loss': loss_value})

            def checkpoint_pretrain(step, loss_value, params_, opt_state_, data_, key_):
                checkpoint_state = build_checkpoint_state(
                    stage='pretrain',
                    step=step,
                    params=params_,
                    data=data_,
                    key=key_,
                    pretrain_opt_state=opt_state_,
                    train_opt_state=None,
                )
                if run_manager.should_checkpoint(step, preiterations):
                    run_manager.checkpoints.save_step('pretrain', step, checkpoint_state)

            if pretrain_method == 'hf':
                # Prepare the Hartree-Fock reference used for orbital pretraining.
                hartree_fock = pretrain_HF.get_hf(
                    pyscf_mol=pyscf_mol,
                    molecule=molecule,
                    nspins=electrons,
                    restricted=pretrain_restricted,
                    basis=pretrain_basis,
                    ecp={},
                    core_electrons=core_electrons,
                    states=hf_states,
                    excitation_type=hf_excitation_type,
                )
                if debug:
                    jax.debug.print("hartree_fock:{}", hartree_fock)

                params, data, pretrain_opt_state, sharded_key = pretrain_HF.pretrain_hartree_fock(
                    params=params,
                    positions=data.positions,
                    spins=data.spins,
                    charges=data.charges,
                    atoms=data.atoms,
                    batch_network=batch_network,
                    batch_orbitals=orbitals_vmap,
                    sharded_key=sharded_key,
                    electrons=electrons,
                    scf_approx=hartree_fock,
                    iterations=preiterations,
                    batch_size=batch_size,
                    logger=log_pretrain,
                    checkpoint_callback=checkpoint_pretrain,
                    scf_fraction=scf_fraction,
                    states=hf_states,
                    start_iteration=pretrain_start_step,
                    opt_state=pretrain_opt_state,
                    data=data,
                )
            elif pretrain_method == 'dft':
                if hf_states != 0:
                    raise ValueError(
                        'DFT pretraining currently supports only ground states; set hf_states=0.'
                    )
                dft_reference = pretrain_DFT.get_dft(
                    pyscf_mol=pyscf_mol,
                    molecule=molecule,
                    nspins=electrons,
                    restricted=pretrain_restricted,
                    basis=pretrain_basis,
                    ecp={},
                    core_electrons=core_electrons,
                    xc=dft_xc,
                    grid_level=dft_grid_level,
                    states=0,
                )
                if debug:
                    jax.debug.print("dft_reference:{}", dft_reference)

                params, data, pretrain_opt_state, sharded_key = pretrain_DFT.pretrain_ks_dft(
                    params=params,
                    positions=data.positions,
                    spins=data.spins,
                    charges=data.charges,
                    atoms=data.atoms,
                    batch_network=batch_network,
                    batch_orbitals=orbitals_vmap,
                    sharded_key=sharded_key,
                    electrons=electrons,
                    dft_approx=dft_reference,
                    iterations=preiterations,
                    batch_size=batch_size,
                    logger=log_pretrain,
                    checkpoint_callback=checkpoint_pretrain,
                    scf_fraction=scf_fraction,
                    start_iteration=pretrain_start_step,
                    opt_state=pretrain_opt_state,
                    data=data,
                )
            else:
                raise ValueError(
                    f"Unsupported pretrain_method: {pretrain_method}. Expected 'hf' or 'dft'."
                )

            train_start_step = t_init

        # Build the local-energy estimator and the variational loss.
        loss_network = log_network if complex_output else logabs_network
        local_energy = hamiltonian.local_energy(
            f=signed_network,
            nspins=electrons,
            charges=charges,
            use_scan=use_scan,
            complex_output=complex_output,
            laplacian_method=laplacian_method,
        )
        evaluate_loss = qmc_loss_functions.make_loss(
            loss_network,
            local_energy,
            clip_local_energy=clip_local_energy,
            clip_from_median=True,
            center_at_clipped_energy=True,
            complex_output=complex_output,
        )

        def learning_rate_schedule(t_: jnp.ndarray) -> jnp.ndarray:
            # inverse_time
            # return learning_rate / (1.0 + t_ / learning_rate_decay)
            # sqrt
            # return learning_rate / jnp.sqrt(1.0 + t_ / learning_rate_decay)
            # polynomial Decay
            return learning_rate * jnp.power((1.0 / (1.0 + (t_ / 1.0))), learning_rate_decay)


        # Adam with a scalar learning-rate schedule for VMC optimization.
        optimizer = optax.chain(
            optax.scale_by_adam(b1=0.9, b2=0.999, eps=1e-6),
            optax.scale_by_schedule(learning_rate_schedule),
            optax.scale(-1.),
        )
        if train_opt_state is None:
            train_opt_state = optimizer.init(params)

        batch_signed_network = jax.vmap(
            signed_network, in_axes=(None, 0, None, None, None), out_axes=(0, 0)
        )
        monte_carlo = VMCmcstep.make_mcmc_step(
            f=batch_signed_network,
            ndim=3,
            nelectrons=nelectrons,
            steps=mcmc_steps,
        )

        step = make_training_step(
            mcmc_step=monte_carlo,
            optimizer_step=make_opt_update_step(evaluate_loss, optimizer),
            reset_if_nan=True,
        )

        initial_state = build_checkpoint_state(
            stage='train',
            step=train_start_step,
            params=params,
            data=data,
            key=sharded_key,
            train_opt_state=train_opt_state,
        )
        run_manager.checkpoints.save_last(initial_state)

        # Main VMC training loop.
        if debug:
            jax.debug.print("sharded_key:{}", sharded_key)

        mcmc_width_val = jnp.asarray(mcmc_width)
        adapt_frequency = int(cfg.get('mcmc_adapt_frequency', 20))
        pmove_min = float(cfg.get('mcmc_pmove_min', 0.50))
        pmove_max = float(cfg.get('mcmc_pmove_max', 0.60))
        width_scale = float(cfg.get('mcmc_width_scale', 1.05))
        pmoves = np.zeros((adapt_frequency,), dtype=np.float32)

        iterator: Any
        iterator = trange(train_start_step, iterations, desc='Training', dynamic_ncols=True)

        for t in iterator:
            sharded_key, subkeys = jax.random.split(sharded_key, 2)
            data, params, train_opt_state, loss, aux_data, pmove = step(
                data,
                params,
                train_opt_state,
                subkeys,
                mcmc_width_val,
            )

            pmove_mean = jnp.mean(pmove)
            t_since_update = t % adapt_frequency
            pmoves[t_since_update] = _to_scalar(pmove_mean)
            if t > 0 and t_since_update == 0:
                mean_pmove = float(np.mean(pmoves))
                if mean_pmove > pmove_max:
                    mcmc_width_val = mcmc_width_val * width_scale
                elif mean_pmove < pmove_min:
                    mcmc_width_val = mcmc_width_val / width_scale
            window_size = min(t + 1, adapt_frequency)
            pmove_window_mean = float(np.mean(pmoves[:window_size]))

            step_id = t + 1
            loss_value = float(jnp.real(loss))
            variance_value = float(aux_data.variance)
            iterator.set_postfix(iter=step_id, loss=f'{loss_value:.6f}')

            if run_manager.should_log(step_id, iterations):
                run_manager.log_scalars(
                    'train',
                    step_id,
                    {
                        'loss': loss_value,
                        'variance': variance_value,
                        'pmove': _to_scalar(pmove_mean),
                        'pmove_window': pmove_window_mean,
                        'mcmc_width': _to_scalar(mcmc_width_val),
                    },
                )

            checkpoint_state = build_checkpoint_state(
                stage='train',
                step=step_id,
                params=params,
                data=data,
                key=sharded_key,
                train_opt_state=train_opt_state,
            )
            if run_manager.should_checkpoint(step_id, iterations):
                run_manager.checkpoints.save_step('train', step_id, checkpoint_state)
    finally:
        run_manager.close()
