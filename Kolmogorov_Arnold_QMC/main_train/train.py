"""Currently, we only develop the single stream version.
Because the parallel strategy of JAX changed a lot since last year. We need spend a long time to reconstruct it.
one more thing is that we do not pretrain it currently.

24.10.2025."""

import ml_collections
import jax.numpy as jnp
import jax
import time
import os
import numpy as np
import optax
import kfac_jax
from Kolmogorov_Arnold_QMC.optimizer.opt import make_training_step, make_opt_update_step
from Kolmogorov_Arnold_QMC.kan_wavefunction_case_one.kan_networks_case_one import make_kan_net, KANetsData
from Kolmogorov_Arnold_QMC.kan_wavefunction_case_one.spin_indices import jastrow_indices_ee, jastrow_indices_ae
from Kolmogorov_Arnold_QMC.monte_carlo_step import VMCmcstep
from Kolmogorov_Arnold_QMC.hamiltonian import hamiltonian
from Kolmogorov_Arnold_QMC.loss_function import loss as qmc_loss_functions
from Kolmogorov_Arnold_QMC.initialization import electrons_initialization
from Kolmogorov_Arnold_QMC.pretrain import pretrain_HF


def _to_scalar(x):
    return float(jnp.asarray(x).reshape(-1)[0])


def _init_swanlab(cfg: ml_collections.ConfigDict):
    swan_cfg = cfg.get('swanlab', ml_collections.ConfigDict())
    if not bool(swan_cfg.get('enabled', False)):
        return None

    os.environ.setdefault('SWANLAB_SAVE_DIR', os.path.join(os.getcwd(), '.swanlab'))
    os.environ.setdefault('SWANLAB_LOG_DIR', os.path.join(os.getcwd(), 'swanlog'))

    try:
        import swanlab
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            'SwanLab is enabled but not installed. Please run: '
            '`conda run -n jax pip install swanlab`.'
        ) from exc

    run_config = {
        'batch_size': int(cfg.batch_size),
        'iterations': int(cfg.iterations),
        'preiterations': int(cfg.preiterations),
        'nelectrons': int(cfg.system.nelectrons),
        'electrons': tuple(cfg.system.electrons),
        'layer_dims': tuple(cfg.layer_dims),
        'g': tuple(cfg.g),
        'k': tuple(cfg.k),
        'chebyshev': bool(cfg.chebyshev),
        'spline': bool(cfg.spline),
    }

    try:
        swanlab.init(
            project=swan_cfg.get('project', 'Kolmogorov_Arnold_QMC'),
            workspace=swan_cfg.get('workspace', None),
            experiment_name=swan_cfg.get('experiment_name', None),
            description=swan_cfg.get('description', None),
            mode=swan_cfg.get('mode', 'cloud'),
            logdir=os.environ['SWANLAB_LOG_DIR'],
            config=run_config,
        )
    except Exception as exc:  # pylint: disable=broad-except
        raise RuntimeError(
            'Failed to initialize SwanLab. For cloud mode, run '
            '`conda run -n jax swanlab login` first. '
            "Or set `cfg.swanlab.mode = 'offline'`."
        ) from exc
    return swanlab


def train(cfg: ml_collections.ConfigDict,):
    swanlab = _init_swanlab(cfg)
    swan_state = {'module': swanlab}

    def swan_log(metrics, step):
        if swan_state['module'] is None:
            return
        try:
            swan_state['module'].log(metrics, step=step)
        except Exception as exc:  # pylint: disable=broad-except
            print(f'[SwanLab] log failed and will be disabled: {exc}')
            swan_state['module'] = None

    def pretrain_logger(step, value):
        swan_log({'pretrain/loss': _to_scalar(value)}, int(step))

    spins_jastrow = jnp.array(cfg.spins)
    #jax.debug.print("spins:{}", spins_jastrow)
    parallel_indices, antiparallel_indices, n_parallel, n_antiparallel = jastrow_indices_ee(spins=spins_jastrow,
                                                                                            nelectrons=6)
    #jax.debug.print("parallel_indices:{}", parallel_indices)
    g = jnp.array(cfg.g)
    k = jnp.array(cfg.k)
    layer_dims = jnp.array(cfg.layer_dims)
    """electron coordinates initialization. 10.11.2025."""
    seed_electrons_coords = 22
    key_electrons_coords = jax.random.PRNGKey(seed_electrons_coords)
    key_electrons_coords, subkey_electrons_coords = jax.random.split(key_electrons_coords)
    pos, spins_test = electrons_initialization.init_electrons(
        subkey_electrons_coords,
        cfg.system.molecule,
        cfg.system.electrons,
        batch_size=cfg.batch_size,
        init_width=0.1,
        core_electrons={},
    )

    #jax.debug.print("pos_test:{}", pos_test)
    """test pretrain. 10.11.2025."""
    hartree_fock = pretrain_HF.get_hf(
        pyscf_mol=cfg.system.get('pyscf_mol'),
        molecule=cfg.system.molecule,
        nspins=(3, 3),
        restricted=False,
        basis='ccpvdz',
        ecp={},
        core_electrons={},
        states=0,
        excitation_type='ordered')
    # broadcast the result of PySCF from host 0 to all other hosts
    jax.debug.print("hartree_fock:{}", hartree_fock)
    """we need check the next fitting step.10.11.2025."""

    charges = jnp.array(cfg.charges)
    atoms = jnp.array(cfg.atoms)
    #pos = jnp.array(cfg.pos)
    #jax.debug.print("g:{}", g)
    grid_range_envelope = jnp.array(cfg.envelope.grid_range_envelope)
    kan_init, kan_apply, orbitals_apply = make_kan_net(nspins=(3, 3),
                                                       charges=charges,
                                                       nelectrons=6,
                                                       nfeatures=4,
                                                       n_parallel=n_parallel,
                                                       n_antiparallel=n_antiparallel,
                                                       parallel_indices=parallel_indices,
                                                       antiparallel_indices=antiparallel_indices,
                                                       grid_range=cfg.grid_range,
                                                       g=g,
                                                       k=k,
                                                       natoms=1,
                                                       ndims=3,
                                                       layer_dims=layer_dims,
                                                       g_envelope=cfg.envelope.g_envelope,
                                                       k_envelope=cfg.envelope.k_envelope,
                                                       grid_range_envelope=grid_range_envelope,
                                                       chebyshev=cfg.chebyshev,
                                                       spline=cfg.spline,
                                                       add_residual=cfg.add_residual,
                                                       add_bias=cfg.add_bias,
                                                       external_weights=cfg.external_weights,
                                                       envelope_chebyshev=cfg.envelope_chebyshev,
                                                       envelope_spline=cfg.envelope_spline,
                                                       envelope_simple=cfg.envelope_simple,)

    seed = 42
    key = jax.random.PRNGKey(seed)
    key, subkey = jax.random.split(key)
    params = kan_init(subkey)
    signed_network = kan_apply
    logabs_network = lambda *args, **kwargs: signed_network(*args, **kwargs)[1]
    """these are for real orbitals. not for complex orbitals. to be continued...3.12.2025.!!!"""
    """complex version can run. 5.12.2025."""
    spins = jnp.array([cfg.spins])
    #jax.debug.print("spins:{}", spins)
    batch_network = jax.vmap(
        logabs_network, in_axes=(None, 0, None, None, None), out_axes=0
    )
    jax.debug.print("pos:{}", pos)
    jax.debug.print("spis:{}", spins)
    jax.debug.print("atoms:{}", atoms)
    jax.debug.print("charges:{}", charges)
    #value_wavefunction = batch_network(params, pos, spins, atoms, charges)

    def log_network(*args, **kwargs):
        phase, mag = signed_network(*args, **kwargs)
        return mag + 1.j * phase

    key, hartree_fock_key = jax.random.split(key, 2)
    orbitals_vmap = jax.vmap(orbitals_apply, in_axes=(None, 0, None, None, None), out_axes=0)
    try:
        params, pos = pretrain_HF.pretrain_hartree_fock(
            params=params,
            positions=pos,
            spins=spins,
            charges=charges,
            atoms=atoms,
            batch_network=batch_network,
            batch_orbitals=orbitals_vmap,
            sharded_key=hartree_fock_key,
            electrons=cfg.system.electrons,
            scf_approx=hartree_fock,
            iterations=cfg.preiterations,
            batch_size=cfg.batch_size,
            logger=pretrain_logger if swan_state['module'] is not None else None,
            scf_fraction=1.0,
            states=0,
        )

        #jax.debug.print("pos:{}", pos)
        #jax.debug.print("params:{}", params)
        #jax.debug.print("atoms:{}", atoms)

        #jax.debug.print("wavefunction_value:{}", wavefunction_value)
        """we need do batch for pos."""
        data = KANetsData(positions=pos, spins=spins, atoms=atoms, charges=charges)

        # Main-chain MCMC now uses VMC all-electron kernel from VMCmcstep.py.
        batch_signed_network = jax.vmap(
            signed_network, in_axes=(None, 0, None, None, None), out_axes=(0, 0)
        )
        monte_carlo = VMCmcstep.make_mcmc_step(
            f=batch_signed_network,
            ndim=3,
            nelectrons=cfg.system.nelectrons,
            steps=10,
        )
        """the following two lines for testing the walker move process.31.10.2025."""
        #key, monte_carlo_key = jax.random.split(subkey)
        #new_data = monte_carlo(params, data, monte_carlo_key, 0.1)
        """now we need move to the energy calculation method."""
        """the following is energy calculation test. 31.10.2025."""
        #jax.debug.print("charges:{}", charges)
        key, energy_key = jax.random.split(key)
        complex_output = bool(cfg.get('complex_output', False))
        loss_network = log_network if complex_output else logabs_network
        local_energy = hamiltonian.local_energy(f=signed_network,
                                                nspins=(3, 3),
                                                charges=charges,
                                                use_scan=False,
                                                complex_output=complex_output,
                                                laplacian_method='default')
        """the reason for the error is local_energy can not accept the batched input. 3.11.2025."""
        """we solved it by a simple reconstruction of input axes."""
        #local_energy_vmap = jax.vmap(local_energy, in_axes=(None, None, 0, None, None, None))
        #output = local_energy_vmap(params, energy_key, data.positions, data.spins, data.atoms, data.charges,)
        #jax.debug.print("output:{}", output)
        """next, we need construction the loss function. 3.11.2025."""
        evaluate_loss = qmc_loss_functions.make_loss(loss_network,
                                                     local_energy,
                                                     clip_local_energy=5.0,
                                                     clip_from_median=True,
                                                     center_at_clipped_energy=True,
                                                     complex_output=complex_output,
                                                     )

        def learning_rate_schedule(t_: jnp.ndarray) -> jnp.ndarray:
            return 0.05 * jnp.power(
                (1.0 / (1.0 + (t_ / 1.0))), 10000.0)

        optimizer = optax.chain(
            optax.scale_by_adam(b1=0.9, b2=0.999,eps=1e-6),
            optax.scale_by_schedule(learning_rate_schedule),
            optax.scale(-1.))
        #jax.debug.print("type_of_optimizer:{}",type(optimizer))
        if isinstance(optimizer, optax.GradientTransformation):
            opt_state = optimizer.init(params)
            #jax.debug.print("opt_state:{}", opt_state)
            """because we dont set any parallel strategy for monte carlo step. We also need rewrite the parallel strategy for optimization.4.11.2025."""
            step = make_training_step(mcmc_step=monte_carlo,
                                      optimizer_step=make_opt_update_step(evaluate_loss, optimizer),
                                      reset_if_nan=True)

            mcmc_width = jnp.asarray(0.1)
            adapt_frequency = int(cfg.get('mcmc_adapt_frequency', 20))
            pmove_min = float(cfg.get('mcmc_pmove_min', 0.50))
            pmove_max = float(cfg.get('mcmc_pmove_max', 0.60))
            width_scale = float(cfg.get('mcmc_width_scale', 1.05))
            pmoves = np.zeros((adapt_frequency,), dtype=np.float32)
            t_init = 0
            sharded_key = key
            jax.debug.print("sharded_key:{}", sharded_key)
            """to be continued... 3.11.2025."""
            for t in range(t_init, cfg.iterations):
                sharded_key, subkeys = jax.random.split(sharded_key, 2)

                data, params, opt_state, loss, aux_data, pmove = step(data, params, opt_state, subkeys, mcmc_width,)

                pmove_mean = jnp.mean(pmove)
                t_since_update = t % adapt_frequency
                pmoves[t_since_update] = _to_scalar(pmove_mean)
                if t > 0 and t_since_update == 0:
                    mean_pmove = float(np.mean(pmoves))
                    if mean_pmove > pmove_max:
                        mcmc_width = mcmc_width * width_scale
                    elif mean_pmove < pmove_min:
                        mcmc_width = mcmc_width / width_scale
                window_size = min(t + 1, adapt_frequency)
                pmove_window_mean = float(np.mean(pmoves[:window_size]))

                #loss = loss[0]
                jax.debug.print(
                    "loss: {}, pmove: {}, pmove_window: {}, mcmc_width: {}",
                    loss,
                    pmove_mean,
                    pmove_window_mean,
                    mcmc_width,
                )
                swan_log(
                    {
                        'train/loss': _to_scalar(loss),
                        'train/pmove': _to_scalar(pmove_mean),
                        'train/pmove_window': pmove_window_mean,
                        'train/mcmc_width': _to_scalar(mcmc_width),
                    },
                    int(t),
                )
    finally:
        if swan_state['module'] is not None:
            swan_state['module'].finish()


