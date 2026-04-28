from typing import Any, Callable, Optional

import jax

from pretrain import pretrain_DFT, pretrain_HF, pretrain_mkan


class PretrainRunner:
    """Encapsulates HF/DFT pretraining stage orchestration."""

    def __init__(
        self,
        *,
        run_manager,
        build_checkpoint_state: Callable[..., dict[str, Any]],
        enabled: bool,
        preiterations: int,
        method: str,
        pyscf_mol,
        molecule,
        electrons,
        restricted: bool,
        basis: str,
        core_electrons,
        hf_states: int,
        hf_excitation_type: str,
        dft_xc: str,
        dft_grid_level: int,
        scf_fraction: float,
        batch_size: int,
        pretrain_mcmc_steps: int,
        pretrain_mcmc_width: float,
        debug: bool,
        scalar_pretrain: bool = False,
        phase_weight: float = 1.0e-2,
    ):
        self.run_manager = run_manager
        self.build_checkpoint_state = build_checkpoint_state

        self.enabled = enabled
        self.preiterations = preiterations
        self.method = method

        self.pyscf_mol = pyscf_mol
        self.molecule = molecule
        self.electrons = electrons
        self.restricted = restricted
        self.basis = basis
        self.core_electrons = core_electrons

        self.hf_states = hf_states
        self.hf_excitation_type = hf_excitation_type
        self.dft_xc = dft_xc
        self.dft_grid_level = dft_grid_level

        self.scf_fraction = scf_fraction
        self.batch_size = batch_size
        self.pretrain_mcmc_steps = pretrain_mcmc_steps
        self.pretrain_mcmc_width = pretrain_mcmc_width
        self.debug = debug
        self.scalar_pretrain = scalar_pretrain
        self.phase_weight = phase_weight

    def run(
        self,
        *,
        params,
        data,
        sharded_key,
        pretrain_start_step: int,
        train_opt_state,
        pretrain_opt_state,
        batch_network,
        orbitals_vmap,
        batch_log_network=None,
        t_init: int,
    ):
        needs_pretrain = (
            self.enabled
            and train_opt_state is None
            and pretrain_start_step < self.preiterations
        )
        if not needs_pretrain:
            return params, data, sharded_key, pretrain_opt_state, t_init

        def log_pretrain(step: int, loss_value: float) -> None:
            if self.run_manager.should_log(step, self.preiterations):
                self.run_manager.log_scalars('pretrain', step, {'loss': loss_value})

        def checkpoint_pretrain(step, loss_value, params_, opt_state_, data_, key_):
            del loss_value
            checkpoint_state = self.build_checkpoint_state(
                stage='pretrain',
                step=step,
                params=params_,
                data=data_,
                key=key_,
                pretrain_opt_state=opt_state_,
                train_opt_state=None,
            )
            if self.run_manager.should_checkpoint(step, self.preiterations):
                self.run_manager.checkpoints.save_step('pretrain', step, checkpoint_state)

        if self.method == 'hf':
            reference = pretrain_HF.get_hf(
                pyscf_mol=self.pyscf_mol,
                molecule=self.molecule,
                nspins=self.electrons,
                restricted=self.restricted,
                basis=self.basis,
                ecp={},
                core_electrons=self.core_electrons,
                states=self.hf_states,
                excitation_type=self.hf_excitation_type,
            )
            if self.debug:
                jax.debug.print('hartree_fock:{}', reference)

            if self.scalar_pretrain:
                params, data, pretrain_opt_state, sharded_key = (
                    pretrain_mkan.pretrain_scalar_wavefunction(
                        params=params,
                        positions=data.positions,
                        spins=data.spins,
                        charges=data.charges,
                        atoms=data.atoms,
                        batch_network=batch_network,
                        batch_log_network=batch_log_network,
                        sharded_key=sharded_key,
                        electrons=self.electrons,
                        scf_approx=reference,
                        iterations=self.preiterations,
                        logger=log_pretrain,
                        checkpoint_callback=checkpoint_pretrain,
                        scf_fraction=self.scf_fraction,
                        phase_weight=self.phase_weight,
                        mcmc_steps=self.pretrain_mcmc_steps,
                        mcmc_width=self.pretrain_mcmc_width,
                        start_iteration=pretrain_start_step,
                        opt_state=pretrain_opt_state,
                        data=data,
                    )
                )
            else:
                params, data, pretrain_opt_state, sharded_key = pretrain_HF.pretrain_hartree_fock(
                    params=params,
                    positions=data.positions,
                    spins=data.spins,
                    charges=data.charges,
                    atoms=data.atoms,
                    batch_network=batch_network,
                    batch_orbitals=orbitals_vmap,
                    sharded_key=sharded_key,
                    electrons=self.electrons,
                    scf_approx=reference,
                    iterations=self.preiterations,
                    batch_size=self.batch_size,
                    logger=log_pretrain,
                    checkpoint_callback=checkpoint_pretrain,
                    scf_fraction=self.scf_fraction,
                    states=self.hf_states,
                    mcmc_steps=self.pretrain_mcmc_steps,
                    mcmc_width=self.pretrain_mcmc_width,
                    start_iteration=pretrain_start_step,
                    opt_state=pretrain_opt_state,
                    data=data,
                )
        elif self.method == 'dft':
            if self.hf_states != 0:
                raise ValueError('DFT pretraining currently supports only ground states; set hf_states=0.')
            reference = pretrain_DFT.get_dft(
                pyscf_mol=self.pyscf_mol,
                molecule=self.molecule,
                nspins=self.electrons,
                restricted=self.restricted,
                basis=self.basis,
                ecp={},
                core_electrons=self.core_electrons,
                xc=self.dft_xc,
                grid_level=self.dft_grid_level,
                states=0,
            )
            if self.debug:
                jax.debug.print('dft_reference:{}', reference)

            if self.scalar_pretrain:
                params, data, pretrain_opt_state, sharded_key = (
                    pretrain_mkan.pretrain_scalar_wavefunction(
                        params=params,
                        positions=data.positions,
                        spins=data.spins,
                        charges=data.charges,
                        atoms=data.atoms,
                        batch_network=batch_network,
                        batch_log_network=batch_log_network,
                        sharded_key=sharded_key,
                        electrons=self.electrons,
                        scf_approx=reference,
                        iterations=self.preiterations,
                        logger=log_pretrain,
                        checkpoint_callback=checkpoint_pretrain,
                        scf_fraction=self.scf_fraction,
                        phase_weight=self.phase_weight,
                        mcmc_steps=self.pretrain_mcmc_steps,
                        mcmc_width=self.pretrain_mcmc_width,
                        start_iteration=pretrain_start_step,
                        opt_state=pretrain_opt_state,
                        data=data,
                    )
                )
            else:
                params, data, pretrain_opt_state, sharded_key = pretrain_DFT.pretrain_ks_dft(
                    params=params,
                    positions=data.positions,
                    spins=data.spins,
                    charges=data.charges,
                    atoms=data.atoms,
                    batch_network=batch_network,
                    batch_orbitals=orbitals_vmap,
                    sharded_key=sharded_key,
                    electrons=self.electrons,
                    dft_approx=reference,
                    iterations=self.preiterations,
                    batch_size=self.batch_size,
                    logger=log_pretrain,
                    checkpoint_callback=checkpoint_pretrain,
                    scf_fraction=self.scf_fraction,
                    mcmc_steps=self.pretrain_mcmc_steps,
                    mcmc_width=self.pretrain_mcmc_width,
                    start_iteration=pretrain_start_step,
                    opt_state=pretrain_opt_state,
                    data=data,
                )
        else:
            raise ValueError(f"Unsupported pretrain_method: {self.method}. Expected 'hf' or 'dft'.")

        return params, data, sharded_key, pretrain_opt_state, t_init
