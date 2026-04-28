import functools
import jax

try:
    import kfac_jax
except Exception:  # kfac_jax pulls in TFP/distrax, which can lag JAX releases.
    kfac_jax = None


# Axis name we pmap over.
PMAP_AXIS_NAME = 'qmc_pmap_axis'

# Shortcut for jax.pmap over PMAP_AXIS_NAME. Prefer this if pmapping any
# function which does communications or reductions.
pmap = functools.partial(jax.pmap, axis_name=PMAP_AXIS_NAME)

if kfac_jax is not None:
    psum = functools.partial(kfac_jax.utils.psum_if_pmap, axis_name=PMAP_AXIS_NAME)
    pmean = functools.partial(
        kfac_jax.utils.pmean_if_pmap, axis_name=PMAP_AXIS_NAME)
    all_gather = functools.partial(kfac_jax.utils.wrap_if_pmap(jax.lax.all_gather),
                                   axis_name=PMAP_AXIS_NAME)
else:
    def psum(value):
        return value

    def pmean(value):
        return value

    def all_gather(value):
        return value
