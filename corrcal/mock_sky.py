import numpy as np
import healpy
from astropy import units, constants

def gaussian_diffuse(
    n_side, C_l=1, max_intensity=None, force_nonneg=True
):
    """Generate a mock diffuse sky given an angular power spectrum."""
    if callable(C_l):
        ells = np.arange(3*n_side)
        C_l = C_l(ells)
    else:
        C_l = np.atleast_1d(C_l)

    intensity = healpy.synfast(C_l, n_side)
    if force_nonneg and intensity.min() < 0:
        intensity += np.abs(intensity.min())
    
    if max_intensity is not None:
        intensity *= max_intensity / intensity.max()
    return intensity


def random_sources(
    n_src,
    min_flux=10*units.mJy.to(units.Jy),
    max_flux=1000,
    flux_index=-2,
    min_dec=-np.pi/2,
    max_dec=np.pi/2,
    min_ra=0,
    max_ra=2*np.pi,
):
    """Generate a catalog of point sources."""
    src_fluxes = np.random.uniform(size=n_src)
    src_fluxes = (
        min_flux**flux_index + (
            max_flux**flux_index - min_flux**flux_index
        ) * src_fluxes
    ) ** (1/flux_index)
    src_dec = np.random.uniform(min_dec, max_dec, n_src)
    src_ra = np.random.uniform(min_ra, max_ra, n_src)
    return src_ra, src_dec, src_fluxes
