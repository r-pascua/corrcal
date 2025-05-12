"""Module for calculating model covariance."""
from astropy import constants, units
import healpy
import numpy as np
from typing import Optional, Union
from pathlib import Path
from pyuvdata import UVBeam, UVData
from pyradiosky import SkyModel
from .covdata import UVCov
from . import utils
from scipy.special import spherical_jn, sph_harm_y


def build_source_model(
    *,
    freq_array: Optional[np.ndarray] = None,
    lst_array: Optional[np.ndarray] = None,
    time_array: Optional[np.ndarray] = None,
    telescope_location: Optional[np.ndarray] = None,
    baselines: Optional[np.ndarray] = None,
    data: Optional[Union[UVData, str]] = None,
    beam: Union[UVBeam, str] = None,
    sources: SkyModel = None,
) -> np.ndarray:
    r"""Construct the point-source matrix.

    This is the :math:`\sigma` matrix in Eq. ?? of <ref paper>. In particular,
    each matrix element has the following form:

    .. math::

        \sigma_{km}(\nu,t) = S_m(\nu) A_\nu\bigl(\hat{n}_m(t)\bigr)
        \exp\bigl(-i2\pi\nu\vec {b}_k \cdot \hat{n}_m(t) / c\bigr).

    For computational simplicity, we work in a frame where the array is fixed,
    so the positions of the sources change for each observed time. For now,
    polarization is handled under the assumption that the sky is pure Stokes-I,
    and so any polarization comes solely from the antenna response.

    Parameters
    ----------
    freq_array
        Frequencies, in Hz, at which to evaluate the beam/source models.
        Does not need to be provided if ``data`` is provided.
    lst_array
        Local Sidereal Time for each observation, in radians. Does not need
        to be provided if ``data`` is provided.
    time_array
        Observation times, in JD. Does not need to be provided if ``data``
        is provided.
    telescope_location
        Latitude, longitude, in radians, and altitude, in meters, of the
        telescope site. Does not need to be provided if ``data`` is provided.
    baseline_array
        Array containing the baseline vectors for the entire array, in units
        of meters. Does not need to be provided if ``data`` is provided. Should
        be shape (Nbls,3) and in topocentric (ENU) coordinates.
    data
        ``UVData`` object or path to a file that can be loaded into a
        ``UVData`` object. Does not need to be provided if ``freq_array``,
        ``telescope_location``, and either ``lst_array`` or ``time_array`` are
        all provided.
    beam
        ``UVBeam`` object or path to a file that can be loaded into a
        ``UVBeam`` object.
    sources
        ``SkyModel`` object containing information about the point source
        catalog (positions, fluxes, etc). Some pre-processing should be done
        on this so that only a handful of sources are present. The source
        positions should be provided in (RA,dec) coordinates.

    Returns
    -------
    source_model
        Array with shape (Ntimes, Nfreqs, Npols, Nbls, Nsrc) representing the
        :math:`\sigma` matrix from Eq. ?? of <ref paper> at each time,
        frequency, and polarization.

    Notes
    -----
    The calculated array will *not* be sorted by redundancy. In order to use
    the output of this function to create a ``UVCov`` object, you must sort
    the array appropriately after it is calculated (or provided a sorted
    baseline array or ``UVData`` object that has had its metadata adjusted in
    accordance with a grouping by redundancy). For the greatest control, it is
    recommended to provide a ``UVData`` object that has already had a number
    of adjustments made based on its metadata.
    """
    # Setup
    if beam is None or sources is None:
        raise ValueError(
            "Both a beam model and source model must be provided."
        )

    if data is None:
        freqs_ok = isinstance(freq_array, np.ndarray)
        times_ok = isinstance(lst_array, np.ndarray) or isinstance(
            time_array, np.ndarray
        )
        loc_ok = isinstance(telescope_location, np.ndarray)
        bls_ok = isinstance(baselines, np.ndarray)
        if not all(freqs_ok, times_ok, loc_ok, bls_ok):
            raise ValueError(
                "Not enough information provided to construct model."
            )
    else:
        if not isinstance(data, UVData):
            uvdata = UVData()
            uvdata.read(data, read_data=False)
        else:
            uvdata = data
        freq_array = np.unique(uvdata.freq_array)
        lst_array = uvdata.lst_array[
            np.unique(uvdata.time_array, return_index=True)[1]
        ]
        telescope_location = uvdata.telescope_lat_lon_alt
        baseline_array = np.zeros((uvdata.Nbls, 3), dtype=float)
        antpos, antnums = uvdata.get_ENU_baselines()
        baseline_array = utils.build_baseline_array(
            ant_1_array=uvdata.ant_1_array,
            ant_2_array=uvdata.ant_2_array,
            antpos=antpos,
            antnums=antnums,
        )
        _ = len(baseline_array)  # Just to make flake8 not complain.

    if not isinstance(beam, UVBeam):
        uvbeam = UVBeam()
        uvbeam.read(beam)
    else:
        uvbeam = beam.copy()


def compute_diffuse_matrix(
    Tsky,
    nside,
    beam,
    telescope_location,
    obstime,
    freq,
    enu_antpos,
    ant_1_array,
    ant_2_array,
    edges,
    n_eig,
    flat_sky=True,
):
    """TODO: write docs"""
    # TODO: switch over from healpy to astropy-healpix
    if flat_sky:
        return _compute_diffuse_matrix_from_flat_sky(
            Tsky=Tsky,
            nside=nside,
            beam=beam,
            telescope_location=telescope_location,
            obstime=obstime,
            freq=freq,
            enu_antpos=enu_antpos,
            ant_1_array=ant_1_array,
            ant_2_array=ant_2_array,
            edges=edges,
            n_eig=n_eig,
        )
    else:
        return _compute_diffuse_matrix_from_harmonics(
            Tsky=Tsky,
            nside=nside,
            beam=beam,
            telescope_location=telescope_location,
            obstime=obstime,
            freq=freq,
            enu_antpos=enu_antpos,
            ant_1_array=ant_1_array,
            ant_2_array=ant_2_array,
            edges=edges,
            n_eig=n_eig,
        )


def _compute_diffuse_matrix_from_harmonics(
    Tsky,
    nside,
    beam,
    telescope_location,
    obstime,
    freq,
    enu_antpos,
    ant_1_array,
    ant_2_array,
    edges,
    n_eig,
):
    raise NotImplementedError("Need to finish debugging.")


def _compute_diffuse_matrix_from_flat_sky(
    sky_pspec,
    nside,
    beam,
    freq,
    enu_antpos,
    ant_1_array,
    ant_2_array,
    edges,
    n_eig=1,
    lmax=10,
    pixwgt_datapath=None,
    bl_convention="i-j",
):
    """Compute the diffuse matrix assuming we can flat sky the integrals."""
    # First, compute the sph. harm. coefficients for the beam-squared.
    pixels = np.arange(healpy.nside2npix(nside))
    theta, phi = healpy.pix2ang(nside, pixels)
    hpx_bsq_vals = np.abs(
        beam.power_eval(
            az_array=phi,
            za_array=theta,
            freq_array=np.array([freq]),
        )[0][0,0]
    ) ** 2
    hpx_bsq_vals[theta>np.pi/2] = 0  # Horizon cut
    Blm = healpy.map2alm(
        hpx_bsq_vals,
        use_pixel_weights=True,
        lmax=lmax,
        datapath=pixwgt_datapath,
    )[:,None,None]

    # Next, prepare the multipole integers for use later.
    ells, emms = healpy.Alm.getlm(lmax)
    ells = ells[:,None,None]
    emms = emms[:,None,None]
    zero_m = emms[:,0,0] == 0

    # Compute the uvws.
    uvws = freq * (
        enu_antpos[ant_1_array] - enu_antpos[ant_2_array]
    ) / constants.c.si.value
    if bl_convention == "j-i":
        uvws = -uvws

    # Now prepare the diffuse matrix and fill it in block-by-block.
    sky_ells = np.arange(sky_pspec.size)
    diff_mat = np.zeros((edges[-1], 2*n_eig), dtype=float)
    for start, stop in zip(edges, edges[1:]):
        # Compute the baseline differences.
        uvw_here = uvws[start//2:stop//2]
        uvw_diffs = uvw_here[:,None] - uvw_here[None,:]
        uvw_diff_mags = np.linalg.norm(uvw_diffs, axis=2)

        # Compute the angles associated with the differences.
        normed_uvw_diffs = uvw_diffs / np.where(
            np.isclose(uvw_diff_mags, 0), 1, uvw_diff_mags
        )[:,:,None]
        thetas = np.arccos(normed_uvw_diffs[...,2])[None,:,:]
        phis = np.arctan2(
            normed_uvw_diffs[...,1], normed_uvw_diffs[...,0]
        )[None,:,:]

        # Compute the bandpower measured by this group.
        avg_uvw_mag = np.linalg.norm(uvw_here.mean(axis=0))
        pspec_bessels = spherical_jn(sky_ells, 2*np.pi*avg_uvw_mag)
        band_power = 4 * np.pi * np.sum(
            sky_pspec * (2*sky_ells + 1) * pspec_bessels**2
        )

        # Compute the fringe harmonics.
        bessels = spherical_jn(ells, 2*np.pi*uvw_diff_mags[None,:,:])
        Ylms = sph_harm_y(ells, emms, thetas, phis)
        flm = 1j**ells * bessels * Ylms
        
        # Now compute the complex covariance elements.
        offset = np.sum(flm[zero_m] * Blm[zero_m], axis=0)
        B_kk = np.sum(
            flm*Blm + (-1)**ells * flm.conj() * Blm.conj(), axis=0
        ) - offset

        # Now fill in the real-valued covariance.
        block = np.zeros(2*np.array(B_kk.shape), dtype=float)
        block[::2,::2] = B_kk.real
        block[1::2,1::2] = B_kk.real
        block[1::2,::2] = B_kk.imag
        block[::2,1::2] = -B_kk.imag
        block *= band_power

        # Now take the eigendecomposition.
        eigvals, eigvecs = np.linalg.eigh(block)
        sort = np.argsort(eigvals)[::-1]
        eigvals = eigvals[sort][:2*n_eig]
        eigvecs = eigvecs[:,sort][:,:2*n_eig]
        block = np.sqrt(eigvals)[None,:] * eigvecs
        diff_mat[start:stop] = np.where(np.isnan(block), 0, block)

    return diff_mat
