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
    from astropy.coordinates import Latitude, Longitude, AltAz
    from astropy.coordinates import EarthLocation, SkyCoord
    from astropy.time import Time
    from astropy_healpix import HEALPix
    import healpy

    # TODO: insert a check that the beam is OK.
    # Construct the AltAz frame for coordinate transformations.
    observatory = EarthLocation(*telescope_location)
    local_frame = AltAz(location=observatory, obstime=Time(obstime, format="jd"))

    # Prepare the direction cosine grid.
    uvws = freq * (
        enu_antpos[ant_2_array] - enu_antpos[ant_1_array]
    ) / constants.c.si.value
    umax = np.linalg.norm(uvws, axis=1).max()
    dl = 1 / (4*umax)
    n_l = int(2 // dl)
    if n_l % 2 == 0:
        n_l += 1  # Ensure we get (l,m) = (0,0)
    lm_grid = np.linspace(-1, 1, n_l)
    measure = (lm_grid[1]-lm_grid[0]) ** 2

    # Put the beam and sky onto the (l,m) grid.
    gridded_beam = np.zeros((n_l, n_l), dtype=float)
    flat_Tsky = np.zeros(gridded_beam.shape, dtype=float)
    hpx_grid = HEALPix(nside=nside, order="ring", frame="icrs")
    for row, m in enumerate(lm_grid):
        # Figure out which sky positions are above the horizon.
        lmag = np.sqrt(m**2 + lm_grid**2)
        select = lmag < 1
        if select.sum() == 0:
            continue

        # Compute the azimuth and zenith angle for each of these points.
        indices = np.argwhere(select).flatten()
        za = np.arcsin(lmag[select])
        az = np.arctan2(m, lm_grid[select])

        # Interpolate the beam.
        beam_vals = beam.interp(
            az_array=az, za_array=za, freq_array=np.array([freq])
        )[0][0,0]
        if beam_vals.ndim == 3:
            # For some versions of pyuvsim, this returns an array with shape
            # (Nvispol, Nfreq, Npix); for now I'll be assuming that we want
            # the XX polarization.
            # TODO: Update this to correctly handle different polarizations
            beam_vals = beam_vals[0,0]
        gridded_beam[row,select] = beam_vals
        
        # Interpolate the sky intensity.
        coords = SkyCoord(
            Longitude(np.pi/2 - az, unit="rad"),  # astropy uses E of N
            Latitude(np.pi/2 - za, unit="rad"),
            frame=local_frame,
        ).transform_to("icrs")
        flat_Tsky[row,select] = hpx_grid.interpolate_bilinear_skycoord(
            coords, Tsky
        ) / np.sqrt(1 - lmag[select]**2)  # Apply projection effect to the sky.

    # Now compute the sky power spectrum.
    sky_pspec = np.abs(
        np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(flat_Tsky * measure)))
    ) ** 2

    # Now compute the diffuse matrix.
    LM = np.array(np.meshgrid(lm_grid, lm_grid))
    diff_mat = np.zeros((edges[-1], n_eig), dtype=complex)
    for grp, (start, stop) in enumerate(zip(edges, edges[1:])):
        fringe = np.exp(
            2j * np.pi * np.einsum("bi,ilm->blm", uvws[start:stop,:2], LM)
        )
        kernel = np.fft.fftshift(
            np.fft.fft2(
                np.fft.ifftshift(
                    gridded_beam[None,...] * fringe, axes=(1,2)
                ), axes=(1,2)
            ), axes=(1,2)
        )
        block = measure * np.einsum(
            "puv,quv,uv->pq", kernel, kernel.conj(), sky_pspec
        )

        # Compute the eigenvalues/vectors for this block.
        eigvals, eigvecs = np.linalg.eigh(block, "U")
        if eigvals[0] < eigvals[-1]:
            eigvals = eigvals[::-1]
            eigvecs = eigvecs[:,::-1]
        diff_mat[start:stop] = eigvecs[:,:n_eig] * np.sqrt(
            eigvals[None,:n_eig]
        )

    return diff_mat


def compute_source_matrix(
   tmp 
):
    pass
