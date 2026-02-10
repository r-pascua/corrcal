"""Module for calculating model covariance."""
import numpy as np
from numpy.typing import NDArray
from typing import Type
from pathlib import Path

from . import utils
from scipy.special import spherical_jn, sph_harm_y

try:
    from pyuvdata import UVBeam, UVData, BeamInterface
    from pyuvdata.analytic_beam import AnalyticBeam
    HAVE_PYUVDATA = True
except ImportError:
    HAVE_PYVDATA = False

try:
    from pyradiosky import SkyModel
    HAVE_PYRADIOSKY = True
except ImportError:
    HAVE_PYRADIOSKY = False
   
try:
    import healpy
    HAVE_HEALPY = True
except ImportError:
    HAVE_HEALPY = False

try:
    from astropy import constants, units
    from astropy.coordinates import Latitude, Longitude, AltAz
    from astropy.coordinates import EarthLocation, SkyCoord
    from astropy.time import Time
    HAVE_ASTROPY = True
except ImportError:
    HAVE_ASTROPY = False

def build_source_matrix(
    *,
    freq_array: NDArray[float] | None = None,
    lst_array: NDArray[float] | None = None,
    time_array: NDArray[float] | None = None,
    pol_array: NDArray[str] | None = None,
    telescope_location: NDArray[float] | EarthLocation | None = None,
    baseline_array: NDArray[float] | None = None,
    uvdata: UVData | str | None = None,
    beam: UVBeam | Type[AnalyticBeam] | str | None = None,
    Nsrc: int = 1,
    source_model: SkyModel | None = None,
    fluxes: NDArray[float] | None = None,
    src_ra: NDArray[float] | None = None,
    src_dec: NDArray[float] | None = None,
    spec_inds: NDArray[float] | None = None,
    ref_freqs: NDArray[float] | None = None,
    bright_source_cutoff: float = 100,
    altitude_cutoff: float = 80,
) -> NDArray[float]:
    r"""Construct the point-source matrix.


    This function computes the source matrix according to Equation 26 from
    Pascua+ 2025. Each matrix element has the following form:

    .. math::

        \Sigma_{kj}(\nu,t) = S_j(\nu) A_\nu\bigl(\hat{r}_j(t)\bigr)
        \exp\bigl(-i2\pi\nu\vec {b}_k \cdot \hat{n}_j(t) / c\bigr).

    For computational simplicity, we work in a frame where the array is fixed,
    so the positions of the sources change for each observed time. This
    function assumes that the sky is purely Stokes-I (i.e., :math:`S_j(\nu)`
    is the total flux density for source :math:`j` at frequency :math:`\nu`).

    Parameters
    ----------
    freq_array
        Frequencies, in Hz, at which to evaluate the beam/source models.
        Does not need to be provided if ``uvdata`` is provided.
    lst_array
        Local Sidereal Time for each observation, in radians. Does not need
        to be provided if ``uvdata`` is provided.
    time_array
        Observation times, in JD. Does not need to be provided if ``uvdata``
        is provided.
    pol_array
        Array indicating which visibility polarizations to use. Does not need
        to be provided if ``uvdata`` is provided.
    telescope_location
        Latitude, longitude, in radians, and altitude, in meters, of the
        telescope site. Does not need to be provided if ``uvdata`` is provided.
    baseline_array
        Array containing the baseline vectors for the entire array, in units
        of meters. Does not need to be provided if ``uvdata`` is provided.
        Should be shape (Nbls,3) and in topocentric (ENU) coordinates.
    uvdata
        ``pyuvdata.UVData`` object or path to a file that can be loaded into a
        ``pyuvdata.UVData`` object. Does not need to be provided if
        ``freq_array``, ``telescope_location``, and either ``lst_array``
        or ``time_array`` are all provided.
    beam
        The beam model to be used for computing the perceived source fluxes.
        May be provided as a ``pyuvdata.UVBeam`` object, a path to a file that
        may be read into a ``pyuvdata.UVBeam`` object, or a
        ``pyuvdata.AnalyticBeam`` object.
    Nsrc
        Number of sources to use for calibration. The top ``n_src``
        contributors to the observed source flux will be used for building
        the source matrix. The default option is to use the brightest source.
    source_model
        ``pyradiosky.SkyModel`` object containing information about the point
        source catalog (positions, fluxes, etc). Some pre-processing should be
        done on this so that only a handful of sources are present. The source
        positions should be provided in (RA,dec) coordinates in the J2000 epoch.
        Not required if enough information is provided to construct the source
        catalog (see documentation for the ``fluxes``, ``src_ra``, and
        ``src_dec`` arguments).
    fluxes
        The source fluxes, in Janskys, either at the corresponding reference
        frequencies (when providing spectral indices) or at the measured
        frequencies in ``freq_array``. Not required if ``source_model`` is
        provided. If ``ref_freqs`` and ``spec_inds`` are provided, then it
        is assumed that each source flux evolves as a power law in frequency;
        otherwise the provided ``fluxes`` must provide the source flux at each
        frequency in ``freq_array`` with shape ``(freq_array.size, cat_size)``,
        where ``cat_size`` is the total number of sources provided.
    src_ra
        Right ascension of each source measured in the J2000 epoch in units
        of radians. Not required if ``source_model`` is provided.
    src_dec
        Declination of each source measured in the J2000 epoch in units of
        radians. Not required if ``source_model`` is provided.
    spec_inds
        Spectral index of each source, if source fluxes are provided only at
        a single frequency. Not required if ``source_model`` is provided or if
        source fluxes are provided for each frequency in ``freq_array``.
    ref_freqs
        Reference frequency for each source flux if source spectral evolution
        is treated as a power law. Only required if ``spec_inds`` is provided.
    bright_source_cutoff
        Minimum band-averaged flux density, in Janskys, required for a source
        to always be considered a candidate for calibration when it is above
        the horizon. Default is 100 Jy.
    altitude_cutoff
        Altitude below which sources should not be considered as calibration
        candidates, measured in degrees. This should roughly be set to the
        width of the primary beam main lobe. Default is 10 degrees.


    Returns
    -------
    source_matrix
        Array with shape (Ntimes, Nfreqs, Npols, 2*Nbls, Nsrc) representing the
        :math:`\Sigma` matrix from Equation 26 of Pascua+ 2025 at each time,
        frequency, and polarization.

    Notes
    -----
    This calculation is currently tuned for zenith-pointing arrays. Support
    for off-zenith pointings will be supported in a future update.

    The baseline axis of the source matrix is sorted according to however the
    provided ``baseline_array`` is sorted. If the source matrix is constructed
    from a ``pyuvdata.UVData`` object, then the baseline axis sorting is
    determined by however the baselines are sorted in the ``pyuvdata.UVData``
    object.
    """
    if not (HAVE_PYUVDATA and HAVE_PYRADIOSKY and HAVE_ASTROPY):
        raise NotImplementedError(
            "pyuvdata, pyradiosky, and astropy must be installed to use the "
            "model building utility functions."
        )

    # Setup
    if beam is None:
        raise ValueError("A beam model must be provided.")

    if uvdata is None:
        freqs_ok = isinstance(freq_array, np.ndarray)
        times_ok = isinstance(lst_array, np.ndarray) and isinstance(
            time_array, np.ndarray
        )
        loc_ok = isinstance(telescope_location, (np.ndarray, EarthLocation))
        bls_ok = isinstance(
            baseline_array, np.ndarray
        ) and baseline_array.shape[1] == 3
        pols_ok = isinstance(pol_array, np.ndarray)
        if not all(freqs_ok, times_ok, loc_ok, bls_ok, pols_ok):
            raise ValueError(
                "Not enough information provided to construct model."
            )
    elif isinstance(uvdata, (str, UVData)):
        if not isinstance(uvdata, UVData):
            uvdata = UVData.from_file(uvdata, read_data=False)
        freq_array = np.unique(uvdata.freq_array)
        time_array, inds = np.unique(uvdata.time_array, return_index=True)
        lst_array = uvdata.lst_array[inds]
        pol_array = np.array(uvdata.get_pols())
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
    else:
        raise ValueError(
            "Not enough telescope information provided to construct model."
        )

    if source_model is None:
        if any(arg is None for arg in (fluxes, src_ra, src_dec)):
            raise ValueError("Insufficient source information provided.")
        fluxes = np.atleast_2d(fluxes)
        if fluxes.shape[0] == 1 and freq_array.size != 1:
            if spec_inds is None and ref_freqs is None:
                raise ValueError(
                    "When providing sources for a single frequency, spectral "
                    "indices and reference frequencies must also be provided."
                )
            freq_scaling = (
                freq_array[:,None] / ref_freqs[None,:]
            ) ** spec_inds[None,:]
            fluxes = fluxes * freq_scaling
        elif fluxes.shape[0] != freq_array.size:
            raise ValueError(
                "Provided fluxes array does not conform to the expected shape."
            )
    else:
        source_model = source_model.at_frequencies(
            freq_array*units.Hz, inplace=False
        )
        fluxes = source_model.stokes[0].to(units.Jy).value  # Only use Stokes-I
        src_ra = source_model.ra.to(units.rad).value
        src_dec = source_model.ra.to(units.rad).value

    if not isinstance(beam, (UVBeam, AnalyticBeam)):
        beam = UVBeam.from_file(beam)

    # Convert to a power beam; this will need to change for polarized CorrCal.
    beam = BeamInterface(beam).as_power_beam(include_cross_pols=True)

    # Everything should be OK to use at this point, so start computing things.
    if isinstance(telescope_location, np.ndarray):
        telescope_location = EarthLocation(*telescope_location)

    # Initialize the source matrix.
    Ntimes = time_array.size
    Nfreqs = freq_array.size
    Npols = pol_array.size
    Nbls = baseline_array.shape[0]
    source_matrix = np.zeros((Ntimes, Nfreqs, Npols, 2*Nbls, Nsrc), dtype=float)

    # Figure out which sources to always consider as candidates when they're up
    is_bright = fluxes.mean(axis=0) > bright_source_cutoff

    # Fill in the source matrix time by time.
    for t, obstime in enumerate(time_array):
        # Convert source positions to local Altitude-Azimuth coordinates.
        local_frame = AltAz(
            location=telescope_location, obstime=Time(obstime, format="jd")
        )
        source_positions = SkyCoord(
            Longitude(src_ra*units.rad),
            Latitude(src_dec*units.rad),
            frame="icrs",
        ).transform_to(local_frame)

        # Choose which sources to keep as calibration candidates.
        above_horizon = source_positions.alt > 0
        near_zenith = source_positions.alt.deg > altitude_cutoff
        select = near_zenith | (above_horizon & is_bright)

        # Compute apparent flux for each calibration candidate.
        src_az = Longitude(
            np.pi/2*units.rad - source_positions.az
        )[select].to(units.rad).value  # astropy -> pyuvdata az convention
        src_za = np.pi/2 - source_positions.alt.rad[select]
        src_flux = fluxes[select]
        beam_vals = beam.compute_response(
            az_array=src_az,
            za_array=src_za,
            freq_array=freq_array,
            az_za_grid=False,
            freq_interp_kind="cubic",
            reuse_spline=True,
        )[0]  # Output shape is (1, Npol, Nfreq, Nsrc) for power beam.
        weighted_fluxes = beam_vals * src_flux[None]

        # Use band-averaged pI flux to choose calibration sources, if possible.
        beam_pols = beam.polarization_array
        if beam_pols.size == 1:
            pol_select = slice()
        elif 1 in beam_pols:  # 1 is Stokes-I
            pol_select = np.argwhere(beam_pols == 1)
        elif -5 in beam_pols or -6 in beam_pols:  # -5 = XX, -6 = YY
            pol_select = np.argwhere(beam_pols == -5) | np.argwhere(
                beam_pols == -6
            )
        elif -1 in beam_pols or -2 in beam_pols:  # -1 = RR, -2 = LL
            pol_select = np.argwhere(beam_pols == -1) | np.argwhere(
                beam_pols == -2
            )
        else:  # only have cross-polarized beam available, just use one
            pol_select = slice(None,1)
        cal_src = np.argsort(
            weighted_fluxes[pol_select].sum(axis=0).mean(axis=0)
        )[::-1][:Nsrc]
        src_az = src_az[cal_src]
        src_za = src_za[cal_src]
        weighted_fluxes = weighted_fluxes[:,:,cal_src].transpose(1,0,2)
        # The transpose reorders the fluxes to (freq,pol,src).

        # Now compute the fringe factors.
        wavelengths = constants.c.si.value / freq_array
        uvws = baseline_array[None] / wavelengths[:,None,None]
        src_nhat = np.array(
            [
                np.sin(src_za)*np.cos(src_az),
                np.sin(src_za)*np.sin(src_az),
                np.sin(src_za),
            ]
        )
        src_fringe = np.exp(-2j * np.pi * uvws @ src_nhat)  # (freq,bl,src)
        
        # Finally, compute the visibilities and populate the source matrix.
        src_vis = weighted_fluxes[:,:,None,:] * src_fringe[:,None,:,:]
        source_matrix[t,:,:,::2,:] = src_vis.real
        source_matrix[t,:,:,1::2,:] = src_vis.imag

    return source_matrix

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
    if not (HAVE_PYUVDATA and HAVE_PYRADIOSKY and HAVE_HEALPY):
        raise NotImplementedError(
            "pyuvdata, pyradiosky, astropy, and healpy must be installed to "
            "use the diffuse covariance model building utility functions."
        )
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
    sky_pspec: NDArray[float],
    nside: int,
    beam: UVBeam | Type[AnalyticBeam],
    freq: float,
    enu_antpos: NDArray[float],
    ant_1_array: NDArray[int],
    ant_2_array: NDArray[int],
    edges: NDArray[int],
    n_eig: int | None = 1,
    lmax: int | None = 10,
    pixwgt_datapath: str | None = None,
    bl_convention: str | None = "i-j",
) -> NDArray[float]:
    """Compute the diffuse matrix assuming we can flat sky the integrals.

    This function computes the diffuse matrix according to the discussion
    in Section 3.3 of Pascua+ 2025.

    Parameters
    ----------
    sky_pspec
        Angular power spectrum of the diffuse emission on the sky at the
        provided frequency ``freq``.
    nside
        Resolution parameter to use for computing the beam harmonics.
    freq
        Observed frequency, in Hz.
    enu_antpos
        Antenna positions in local ENU coordinates with shape (Nants, 3).
    ant_1_array, ant_2_array
        Index arrays indicating which pair of antennas are used for each
        baseline.
    edges
        Array indicating the start and end of each redundant group.
    n_eig
        Number of eigenmodes to use for each redundant block.
    lmax
        The maximum multipole to use when evaluating the sum in Equation
        60 from Pascua+ 2025.
    pixwgt_datapath
        Path to files containing the pixel weight arrays used when taking
        spherical harmonic transformations with `healpy.map2alm`.
    bl_convention
        Whether baselines are computed as x_j - x_i ("j-i" convention) or
        x_i - x_j ("i-j" convention) for baseline (i,j).

    Returns
    -------
    diff_mat
        The diffuse matrix, with shape `(N_baseline, 2*n_eig)`, evaluated
        at the provided frequency. There are `2*n_eig` columns, since
        `n_eig` modes are used for each of the real-real and imag-imag
        covariance components.

    Notes
    -----
    The antenna number arrays should be sorted by redundancy so that 
    the following is always true:
        
    .. code-block:: python

        group_num = <some integer>
        start, stop = edges[group_num:group_num+2]
        ai, aj = ant_1_array[start:stop], ant_2_array[start:stop]
        bl_vecs = enu_antpos[aj] - enu_antpos[ai]
        np.allclose(bl_vecs[0], bl_vecs)
    
    """
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
        band_power = np.sum(sky_pspec * (2*sky_ells + 1) * pspec_bessels**2)

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

        # eigh returns in ascending eigenvalue order; we  want the biggest.
        eigvals = eigvals[::-1][:2*n_eig]

        # Need to sort the real/imag separately to maintain parity.
        _eigvecs = eigvecs.copy()
        _eigvecs[:,::2] = eigvecs[:,::2][:,::-1]
        _eigvecs[:,1::2] = eigvecs[:,1::2][:,::-1]
        eigvecs = _eigvecs.copy()[:,:2*n_eig]

        # We want each 2x2 block to look like the covariance of a complex
        # number, but sometimes the eigenvector order is swapped. For a GRF,
        # the diagonal should be larger than the off-diagonal, so we ensure
        # that is always the case. It suffices to check the first two modes,
        # because these roughly represent the redundantly averaged visibility.
        if np.abs(eigvecs[0,0]) < np.abs(eigvecs[1,0]):
            _eigvecs = eigvecs.copy()
            _eigvecs[::2,::2] = eigvecs[1::2,::2]
            _eigvecs[1::2,::2] = eigvecs[::2,::2]
            eigvecs = _eigvecs.copy()

        if np.abs(eigvecs[1,1]) < np.abs(eigvecs[0,1]):
            _eigvecs = eigvecs.copy()
            _eigvecs[::2,1::2] = eigvecs[1::2,1::2]
            _eigvecs[1::2,1::2] = eigvecs[::2,1::2]
            eigvecs = _eigvecs.copy()

        # This block of the diffuse matrix is just the truncated
        # eigendecomposition of the expected covariance for this block.
        block = np.sqrt(eigvals)[None,:] * eigvecs
        diff_mat[start:stop] = np.where(np.isnan(block), 0, block)

    return diff_mat
