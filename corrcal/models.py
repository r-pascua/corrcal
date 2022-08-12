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
