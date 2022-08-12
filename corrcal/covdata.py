"""Container for covariance matrices used in Correlation Calibration."""
import numpy as np
from pyuvdata.uvbase import UVBase
import pyuvdata.parameter as uvp
from pyuvdata.parameter import UVParameter


class UVCov(UVBase):
    """
    A class for interfacing with CorrCal-style covariance matrices.

    Supports reading covh5 file types. Legacy support for binary files used in
    ``corrcal2``.

    Attributes
    ----------
    ``pyuvdata.UVParameter`` objects
        < fill this out later >
    """

    def __init__(self):
        # Basic counting parameters
        self._Ntimes = UVParameter(
            "Ntimes", description="Number of times.", expected_type=int
        )
        self._Nbls = UVParameter(
            "Nbls", description="Number of baselines.", expected_type=int
        )
        self._Nfreqs = UVParameter(
            "Nfreqs", description="Number of frequencies.", expected_type=int
        )
        # TODO: figure out if it's possible to write CorrCal for BDA data.
        self._Npols = UVParameter(
            "Npols", description="Number of polarizations.", expected_type=int
        )
        self._Nsrc = UVParameter(
            "Nsrc",
            description="Number of point-source terms.",
            expected_type=int,
        )
        self._Nmodes = UVParameter(
            "Nmodes",
            description="Number of eigenmodes used to describe each redundant group.",
            expected_type=int,
        )
        self._Nblocks = UVParameter(
            "Nblocks",
            description="Number of quasi-redundant groups.",
            expected_type=int,
        )

        # Start array-like parameters
        desc = (
            "Array of point-source terms that contribute to the covariance. "
            "Has shape (Ntimes, Nfreqs, Npols, Nbls, Nsrc), is type complex, "
            "in units of Jy. See Eq. X in <paper link placeholder>."
        )
        self._sources = UVParameter(
            "sources",
            description=desc,
            form=("Ntimes", "Nfreqs", "Npols", "Nbls", "Nsrc"),
            expected_type=complex,
        )

        desc = (
            "Array of 'diffuse' or 'redundant' terms that contribute to the "
            "covariance. Has shape (Ntimes, Nfreqs, Npols, Nbls, Nmodes), "
            "is type complex float, in units of Jy. In particular, this array "
            "specifies the eigenvalues for each quasi-redundant group at each "
            "time, frequency, and polarization. See Eq. X in <paper>."
        )
        # Here, each quasi-redundant block is being treated as though it can
        # be written as the outer product of a shape (group_size, Nmodes)
        # matrix with its transpose (or Hermitian conjugate).
        self._eigvals = UVParameter(
            "eigvals",
            description=desc,
            form=("Ntimes", "Nfreqs", "Npols", "Nbls", "Nmodes"),
            expected_type=complex,
        )

        desc = (
            "Array of eigenvectors describing each quasi-redundant block in "
            "the 'diffuse' or 'redundant' terms. This should be a block-diagonal "
            "matrix with each block giving the eigenvectors describing that "
            "block. Has shape (Ntimes, Nfreqs, Npols, Nbls, Nmodes), is type "
            "complex float, and is dimensionless. See Eq. X in <paper>."
        )
        self._modes = UVParameter(
            "modes",
            description=desc,
            form=("Ntimes", "Nfreqs", "Npols", "Nbls", "Nmodes"),
            expected_type=complex,
        )

        # Noise will be shaped like [time, freq, pol, bl]

        # notes: define the object first, and let the object determine how
        # the filetype is written. axes most frequently sliced over should
        # be the last axes. prioritize shapes for operations with worst
        # scaling.
