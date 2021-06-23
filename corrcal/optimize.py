import numpy as np

from . import cfuncs


def get_chisq(gains, data, cov, ant1, ant2, scale=1, norm=1):
    """Calculate chi-squared using a Sparse2Level object.

    We define chi-squared as .. math::

        \\chi^2 = d^\\dagger \\bigl(N + S\\bigr)^{-1} d

    < fill out the rest of this later >

    Parameters
    ----------
    gains: np.ndarray
        Complex gains, separated into real and imaginary parts.
        < add a note about indexing, etc >
    data: np.ndarray
        Raw visibility data.
    cov: :class:`~.Sparse2Level`
        Sparse covariance matrix.
    ant1: np.ndarray of int
        < fill out later >
    ant2: np.ndarray of int
        < fill out later >
    scale: float, optional
        Amount to scale the gains by (to help with convergence?).
    norm: float, optional
        Factor setting how strongly the amplitude of the gains affects
        the value for chi-squared.

    Returns
    -------
    chisq: float
        The calculated chi-squared.
    """
    gains /= scale
    cov = cov.copy()
    cov.apply_gains(gains, ant1, ant2)
    inverse_cov = cov.inverse()
    raw_chisq = data @ (inverse_cov * data)
    norm *= (
        np.sum(gains[1::2]) ** 2 + (np.sum(gains[::2]) - gains.size / 2) ** 2
    )
    return raw_chisq + norm


def get_chisq_dense(gains, data, noise, sky_cov, ant1, ant2, scale=1, norm=1):
    """Calculate chi-squared from the dense covariance.

    Parameters
    ----------
    gains: np.ndarray of float
        Diagonal of the gain matrix?
    data: np.ndarray of float
        Raw visibility data, split into real and imaginary parts.
        Even indices correspond to real component, while odd indices
        correspond to imaginary component.
    noise: np.ndarray of float
        Diagonal of the noise variance.
    sky_cov: np.ndarray of float
        Dense sky covariance matrix.
    ant1: np.ndarray of int
        Array denoting antenna 1 in the visibility.
    ant2: np.ndarray of int
        Array denoting antenna 2 in the visibility.
    scale: float, optional
        Scale factor applied to the gains (for numerical stability?
        convergence?).
    norm: float, optional
        Factor dictating how much the gain amplitude contributes to
        the chi-squared calculation.

    Returns
    -------
    chisq: float
        The calculated chi-squared.
    """
    gains /= scale
    sky_cov = sky_cov.copy()  # To be safe with the covariance.
    if sky_cov.shape[0] != sky_cov.shape[1]:
        raise ValueError("Sky covariance must be square.")
    Nbls = sky_cov.shape[0]

    # Apply the gains to the sky covariance.
    cfuncs.apply_gains_to_matrix(
        sky_cov.ctypes.data,
        gains.ctypes.data,
        ant1.ctypes.data,
        ant2.ctypes.data,
        Nbls // 2,
        Nbls,
    )
    sky_cov = sky_cov.T
    cfuncs.apply_gains_to_matrix(
        sky_cov.ctypes.data,
        gains.ctypes.data,
        ant1.ctypes.data,
        ant2.ctypes.data,
        Nbls // 2,
        Nbls,
    )

    # Add in the noise.
    sky_cov = sky_cov.T + noise
    # Force the covariance to be symmetric (seems a little sus?)
    sky_cov = 0.5 * (sky_cov + sky_cov.T)
    raw_chisq = data @ sky_cov @ data
    norm *= (
        np.sum(gains[1::2]) ** 2 + (np.sum(gains[::2]) - gains.size // 2) ** 2
    )
    return raw_chisq + norm
