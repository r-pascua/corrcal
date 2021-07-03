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


def get_chisq_gradient(gains, data, cov, ant1, ant2, scale=1, norm=1):
    """Calculate the gradient of chi-squared with respect to antenna gains.

    This function calculates the chi-squared gradient according to .. math::

        \\nabla \\chi^2 = 2q^T H' p + {\\rm gain normalization},

    where :math:`H` is the matrix of inverse gains, the prime denotes the
    derivative with respect to the antenna gains, and the vectors :math:`p`
    and :math:`q` are defined as: .. math::

        p \\equiv \\bigl(N + H^T C H\\bigr)^{-1} d,
        q \\equiv C H p.

    In the above equations, :math:`N` is the (diagonal) matrix giving the
    per-baseline thermal noise variance, :math:`C` is the baseline-baseline
    covariance one would measure in the absence of noise, and :math:`d` is
    the visibility data from the instrument. This is Equation 4 in Sievers
    2017. The normalization term prevents chi-squared from setting the
    gains arbitrarily high or low.

    Parameters
    ----------
    Same as chi-squared params. Update this later.

    Returns
    -------
    gradient: np.ndarray
        Gradient of chi-squared with respect to antenna gains.
    """
    # Apply the gains to the covariance and take the inverse.
    gains /= scale
    cov = cov.copy()
    cov.apply_gains(gains, ant1, ant2)
    inverse_cov = cov.inverse()

    # Calculate the p and q vectors from Sievers 2017 (Equation 4).
    pvec = inverse_cov * data
    inv_gain_times_pvec = pvec.copy()
    cfuncs.apply_gains_to_matrix(
        inv_gain_times_pvec.ctypes.data,
        gains.ctypes.data,
        ant2.ctypes.data,
        ant1.ctypes.data,
        inv_gain_times_pvec.size // 2,
        1,
    )
    # This is a little bit of a hack to save some memory.
    qvec = cov.copy()  # This isn't a vector yet, but...
    qvec.noise_variance = np.zeros_like(qvec.noise_variance)
    qvec *= inv_gain_times_pvec  # this gives a *vector*.

    return _calculate_gradient(gains, pvec, qvec, ant1, ant2, scale, norm)


def _calculate_gradient(gains, pvec, qvec, ant1, ant2, scale=1, norm=1):
    """Calculate Equation 4 of Sievers 2017.

    See :func:`get_chisq_gradient` for a brief discussion of the terms.
    """
    Nant = len(set(ant1).union(set(ant2)))
    gradient = np.zeros(2 * Nant, dtype=float)

    # Extract the gains for ease of reference.
    re_g1 = gains[2 * ant1]
    re_g2 = gains[2 * ant2]
    im_g1 = gains[2 * ant1 + 1]
    im_g2 = gains[2 * ant2 + 1]

    # Initialize the various derivatives. This method is a bit faster
    # than Jon's implementation, especially for large arrays.
    v1_m1r_v2 = np.empty_like(qvec)
    v1_m1i_v2 = np.empty_like(qvec)
    v1_m2r_v2 = np.empty_like(qvec)
    v1_m2i_v2 = np.empty_like(qvec)

    # Real/imaginary slices for notational convenience/clarity.
    re = slice(0, None, 2)  # Equivalent to [0::2]
    im = slice(1, None, 2)  # Equivalent to [1::2]

    # Populate the vectors previously initialized.
    v1_m1r_v2[re] = re_g2 * pvec[re] - im_g2 * pvec[im]
    v1_m1r_v2[im] = im_g2 * pvec[re] + re_g2 * pvec[im]
    v1_m1i_v2[re] = im_g2 * pvec[re] + re_g2 * pvec[im]
    v1_m1i_v2[im] = -re_g2 * pvec[re] + im_g2 * pvec[im]

    v1_m2r_v2[re] = re_g1 * pvec[re] + im_g1 * pvec[im]
    v1_m2r_v2[im] = -im_g1 * pvec[re] + re_g1 * pvec[im]
    v1_m2i_v2[re] = im_g1 * pvec[re] - re_g1 * pvec[im]
    v1_m2i_v2[im] = re_g1 * pvec[re] + im_g1 * pvec[im]

    # TODO: Figure out how to unpack this math to make it make sense.
    v1_m1r_v2 *= qvec
    v1_m1i_v2 *= qvec
    v1_m2r_v2 *= qvec
    v1_m2i_v2 *= qvec

    # Sum the real and imaginary parts for some reason?
    v1_m1r_v2 = v1_m1r_v2[re] + v1_m1r_v2[im]
    v1_m1i_v2 = v1_m1i_v2[re] + v1_m1i_v2[im]
    v1_m2r_v2 = v1_m2r_v2[re] + v1_m2r_v2[im]
    v1_m2i_v2 = v1_m2i_v2[re] + v1_m2i_v2[im]

    cfuncs.sum_grads_c(
        gradient.ctypes.data,
        v1_m1r_v2.ctypes.data,
        v1_m1i_v2.ctypes.data,
        ant1.ctypes.data,
        v1_m2i_v2.size,
    )
    cfuncs.sum_grads_c(
        gradient.ctypes.data,
        v1_m2r_v2.ctypes.data,
        v1_m2i_v2.ctypes.data,
        ant2.ctyptes.data,
        v1_m2i_v2.size,
    )

    # Finish calculating the gradient.
    gain_norm_real = (
        2 * (np.sum(gains[re]) - gains.size / 2) / (gains.size / 2)
    )
    gain_norm_imag = 2 * np.sum(gains[im])
    return (-2 * gradient + norm * (gain_norm_real + gain_norm_imag)) / scale


def get_chisq_gradient_dense(
    gains, data, noise, sky_cov, ant1, ant2, scale=1, norm=1
):
    """Calculate chi-squared for the dense covariance.

    This function uses the actual thermal noise and sky covariance matrices
    directly, instead of using a :class:`~.Sparse2Level` object as is done
    in :func:`get_chisq_gradient`.

    Parameters
    ----------
    Same as :func:`get_chisq_dense`. Fill in later.

    Returns
    -------
    chisq_gradient
        Fill this in later.

    See Also
    --------
    :func:`get_chisq_gradient`
    """
    # Setup for applying gains to sky covariance, but don't do it in-place.
    gains = gains / scale
    cov = sky_cov.copy()
    Nbls = sky_cov.shape[0]

    # Calculate H^T C H to compute the p-vector from Eq'n 4 of Sievers 2017.
    # TODO: make sure the gain convention is correct.
    cfuncs.apply_gains_to_matrix(
        cov.ctypes.data,
        gains.ctypes.data,
        ant1.ctypes.data,
        ant2.ctypes.data,
        Nbls // 2,
        Nbls,
    )
    cov = cov.T
    cfuncs.apply_gains_to_matrix(
        cov.ctypes.data,
        gains.ctypes.data,
        ant1.ctypes.data,
        ant2.ctypes.data,
        Nbls // 2,
        Nbls,
    )

    # Complete the covariance matrix and calculate p.
    cov += noise
    cov = 0.5 * (cov + cov.T)  # Enforce symmetry just in case...
    pvec = np.linalg.inv(cov) @ data

    # Now calculate the q-vector from Equation 4, q = C H p.
    qvec = pvec.copy()
    cfuncs.apply_gains_to_matrix(
        qvec.ctypes.data,
        gains.ctypes.data,
        ant2.ctypes.data,
        ant1.ctypes.data,
        Nbls // 2,
        1,
    )
    qvec = cov @ qvec
    return _calculate_gradient(gains, pvec, qvec, ant1, ant2, scale, norm)
