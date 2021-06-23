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


def chisq_gradient(gains, data, cov, ant1, ant2, scale=1, norm=1):
    """Calculate the gradient of chi-squared.

    Parameters
    ----------
    Same as chi-squared params. Update this later.

    Returns
    -------
    gradient: np.ndarray
        Gradient of chi-squared with respect to antenna gains.
    """
    gains /= scale
    cov = cov.copy()
    cov.apply_gains(gains, ant1, ant2)
    inverse_cov = cov.inverse()
    # TODO: figure out better names for this... just work out the math
    # then figure out physically meaningful ways to talk about it.
    sd = inverse_cov * data
    gsd = sd.copy()
    cfuncs.apply_gains_to_matrix(
        gsd.ctypes.data,
        gains.ctypes.data,
        ant2.ctypes.data,
        ant1.ctypes.data,
        gsd.size // 2,
        1,
    )
    # This is a little bit of a hack to save some memory.
    cgsd = cov.copy()
    cgsd.noise_variance = np.zeros_like(cgsd.noise_variance)
    cgsd *= gsd

    Nant = max([ant1.max(), ant2.max()]) + 1
    gradient = np.zeros(2 * Nant, dtype=float)

    # Extract the gains for ease of reference.
    re_g1 = gains[2 * ant1]
    re_g2 = gains[2 * ant2]
    im_g1 = gains[2 * ant1 + 1]
    im_g2 = gains[2 * ant2 + 1]

    # This is a term in the gradient... Make the code nicer.
    # This is a bit faster than Jon's implementation, especially
    # for large arrays.
    v1_m1r_v2 = np.empty_like(cgsd)
    v1_m1i_v2 = np.empty_like(cgsd)
    v1_m2r_v2 = np.empty_like(cgsd)
    v1_m2i_v2 = np.empty_like(cgsd)

    # Real/imaginary slices for notational convenience/clarity.
    re = slice(0, None, 2)  # Equivalent to [0::2]
    im = slice(1, None, 2)  # Equivalent to [1::2]

    # Populate the vectors previously initialized.
    v1_m1r_v2[re] = re_g2 * sd[re] - im_g2 * sd[im]
    v1_m1r_v2[im] = im_g2 * sd[re] + re_g2 * sd[im]
    v1_m1i_v2[re] = im_g2 * sd[re] + re_g2 * sd[im]
    v1_m1i_v2[im] = -re_g2 * sd[re] + im_g2 * sd[im]

    v1_m2r_v2[re] = re_g1 * sd[re] + im_g1 * sd[im]
    v1_m2r_v2[im] = -im_g1 * sd[re] + re_g1 * sd[im]
    v1_m2i_v2[re] = im_g1 * sd[re] - re_g1 * sd[im]
    v1_m2i_v2[im] = re_g1 * sd[re] + im_g1 * sd[im]

    # Unpack this math to make it make sense.
    v1_m1r_v2 *= cgsd
    v1_m1i_v2 *= cgsd
    v1_m2r_v2 *= cgsd
    v1_m2i_v2 *= cgsd

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
    grad_real = 2 * (np.sum(gains[re]) - gains.size / 2) / (gains.size / 2)
    grad_im = 2 * np.sum(gains[im])
    return -2 * gradient / scale + norm * (grad_real + grad_im) / scale
