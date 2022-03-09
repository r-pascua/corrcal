import numpy as np
import warnings

from . import cfuncs
from . import linalg
from . import utils


def simple_chisq(gain_mat, data, noise, cov):
    """Simple chi-squared routine using a dense covariance representation.

    Parameters
    ----------
    gain_mat
        One-dimensional array containing the products of the complex gains.
    data
        The complex-valued visibility data.
    noise
        One-dimensional array containing the per-baseline thermal noise
        variance.
    cov
        Dense representation of the complex-valued covariance matrix.

    Returns
    -------
    chisq
        Calculated value of chi-squared.
    """
    scaled_cov = utils.scale_cov_by_gains(cov, gain_mat)
    diag_inds = np.arange(scaled_cov.shape[0])
    scaled_cov[diag_inds, diag_inds] += noise
    chisq = data.conj() @ np.linalg.inv(scaled_cov) @ data
    if np.abs(chisq.imag / chisq.real) > 1e-8:
        warnings.warn(
            "Imaginary part of chi-squared is not consistent with zero."
        )
    return chisq.real


def simple_nll(
    gains, data, noise, cov, ant_1_inds, ant_2_inds, norm="simple", scale=None,
):
    """Simple negative log-likelihood function.

    Parameters
    ----------
    gains
        Array of per-antenna gains. The first half of the array should contain
        the real part of the gains, while the second half of the array should
        contain the imaginary part of the gains.
    data
        Complex-valued visibility data.
    noise
        One-dimensional array containing the per-baseline noise variance.
    cov
        Dense representation of the covariance matrix.
    ant_1_inds
        Indices specifying which gain to use for the first antenna in each
        visibility.
    ant_2_inds
        Indices specifying which gain to use for the second antenna in each
        visibility.
    norm
        How to normalize the likelihood. "simple" sets the unweighted average
        gain to unity. "det" uses the usual Gaussian likelihood normalization.
    scale
        Not used in this function, but needed for the scipy minimizer.

    Returns
    -------
    nll
        Negative log-likelihood for the model parameters given the data.
    """
    gain_mat = utils.build_gain_mat(gains, ant_1_inds, ant_2_inds)
    chisq = simple_chisq(gain_mat, data, noise, cov)
    if norm == "simple":
        n_ants = gains.size // 2
        _norm = (
            gains[n_ants:].sum() ** 2 + (gains[:n_ants].sum() - n_ants) ** 2
        )
    elif norm == "det":
        scaled_cov = utils.scale_cov_by_gains(cov, gain_mat)
        try:
            _norm = (
                2 * np.log(np.diag(np.linalg.cholesky(scaled_cov)).real).sum()
            )
        except np.linalg.LinAlgError:
            warnings.warn("Covariance is not positive-definite.")
            _norm = 1e10
    else:
        raise NotImplementedError
    return chisq + _norm


def simple_grad_nll(
    gains, data, noise, cov, ant_1_inds, ant_2_inds, norm="simple", scale=1,
):
    """
    Simple function calculating the gradient of the negative log-likelihood.

    Parameters
    ----------
    gains
        Per-antenna gains. First half contains the real part, second half
        contains the imaginary part.
    data
        Complex-valued visibilities.
    noise
        Per-baseline noise variance.
    cov
        Dense representation of the covariance.
    ant_1_inds
        Index of the first antenna for each baseline.
    ant_2_inds
        Index of the second antenna for each baseline.
    norm
        Normalization scheme to use. See :func:~`simple_nll` for details.
    scale
        Scaling to apply to the calculated gradient. Intended to be used to
        prevent the minimizer from searching too broad of a space in the
        chi-squared surface.

    Returns
    -------
    grad_nll
        Gradient of the negative log-likelihood with respect to the per-antenna
        gains. The first half contains the derivatives with respect to the real
        part of the gains, while the second half contains the derivatives with
        respect to the imaginary part of the gains.

    See Also
    --------
    :func:~`simple_nll`
    """
    # Setup with the gain parameters.
    gain_mat = utils.build_gain_mat(gains, ant_1_inds, ant_2_inds)
    complex_gains = utils.build_complex_gains(gains)
    n_ants = complex_gains.size

    # Apply gains/noise.
    scaled_cov = utils.scale_cov_by_gains(cov, gain_mat)
    diag_inds = np.arange(scaled_cov.shape[0])
    scaled_cov[diag_inds, diag_inds] += noise
    cinv = np.linalg.inv(scaled_cov)
    weighted_data = cinv @ data

    # Calculate the gradient on a per-antenna basis.
    grad_nll = np.zeros_like(gains)
    for m in range(n_ants):
        for n in range(2):  # Iterate over real/imag
            # Figure out which half of the gradient to fill in.
            here = (slice(None, n_ants), slice(n_ants, None))[n]

            # Calculate the derivatives of the gain matrix and its conjugate.
            gain_mat_deriv = np.zeros_like(gain_mat)
            conj_gain_mat_deriv = np.zeros_like(gain_mat_deriv)
            delta_a1_m = ant_1_inds == m
            delta_a2_m = ant_2_inds == m
            ant_1_gains = complex_gains[ant_1_inds[delta_a2_m]]
            ant_2_gains = complex_gains[ant_2_inds[delta_a1_m]]
            if n == 0:  # Derivative w.r.t. real part of the gain for antenna m
                gain_mat_deriv[delta_a1_m] += np.conj(ant_2_gains)
                gain_mat_deriv[delta_a2_m] += ant_1_gains
                conj_gain_mat_deriv[delta_a1_m] += ant_2_gains
                conj_gain_mat_deriv[delta_a2_m] += np.conj(ant_1_gains)
            else:  # Derivative w.r.t. imag part of the gain for antenna m
                gain_mat_deriv[delta_a1_m] += 1j * np.conj(ant_2_gains)
                gain_mat_deriv[delta_a2_m] -= 1j * ant_1_gains
                conj_gain_mat_deriv[delta_a1_m] -= 1j * ant_2_gains
                conj_gain_mat_deriv[delta_a2_m] += 1j * np.conj(ant_1_gains)
            cov_deriv = linalg.diagmul(
                gain_mat_deriv, linalg.diagmul(cov, gain_mat.conj())
            )
            chisq_deriv = weighted_data.conj() @ cov_deriv @ weighted_data
            if norm == "simple":
                if n == 0:
                    _norm = 2 * (complex_gains.real.sum() - n_ants)
                else:
                    _norm = 2 * complex_gains.imag.sum()
            elif norm == "det":
                _norm = np.diag(cinv @ cov_deriv).sum()
            else:
                raise NotImplementedError
            grad_nll_here = chisq_deriv + _norm
            if np.abs(grad_nll_here.imag / grad_nll_here.real) > 1e-8:
                warnings.warn("Gradient may not be well-behaved.")
            grad_nll[here][m] = grad_nll_here.real
    return grad_nll * scale


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
