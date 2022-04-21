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
    scaled_cov = 0.5*(scaled_cov + scaled_cov.T)
    chisq = data.conj() @ np.linalg.inv(scaled_cov) @ data
    if np.abs(chisq.imag / chisq.real) > 1e-8:
        warnings.warn(
            "Imaginary part of chi-squared is not consistent with zero."
        )
    return chisq.real


def dense_nll(
    gains: np.ndarray,
    data: linalg.SplitVec,
    noise: linalg.SplitMat,
    cov: linalg.SplitMat,
    ant_1_inds: np.ndarray,
    ant_2_inds: np.ndarray,
    norm: str = "simple",
    gain_scale: complex = 1,
    grad_scale: None = None,
) -> float:
    """Calculate the negative log-likelihood using dense covariance.

    Parameters
    ----------
    gains
        Array of per-antenna gains. The first half of the array should contain
        the real part of the gains, while the second half of the array should
        contain the imaginary part of the gains.
    data
        Visibilities to calibrate.
    noise
        Per-baseline variance due to thermal noise.
    cov
        Modeled sky covariance.
    ant_1_inds
        Indices specifying which gain to use for the first antenna in each
        visibility.
    ant_2_inds
        Indices specifying which gain to use for the second antenna in each
        visibility.
    norm
        How to normalize the likelihood. "simple" sets the unweighted average
        gain to unity. "det" uses the usual Gaussian likelihood normalization.
    gain_scale
        Per-antenna gain scaling to be removed before calculating the likelihood.
    grad_scale
        Not used in this function, but needed for the scipy minimizer.

    Returns
    -------
    nll
        Negative log-likelihood for the model parameters given the data.
    """
    n_ants = gains.size // 2
    complex_gains = gains[:n_ants] + 1j*gains[n_ants:]
    complex_gains /= gain_scale
    gain_mat = gains[ant_1_inds] * gains[ant_2_inds].conj()
    gain_mat = linalg.SplitMat(np.diag(gain_mat))
    full_cov = noise + gain_mat @ cov @ gain_mat.conj()
    full_cov = 0.5 * (full_cov+full_cov.T)
    chisq = data.conj() @ full_cov.inv() @ data
    if np.abs(chisq.imag / chisq.real) > 1e-8:
        warnings.warn("Chi-squared isn't purely real.")
    chisq = chisq.real
    if norm == "simple":
        _norm = np.sum(gains.imag)**2 + np.sum(gains.real-1)**2
    elif norm == "det":
        try:
            L = np.linalg.cholesky(full_cov.data)
            _norm = 2 * np.log(np.diag(L)).sum()
        except np.linalg.LinAlgError:
            _norm = 0.5 * np.log(np.linalg.det(cov.data))
    else:
        _norm = 0
    return chisq + _norm


def dense_grad_nll(
    gains: np.ndarray,
    data: linalg.SplitVec,
    noise: linalg.SplitMat,
    cov: linalg.SplitMat,
    ant_1_inds: np.ndarray,
    ant_2_inds: np.ndarray,
    norm: str = "simple",
    gain_scale: complex = 1,
    grad_scale: float = 1,
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
    gain_scale
        Amount by which the gains have been pre-scaled.
    grad_scale
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
    :func:~`dense_nll`
    """
    # Get some array lengths for proper bookkeeping.
    n_bls = ant_1_inds.size
    n_ants = gains.size // 2

    # Setup the gain matrix.
    complex_gains = gains[:n_ants] + 1j*gains[n_ants:]
    complex_gains /= gain_scale
    gain_mat = complex_gains[ant_1_inds] * complex_gains[ant_2_inds].conj()
    gain_mat = linalg.SplitMat(np.diag(gain_mat))

    # Construct the full covariance.
    full_cov = noise + gain_mat @ cov @ gain_mat.conj()
    full_cov = 0.5*(full_cov + full_cov.T)
    cinv = np.linalg.inv(scaled_cov)

    # Get some auxilliary parameters required for the calculation.
    re_gain = complex_gains.real
    im_gain = complex_gains.imag
    weighted_data = cinv @ data

    # Calculate the gradient on a per-antenna basis.
    grad_nll = np.zeros_like(gains)
    _norm = 0
    for i in range(n_ants):
        for k in range(2):
            # Determine whether we're looking at the real or imaginary part.
            here = (slice(None, n_ants), slice(n_ants, None))[k]

            # Figure out which visibilities contain this antenna.
            delta_a1_i = ant_1_inds == i
            delta_a2_i = ant_2_inds == i

            # Calculate the derivative of the gain matrix.
            gain_mat_grad = np.zeros(n_bls, dtype=complex)
            if k == 0:
                gain_mat_grad[delta_a1_i] = (
                    re_gain[ant_2_inds][delta_a1_i]
                    - 1j*im_gain[ant_2_inds][delta_a1_i]
                )
                gain_mat_grad[delta_a2_i] = (
                    re_gain[ant_1_inds][delta_a2_i]
                    + 1j*im_gain[ant_1_inds][delta_a2_i]
                )
            else:
                gain_mat_grad[delta_a1_i] = (
                    im_gain[ant_2_inds][delta_a1_i]
                    + 1j*re_gain[ant_2_inds][delta_a1_i]
                )
                gain_mat_grad[delta_a2_i] = (
                    im_gain[ant_1_inds][delta_a2_i]
                    - 1j*re_gain[ant_1_inds][delta_a2_i]
                )
            gain_mat_grad = linalg.SplitMat(np.diag(gain_mat_grad))
            cov_grad = (
                gain_mat_grad @ full_cov @ gain_mat.conj()
                + gain_mat @ cov @ gain_mat_grad.conj()
            )
            grad_nll[here][i] = np.real(
                weighted_data.conj() @ cov_grad @ weighted_data
            )

            # Calculate the gradient of the normalization term.
            if norm == "simple":
                # TOOD: make sure this is actually the right thing to do.
                if k == 0:
                    _norm += 2 * (re_gain.sum()-n_ants) / n_ants
                else:
                    _norm += 2 * im_gain.sum()
            elif norm == "det":
                _norm = cinv @ cov_grad
                _norm = np.diag(_norm.real).sum() + np.diag(_norm.imag).sum()
                grad_nll[here][i] -= _norm

    if norm == "simple":
        # TODO: see note above.
        grad_nll -= _norm
        
    # Get the sign and scaling right for the gradient.
    grad_nll = grad_nll[:n_ants] + 1j*grad_nll[n_ants:]
    grad_nll *= -grad_scale / gain_scale
    return linalg.SplitVec(grad_nll).data


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
