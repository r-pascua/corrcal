import numpy as np
import warnings

from . import cfuncs
from . import linalg
from . import sparse
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
    full_cov = (full_cov+full_cov.T) * 0.5
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
        Dense representation of the sky covariance.
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
    full_cov = (full_cov+full_cov.T) * 0.5
    cinv = full_cov.inv()

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
                gain_mat_grad[delta_a1_i] = np.conj(
                    complex_gains[ant_2_inds][delta_a1_i]
                )
                gain_mat_grad[delta_a2_i] = (
                    complex_gains[ant_1_inds][delta_a2_i]
                )
            else:
                gain_mat_grad[delta_a1_i] = 1j * np.conj(
                    complex_gains[ant_2_inds][delta_a1_i]
                )
                gain_mat_grad[delta_a2_i] = -1j * (
                    complex_gains[ant_1_inds][delta_a2_i]
                )
            gain_mat_grad = linalg.SplitMat(np.diag(gain_mat_grad))
            cov_grad = (
                gain_mat_grad @ cov @ gain_mat.conj()
                + gain_mat @ cov @ gain_mat_grad.conj()
            )
            grad_nll[here][i] = np.real(
                weighted_data.conj() @ cov_grad @ weighted_data
            )

            # Calculate the gradient of the normalization term.
            if norm == "simple":
                # TODO: make sure this is actually the right thing to do.
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


def nll(gains, cov, data, ant_1_inds, ant_2_inds, scale=1):
    """Calculate the negative log-likelihood.

    Fill this in later.

    Parameters
    ----------
    gains
        Double-length array of real numbers. The first half of the array gives
        the real part of the gains; the second half gives the imaginary part.
        This array is assumed to be sorted so that it can be sensibly sliced
        into with ``ant_1_inds`` and ``ant_2_inds``.
    cov
        :class:`~.sparse.SparseCov` object containing the sparse representation
        of the covaraince matrix.
    data
        Complex-valued array containing the data sorted into quasi-redundant
        groups.
    ant_1_inds
        Indices of the first antenna in each baseline in the sorted data.
    ant_2_inds
        Indices of the second antenna in each baseline in the sorted data.
    scale
        Amount that the gains were scaled prior to starting the conjugate
        gradient routine.

    Returns
    -------
    nll
        The negative log-likelihood (up to a constant offset).
    """
    n_ants = gains.size // 2
    complex_gains = gains[:n_ants] + 1j*gains[n_ants:]
    complex_gains /= scale
    cov.gains = complex_gains[ant_1_inds] * complex_gains[ant_2_inds].conj()
    cinv, logdet = cov.inv(return_det=True)
    chisq = data.conj() @ cinv @ data
    if np.abs(chisq.imag / chisq.real) > 1e-8:
        warnings.warn("Chi-squared isn't purely real!")
    return np.real(chisq + logdet)


def grad_nll(gains, cov, data, ant_1_inds, ant_2_inds, scale=1):
    """Calculate the gradient of the negative log-likelihood.

    This is the gradient with respect to the real/imaginary per-antenna gains.
    See Eq. ?? of Pascua+ 22 for details of what is being calculated.

    Parameters
    ----------
    same as nll. fill this out later.
    """
    # Prepare the gain matrix.
    n_ants = gains.size // 2
    complex_gains = gains[:n_ants] + 1j*gains[n_ants:]
    complex_gains /= scale
    cov.gains = complex_gains[ant_1_inds] * complex_gains[ant_2_inds].conj()
    
    # Prepare some auxiliary matrices/vectors.
    cinv = cov.inv()
    wgted_data = cinv @ data
    src_rhs = cov.src_mat.T.conj() * cov.gains[None,:].conj()
    diff_rhs = cov.diff_mat.T.conj() * cov.gains[None,:].conj()

    # Initialize important arrays for the gradient calculation.
    gradient = np.zeros(2*n_ants, dtype=float)
    gain_mat_grad = np.zeros_like(cov.gains)
    cov_grad = np.zeros_like(cinv)

    # Calculate the gradient antenna by antenna.
    # TODO: see if this is the fastest way to calculate the gradient.
    for ant in range(n_ants):
        for i, sl in enumerate((slice(None,n_ants), slice(n_ants,None))):
            delta_ant1 = ant_1_inds == ant
            delta_ant2 = ant_2_inds == ant
            gain_mat_grad[:] = 0
            if i == 0:  # Gradient w.r.t. real part of gains
                gain_mat_grad[delta_ant1] += gains[ant_2_inds][delta_ant1].conj()
                gain_mat_grad[delta_ant2] += gains[ant_1_inds][delta_ant2]
            else:
                gain_mat_grad[delta_ant1] += 1j*gains[ant_2_inds][delta_ant1].conj()
                gain_mat_grad[delta_ant2] -= 1j*gains[ant_1_inds][delta_ant2]

            # Calculate Eq. ?? from Pascua+ 22.
            cov_grad[...] = 0
            cov_grad += (gain_mat_grad[:,None] * cov.src_mat) @ src_rhs
            cov_grad += (gain_mat_grad[:,None] * cov.diff_mat) @ diff_rhs
            cov_grad += cov_grad.T.conj()

            # Calculate the gradient of the normalization, but do it fast.
            grad_norm = np.sum(cinv * cov_grad.T)
            grad_chisq = -wgted_data.conj() @ cov_grad @ wgted_data
            grad = grad_norm + grad_chisq

            if np.abs(grad.imag / grad.real) > 1e-8:
                warnings.warn("Gradient isn't purely real!")
            gradient[sl][ant] = grad.real
    return gradient / scale
