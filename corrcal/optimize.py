import numpy as np
import warnings

from . import _cfuncs
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
    scaled_cov = 0.5 * (scaled_cov + scaled_cov.T)
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
        Per-antenna gain scaling to be divided out before the calculation.
    grad_scale
        Not used in this function, but needed for the scipy minimizer.

    Returns
    -------
    nll
        Negative log-likelihood for the model parameters given the data.
    """
    n_ants = gains.size // 2
    complex_gains = gains[:n_ants] + 1j * gains[n_ants:]
    complex_gains /= gain_scale
    gain_mat = gains[ant_1_inds] * gains[ant_2_inds].conj()
    gain_mat = linalg.SplitMat(np.diag(gain_mat))
    full_cov = noise + gain_mat @ cov @ gain_mat.conj()
    full_cov = (full_cov + full_cov.T) * 0.5
    chisq = data.conj() @ full_cov.inv() @ data
    if np.abs(chisq.imag / chisq.real) > 1e-8:
        warnings.warn("Chi-squared isn't purely real.")
    chisq = chisq.real
    if norm == "simple":
        _norm = np.sum(gains.imag) ** 2 + np.sum(gains.real - 1) ** 2
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
    complex_gains = gains[:n_ants] + 1j * gains[n_ants:]
    complex_gains /= gain_scale
    gain_mat = complex_gains[ant_1_inds] * complex_gains[ant_2_inds].conj()
    gain_mat = linalg.SplitMat(np.diag(gain_mat))

    # Construct the full covariance.
    full_cov = noise + gain_mat @ cov @ gain_mat.conj()
    full_cov = (full_cov + full_cov.T) * 0.5
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
                gain_mat_grad[delta_a2_i] = complex_gains[ant_1_inds][
                    delta_a2_i
                ]
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
                    _norm += 2 * (re_gain.sum() - n_ants) / n_ants
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
    grad_nll = grad_nll[:n_ants] + 1j * grad_nll[n_ants:]
    grad_nll *= -grad_scale / gain_scale
    return linalg.SplitVec(grad_nll).data


def nll(gains, cov, data, ant_1_inds, ant_2_inds, scale=1, phs_norm_fac=np.inf):
    """Calculate the negative log-likelihood.

    TODO: Improve the docs at some point.

    Parameters
    ----------
    gains
        Per-antenna gains, alternating real/imaginary so that the even indices
        correspond to the real part and the odd indices correspond to the
        imaginary part.
    cov
        Covariance matrix with inversion methods and methods for building
        gain matrices.
    data
        Real-valued array containing the data sorted into quasi-redundant
        groups in alternating real/imaginary format.
    ant_1_inds
        Indices of the first antenna in each baseline in the sorted data.
    ant_2_inds
        Indices of the second antenna in each baseline in the sorted data.
    scale
        Amount that the gains were scaled prior to starting the conjugate
        gradient routine.
    phs_norm_fac
        Scale of the Gaussian prior placed on the average gain phase. Default
        is to not apply a prior to the gain phases.

    Returns
    -------
    nll
        The negative log-likelihood (up to a constant offset).
    """
    cov = cov.copy()
    cov.apply_gains(gains / scale, ant_1_inds, ant_2_inds)
    cinv, logdet = cov.inv(return_det=True)
    chisq = data @ (cinv @ data)

    # Use a Gaussian prior that the average phase should be nearly zero
    phases = np.arctan2(gains[1::2], gains[::2])
    phs_norm = np.sum(phases)**2 / phs_norm_fac**2
    return np.real(chisq) + logdet + phs_norm


def grad_nll(gains, cov, data, ant_1_inds, ant_2_inds, scale=1, phs_norm_fac=np.inf):
    """Calculate the gradient of the negative log-likelihood.

    This is the gradient with respect to the real/imaginary per-antenna gains.
    See Eq. ?? of Pascua+ 25 for details of what is being calculated.

    Parameters
    ----------
    same as nll. fill this out later.
    """
    # Prepare the gain matrix.
    gains = gains / scale
    complex_gains = gains[::2] + 1j*gains[1::2]
    gain_mat = complex_gains[ant_1_inds] * complex_gains[ant_2_inds].conj()

    # Prepare some auxiliary matrices/vectors.
    cinv = cov.copy()
    cinv.apply_gains(gains/scale, ant_1_inds, ant_2_inds)
    cinv = cinv.inv(return_det=False)
    p = cinv @ data
    cov = cov.copy()
    cov.noise = np.zeros_like(cov.noise)
    
    # Compute p = (C-N) @ G^T @ p.
    q = p.copy()
    q[::2] = gain_mat.real * p[::2] + gain_mat.imag * p[1::2]
    q[1::2] = gain_mat.imag * p[::2] - gain_mat.real * p[1::2]
    q = cov @ q

    # Now compute s = Re(q^\dag @ p), t = Im(q^\dag @ p).
    s = p[::2]*q[::2] + p[1::2]*q[1::2]
    t = p[1::2]*q[::2] - p[::2]*q[1::2]
    grad_chisq = linalg.accumulate_grad_chisq(
        gains, s, t, ant_1_inds, ant_2_inds
    )

    # Compute the contribution from the gradient of the determinant.
    grad_logdet = np.zeros_like(gains)

    # Accumulate the contributions from the phase normalization.
    amps = np.sqrt(gains[::2]**2 + gains[1::2]**2)
    phases = np.arctan2(gains[1::2], gains[::2])
    n_ants = complex_gains.size
    grad_phs_prefac = 2 * phases / (n_ants * phs_norm_fac**2)
    grad_phs = np.zeros_like(gains)
    grad_phs[::2] = -grad_phs_prefac * np.sin(phases)
    grad_phs[1::2] = grad_phs_prefac * np.cos(phases)


    return (0.5*grad_logdet + grad_chisq + grad_phs) / scale


def compute_trace(cov, inv_cov, gain_mat, grad_gain):
    r"""Helper function for the gradient routine.

    This function computes the trace of :math:`$C^{-1} \partial C$`. After
    exploiting trace properties, we're left with the following quantities to
    compute:

    .. math::

        \begin{align}
            {\rm Tr}(N^{-1} \partial G \Delta \Delta^\dag G^\dag) &= \sum_{i,m}
            N^{-1}_{ii} \partial G_{ii} G^{*}_{ii} |\Delta_{im}|^2, \\
            
            {\rm Tr}(N^{-1} \partial G \Sigma \Sigma^\dag G^\dag) &= \sum_{i,j}
            N^{-1}_{ii} \partial G_{ii} G^{*}_{ii} |\Sigma_{ij}|^2, \\
            
            {\rm Tr}(\Delta' \Delta'^\dag \partial G \Delta \Delta^\dag
            G^\dag) &= \sum_{m,n} (\Delta'^\dag \partial G \Delta)_{mn}
            (\Delta^\dag G^\dag \Delta')_{nm}, \\

            {\rm Tr}(\Sigma' \Sigma'^\dag \partial G \Sigma \Sigma^\dag
            G^\dag) &= \sum_{i,j} (\Sigma'^\dag \partial G \Sigma)_{ij}
            (\Sigma^\dag G^\dag \Sigma')_{ji}, \\

            {\rm Tr}(\Delta' \Delta'^\dag \partial G \Sigma \Sigma^\dag
            G^\dag) &= \sum_{m,j} (\Delta'^\dag \partial G \Sigma)_{mj}
            (\Sigma^\dag G^\dag \Delta')_{jm}, \\

            {\rm Tr}(\Sigma' \Sigma'^\dag \partial G \Delta \Delta^\dag
            G^\dag) &= \sum_{j,m} (\Sigma'^\dag \partial G \Delta)_{jm}
            (\Delta^\dag G^\dag \Sigma')_{mj},
        \end{align}

    where :math:`C^{-1} = N^{-1} - \Delta'\Delta'^\dag - \Sigma'\Sigma'^\dag`
    (with the gains already applied prior to inversion), :math:`\Delta` is the
    diffuse matrix, :math:`\Sigma` is the source matrix, :math:`N` is the noise
    matrix, and :math:`G` is the gain matrix. The terms in the trace are
    written this way to motivate how they can be computed efficiently: the
    first two terms are straightforward to compute quickly; the other terms can
    all be computed with C-routines that are minor modifications of routines
    used in the inversion routine.

    Parameters
    ----------
    cov
        Sparse covariance object.
    inv_cov
        Sparse inverse covariance object.
    gain_mat
        Diagonal of the gain matrix.
    grad_gain
        Diagonal of the derivative of the gain matrix.

    Returns
    -------
    trace
        Trace of the product of the inverse covariance and the derivative of
        the covariance.
    """
    # Compute the first two terms.
    tmp = inv_cov.noise * gain_mat.conj() * grad_gain
    trace = tmp @ np.sum(np.abs(cov.diff_mat)**2, axis=1)
    trace += tmp @ np.sum(np.abs(cov.src_mat)**2, axis=1)

    # Compute the auxiliary terms.
    dG_Delta = grad_gain[:,None] * cov.diff_mat
    dG_Sigma = grad_gain[:,None] * cov.src_mat
    G_Delta = gain_mat[:,None] * cov.diff_mat
    G_Sigma = gain_mat[:,None] * cov.src_mat

    # TODO: write an alternate routine for computing the trace, but stick
    # it in the gradient routine. Since the gradient is diagonal, we actually
    # only need the diagonal entries from the (Nbl,Nbl) matrices in order to
    # compute the trace. There should be a reasonably fast way of computing
    # these--something that goes like O(Nbl*Neig**2) (i.e., same scaling as
    # the likelihood evaluation)
    # i.e. tr(dG D (D^\dag G^\dag D') D'^\dag)
    # Compute the third term: tr(D'^\dag dG D D^\dag G^\dag D')
    tmp = linalg.mult_diff_mats(
        inv_cov.diff_mat.T.conj(), dG_Delta, cov.edges
    )
    tmp2 = linalg.mult_diff_mats(G_Delta.T.conj(), inv_cov.diff_mat, cov.edges)
    trace -= np.sum(tmp * tmp2.transpose(0,2,1))

    # Compute the fourth term: tr(S'^\dag dG S S^\dag G^\dag S')
    tmp = inv_cov.src_mat.T.conj() @ dG_Sigma
    tmp2 = inv_cov.src_mat.T.conj() @ G_Sigma
    trace -= np.sum(tmp * tmp2.conj())

    # Compute the fifth term: tr(D'^\dag dG S S^\dag G^\dag D')
    tmp = linalg.mult_src_by_blocks(
        inv_cov.diff_mat.T.conj(), dG_Sigma, cov.edges
    )
    tmp2 = linalg.mult_src_by_blocks(
        inv_cov.diff_mat.T.conj(), G_Sigma, cov.edges
    ).conj()
    trace -= np.sum(tmp * tmp2)

    # Compute the last term: tr(D^\dag G^\dag S' S'^\dag dG D)
    tmp = linalg.mult_src_by_blocks(
        G_Delta.T.conj(), inv_cov.src_mat, cov.edges
    )
    tmp2 = linalg.mult_src_by_blocks(
        dG_Delta.T.conj(), inv_cov.src_mat, cov.edges
    ).conj()
    trace -= np.sum(tmp * tmp2)

    return 2 * trace.real
