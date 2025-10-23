import numpy as np
import warnings

from numpy.typing import NDArray
from .sparse import SparseCov
from . import _cfuncs
from . import linalg
from . import utils


def nll(
    gains: NDArray[float],
    cov: SparseCov,
    data: NDArray[float],
    ant_1_inds: NDArray[int],
    ant_2_inds: NDArray[int],
    scale: float = 1,
    phs_norm_fac: float = np.inf,
) -> float:
    """Calculate the CorrCal negative log-likelihood.

    Evaluates Equation 61 from Pascua+ 2025,

    .. math::

        -\log\mathcal{L} = \log{\rm det} \mathbf{C}
        + \mathbfit{d}^T \mathbf{C}^{-1} \mathbfit{d}
        + N_{\rm ant}^{-2} \sigma_{\rm phs}^{-2} \biggl\sum_a \phi_a\biggr|^2.

    The steps followed by this function are discussed in Section 4.2.1 of
    Pascua+ 2025. Essentially, it inverts the covariance (and picks up the
    log-determinant in the process), then computes chi-squared and the phase
    normalization.

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
    phs_norm = np.mean(phases)**2 / phs_norm_fac**2
    return np.real(chisq) + logdet + phs_norm


def grad_nll(
    gains: NDArray[float],
    cov: SparseCov,
    data: NDArray[float],
    ant_1_inds: NDArray[int],
    ant_2_inds: NDArray[int],
    scale: float = 1,
    phs_norm_fac: float = np.inf,
) -> NDArray[float]:
    """Calculate the gradient of the CorrCal negative log-likelihood.

    This function computes the gradient of the negative log-likelihood with
    respect to the real/imaginary parts of the gains, as given in Equation 71
    of Pascua+ 2025,

    .. math::

        -\partial\log\mathcal{L} = {\rm Tr}\bigl(
            \mathbf{C}^{-1} \partial \mathbf{C}
        \bigr) + \mathbfit{d}^T \partial\mathbf{C}^{-1} \mathbfit{d}
        - \partial\log\mathcal{L}_\phi,

    where :math:`\mathcal{L}_\phi` is the phase normalization prior. This
    function follows the steps described in Section 4.2.2 of Pascua+ 2025.

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
    grad_nll
        Gradient of the negative log-likelihood.
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
    noise = cov.noise
    cov = cov.copy()
    cov.noise = np.zeros_like(cov.noise)
    
    # Compute q = (C-N) @ G.T @ p.
    q = p.copy()
    q[::2] = gain_mat.real*p[::2] + gain_mat.imag*p[1::2]
    q[1::2] = -gain_mat.imag*p[::2] + gain_mat.real*p[1::2]
    q = cov @ q

    # Now compute s = Re(q.conj() * p), t = Im(q.conj() * p).
    s = p[::2]*q[::2] + p[1::2]*q[1::2]
    t = p[1::2]*q[::2] - p[::2]*q[1::2]

    # Compute the "inverse power" for use in the trace calculation.
    inv_power = np.sum(
        cinv.diff_mat[::2]**2 + cinv.diff_mat[1::2]**2, axis=1
    ) + np.sum(
        cinv.src_mat[::2]**2 + cinv.src_mat[1::2]**2, axis=1
    )

    gradient = accumulate_gradient(
        gains, s, t, inv_power, noise, ant_1_inds, ant_2_inds
    )

    # Accumulate the contributions from the phase normalization.
    amps = np.sqrt(gains[::2]**2 + gains[1::2]**2)
    phases = np.arctan2(gains[1::2], gains[::2])
    n_ants = complex_gains.size
    grad_phs_prefac = 2 * np.sum(phases) / (amps * n_ants**2 * phs_norm_fac**2)
    gradient[::2] -= grad_phs_prefac * np.sin(phases)
    gradient[1::2] += grad_phs_prefac * np.cos(phases)


    return gradient / scale


def accumulate_gradient(
    gains: NDArray[float],
    s: NDArray[float],
    t: NDArray[float],
    inv_P: NDArray[float],
    noise: NDArray[float],
    ant_1_inds: NDArray[int],
    ant_2_inds: NDArray[int],
) -> NDArray[float]:
    """Accumulate gradient components by looping over baselines.

    This function computes the sums in Equations 79 and 89 of Pascua+ 2025,
    provided the gains and appropriate auxiliary vectors. The chi-squared
    gradient is accumulated via

    .. math::
        
        \mathbfit{d}^T \partial \mathbf{C}^{-1} \mathbfit{d} = -2 \sum_k
            s_k \partial G_k^R + t_k \partial G_k^I,

    while the gradient of the log-determinant is accumulated via

    .. math::

        {\rm Tr}\bigl(\mathbf{C}^{-1} \partial \mathbfit{C}\bigr) = 2 \sum_k
            \frac{\sigma_k^2 \bar{P}_k}{|G_k|^2} \Bigl( G_k^R \partial G_k^R +
            G_k^I \partial G_k^I \Bigr).


    Parameters
    ----------
    gains
        Per-antenna gains, alternating real/imaginary so that the even indices
        correspond to the real part and the odd indices correspond to the
        imaginary part.
    s, t
        Auxiliary vectors formed from the covariance, data, and gains. See
        Section 4.2.2 of Pascua+ 2025 for details.
    inv_P
        Total ``sky power'' in the inverse covariance for each baseline.
    noise
        Thermal noise variance.
    ant_1_inds, ant_2_inds
        Index arrays mapping baseline indices to antenna indices.

    Returns
    -------
    grad_nll
        Gradient of the negative log-likelihood (excluding phase prior).

    Notes
    -----
    This is just a thin wrapper around the accumulate_gradient C function.
    """
    gradient = np.zeros_like(gains)
    n_bls = ant_1_inds.size
    _cfuncs.accumulate_gradient(
        gains.ctypes.data,
        s.ctypes.data,
        t.ctypes.data,
        inv_P.ctypes.data,
        noise.ctypes.data,
        gradient.ctypes.data,
        ant_1_inds.ctypes.data,
        ant_2_inds.ctypes.data,
        n_bls,
    )
    return gradient
