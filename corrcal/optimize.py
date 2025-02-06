import numpy as np
import warnings

from . import _cfuncs
from . import linalg
from . import sparse
from . import utils


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
    phs_norm = np.mean(phases)**2 / phs_norm_fac**2
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


def accumulate_gradient(gains, s, t, P, noise, ant_1_inds, ant_2_inds):
    """Loop over baselines and accumulate the per-antenna gradient contribs.

    Thin wrapper around the accumulate_gradient C function.

    Parameters
    ----------
    TODO

    Returns
    -------
    TODO
    """
    gradient = np.zeros_like(gains)
    n_bls = ant_1_inds.size
    _cfuncs.accumulate_gradient(
        gains.ctypes.data,
        s.ctypes.data,
        t.ctypes.data,
        P.ctypes.data,
        noise.ctypes.data,
        gradient.ctypes.data,
        ant_1_inds.ctypes.data,
        ant_2_inds.ctypes.data,
        n_bls,
    )
    return gradient
