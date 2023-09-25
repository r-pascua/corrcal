import ctypes
import numpy as np
from typing import Sequence
from . import linalg
from . import _cfuncs


def check_parallel(parallel, gpu):
    """
    Ensure that only parallelization or GPU acceleration is requested.

    Parameters
    ----------
    parallel: bool
        Whether to perform the operation in parallel.
    gpu: bool
        Whether to use GPU acceleration.
    """
    if parallel and gpu:
        raise ValueError(
            "CPU parallelization and GPU acceleration cannot be "
            "performed simultaneously."
        )


def build_baseline_array(
    ant_1_array: np.ndarray,
    ant_2_array: np.ndarray,
    antpos: np.ndarray,
    antnums: Sequence,
):
    """Calculate all the baseline vectors for the provided parameters.

    Parameters
    ----------
    ant_1_array
        Array specifying the first antenna in each baseline.
    ant_2_array
        Array specifying the second antenna in each baseline.
    antpos
        Array with shape (Nants, 3) giving the ENU position, in meters, of
        each antenna in the array.
    antnums
        Iterable giving the number of each antenna in the order that the terms
        of ``antpos`` appear.

    Returns
    -------
    baselines
        Array with shape (Nbls, 3) giving all of the baselines for the provided
        parameters. This is calculated using the convention that the baseline
        is formed by subtracting the position of antenna 1 from the position of
        antenna 2, i.e. :math:`b_{ij} = x_j - x_i`.
    """
    ant_1_inds = np.zeros_like(ant_1_array)
    ant_2_inds = np.zeros_like(ant_2_array)
    for i, ant in enumerate(antnums):
        ant_1_inds[ant_1_array == ant] = i
        ant_2_inds[ant_2_array == ant] = i
    return antpos[ant_2_inds] - antpos[ant_1_inds]


# TODO: rewrite these using the new SplitMat/SplitVec formalism
def build_gain_mat(gains, ant_1_inds, ant_2_inds):
    """Build the matrix of products of per-antenna gains."""
    complex_gains = build_complex_gains(gains)
    return complex_gains[ant_1_inds] * complex_gains[ant_2_inds].conj()


def scale_cov_by_gains(cov, gain_mat):
    return linalg.diagmul(gain_mat, linalg.diagmul(cov, gain_mat.conj()))


def build_complex_gains(gains : np.ndarray) -> np.ndarray:
    """Turn split real/imag gain array into complex gains.

    Parameters
    ----------
    gains
        Array of per-antenna gains, arranged so that even elements are the
        real part of the gain and odd elements are the imaginary part.

    Returns
    -------
    complex_gains
        Complex per-antenna gains.
    """
    return gains[::2] + 1j*gains[1::2]


def rephase_to_ant(gains, ant=0):
    """Rephase gains to a reference antenna."""
    if np.iscomplexobj(gains):
        complex_gains = gains.copy()
    else:
        complex_gains = build_complex_gains(gains)
    ref_gain = complex_gains[ant]
    conj_phase = ref_gain.conj() / np.abs(ref_gain)
    rephased_complex_gains = complex_gains * conj_phase
    if np.iscomplexobj(gains):
        # If input is complex, output should be as well
        return rephased_complex_gains
    else:
        rephased_gains = np.zeros(2*complex_gains.size)
        rephased_gains[::2] = rephased_complex_gains.real
        rephased_gains[1::2] = rephased_complex_gains.imag
        return rephased_gains


def comply_shape(mat):
    """Check that the provided array has the right shape."""
    if mat.shape[-1] != mat.shape[-2]:
        raise ValueError("Array is not square!")


def make_small_blocks(
    noise_diag: np.ndarray, diff_mat: np.ndarray, edges: np.ndarray
):
    """Make small blocks for use in inverting the diffuse matrix.

    This routine calculates :math:`\Delta^\dag N^{-1} \Delta` for a diffuse
    matrix that is block-diagonal. It is a thin wrapper around the C-code
    that performs the actual computation.

    Parameters
    ----------
    noise_diag
        Diagonal of the noise variance matrix. The array should consist of
        double precision complex numbers.
    diff_mat
        Diffuse matrix sorted into redundant groups. The rows correspond to
        different baselines (and this is the axis it is sorted along), while
        the columns correspond to different eigenmodes. The array should
        consist of double precision complex numbers.
    edges
        Array specifying the edges of each redundant group. The array should
        consist of 64-bit integers.

    Returns
    -------
    small_blocks
        Array containing the small blocks resulting from the matrix product.
        The array is 3-dimensional; indexing along the zeroth-axis accesses
        blocks for different redundant groups.
    """
    n_eig = diff_mat.shape[-1]
    n_grp = edges.size - 1
    out = np.zeros((n_grp, n_eig, n_eig), dtype=complex)
    _cfuncs.make_all_small_blocks(
        noise_diag.ctypes.data,
        diff_mat.ctypes.data,
        out.ctypes.data,
        edges.ctypes.data,
        n_eig,
        n_grp,
    )
    return out
