"""
Module containing various linear algebra tools.
"""
import numpy as np
import ctypes
from . import _cfuncs
from . import utils

# no-ops to keep things from crashing on import
def SplitVec(foo):
    pass

def SplitMat(foo):
    pass

def tril_inv(mat: np.ndarray):
    """Invert a lower triangular matrix.

    Parameters
    ----------
    mat
        Matrix to invert. If the provided array is 2-dimensional, then it
        must be square. If it is 3-dimensional, then it is interpreted as
        a collection of square matrices to be inverted in parallel.

    Returns
    -------
    inv
        Inverse of the provided matrix.
    """
    utils.comply_shape(mat)
    inv = np.zeros_like(mat)
    if mat.ndim == 2:
        _cfuncs.tril_inv(mat.ctypes.data, inv.ctypes.data, mat.shape[0])
    else:
        _cfuncs.many_tril_inv(
            mat.ctypes.data, inv.ctypes.data, mat.shape[-1], mat.shape[0]
        )
    return inv


def cholesky(mat: np.ndarray, inplace: bool = False):
    """Take the Cholesky decomposition of a matrix.

    Parameters
    ----------
    mat
        Matrix to decompose. If the provided array is 2-dimensional, then it
        must be square. If it is 3-dimensional, then it is interpreted as
        a collection of square matrices to be decomposed in parallel.

    Returns
    -------
    chol
        Cholesky decomposition of the provided matrix.

    Notes
    -----
    Some testing suggests that this routine is slower than the routine
    provided by numpy, so it is recommended to instead use numpy for Cholesky
    decomposition.
    """
    utils.comply_shape(mat)
    if not inplace:
        chol = np.zeros_like(mat)
    if mat.ndim == 2:
        if inplace:
            _cfuncs.cholesky_inplace(mat.ctypes.data, mat.shape[0])
        else:
            _cfuncs.cholesky(mat.ctypes.data, out.ctypes.data, mat.shape[0])
    else:
        if inplace:
            _cfuncs.many_chol_inplace(
                mat.ctypes.data, mat.shape[-1], mat.shape[0]
            )
        else:
            _cfuncs.many_chol(
                mat.ctypes.data, out.ctypes.data, mat.shape[-1], mat.shape[0]
            )
    if not inplace:
        return out


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


def mult_src_by_blocks(
    blocks_H: np.ndarray, src_mat: np.ndarray, edges: np.ndarray
):
    """Prepare the source matrix for inversion.

    This routine calculates :math:`{\Delta'}^\dag \Sigma` as part of the
    source matrix "inversion" step in the case where the diffuse matrix is
    block-diagonal. It is a thin wrapper around the C-code that performs the
    actual computation.

    Parameters
    ----------
    blocks_H
        Hermitian conjugate of the "inverse" diffuse matrix. This should be a
        3-dimensional array of double precision complex numbers, with the
        zeroth axis indexing over redundant groups.
    src_mat
        Source matrix, sorted by redundant groups. This should be an array of
        double precision complex numbers.
    edges
        Array specifying the edges of each redundant group. The array should
        consist of 64-bit integers.
        
    Returns
    -------
    out
        Product of the Hermitian conjugate of the "inverse" diffuse matrix and
        the source matrix.
    """
    n_eig = blocks_H.shape[-1]
    n_grp = edges.size - 1
    n_bl = edges[-1]
    n_src = src_mat.shape[-1]
    out = np.zeros_like(src_mat)
    _cfuncs.mult_src_by_blocks(
        blocks_H.ctypes.data,
        src_mat.ctypes.data,
        out.ctypes.data,
        edges.ctypes.data,
        n_bl,
        n_src,
        n_eig,
        n_grp,
    )
    return out
