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
            _cfuncs.cholesky(mat.ctypes.data, chol.ctypes.data, mat.shape[0])
    else:
        if inplace:
            _cfuncs.many_chol_inplace(
                mat.ctypes.data, mat.shape[-1], mat.shape[0]
            )
        else:
            _cfuncs.many_chol(
                mat.ctypes.data, chol.ctypes.data, mat.shape[-1], mat.shape[0]
            )
    if not inplace:
        return chol


def block_multiply(
    diff_mat: np.ndarray, blocks: np.ndarray, edges: np.ndarray
):
    r"""Helper function for diffuse matrix inversion routine.

    This routine calculates :math:`N^{-1} \Delta {L_\Delta}^{-1\dagger}` as
    part of the diffuse matrix "inversion" routine in the case where the
    diffuse matrix is block-diagonal. It is a thin wrapper around the C-code
    that performs the actual computation.

    Parameters
    ----------
    diff_mat
        Diffuse matrix, sorted by redundant groups. Should be an array of
        double precision complex numbers.
    blocks
        Array of many small square matrices. Should represent the block-
        diagonal entries in the Cholesky decomposition of the small matrix
        :math:`1 \pm \Delta^\dagger N^{-1} \Delta` that is computed during
        the first application of the Woodbury identity.
    edges
        Array specifying the edges of each redundant group. The array should
        consist of 64-bit integers.
        
    Returns
    -------
    out
        Product of the diffuse matrix and small blocks--this is the "inverse"
        of the diffuse matrix.
    """
    out = np.zeros_like(diff_mat)
    _cfuncs.block_multiply(
        blocks.ctypes.data,
        diff_mat.ctypes.data,
        out.ctypes.data,
        edges.ctypes.data,
        diff_mat.shape[-1],
        edges.size - 1,
    )
    return out
    

def mult_diff_mats(
    diff_mat_H: np.ndarray, inv_diff_mat: np.ndarray, edges: np.ndarray
):
    r"""Helper function for trace computation.
    
    This routine calculates :math:`\Delta^\dag \Delta'` as part of the trace
    computation routine.

    Parameters
    ----------
    diff_mat_H
        Hermitian conjugate of the diffuse matrix.
    inv_diff_mat
        "Inverse" of the diffuse matrix.
    edges
        Array specifying the edges of each redundant group.

    Returns
    -------
    out
        Product of the Hermitian conjugate of the diffuse matrix and the
        "inverse" diffuse matrix.
    """
    n_bl, n_eig = inv_diff_mat.shape
    n_grp = edges.size - 1
    out = np.zeros((n_grp, n_eig, n_eig), dtype=float)
    _cfuncs.mult_diff_mats(
        diff_mat_H.ctypes.data,
        inv_diff_mat.ctypes.data,
        out.ctypes.data,
        edges.ctypes.data,
        n_bl,
        n_eig,
        n_grp,
    )
    return out


def mult_src_by_blocks(
    blocks_H: np.ndarray, src_mat: np.ndarray, edges: np.ndarray
):
    r"""Prepare the source matrix for inversion.

    This routine calculates :math:`{\Delta'}^\dag \Sigma` as part of the
    source matrix "inversion" step in the case where the diffuse matrix is
    block-diagonal. It is a thin wrapper around the C-code that performs the
    actual computation.

    Parameters
    ----------
    blocks_H
        Hermitian conjugate of the "inverse" diffuse matrix. This should be
        an array of double precision floats.
    src_mat
        Source matrix, sorted by redundant groups. This should be an array of
        double precision floats.
    edges
        Array specifying the edges of each redundant group. The array should
        consist of 64-bit integers.
        
    Returns
    -------
    out
        Product of the transpose of the "inverse" diffuse matrix and the
        source matrix.
    """
    n_eig = blocks_H.shape[0]
    n_grp = edges.size - 1
    n_bl = edges[-1]
    n_src = src_mat.shape[-1]
    out = np.zeros((n_eig*n_grp, n_src), dtype=float)
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


def mult_src_blocks_by_diffuse(inv_diff_mat, src_blocks, edges):
    """Compute the inverse diffuse covariance times the source matrix.

    This function computes the product :math:`\Delta'\Delta'^\dag\Sigma` when
    provided with :math:`\Delta'` and :math:`\Delta'^\dag\Sigma` (as well as
    the redundant group edges) as inputs. It is required in two different
    parts of the second application of the Woodbury identity.

    Parameters
    ----------
    inv_diff_mat
        The "inverse" of the diffuse matrix.
    src_blocks
        The product of the Hermitian conjugate of the "inverse" diffuse matrix
        and the source matrix.
    edges
        Array specifying the edges of each redundant group. The array should
        consist of 64-bit integers.
        
    Returns
    -------
    out
        Product of the inverse diffuse covariance and the source matrix.
    """
    n_src = src_blocks.shape[-1]
    n_grp = edges.size - 1
    n_bls, n_eig = inv_diff_mat.shape
    out = np.zeros((n_bls, n_src), dtype=float)
    _cfuncs.mult_src_blocks_by_diffuse(
        inv_diff_mat.ctypes.data,
        src_blocks.ctypes.data,
        out.ctypes.data,
        edges.ctypes.data,
        n_src,
        n_eig,
        n_grp,
    )
    return out


def sparse_cov_times_vec(sparse_cov, vec):
    """Multiply a vector by a sparse covariance matrix.
    
    Parameters
    ----------
    sparse_cov
        Object containing all of the sparse covariance information.
    vec
        Vector to multiply by the covariance.

    Returns
    -------
    out
        Product of the covariance matrix and the provided vector.
    """
    out = np.zeros_like(vec)
    _cfuncs.sparse_cov_times_vec(
        sparse_cov.noise.ctypes.data,
        sparse_cov.diff_mat.ctypes.data,
        sparse_cov.src_mat.ctypes.data,
        sparse_cov.n_bls,
        sparse_cov.n_eig,
        sparse_cov.n_src,
        sparse_cov.n_grp,
        sparse_cov.edges.ctypes.data,
        sparse_cov.isinv,
        vec.ctypes.data,
        out.ctypes.data,
    )
    return out
