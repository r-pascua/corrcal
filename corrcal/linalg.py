"""
Module containing various linear algebra tools.

All of the functions provided in this module are thin wrappers around their
C counterparts.
"""
import numpy as np
from numpy.typing import NDArray
import ctypes
from typing import Type
from . import _cfuncs
from . import utils

def tril_inv(mat: NDArray[float]) -> NDArray[float]:
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


def cholesky(
        mat: NDArray[float], inplace: bool = False
    ) -> NDArray[float] | None:
    """Take the (lower-triangular) Cholesky decomposition of a matrix.

    Parameters
    ----------
    mat
        Matrix to decompose. If the provided array is 2-dimensional, then it
        must be square. If it is 3-dimensional, then it is interpreted as
        a collection of square matrices to be decomposed in parallel.

    inplace
        Whether to replace the provided matrix with its Cholesky factorization
        or return a new object. Returns a new object by default.

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
    diff_mat: NDArray[float], blocks: NDArray[float], edges: NDArray[int]
) -> NDArray[float]:
    r"""Helper function for diffuse matrix inversion routine.

    This routine calculates :math:`N^{-1} \Delta {L_\Delta}^{-1\dagger}` as
    part of the diffuse matrix "inversion" routine in the case where the
    diffuse matrix is block-diagonal. It is a thin wrapper around the C-code
    that performs the actual computation.

    Parameters
    ----------
    diff_mat
        Diffuse matrix, sorted by redundant groups. Should be an array of
        double precision floats.
    blocks
        Array of many small square matrices that represent the block-diagonal
        entries in the Cholesky decomposition of the small matrix
        :math:`1 \pm \Delta^\dagger N^{-1} \Delta` that is computed during
        the first application of the Woodbury identity.
    edges
        Array specifying the edges of each redundant group. The array should
        consist of 64-bit integers.
        
    Returns
    -------
    out
        Product of the diffuse matrix and small blocks.
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
    

def mult_src_by_blocks(
    blocks_T: NDArray[float], src_mat: NDArray[float], edges: NDArray[int]
) -> NDArray[float]:
    r"""Prepare the source matrix for inversion.

    This routine calculates :math:`{\Delta'}^\dag \Sigma` as part of the
    source matrix "inversion" step in the case where the diffuse matrix is
    block-diagonal. It is a thin wrapper around the C-code that performs the
    actual computation.

    Parameters
    ----------
    blocks_T
        Transpose of the "inverse" diffuse matrix. This should be an array of
        double precision floats.
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
    n_eig = blocks_T.shape[0]
    n_grp = edges.size - 1
    n_bl = edges[-1]
    n_src = src_mat.shape[-1]
    out = np.zeros((n_eig*n_grp, n_src), dtype=float)
    _cfuncs.mult_src_by_blocks(
        blocks_T.ctypes.data,
        src_mat.ctypes.data,
        out.ctypes.data,
        edges.ctypes.data,
        n_bl,
        n_src,
        n_eig,
        n_grp,
    )
    return out


def mult_src_blocks_by_diffuse(
        inv_diff_mat: NDArray[float],
        src_blocks: NDArray[float],
        edges: NDArray[int],
    ) -> NDArray[float]:
    """Compute the inverse diffuse covariance times the source matrix.

    This function computes the product
    :math:`\bar{\Delta}\bar{\Delta}^\dag\Sigma` when provided with
    :math:`\bar{\Delta}` and :math:`\bar{\Delta}^\dag\Sigma` (as well as
    the redundant group edges) as inputs. It is required in two different
    parts of the second application of the Woodbury identity.

    Parameters
    ----------
    inv_diff_mat
        The "inverse" of the diffuse matrix.
    src_blocks
        The product of the transpose of the "inverse" diffuse matrix
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
        int(n_src),
        int(n_eig),
        int(n_grp),
    )
    return out


def sparse_cov_times_vec(
        sparse_cov: Type[SparseCov], vec: NDArray[float]
    ) -> NDArray[float]:
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


def make_small_blocks(
    noise_diag: NDArray[float], diff_mat: NDArray[float], edges: NDArray[int]
) -> NDArray[float]:
    """Make small blocks for use in inverting the diffuse matrix.

    This routine calculates :math:`\Delta^\dag N^{-1} \Delta` for a diffuse
    matrix that is block-diagonal. It is a thin wrapper around the C-code
    that performs the actual computation.

    Parameters
    ----------
    noise_diag
        Diagonal of the noise variance matrix. The array should consist of
        double precision floats.
    diff_mat
        Diffuse matrix sorted into redundant groups. The rows correspond to
        different baselines (and this is the axis it is sorted along), while
        the columns correspond to different eigenmodes. The array should
        consist of double precision floats.
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
    out = np.zeros((n_grp, n_eig, n_eig), dtype=float)
    _cfuncs.make_all_small_blocks(
        noise_diag.ctypes.data,
        diff_mat.ctypes.data,
        out.ctypes.data,
        edges.ctypes.data,
        n_eig,
        n_grp,
    )
    return out


def sum_diags(blocks: NDArray[float]) -> float:
    """Helper function for computing log-determinant of covariance."""
    n_grps, n_eig = blocks.shape[:2]
    return _cfuncs.sum_diags(blocks.ctypes.data, int(n_grps), int(n_eig))
