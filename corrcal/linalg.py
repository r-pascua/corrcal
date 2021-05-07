"""
Module containing various linear algebra tools.
"""
import numpy as np
from . import cfuncs
from . import utils


# TODO: fix this; it doesn't actually operate on a Sparse2Level
# Or at least, this implementation is *wrong* since the
# Sparse2Level object does not have a shape attribute.
# TODO: make this not be in-place, or make that optional
def cholesky(matrix, parallel=False, gpu=False):
    """
    Perform Cholesky factorization on a matrix.

    Parameters
    ----------
    matrix: TBD
        Matrix to be factorized. Matrix is factored in-place.
    parallel: bool, optional
        Whether to perform the computation in parallel. Default is to
        perform the calculation in serial.
    gpu: bool, optional
        Whether to use GPU acceleration to calculate the factorization.
        Currently not implemented.
    """
    raise NotImplementedError(
        "Unused in public branch. Documentation there is incorrect."
    )
    utils.check_parallel(parallel, gpu)
    if parallel:
        cfuncs.cholesky_factorization_parallel(
            matrix.ctypes.data, *matrix.shape[::-1]
        )
    elif gpu:
        raise NotImplementedError
    else:
        cfuncs.cholesky_factorization(matrix.ctypes.data, matrix.shape[0])


# TODO: figure out how to make this work. It has the same issue
# as the cholesky decomposition function.
def tri_inv(matrix, parallel=False, gpu=False):
    """
    Invert a (upper? lower?) triangular matrix.

    Parameters
    ----------
    matrix: TBD
        Matrix to be inverted.
    parallel: bool, optional
        Whether to perform the operation in parallel. Default is to
        perform the operation in serial.
    gpu: bool, optional
        Whether to use GPU acceleration. Currently not implemented.

    Returns
    -------
    inverse: TBD
        Inverse of the provided matrix.
    """
    raise NotImplementedError("Needs to be fixed.")
    utils.check_parallel(parallel, gpu)
    inverse = 0 * matrix
    if parallel and matrix.ndim > 2:
        cfuncs.many_tri_inv_c(
            matrix.ctypes.data, inverse.ctypes.data, *matrix.shape[::-1]
        )
    elif gpu:
        raise NotImplementedError
    else:
        cfuncs.tri_inv_c(
            matrix.ctypes.data, inverse.ctypes.data, matrix.shape[0]
        )
    return inverse


def multiply(left, right):
    """
    Perform matrix multiplication left @ right.

    Parameters
    ----------
    left: TBD
        Left-hand matrix in the product.
    right: TBD
        Right-hand matrix in the product.

    Returns
    -------
    product: TBD
        Product of ``left`` and ``right`` matrices.
    """
    if left.shape[1] != right.shape[0]:
        raise ValueError("Matrices cannot be multiplied.")
    # TODO: figure out a way to check data types are correct?
    n, k = left.shape
    k, m = right.shape
    product = np.zeros((n, m))
    cfuncs.mymatmul_c(
        left.ctypes.data,
        k,
        right.ctypes.data,
        m,
        n,
        m,
        k,
        product.ctypes.data,
        m,
    )
    return product


def block_multiply(vectors, blocks, edges):
    """
    Figure out exactly what this does.

    Parameters
    ----------
    vectors: np.ndarray
        ???
    blocks: ??
        ???
    edges: array-like of int
        ???

    Returns
    -------
    product: np.ndarray
        ???
    """
    product = np.zeros_like(vectors)
    nblocks = len(edges) - 1
    edges = np.asarray(edges, dtype="int64")
    cfuncs.mult_vecs_by_blocs(
        vectors.ctypes.data,
        blocks.ctypes.data,
        *product.shape[::-1],
        nblocks,
        edges.ctypes.data,
        product.ctypes.data
    )
    return product
