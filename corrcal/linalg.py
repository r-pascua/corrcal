"""
Module containing various linear algebra tools.
"""
import numpy as np
from numba import njit, prange
from . import cfuncs
from . import utils


def make_lower(n):
    A = np.random.normal(size=(n,n)) * 1j * np.random.normal(size=(n,n))
    return np.linalg.cholesky(A @ A.T.conj())


@njit(parallel=True)
def matmul(a, b, c):
    an, am = a.shape
    bn, bm = b.shape
    for i in range(an):
        for j in range(bm):
            s = 0
            for k in prange(bn):
                s += a[i,k] * b[k,j]
            c[i,j] = s


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
    Nvec, Nmodes = vectors.shape
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

def many_tri_inv(mat):
    inv = np.zeros_like(mat)
    if mat.ndim == 2:
        cfuncs.tri_inv_c(
            mat.ctypes.data, inv.ctypes.data, mat.shape[0]
        )
        return inv

    # TODO: convert this to numba-fied python code
    Nmat = mat.shape[0]
    n = mat.shape[1]
    cfuncs.many_tri_inv_c(
        mat.ctypes.data, inv.ctypes.data, n, Nmat
    )
    return inv
