"""
Module containing various linear algebra tools.

Note: it looks like trying to numba-fy the C-code results in functions that
are generally slower than the numpy linear algebra tools.
"""
import numpy as np
from numba import njit, prange
from typing import Sequence
from . import _cfuncs
from . import utils


def make_lower(n):
    A = np.random.normal(size=(n, n)) * 1j * np.random.normal(size=(n, n))
    return np.linalg.cholesky(A @ A.T.conj())


@njit
def _diagmul(mat, diag):
    """Multiply a matrix by a diagonal matrix from the left."""
    out = np.zeros_like(mat)
    for i in prange(mat.shape[0]):
        out[i] = mat[i] * diag[i]
    return out


def diagmul(left, right):
    """Matrix multiplication with one matrix diagonal."""
    if left.ndim == 1:
        return _diagmul(right, left)
    else:
        return _diagmul(left.T, right).T
