"""
Module containing various linear algebra tools.

Note: it looks like trying to numba-fy the C-code results in functions that
are generally slower than the numpy linear algebra tools.
"""
import numpy as np
from numba import njit, prange
from typing import Sequence
from . import cfuncs
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


# TODO: write a SplitBase class to tidy things up
class SplitMat:
    """Class for managing real/imaginary split complex matrices."""

    def __init__(self, mat: np.ndarray):
        n_row, n_col = mat.shape
        self.n_row = n_row
        self.n_col = n_col
        self.data = np.zeros((2 * n_row, 2 * n_col), dtype=float)
        self.data[:n_row, :n_col] = mat.real
        self.data[:n_row, n_col:] = -mat.imag
        self.data[n_row:, :n_col] = mat.imag
        self.data[n_row:, n_col:] = mat.real

    def __add__(self, other):
        out = self.data + other.data
        return SplitMat(
            out[: self.n_row, : self.n_col]
            + 1j * out[self.n_row :, : self.n_col]
        )

    def __matmul__(self, other):
        if isinstance(other, SplitMat):
            out = self.data @ other.data
            return SplitMat(
                out[: self.n_row, : other.n_col]
                + 1j * out[self.n_row :, : other.n_col]
            )
        elif isinstance(other, SplitVec):
            out = self.data @ other.data
            return SplitVec(out[: self.n_row] + 1j * out[self.n_row :])
        else:
            raise NotImplementedError

    def __mul__(self, other):
        return SplitMat(other * (self.real + 1j * self.imag))

    def copy(self):
        return SplitMat(self.real + 1j * self.imag)

    def conj(self):
        return SplitMat(self.real - 1j * self.imag)

    def inv(self):
        out = np.linalg.inv(self.real + 1j * self.imag)
        return SplitMat(out)

    @property
    def T(self):
        return SplitMat(self.real.T + 1j * self.imag.T)

    @property
    def H(self):
        return SplitMat(self.real.T - 1j * self.imag.T)

    @property
    def real(self):
        return self.data[: self.n_row, : self.n_col]

    @property
    def imag(self):
        return self.data[self.n_row :, : self.n_col]


class SplitVec:
    """Class for managing real/imaginary split complex vectors."""

    def __init__(self, vec: np.ndarray):
        self.len = vec.size
        self.data = np.zeros(2 * self.len, dtype=float)
        self.data[: self.len] = vec.real
        self.data[self.len :] = vec.imag

    def __matmul__(self, other):
        if isinstance(other, SplitMat):
            out = self.data @ other.conj().data
            return SplitVec(out[: self.len] + 1j * out[self.len :])
        elif isinstance(other, SplitVec):
            re_out = self.real @ other.real - self.imag @ other.imag
            im_out = self.real @ other.imag + self.imag @ other.real
            return re_out + 1j * im_out
        else:
            raise NotImplementedError

    def conj(self):
        return SplitVec(self.real - 1j * self.imag)

    def copy(self):
        return SplitVec(self.real + 1j * self.imag)

    @property
    def real(self):
        return self.data[: self.len]

    @property
    def imag(self):
        return self.data[self.len :]
