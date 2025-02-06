import pytest
import numpy as np
from corrcal import linalg


def test_compute_inv_diff_mat(diff_mat, dense_diff_mat, noise, edges):
    """Validate the sparse calculation of the "inverse" diffuse matrix."""
    n_eig = diff_mat.shape[1]
    N_inv = 1 / noise

    # Compute the inverse diffuse matrix as it's done in SparseCov.inv
    small_blocks = linalg.make_small_blocks(
        noise_diag=noise, diff_mat=diff_mat, edges=edges
    )
    small_blocks = np.eye(n_eig)[None,...] + small_blocks
    small_blocks = np.linalg.cholesky(small_blocks)
    small_inv = linalg.tril_inv(small_blocks).transpose(0,2,1).copy()
    tmp = N_inv[:,None] * diff_mat
    inv_diff_mat = linalg.block_multiply(tmp, small_inv, edges)
    dense_inv_diff_mat = np.zeros_like(dense_diff_mat)
    for grp, (start, stop) in enumerate(zip(edges, edges[1:])):
        sl = slice(grp*n_eig, (grp+1)*n_eig)
        dense_inv_diff_mat[start:stop,sl] = inv_diff_mat[start:stop]
    
    # Now compute it manually.
    dense_small_blocks = np.eye(
        n_eig * (edges.size-1)
    ) + dense_diff_mat.T @ (N_inv[:,None] * dense_diff_mat)
    inv_diff_mat = N_inv[:,None] * (
        dense_diff_mat @ np.linalg.inv(np.linalg.cholesky(dense_small_blocks)).T
    )
    assert np.allclose(dense_inv_diff_mat, inv_diff_mat)


def test_inv(cov):
    assert np.allclose(np.linalg.inv(cov.expand()), cov.inv().expand())


def test_double_inversion(cov):
    assert np.allclose(cov.expand(), cov.inv().inv().expand())


def test_logdet(cov):
    sparse_logdet = cov.inv(return_det=True)[1]
    L = np.linalg.cholesky(cov.expand())
    dense_logdet = 2*np.log(np.diag(L)).sum() - np.log(cov.noise).sum()
    assert np.isclose(dense_logdet, sparse_logdet)
