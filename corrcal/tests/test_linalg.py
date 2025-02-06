import pytest
import numpy as np
from corrcal import linalg

def test_make_small_blocks(noise, diff_mat, dense_diff_mat, edges, n_eig):
    small_blocks = linalg.make_small_blocks(
        noise_diag=noise, diff_mat=diff_mat, edges=edges
    )
    dense_small_blocks = dense_diff_mat.T @ np.diag(1/noise) @ dense_diff_mat
    blocks = np.zeros((edges.size-1, 2*n_eig, 2*n_eig), dtype=float)
    for grp in range(edges.size-1):
        sl = slice(2*grp*n_eig, 2*(grp+1)*n_eig)
        blocks[grp] = dense_small_blocks[sl,sl]
    assert np.allclose(blocks, small_blocks)


def test_tril_inv(noise, diff_mat, edges):
    blocks = np.eye(diff_mat.shape[1])[None,...] + linalg.make_small_blocks(
        noise_diag=noise, diff_mat=diff_mat, edges=edges
    )
    blocks = np.linalg.cholesky(blocks)
    inv_blocks = np.linalg.inv(blocks)
    assert np.allclose(linalg.tril_inv(blocks), inv_blocks)


def test_block_multiply(noise, diff_mat, dense_diff_mat, edges, n_eig):
    small_blocks = linalg.make_small_blocks(
        noise_diag=noise, diff_mat=diff_mat, edges=edges
    )
    N_inv = 1 / noise
    dense_blocks = dense_diff_mat.T @ (N_inv[:,None] * dense_diff_mat)
    dense_prod = dense_diff_mat @ dense_blocks
    _sparse_prod = linalg.block_multiply(diff_mat, small_blocks, edges)
    sparse_prod = np.zeros_like(dense_prod)
    for grp, (start, stop) in enumerate(zip(edges, edges[1:])):
        sl = slice(2*grp*n_eig, 2*(grp+1)*n_eig)
        sparse_prod[start:stop,sl] = _sparse_prod[start:stop]
    assert np.allclose(sparse_prod, dense_prod)
    
    
def test_sparse_cov_times_vec(cov):
    data = np.random.normal(size=cov.diff_mat.shape[0])
    assert np.allclose(cov.expand() @ data, cov @ data)


def test_sum_diags(noise, diff_mat, edges):
    blocks = np.eye(diff_mat.shape[1])[None,...] + linalg.make_small_blocks(
        noise_diag=noise, diff_mat=diff_mat, edges=edges
    )
    blocks = np.linalg.cholesky(blocks)
    logdet_blocks = 0
    for block in blocks:
        logdet_blocks += 2 * np.log(np.diag(block)).sum()
    assert np.isclose(logdet_blocks, linalg.sum_diags(blocks))
