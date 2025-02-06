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


def test_block_multiply(diff_mat, dense_diff_mat, edges, n_eig):
    small_blocks = linalg.make_small_blocks(
        noise_diag=noise, diff_mat=diff_mat, edges=edges
    )
    
