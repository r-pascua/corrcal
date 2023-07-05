import pytest
import numpy as np
from corrcal import linalg

def mock_data(shape):
    return np.random.normal(size=shape) + 1j*np.random.normal(size=shape)
    

@pytest.fixture
def edges():
    return np.array([0,3,5,9,15]).astype(int)


@pytest.fixture
def n_bl():
    return 50


@pytest.fixture
def n_eig():
    return 3


@pytest.fixture
def n_src():
    return 10


@pytest.fixture
def diff_mat(n_bl, n_eig):
    return mock_data((n_bl, n_eig))


@pytest.fixture
def src_mat(n_bl, n_src):
    return mock_data((n_bl, n_src))


@pytest.fixture
def noise(n_bl):
    return np.abs(mock_data(n_bl)).astype(complex)


def test_make_small_blocks(edges, n_bl, n_eig, noise, diff_mat):
    n_grp = edges.size - 1
    dense_diff_mat = np.zeros((n_bl, n_eig*n_grp), dtype=complex)
    for grp in range(n_grp):
        bl_slice = slice(edges[grp], edges[grp+1])
        eig_slice = slice(grp*n_eig, (grp+1)*n_eig)
        dense_diff_mat[bl_slice,eig_slice] = diff_mat[bl_slice]

    N_inv = np.diag(1/noise)
    answer = dense_diff_mat.T.conj() @ N_inv @ dense_diff_mat
    sparse_product = linalg.make_small_blocks(noise, diff_mat, edges)
    dense_product = np.zeros((n_eig*n_grp, n_eig*n_grp), dtype=complex)
    for grp in range(n_grp):
        sl = slice(grp*n_eig, (grp+1)*n_eig)
        dense_product[sl,sl] = sparse_product[grp]
    
    assert np.allclose(dense_product, answer)
