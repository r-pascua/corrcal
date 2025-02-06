import numpy as np
import pytest
from corrcal import SparseCov

@pytest.fixture
def n_eig():
    return 3


@pytest.fixture
def n_src():
    return 10


@pytest.fixture
def n_bl():
    return 50


@pytest.fixture
def edges(n_bl):
    _edges = np.unique(np.random.randint(1, n_bl-1, 8)).astype(int)
    return 2*np.concatenate(([0,], _edges, [n_bl,]))


@pytest.fixture
def diff_mat(n_bl, n_eig):
    _diff_mat = np.random.normal(size=(2*n_bl, 2*n_eig))
    _diff_mat[::2,1::2] = 0
    _diff_mat[1::2,::2] = 0
    return _diff_mat


@pytest.fixture
def src_mat(n_bl, n_src):
    return np.random.normal(size=(2*n_bl, n_src))


@pytest.fixture
def noise(n_bl):
    return np.abs(np.random.normal(size=2*n_bl))


@pytest.fixture
def dense_diff_mat(diff_mat, edges):
    n_eig = diff_mat.shape[1]
    _diff_mat = np.zeros((diff_mat.shape[0], n_eig*(edges.size-1)), dtype=float)
    for grp, (start, stop) in enumerate(zip(edges, edges[1:])):
        _diff_mat[start:stop,grp*n_eig:(grp+1)*n_eig] = diff_mat[start:stop]
    return _diff_mat


@pytest.fixture
def cov(noise, diff_mat, src_mat, edges, n_eig):
    return SparseCov(
        noise=noise,
        diff_mat=diff_mat,
        src_mat=src_mat,
        edges=edges,
        n_eig=2*n_eig,
    )
