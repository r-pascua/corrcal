import numpy as np
import pytest
from corrcal import SparseCov
from corrcal.gridding import make_groups_from_antpos

@pytest.fixture
def n_eig():
    return 3


@pytest.fixture
def n_src():
    return 10


@pytest.fixture
def array_layout():
    n_rows = 5
    n_cols = 4
    dx = 7
    dy = 6
    antpos = {}

    ant = 0
    for row in range(n_rows):
        for col in range(n_cols):
            antpos[ant] = np.array([dx*col, dy*row, 0])
            ant += 1
    return antpos


@pytest.fixture
def min_bl_len():
    return 10


@pytest.fixture
def min_group_size():
    return 4


@pytest.fixture
def ant_1_array(array_layout, min_bl_len, min_group_size):
    return make_groups_from_antpos(
        antpos=array_layout,
        min_bl_len=min_bl_len,
        min_group_size=min_group_size,
    )[0]


@pytest.fixture
def ant_2_array(array_layout, min_bl_len, min_group_size):
    return make_groups_from_antpos(
        antpos=array_layout,
        min_bl_len=min_bl_len,
        min_group_size=min_group_size,
    )[1]


@pytest.fixture
def edges(array_layout, min_bl_len, min_group_size):
    return make_groups_from_antpos(
        antpos=array_layout,
        min_bl_len=min_bl_len,
        min_group_size=min_group_size,
    )[2]


@pytest.fixture
def bls(array_layout, ant_1_array, ant_2_array):
    antpos = np.array(list(array_layout.values()))
    return antpos[ant_2_array] - antpos[ant_1_array]


@pytest.fixture
def n_bl(ant_1_array):
    return ant_1_array.size


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
