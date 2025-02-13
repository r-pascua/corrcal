import numpy as np
import pytest
from corrcal import SparseCov
from corrcal.gridding import make_groups_from_antpos

@pytest.fixture(scope="session")
def n_eig():
    return 3


@pytest.fixture(scope="session")
def n_src():
    return 10


@pytest.fixture(scope="session")
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


@pytest.fixture(scope="session")
def min_bl_len():
    return 10


@pytest.fixture(scope="session")
def min_group_size():
    return 4


@pytest.fixture(scope="session")
def ant_1_array(array_layout, min_bl_len, min_group_size):
    return make_groups_from_antpos(
        antpos=array_layout,
        min_bl_len=min_bl_len,
        min_group_size=min_group_size,
    )[0]


@pytest.fixture(scope="session")
def ant_2_array(array_layout, min_bl_len, min_group_size):
    return make_groups_from_antpos(
        antpos=array_layout,
        min_bl_len=min_bl_len,
        min_group_size=min_group_size,
    )[1]


@pytest.fixture(scope="session")
def edges(array_layout, min_bl_len, min_group_size):
    return make_groups_from_antpos(
        antpos=array_layout,
        min_bl_len=min_bl_len,
        min_group_size=min_group_size,
    )[2]


@pytest.fixture(scope="session")
def bls(array_layout, ant_1_array, ant_2_array):
    antpos = np.array(list(array_layout.values()))
    return antpos[ant_2_array] - antpos[ant_1_array]


@pytest.fixture(scope="session")
def n_bl(ant_1_array):
    return ant_1_array.size


@pytest.fixture(scope="session")
def diff_mat(n_bl, n_eig):
    _diff_mat = np.random.normal(size=(2*n_bl, 2*n_eig))
    _diff_mat[::2,1::2] = 0
    _diff_mat[1::2,::2] = 0
    return _diff_mat


@pytest.fixture(scope="session")
def src_mat(n_bl, n_src):
    return np.random.normal(size=(2*n_bl, n_src))


@pytest.fixture(scope="session")
def noise(n_bl):
    _noise = np.abs(np.random.normal(size=2*n_bl))
    _noise[1::2] = _noise[::2]
    return _noise


@pytest.fixture(scope="session")
def dense_diff_mat(diff_mat, edges):
    n_eig = diff_mat.shape[1]
    _diff_mat = np.zeros((diff_mat.shape[0], n_eig*(edges.size-1)), dtype=float)
    for grp, (start, stop) in enumerate(zip(edges, edges[1:])):
        _diff_mat[start:stop,grp*n_eig:(grp+1)*n_eig] = diff_mat[start:stop]
    return _diff_mat


@pytest.fixture(scope="session")
def cov(noise, diff_mat, src_mat, edges, n_eig):
    return SparseCov(
        noise=noise,
        diff_mat=diff_mat,
        src_mat=src_mat,
        edges=edges,
        n_eig=2*n_eig,
    )


@pytest.fixture(scope="session")
def gains(array_layout):
    n_ants = len(array_layout)
    return np.random.normal(size=2*n_ants)


@pytest.fixture(scope="session")
def gain_mat(gains, ant_1_array, ant_2_array):
    complex_gains = gains[::2] + 1j*gains[1::2]
    return complex_gains[ant_1_array] * complex_gains[ant_2_array].conj()



@pytest.fixture(scope="session")
def data(ant_1_array):
    return np.random.normal(size=2*ant_1_array.size)


@pytest.fixture(scope="session")
def p(cov, gains, ant_1_array, ant_2_array, data):
    cinv = cov.copy()
    cinv.apply_gains(gains, ant_1_array, ant_2_array)
    cinv = cinv.inv(return_det=False)
    return cinv @ data


@pytest.fixture(scope="session")
def q(cov, gain_mat, p):
    cov = cov.copy()
    cov.noise = np.zeros_like(cov.noise)
    _q = p.copy()
    _q[::2] = gain_mat.real*p[::2] + gain_mat.imag*p[1::2]
    _q[1::2] = -gain_mat.imag*p[::2] + gain_mat.real*p[1::2]
    return cov @ _q


@pytest.fixture(scope="session")
def s(p, q):
    return p[::2]*q[::2] + p[1::2]*q[1::2]


@pytest.fixture(scope="session")
def t(p, q):
    return p[1::2]*q[::2] - p[::2]*q[1::2]


@pytest.fixture(scope="session")
def P(cov, gains, ant_1_array, ant_2_array):
    cinv = cov.copy()
    cinv.apply_gains(gains, ant_1_array, ant_2_array)
    cinv = cinv.inv(return_det=False)
    return np.sum(
        cinv.diff_mat[::2]**2 + cinv.diff_mat[1::2]**2, axis=1
    ) + np.sum(
        cinv.src_mat[::2]**2 + cinv.src_mat[1::2]**2, axis=1
    )
