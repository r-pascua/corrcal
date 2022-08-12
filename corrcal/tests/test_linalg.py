import pytest
import numpy as np
from corrcal import linalg


@pytest.fixture
def test_vec():
    return np.random.normal(size=50) + 1j * np.random.normal(size=50)


@pytest.fixture
def test_mat():
    return np.random.normal(size=(50, 50)) + 1j * np.random.normal(
        size=(50, 50)
    )


def test_split_vec_inner_product(test_vec):
    n = test_vec.size
    other_vec = np.random.normal(size=n) + 1j * np.random.normal(size=n)
    ans = test_vec @ other_vec
    test_vec = linalg.SplitVec(test_vec)
    other_vec = linalg.SplitVec(other_vec)
    assert np.isclose(ans, test_vec @ other_vec)


def test_split_matmul(test_mat):
    nn = test_mat.shape
    other_mat = np.random.normal(size=nn) + 1j * np.random.normal(size=nn)
    ans = test_mat @ other_mat
    test_mat = linalg.SplitMat(test_mat)
    other_mat = linalg.SplitMat(other_mat)
    split_ans = test_mat @ other_mat
    assert np.allclose(ans, split_ans.real + 1j * split_ans.imag)


@pytest.mark.parametrize("side", ["left", "right"])
def test_split_vecmul(test_vec, test_mat, side):
    if side == "left":

        def matmul(mat, vec):
            return mat @ vec

    else:

        def matmul(mat, vec):
            return vec @ mat

    ans = matmul(test_mat, test_vec)
    test_vec = linalg.SplitVec(test_vec)
    test_mat = linalg.SplitMat(test_mat)
    split_ans = matmul(test_mat, test_vec)
    assert np.allclose(ans, split_ans.real + 1j * split_ans.imag)
