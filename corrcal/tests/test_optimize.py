import numpy as np
import pytest
from corrcal import optimize

def test_chisq(cov):
    vis = np.random.normal(size=cov.src_mat.shape[0])
    assert np.isclose(vis @ (cov @ vis), vis @ cov.expand() @ vis)


def test_grad_chisq(
    cov, ant_1_array, ant_2_array, s, t, gains, data
):
    # First, compute grad chisq analytically.
    analytic_grad = optimize.accumulate_gradient(
        gains, s, t, np.zeros_like(s), cov.noise, ant_1_array, ant_2_array
    )

    # Now compute it numerically.
    numerical_grad = np.zeros_like(analytic_grad)
    dg = 1e-4
    for i in range(gains.size):
        g_up = gains.copy()
        g_down = gains.copy()
        g_up[i] += dg
        g_down[i] -= dg
        cinv_up = cov.copy()
        cinv_down = cov.copy()
        cinv_up.apply_gains(g_up, ant_1_array, ant_2_array)
        cinv_down.apply_gains(g_down, ant_1_array, ant_2_array)
        cinv_up = cinv_up.inv(return_det=False)
        cinv_down = cinv_down.inv(return_det=False)
        cinv_up.noise = np.zeros_like(cinv_up.noise)
        cinv_down.noise = np.zeros_like(cinv_down.noise)
        chisq_up = data @ (cinv_up @ data)
        chisq_down = data @ (cinv_down @ data)
        numerical_grad[i] = (chisq_up - chisq_down) / (2 * dg)
    assert np.allclose(analytic_grad, numerical_grad)


def test_grad_logdet(cov, ant_1_array, ant_2_array):
    pass
