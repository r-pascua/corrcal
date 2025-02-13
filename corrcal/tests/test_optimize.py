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


def test_grad_logdet(
    cov, ant_1_array, ant_2_array, gains
):
    # Same deal as previous test; analytic gradient, then numerical.
    c = cov.copy()
    c.apply_gains(gains, ant_1_array, ant_2_array)
    c = c.inv(return_det=False)
    P = np.sum(c.diff_mat[::2]**2 + c.diff_mat[1::2]**2, axis=1)
    P += np.sum(c.src_mat[::2]**2 + c.src_mat[1::2]**2, axis=1)
    zeros = np.zeros_like(P)
    analytic_grad = optimize.accumulate_gradient(
        gains, zeros, zeros, P, cov.noise, ant_1_array, ant_2_array
    ) 

    # Now compute numerically.
    numerical_grad = np.zeros_like(analytic_grad)
    dg = 1e-4
    for i in range(gains.size):
        g_up = gains.copy()
        g_down = gains.copy()
        g_up[i] += dg
        g_down[i] -= dg
        cov_up = cov.copy()
        cov_down = cov.copy()
        cov_up.apply_gains(g_up, ant_1_array, ant_2_array)
        cov_down.apply_gains(g_down, ant_1_array, ant_2_array)
        logdet_up = cov_up.inv(return_det=True)[1]
        logdet_down = cov_down.inv(return_det=True)[1]
        numerical_grad[i] = (logdet_up - logdet_down) / (2*dg)
    assert np.allclose(analytic_grad, numerical_grad)


def test_grad_phs_norm(gains, cov, ant_1_array, ant_2_array, data):
    phs_norm_fac = 37
    on_args = dict(
        cov=cov,
        data=data,
        ant_1_inds=ant_1_array,
        ant_2_inds=ant_2_array,
        phs_norm_fac=phs_norm_fac,
    )
    off_args = on_args.copy()
    off_args["phs_norm_fac"] = np.inf  # No phase normalization.
    analytic_grad = optimize.grad_nll(gains, **on_args) 
    analytic_grad -= optimize.grad_nll(gains, **off_args)

    numerical_grad = np.zeros_like(gains)
    dg = 1e-4
    for i in range(gains.size):
        g_up = gains.copy()
        g_down = gains.copy()
        g_up[i] += dg
        g_down[i] -= dg
        norm_up = optimize.nll(g_up, **on_args) 
        norm_up -= optimize.nll(g_up, **off_args)
        norm_down = optimize.nll(g_down, **on_args)
        norm_down -= optimize.nll(g_down, **off_args)
        numerical_grad[i] = (norm_up - norm_down) / (2 * dg)
    assert np.allclose(numerical_grad, analytic_grad)
